"""PLRNN family model implementations (Paradigm B reference implementation, demonstrating
analytic Jacobian / analytic fixed-point capabilities).

Ported from the shallowPLRNN:
    z_t = A \odot z_{t-1} + W1 ReLU(W2 z_{t-1} + h2) + h1 (+ C s_t)
Its analytic Jacobian:
    J(z) = diag(A) + W1 diag(1[W2 z + h2 > 0]) W2
The piecewise-linear structure makes fixed points / k-cycles analytically solvable
(see the analytic backend in analysis/fixed_points.py).

This file demonstrates the standard "Contract A + analytic capability" pattern:
  - supports_analytic_fixed_points = True
  - implement analytic jacobian (should match base-class autodiff, except at ReLU boundaries)
  - expose (A, W1, W2, h1, h2) for the analytic fixed-point algorithm in the analysis layer
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.init import uniform_

from ...modeling_utils import NeuralDynamicsModel
from ...auto.modeling_auto import register_model
from .configuration_plrnn import ShallowPLRNNConfig, DendPLRNNConfig, ALRNNConfig


def _make_z0_lift(config, latent_dim: int):
    """Learnable observation->latent lift B (output_dim, latent_dim), init U(±1/sqrt(output_dim))
    (AL-RNN reference init_uniform). Returns None when learn_z0 is disabled."""
    if not getattr(config, "learn_z0", False):
        return None
    n_obs = config.output_dim
    rB = 1.0 / (n_obs ** 0.5)
    return nn.Parameter(uniform_(torch.empty(n_obs, latent_dim), -rB, rB))


@register_model("shallow_plrnn")
class ShallowPLRNNModel(NeuralDynamicsModel):
    """Shallow piecewise-linear RNN (PLRNN) for dynamical systems reconstruction.

    Single-step update (z: latent state (B, M), x: external input (B, K)):

        z_t = A ⊙ z + W1 @ relu(W2 @ z + h2) + h1 [+ C @ x_t]

    with diagonal A (M,), W1 (M, L), W2 (L, M), h1 (M,), h2 (L,), C (M, K).
    The piecewise-linear ReLU nonlinearity makes the Jacobian and fixed points
    analytic (see jacobian / analytic_parameters). Readout observes the first
    output_dim latent units (identity observation; B = [I_N | 0] when M > N).
    """
    config_class = ShallowPLRNNConfig

    def __init__(self, config: ShallowPLRNNConfig) -> None:
        super().__init__(config)
        M, L, K = config.latent_dim, config.hidden_dim, config.input_dim
        r1, r2 = 1.0 / (L ** 0.5), 1.0 / (M ** 0.5)
        self.W1 = nn.Parameter(uniform_(torch.empty(M, L), -r1, r1))
        self.W2 = nn.Parameter(uniform_(torch.empty(L, M), -r2, r2))
        self.A = nn.Parameter(uniform_(torch.empty(M), a=0.5, b=0.9))  # diagonal
        self.h2 = nn.Parameter(uniform_(torch.empty(L), -r1, r1))
        self.h1 = nn.Parameter(torch.zeros(M))
        if config.autonomous:
            self.register_parameter("C", None)
        else:
            r3 = 1.0 / (K ** 0.5)
            self.C = nn.Parameter(uniform_(torch.empty(M, K), -r3, r3))
        self.apply_freeze_config()

    def _freeze_groups(self) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = {
            "recurrent": [r"^A$", r"^W1\.", r"^W2\.", r"^h1$", r"^h2$"],
            "output": [],  # identity readout: no trainable output parameters
            "h0": [],      # default zero initial state, not trainable
        }
        if self.C is not None:
            groups["input"] = [r"^C\."]
        else:
            groups["input"] = []
        return groups

    # ---------------- hard contract ----------------
    def recurrence(self, x_t, z_prev, *, inputs=None):
        """Single PLRNN step. x_t: (B, K) or None, z_prev: (B, M) -> z_t: (B, M)."""
        # PLRNN is defined by a piecewise-linear ReLU nonlinearity; changing it
        # would invalidate the analytic Jacobian / fixed-point machinery below.
        z = self.A * z_prev + torch.relu(z_prev @ self.W2.T + self.h2) @ self.W1.T + self.h1
        if self.C is not None and x_t is not None:
            z = z + x_t @ self.C.T
        return z

    def readout(self, z_t):
        """Identity observation of the first output_dim latent units. (B, M) -> (B, N)."""
        # observation == "identity": directly observe latent state (standard DSR setting).
        # When output_dim < latent_dim (M > N DSR setting), observations are the first
        # output_dim latent units, matching the reference B = [I_N | 0] readout.
        return z_t[..., :self.config.output_dim]

    # ---------------- analytic analysis support ----------------
    @property
    def supports_analytic_fixed_points(self) -> bool:
        """This family has a closed-form Jacobian / fixed-point solver (scy_fi)."""
        return True

    def jacobian(self, z: torch.Tensor, *, inputs=None) -> torch.Tensor:
        """Analytic Jacobian: diag(A) + W1 diag(1[W2 z + h2 > 0]) W2. z:(M,) -> (M,M).
        Should be allclose to the base-class autodiff result (except at ReLU boundaries)."""
        d = (self.W2 @ z > -self.h2).float()           # (L,) indicator vector
        return torch.diag(self.A) + self.W1 @ torch.diag(d) @ self.W2

    def analytic_parameters(self, task_input: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Expose parameters required by the analytic fixed-point solver (scy_fi),
        all returned as numpy-friendly detached tensors.
        The analytic backend in analysis/fixed_points.py reads (A_diag, W1, W2, h1, h2) from here.

        If a constant task_input is provided, it is folded into the effective bias
        ``h1_eff = h1 + C @ task_input`` so that non-autonomous systems can reuse
        the autonomous SCYFI solver.
        """
        h1_eff = self.h1
        if task_input is not None and self.C is not None:
            s = torch.as_tensor(task_input, dtype=self.C.dtype, device=self.C.device)
            if s.dim() > 1:
                s = s.squeeze(0)
            h1_eff = h1_eff + self.C @ s
        return {
            "A": torch.diag(self.A).detach(),     # (M,M) diagonalized, matching original main(np.diag(A),...) expectation
            "W1": self.W1.detach(),
            "W2": self.W2.detach(),
            "h1": h1_eff.detach(),
            "h2": self.h2.detach(),
        }


@register_model("dend_plrnn")
class DendPLRNNModel(NeuralDynamicsModel):
    """dendritic PLRNN with linear spline basis expansion.

    State equation:
        z_t = A z_{t-1}
              + W [sum_b alpha_b ReLU(z_{t-1} - H_b)]^T
              + h + C s_t

    The term in brackets is a (B, M) matrix computed by broadcasting; the
    recurrence follows the BPTT reference implementation in
    dendPLRNN/BPTT_TF/bptt/PLRNN_model.py.
    """
    config_class = DendPLRNNConfig

    def __init__(self, config: DendPLRNNConfig) -> None:
        super().__init__(config)
        M, B, K = config.latent_dim, config.n_bases, config.input_dim
        r = 1.0 / (M ** 0.5)
        if config.init_scheme == "paper":
            # Talathi & Vartak (2016) init from the BPTT-TF reference: a single
            # AW = (I + R^T R / M) / lambda_max, split into A = diag(AW), W = AW - diag(AW).
            R = torch.randn(M, M)
            AW = torch.eye(M) + R.T @ R / M
            AW = AW / torch.linalg.eigvals(AW).abs().max()
            self.W = nn.Parameter(AW - torch.diag(torch.diagonal(AW)))
            self.A = nn.Parameter(torch.diagonal(AW).contiguous())
            self.h = nn.Parameter(uniform_(torch.empty(M), -r, r))
            # alphas ~ U(±1/sqrt(B)) per the paper (Brenner et al. 2022)
            rb = 1.0 / (B ** 0.5)
            self.alphas = nn.Parameter(uniform_(torch.empty(B), -rb, rb))
        else:
            self.W = nn.Parameter(uniform_(torch.empty(M, M), -r, r))
            self.A = nn.Parameter(uniform_(torch.empty(M), a=0.5, b=0.9))
            self.h = nn.Parameter(torch.zeros(M))
            self.alphas = nn.Parameter(uniform_(torch.empty(B), -r, r))
        if config.threshold_range is not None:
            # Thresholds covering the (normalized) data range, matching the reference
            # init_thetas_uniform (sign-flipped: our basis is relu(z - H) = relu(z + theta)).
            mn, mx = config.threshold_range
            self.H = nn.Parameter(uniform_(torch.empty(M, B), float(mn), float(mx)))
        else:
            self.H = nn.Parameter(uniform_(torch.empty(M, B), -r, r))
        if config.autonomous:
            self.register_parameter("C", None)
        else:
            r3 = 1.0 / (K ** 0.5)
            self.C = nn.Parameter(uniform_(torch.empty(M, K), -r3, r3))
        self.register_parameter("B", _make_z0_lift(config, M))
        self.clip_range = config.clip_range
        self.use_clipping = config.use_clipping
        self.apply_freeze_config()

    def _freeze_groups(self) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = {
            "recurrent": [r"^A$", r"^W\.", r"^h$", r"^H\.", r"^alphas$"],
            "output": [],
            "h0": [],
        }
        if self.C is not None:
            groups["input"] = [r"^C\."]
        else:
            groups["input"] = []
        if self.B is not None:
            groups["output"] = [r"^B\."]
        return groups

    def init_state_from_obs(self, x0: torch.Tensor) -> torch.Tensor:
        """Latent z0 from the first observation x0: (B, N) -> (B, M).

        With learn_z0 a trained lift B (N, M) is applied, then the observed
        dims are hard-set to x0 (reference Z0Model); otherwise falls back to
        the base-class identity lift.
        """
        # Reference Z0Model: learned lift, then hard-set the observed dims
        if self.B is not None:
            z = x0 @ self.B
            z[..., :x0.shape[-1]] = x0
            return z
        return super().init_state_from_obs(x0)

    def _basis(self, z: torch.Tensor) -> torch.Tensor:
        """Compute basis expansion sum_b alpha_b ReLU(z - H_b).

        When use_clipping is True, uses the clipped basis expansion
        sum_b alpha_b (ReLU(z - H_b) - ReLU(z)) (reference PLRNN_Clipping_Step),
        which guarantees bounded orbits when ||A||_2 < 1 (paper Theorem 2).

        z: (B, M) -> (B, M)
        """
        z_ = z.unsqueeze(-1)          # (B, M, 1)
        H_ = self.H.unsqueeze(0)      # (1, M, B)
        a_ = self.alphas.view(1, 1, -1)  # (1, 1, B)
        be = a_ * torch.relu(z_ - H_)
        if self.use_clipping:
            be = be - a_ * torch.relu(z_)
        return be.sum(dim=-1)

    def recurrence(self, x_t, z_prev, *, inputs=None):
        """Single dendPLRNN step. x_t: (B, K) or None, z_prev: (B, M) -> z_t: (B, M).

        z_t = A ⊙ z + basis(z) @ Wᵀ + h [+ C @ x_t]; with clip_range and no
        clipping basis, states are hard-clamped to ±clip_range (reference behavior).
        """
        z = self.A * z_prev + self._basis(z_prev) @ self.W.t() + self.h
        if self.C is not None and x_t is not None:
            z = z + x_t @ self.C.t()
        # Clipped basis expansion is bounded by construction -> no hard state clamp
        # (reference applies clip_z_to_range only to the unclipped variants).
        if self.clip_range is not None and not self.use_clipping:
            z = torch.clamp(z, -self.clip_range, self.clip_range)
        return z

    def readout(self, z_t):
        """Identity observation of the first output_dim latent units. (B, M) -> (B, N)."""
        # Observations are the first output_dim latent units when output_dim < latent_dim.
        return z_t[..., :self.config.output_dim]

    @property
    def supports_analytic_fixed_points(self) -> bool:
        """This family has a closed-form Jacobian / fixed-point solver (scy_fi)."""
        return True

    def jacobian(self, z: torch.Tensor, *, inputs=None) -> torch.Tensor:
        """Analytic Jacobian for dendPLRNN.

        z: (M,) -> (M, M)
        """
        # active indicator per unit and basis: (M, B)
        d = (z.unsqueeze(-1) > self.H).float()
        if self.use_clipping:
            # d/dz [alpha_b (relu(z - H_b) - relu(z))] = alpha_b (1[z > H_b] - 1[z > 0])
            d = d - (z.unsqueeze(-1) > 0).float()
        # effective slope per unit: (M,)
        alpha_d = (self.alphas * d).sum(dim=-1)
        return torch.diag(self.A) + self.W @ torch.diag(alpha_d)

    def analytic_parameters(self, task_input: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Expose effective shallowPLRNN parameters for the analytic solver.

        Equivalent shallowPLRNN:
            z = A z + W1 ReLU(W2 z + h2) + h1
        with hidden_dim = M * B (or M * (B + 1) when use_clipping adds the
        mirrored {-(sum alpha), 0} base),
            W2 = [I; I; ...; I]          (M*B, M)
            h2 = [-H_1; -H_2; ...; -H_B] (M*B,)
            W1 = [alpha_1 W, ..., alpha_B W] (M, M*B)
            A  = diag(A), h1 = h

        If a constant task_input is provided, it is folded into the effective bias:
            h1_eff = h + C @ task_input.
        """
        M, B = self.config.latent_dim, self.config.n_bases
        A_eff = torch.diag(self.A).detach()
        # With use_clipping, phi(z) = sum_b alpha_b relu(z - H_b) - (sum_b alpha_b) relu(z),
        # i.e. B bases {alpha_b, H_b} plus one extra base {-(sum alpha), threshold 0}.
        n_eff = B + 1 if self.use_clipping else B
        W1 = torch.zeros(M, M * n_eff, dtype=self.W.dtype, device=self.W.device)
        W2 = torch.zeros(M * n_eff, M, dtype=self.W.dtype, device=self.W.device)
        h2 = torch.zeros(M * n_eff, dtype=self.h.dtype, device=self.h.device)
        for b in range(B):
            W1[:, b * M:(b + 1) * M] = self.alphas[b].item() * self.W
            W2[b * M:(b + 1) * M, :] = torch.eye(M, dtype=W2.dtype, device=W2.device)
            h2[b * M:(b + 1) * M] = -self.H[:, b]
        if self.use_clipping:
            W1[:, B * M:(B + 1) * M] = -float(self.alphas.sum().item()) * self.W
            W2[B * M:(B + 1) * M, :] = torch.eye(M, dtype=W2.dtype, device=W2.device)
            # h2 block stays 0: relu(z - 0) = relu(z)

        h1_eff = self.h
        if task_input is not None and self.C is not None:
            s = torch.as_tensor(task_input, dtype=self.C.dtype, device=self.C.device)
            if s.dim() > 1:
                s = s.squeeze(0)
            h1_eff = h1_eff + self.C @ s

        return {
            "A": A_eff,
            "W1": W1.detach(),
            "W2": W2.detach(),
            "h1": h1_eff.detach(),
            "h2": h2.detach(),
        }


@register_model("alrnn")
class ALRNNModel(NeuralDynamicsModel):
    """Almost-linear RNN (ALRNN).

    State equation:
        z_t = A z_{t-1} + W Phi*(z_{t-1}) + h + C s_t
        Phi*(z) = [z_1, ..., z_{M-P}, ReLU(z_{M-P+1}), ..., ReLU(z_M)]

    Only the last P = latent_dim - n_linear units are ReLU; the rest are linear.
    """
    config_class = ALRNNConfig

    def __init__(self, config: ALRNNConfig) -> None:
        super().__init__(config)
        M, K = config.latent_dim, config.input_dim
        P = M - config.n_linear
        if P <= 0:
            raise ValueError(
                f"ALRNN requires n_linear < latent_dim, got n_linear={config.n_linear}, "
                f"latent_dim={M}"
            )
        r = 1.0 / (M ** 0.5)
        if config.init_scheme == "paper":
            # AL-RNN reference init (Brenner et al. 2024): A = diag of a normalized
            # positive-definite random matrix (eigenvalues <= 1), W ~ N(0, 0.01^2), h = 0.
            R = torch.randn(M, M)
            K = R.T @ R / M + torch.eye(M)
            K = K / torch.linalg.eigvals(K).abs().max()
            self.A = nn.Parameter(torch.diagonal(K).contiguous())
            self.W = nn.Parameter(torch.randn(M, M) * 0.01)
            self.h = nn.Parameter(torch.zeros(M))
        else:
            self.W = nn.Parameter(uniform_(torch.empty(M, M), -r, r))
            self.A = nn.Parameter(uniform_(torch.empty(M), a=0.5, b=0.9))
            self.h = nn.Parameter(torch.zeros(M))
        if config.autonomous:
            self.register_parameter("C", None)
        else:
            r3 = 1.0 / (K ** 0.5)
            self.C = nn.Parameter(uniform_(torch.empty(M, K), -r3, r3))
        self.register_parameter("B", _make_z0_lift(config, M))
        self.clip_range = config.clip_range
        self.use_clipping = config.use_clipping
        self.apply_freeze_config()

    def _freeze_groups(self) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = {
            "recurrent": [r"^A$", r"^W\.", r"^h$"],
            "output": [],
            "h0": [],
        }
        if self.C is not None:
            groups["input"] = [r"^C\."]
        else:
            groups["input"] = []
        if self.B is not None:
            groups["output"] = [r"^B\."]
        return groups

    def init_state_from_obs(self, x0: torch.Tensor) -> torch.Tensor:
        """Latent z0 from the first observation x0: (B, N) -> (B, M).

        With learn_z0 a trained lift B (N, M) is applied, then the observed
        dims are hard-set to x0 (AL-RNN reference); otherwise falls back to
        the base-class identity lift.
        """
        # AL-RNN reference: z0 = x0 @ B, then hard-set the observed dims
        if self.B is not None:
            z = x0 @ self.B
            z[..., :x0.shape[-1]] = x0
            return z
        return super().init_state_from_obs(x0)

    @property
    def n_linear(self) -> int:
        """Number of linear (identity-activated) latent units (M - P)."""
        return self.config.n_linear

    @property
    def n_relu(self) -> int:
        """Number of ReLU-activated latent units (P)."""
        return self.config.latent_dim - self.config.n_linear

    def _phi_star(self, z: torch.Tensor) -> torch.Tensor:
        """Almost-linear activation: keep first M-P units linear, ReLU the rest."""
        P = self.n_relu
        if P == 0:
            return z
        linear_part = z[..., :-P] if P > 0 else z
        nonlinear_part = torch.relu(z[..., -P:])
        return torch.cat([linear_part, nonlinear_part], dim=-1)

    def recurrence(self, x_t, z_prev, *, inputs=None):
        """Single ALRNN step. x_t: (B, K) or None, z_prev: (B, M) -> z_t: (B, M).

        z_t = A ⊙ z + W @ φ*(z) + h [+ C @ x_t], with φ* keeping the first
        M-P units linear and ReLU-activating the last P; states are clamped
        to ±clip_range when clip_range is set (reference behavior).
        """
        z = self.A * z_prev + self._phi_star(z_prev) @ self.W.t() + self.h
        if self.C is not None and x_t is not None:
            z = z + x_t @ self.C.t()
        if self.clip_range is not None:
            z = torch.clamp(z, -self.clip_range, self.clip_range)
        return z

    def readout(self, z_t):
        """Identity observation of the first output_dim latent units. (B, M) -> (B, N)."""
        # Observations are the first output_dim latent units when output_dim < latent_dim.
        return z_t[..., :self.config.output_dim]

    @property
    def supports_analytic_fixed_points(self) -> bool:
        """This family has a closed-form Jacobian / fixed-point solver (scy_fi)."""
        return True

    def jacobian(self, z: torch.Tensor, *, inputs=None) -> torch.Tensor:
        """Analytic Jacobian for ALRNN.

        z: (M,) -> (M, M)
        """
        P = self.n_relu
        mask = torch.cat([
            torch.ones(self.n_linear, dtype=z.dtype, device=z.device),
            (z[-P:] > 0).float() if P > 0 else torch.empty(0, dtype=z.dtype, device=z.device),
        ])
        return torch.diag(self.A) + self.W @ torch.diag(mask)

    def analytic_parameters(self, task_input: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Expose effective shallowPLRNN parameters for the analytic solver.

        The ALRNN recurrence can be written as:
            z = (diag(A) + W[:, :M-P] @ S_linear) z + W[:, M-P:] ReLU(S_relu z) + h
        where S_linear selects the first M-P rows and S_relu selects the last P rows.
        This is a shallowPLRNN with hidden_dim = P, W2 = S_relu, h2 = 0, and a full
        effective A matrix.

        If a constant task_input is provided, it is folded into the effective bias:
            h_eff = h + C @ task_input.
        """
        M, P = self.config.latent_dim, self.n_relu
        S_linear = torch.eye(M, dtype=self.W.dtype, device=self.W.device)[:self.n_linear, :]
        A_eff = torch.diag(self.A) + self.W[:, :self.n_linear] @ S_linear
        W1 = self.W[:, self.n_linear:]
        W2 = torch.zeros(P, M, dtype=self.W.dtype, device=self.W.device)
        W2[:, self.n_linear:] = torch.eye(P, dtype=W2.dtype, device=W2.device)
        h2 = torch.zeros(P, dtype=self.h.dtype, device=self.h.device)

        h_eff = self.h
        if task_input is not None and self.C is not None:
            s = torch.as_tensor(task_input, dtype=self.C.dtype, device=self.C.device)
            if s.dim() > 1:
                s = s.squeeze(0)
            h_eff = h_eff + self.C @ s

        return {
            "A": A_eff.detach(),
            "W1": W1.detach(),
            "W2": W2.detach(),
            "h1": h_eff.detach(),
            "h2": h2.detach(),
        }