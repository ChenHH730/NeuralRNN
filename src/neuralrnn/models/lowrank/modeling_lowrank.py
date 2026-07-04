"""Low-rank RNN model (Paradigm A: task-optimized RNN).

Port of LowRankRNN from Dubreuil et al. (2022) / Valente et al. (2022).

The recurrent weight matrix is parametrized as:
    W_rec = m @ n^T / N   (when scale_by_hidden_size=True)

where m (N × rank) and n (N × rank) are trainable low-rank factors.
This constrains dynamics to an r-dimensional subspace spanned by columns of m.

Dynamics (Euler discretization):
    r_t     = tanh(z_t + b)
    rec_t   = r_t @ n @ m^T / N
    z_{t+1} = z_t + sigma*xi_t + alpha*(-z_t + rec_t + x_t @ Wi_full)
    y_t     = out_act(z_t) @ Wo_full / N
"""
from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np

from ...modeling_utils import NeuralDynamicsModel, DynamicsModelOutput
from ...auto.modeling_auto import register_model
from .configuration_lowrank import LowrankRNNConfig


_ACT = {"relu": torch.relu, "tanh": torch.tanh, "softplus": torch.nn.functional.softplus}


@register_model("lowrank_rnn")
class LowrankRNNModel(NeuralDynamicsModel):
    """Low-rank recurrent neural network.

    The recurrent connectivity W_rec = m @ n^T / N is factorized into two
    low-rank matrices m (N × rank) and n (N × rank). This constrains the
    recurrent dynamics to the subspace spanned by columns of m, enabling
    transparent analysis of computation in a low-dimensional projection.

    Config class: LowrankRNNConfig
    """

    config_class = LowrankRNNConfig

    def __init__(self, config: LowrankRNNConfig) -> None:
        super().__init__(config)
        N = config.latent_dim
        R = config.rank
        self.alpha = config.alpha
        self.noise_std = config.noise_std
        self.act = _ACT[config.activation]
        self.output_act = _ACT[config.output_activation]

        # ---- Expose attributes for backward compat with reference code ----
        self.hidden_size = N
        self.input_size = config.input_dim
        self.output_size = config.output_dim
        self.rank = config.rank
        self.scale_by_hidden_size = config.scale_by_hidden_size
        self.train_wi = config.train_wi
        self.train_wo = config.train_wo
        self.train_wrec = config.train_wrec
        self.train_h0 = config.train_h0
        self.non_linearity = self.act   # for compat with reference helpers

        # ---- Low-rank recurrent factors: m (N×R), n (N×R) ----
        self.m = nn.Parameter(torch.Tensor(N, R))
        self.n = nn.Parameter(torch.Tensor(N, R))
        if not config.train_wrec:
            self.m.requires_grad = False
            self.n.requires_grad = False

        # ---- Input weights: wi (input_dim, N), si (input_dim) channel scaling ----
        self.wi = nn.Parameter(torch.Tensor(config.input_dim, N))
        self.si = nn.Parameter(torch.Tensor(config.input_dim))
        if config.train_wi:
            self.si.requires_grad = False
        else:
            self.wi.requires_grad = False

        # ---- Output weights: wo (N, output_dim), so (output_dim) channel scaling ----
        self.wo = nn.Parameter(torch.Tensor(N, config.output_dim))
        self.so = nn.Parameter(torch.Tensor(config.output_dim))
        if config.train_wo:
            self.so.requires_grad = False
        else:
            self.wo.requires_grad = False

        # ---- Bias ----
        self.b = nn.Parameter(torch.Tensor(N))
        if not config.add_bias:
            self.b.requires_grad = False

        # ---- Initial state h0 ----
        self.h0 = nn.Parameter(torch.Tensor(N))
        if not config.train_h0:
            self.h0.requires_grad = False

        # ---- Proxy parameters (computed from base params) ----
        self.wi_full: torch.Tensor | None = None
        self.wo_full: torch.Tensor | None = None

        self.reset_parameters()
        self._define_proxy_parameters()
        self.apply_freeze_config()

    def _freeze_groups(self) -> dict[str, list[str]]:
        return {
            "input": [r"^wi$", r"^si$"],
            "recurrent": [r"^m$", r"^n$"],
            "output": [r"^wo$", r"^so$"],
            "h0": [r"^h0$"],
        }

    def extra_repr(self) -> str:
        """Show key model info and parameter shapes."""
        lines = [
            f"input_dim={self.config.input_dim}, latent_dim={self.config.latent_dim}, "
            f"output_dim={self.config.output_dim}, rank={self.config.rank}, "
            f"alpha={self.alpha}, noise_std={self.noise_std}",
            f"  (m): Parameter ({self.config.latent_dim}, {self.config.rank})",
            f"  (n): Parameter ({self.config.latent_dim}, {self.config.rank})",
            f"  (wi): Parameter ({self.config.input_dim}, {self.config.latent_dim})",
            f"  (si): Parameter ({self.config.input_dim})",
            f"  (wo): Parameter ({self.config.latent_dim}, {self.config.output_dim})",
            f"  (so): Parameter ({self.config.output_dim})",
            f"  (b): Parameter ({self.config.latent_dim})",
            f"  (h0): Parameter ({self.config.latent_dim})",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(\n" + self.extra_repr() + "\n)"

    def reset_parameters(self):
        """Initialize parameters matching the reference LowRankRNN."""
        N = self.config.latent_dim
        R = self.config.rank
        with torch.no_grad():
            self.wi.normal_()
            self.si.set_(torch.ones_like(self.si))
            self.m.normal_()
            self.n.normal_()
            self.b.zero_()
            self.wo.normal_(std=4.0)
            self.so.set_(torch.ones_like(self.so))
            self.h0.zero_()

    def _define_proxy_parameters(self):
        """Compute wi_full and wo_full with per-channel scaling.

        References:
            wi_full = (wi^T * si)^T   — applies si scaling per input channel
            wo_full = wo * so          — applies so scaling per output channel
        """
        self.wi_full = (self.wi.t() * self.si).t()  # (input_dim, N)
        self.wo_full = self.wo * self.so              # (N, output_dim)

    # ────────────────────  hard contract  ────────────────────
    def recurrence(self, x_t: torch.Tensor, z_prev: torch.Tensor,
                   *, inputs: torch.Tensor | None = None) -> torch.Tensor:
        """Single-step low-rank transition.

        Args:
            x_t: (B, input_dim) — input at current timestep
            z_prev: (B, N) — hidden state from previous timestep

        Returns:
            z_t: (B, N) — updated hidden state
        """
        self._define_proxy_parameters()
        N = self.config.latent_dim

        # Firing rate from previous state: r = tanh(z_prev + b)
        r = self.act(z_prev + self.b)

        # Low-rank recurrent input: r @ n @ m^T
        rec = r @ self.n @ self.m.t()

        if self.config.scale_by_hidden_size:
            rec = rec / N

        # External input
        inp = x_t @ self.wi_full  # (B, N)

        # Euler update
        z_t = z_prev + self.alpha * (-z_prev + rec + inp)

        # Add noise during training
        if self.noise_std > 0 and self.training:
            z_t = z_t + self.noise_std * torch.randn_like(z_t)

        return z_t

    def readout(self, z_t: torch.Tensor) -> torch.Tensor:
        """Readout from hidden state.

        Args:
            z_t: (B, N) — hidden state

        Returns:
            y_t: (B, output_dim)
        """
        self._define_proxy_parameters()
        N = self.config.latent_dim
        out = self.output_act(z_t) @ self.wo_full
        if self.config.scale_by_hidden_size:
            out = out / N
        return out

    # ────────────────────  override forward for precision ────────────────────
    def forward(self, inputs: torch.Tensor | None = None, *,
                initial_state: torch.Tensor | None = None,
                initial_states: torch.Tensor | None = None,  # alias for reference compat
                n_steps: int | None = None,
                return_states: bool = True,
                return_dynamics: bool = False) -> DynamicsModelOutput:
        """Full rollout with exact reference dynamics.

        Overrides the base class forward() to match the reference LowRankRNN
        exactly: the initial firing rate r = tanh(h0) (no bias for step 0),
        and subsequent r = tanh(h + bias). Output is computed from h after
        the update.

        Args:
            inputs: (B, T, input_dim) — input sequence
            initial_state: (B, N) — initial hidden state (uses h0 if None)
            initial_states: alias for initial_state (reference code compat)
            n_steps: int — required for autonomous rollout (inputs=None)
            return_states: bool — whether to return hidden trajectories
            return_dynamics: bool — if True, return (output, trajectories) tuple
                           matching the reference code convention for
                           plot_trajectories / plot_field compatibility.

        Returns:
            DynamicsModelOutput with outputs and states.
            If return_dynamics=True, returns a (output_tensor, trajectories_tensor)
            tuple for backward compatibility with reference analysis code.
        """
        # Backward compat: initial_states (plural) as alias
        if initial_states is not None and initial_state is None:
            initial_state = initial_states

        if inputs is not None:
            assert inputs.dim() == 3, "inputs must be (batch, T, input_dim)"
            batch_size, T = inputs.shape[0], inputs.shape[1]
            device = inputs.device
        else:
            assert n_steps is not None, "n_steps required for autonomous rollout"
            assert initial_state is not None, "initial_state required for autonomous rollout"
            batch_size, T, device = initial_state.shape[0], n_steps, initial_state.device

        self._define_proxy_parameters()
        N = self.config.latent_dim

        # Initialize
        if initial_state is not None:
            h = initial_state.clone()
        else:
            h = self.h0.to(device).expand(batch_size, -1).contiguous().clone()

        # Initial firing rate: tanh(h0) WITHOUT bias (matching reference)
        r = self.act(h)

        # Noise (pre-sample for efficiency)
        noise = torch.randn(batch_size, T, N, device=device)

        output = torch.zeros(batch_size, T, self.config.output_dim, device=device)
        trajectories = []

        if return_dynamics:
            trajectories.append(h)  # prepend initial state

        for i in range(T):
            rec = r @ self.n @ self.m.t()
            if self.config.scale_by_hidden_size:
                rec = rec / N

            inp = inputs[:, i, :] @ self.wi_full
            h = h + self.alpha * (-h + rec + inp)
            if self.noise_std > 0 and self.training:
                h = h + self.noise_std * noise[:, i, :]
            r = self.act(h + self.b)  # now with bias

            out = self.output_act(h) @ self.wo_full
            if self.config.scale_by_hidden_size:
                out = out / N
            output[:, i, :] = out

            if return_states:
                trajectories.append(h)

        if return_states:
            states_t = torch.stack(trajectories, dim=1)  # (B, T+1, N) with return_dynamics
        else:
            states_t = None

        if return_dynamics:
            # Return tuple (output, trajectories) matching reference convention
            # for compat with plot_trajectories and plot_field
            return output, states_t

        return DynamicsModelOutput(outputs=output, states=states_t)

    # ────────────────────  model-specific utilities  ────────────────────
    @torch.no_grad()
    def svd_reparametrization(self):
        """Orthogonalize m and n via SVD of m @ n^T.

        This makes the columns of m orthogonal (up to scaling) and ensures
        the low-rank structure is represented in its canonical form.
        Useful before analysis (vector fields, fixed points).
        """
        structure = (self.m @ self.n.t()).cpu().numpy()
        u, s, vt = np.linalg.svd(structure, full_matrices=False)
        rank = self.config.rank
        u, s, vt = u[:, :rank], s[:rank], vt[:rank, :]
        self.m.set_(torch.from_numpy(u * np.sqrt(s)).to(self.m.device))
        self.n.set_(torch.from_numpy(vt.T * np.sqrt(s)).to(self.n.device))
        self._define_proxy_parameters()

    def init_state(self, batch_size: int, device: torch.device | str = "cpu") -> torch.Tensor:
        """Return the initial hidden state h0 expanded to batch_size."""
        return self.h0.to(device).expand(batch_size, -1).contiguous()

    def clone(self) -> "LowrankRNNModel":
        """Return a deep copy of this model (for reference train() compat)."""
        import copy
        new_model = LowrankRNNModel(self.config)
        new_model.load_state_dict(copy.deepcopy(self.state_dict()))
        new_model._define_proxy_parameters()
        return new_model

    # ────────────────────  save/load fix (contiguous tensors) ────────────────────
    def save_pretrained(self, save_directory: str, metadata: dict | None = None) -> None:
        """Write config.json + model.safetensors (+ metadata.json).

        Ensures all tensors are contiguous before saving to avoid safetensors errors.
        """
        import os

        # Make all parameters contiguous before saving
        for name, param in self.named_parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()
        # Also ensure proxy params are up-to-date and contiguous
        self._define_proxy_parameters()

        super().save_pretrained(save_directory, metadata=metadata)
