"""Gain RNN family model implementations.

Unified "fixed weights + neuronal gain modulation" paradigm (see
``docs/papers/gain_rnn.md`` and ``gain_rnn.md`` at the repo root):

- ``GainRNNModel``: CTRNN lineage + structural masks + per-neuron gain/bias
  parameterizing the firing-rate map. Gain placement:
      outside: r = g * phi(u + b)  (output gain == presynaptic column scaling;
                                    Beiran & Litwin-Kumar 2025)
      inside:  r = phi(g * u + b)  (input gain == slope modulation;
                                    Stroud et al. 2018)
- ``StpRNNModel``: short-term plasticity as a *dynamic* gain parameterization
  (effective presynaptic gain = syn_x * syn_u, Tsodyks-Markram rate form,
  cell-specific per presynaptic neuron). Defaults reproduce the notebook-11
  (Masse et al. 2019 style) model; nonlinearity_mode="rate" +
  stp_init="random" + per-trial stp_alpha reproduces Zhou & Buonomano 2024.

Freeze invariant (do not break): ``apply_freeze_config()`` is re-called at the
end of every ``__init__`` level in the chain (CTRNN -> Gain -> Stp). Earlier
calls are polymorphic and simply match nothing for not-yet-created parameters;
all calls are idempotent, and freeze flags always win (applied last).
``_freeze_groups()`` at every level merges ``super()._freeze_groups()`` with the
level's new groups — never return only the new groups.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...activations import get_activation
from ...auto.modeling_auto import register_model
from ...modeling_utils import DynamicsModelOutput
from ..constrained_rnn.modeling_constrained_rnn import ConstrainedRNNModel
from .configuration_gain_rnn import GainRNNConfig, StpRNNConfig


def _to_param_vector(value: Any, M: int, name: str) -> torch.Tensor:
    """Broadcast a scalar or validate an (M,) array into a float32 tensor."""
    if isinstance(value, torch.Tensor):
        v = value.detach().clone().float().reshape(-1)
    elif np.isscalar(value):
        v = torch.full((M,), float(value))
    else:
        v = torch.as_tensor(np.asarray(value), dtype=torch.float32).reshape(-1)
    if v.shape != (M,):
        raise ValueError(f"{name} must be a scalar or have shape ({M},), got {tuple(v.shape)}")
    return v


def make_stp_masks(
    latent_dim: int,
    output_dim: int,
    ei_ratio: float = 0.8,
    no_self_connections: bool = True,
    readout_e_only: bool = True,
) -> dict[str, np.ndarray]:
    """Build rec/out masks for STP-style EI networks (notebook 11 conventions).

    Args:
        latent_dim: Number of units M.
        output_dim: Readout dimension.
        ei_ratio: Fraction of excitatory units (first ``round(M*ei_ratio)``).
        no_self_connections: Zero the recurrent diagonal (no autapses).
        readout_e_only: Zero readout rows of inhibitory units.

    Returns:
        dict with ``rec_mask`` (M, M) and ``out_mask`` (M, output_dim) in the
        framework's mask conventions (``out_mask`` multiplies
        ``readout_layer.weight.t()``; note this is the transpose of notebook
        11's ``w_out_mask``).
    """
    M = latent_dim
    e_size = int(round(M * ei_ratio))
    rec_mask = np.ones((M, M), dtype=np.float32)
    if no_self_connections:
        np.fill_diagonal(rec_mask, 0.0)
    out_mask = np.ones((M, output_dim), dtype=np.float32)
    if readout_e_only:
        out_mask[e_size:, :] = 0.0
    return {"rec_mask": rec_mask, "out_mask": out_mask}


@register_model("gain_rnn")
class GainRNNModel(ConstrainedRNNModel):
    """CTRNN with structural masks and a per-neuron gain/bias rate map.

    The rate map is the single entry point for every nonlinearity application:

        outside: rate_map(u) = gain * act(u + bias)
        inside:  rate_map(u) = act(gain * u + bias)

    With gain=1, bias=0 it reduces to ``act`` (plain ConstrainedRNN behavior).

    Family traits (deliberate deviations from the CTRNN base):
      - In ``nonlinearity_mode="rate"`` the readout consumes the firing rates
        ``rate_map(z)`` instead of the raw state (connectome-style; CTRNN's
        rate mode reads the state). Tested and documented.
      - ``recurrence`` accepts ``x_t=None`` for autonomous rollout (the input
        term, including the input bias, is skipped entirely).
      - ``noise_position="post"`` adds recurrent noise to the leaked blend
        before the final nonlinearity (state-level noise) instead of to the
        pre-activation; std follows ``noise_alpha_scaling`` as usual.

    Note (gain_rnn.md §6): for ReLU, inside and outside placements are exactly
    equivalent (scale degeneracy — regularize gains if both are trainable);
    negative gains under Dale constraints flip a neuron's E/I identity.
    """

    config_class = GainRNNConfig

    def __init__(self, config: GainRNNConfig) -> None:
        super().__init__(config)  # CTRNN layers + masks + dale; 1st (polymorphic) apply_freeze_config
        M = config.latent_dim
        # Rebuild the activation with family kwargs (softplus beta, piecewise_tanh r0/rmax).
        self.act = get_activation(config.activation, **(getattr(config, "activation_params", None) or {}))

        self.gain = nn.Parameter(_to_param_vector(config.gain_init, M, "gain_init"))
        self.bias = nn.Parameter(_to_param_vector(config.bias_init, M, "bias_init"))

        with torch.no_grad():
            self.h0.copy_(_to_param_vector(getattr(config, "h0_init", 0.0), M, "h0_init"))

        # 2nd application: gains/biases now exist, so family freeze flags land.
        self.apply_freeze_config()

    # ---------------- freeze plumbing ----------------
    def _freeze_groups(self) -> dict[str, list[str]]:
        groups = super()._freeze_groups()
        groups.update({"gains": [r"^gain$"], "biases": [r"^bias$"]})
        return groups

    def apply_freeze_config(self) -> list[str]:
        """Apply base freeze flags plus family flags (freeze_gain/freeze_bias).

        Freeze flags always win and are applied last; every ``__init__`` level
        re-calls this method (idempotent)."""
        frozen = list(super().apply_freeze_config())
        if getattr(self.config, "freeze_gain", False):
            frozen += self.freeze_parameters(groups="gains")
        if getattr(self.config, "freeze_bias", False):
            frozen += self.freeze_parameters(groups="biases")
        return frozen

    # ---------------- rate map ----------------
    def rate_map(self, u: torch.Tensor) -> torch.Tensor:
        """Parameterized firing-rate map; see class docstring."""
        if self.config.gain_position == "inside":
            return self.act(self.gain * u + self.bias)
        return self.gain * self.act(u + self.bias)

    def firing_rate(self, z: torch.Tensor) -> torch.Tensor:
        """Alias for ``rate_map`` (firing-rate API naming)."""
        return self.rate_map(z)

    def get_firing_rates(self, states: torch.Tensor) -> torch.Tensor:
        """Apply the rate map to a state trajectory (meaningful for nonlinearity_mode="rate")."""
        return self.rate_map(states)

    # ---------------- effective weights / noise ----------------
    def _input_weight(self) -> torch.Tensor:
        w = self.input2h.weight
        if getattr(self.config, "positive_input_weights", False):
            w = F.relu(w)
        if self.in_mask is not None:
            w = w * self.in_mask.t()
        return w

    def _output_weight(self) -> torch.Tensor:
        w = self.readout_layer.weight
        if getattr(self.config, "positive_output_weights", False):
            w = F.relu(w)
        if self.out_mask is not None:
            w = w * self.out_mask.t()
        return w

    def _noise_std(self) -> float:
        if getattr(self.config, "noise_alpha_scaling", False):
            return (2 * self.alpha * self.config.sigma_rec ** 2) ** 0.5
        return self.config.sigma_rec

    def _euler_step(self, rec_in: torch.Tensor, z_prev: torch.Tensor,
                    x_t: torch.Tensor | None) -> torch.Tensor:
        """Shared single Euler transition given the recurrent coupling input.

        ``pre = W @ rec_in + B @ x_t + biases``. ``x_t=None`` skips the input
        term entirely (autonomous rollout). Noise placement follows
        ``config.noise_position``; the nonlinearity always goes through
        ``rate_map``.
        """
        W = self._recurrent_weight()
        pre = F.linear(rec_in, W, self.h2h.bias)
        if x_t is not None:
            pre = pre + F.linear(x_t, self._input_weight(), self.input2h.bias)
        mode = self.config.nonlinearity_mode
        noise_pos = getattr(self.config, "noise_position", "pre")
        add_noise = self.config.sigma_rec > 0 and self.training
        if add_noise and noise_pos == "pre":
            pre = pre + self._noise_std() * torch.randn_like(pre)
        if mode == "pre_activation":
            z = (1 - self.alpha) * z_prev + self.alpha * self.rate_map(pre)
            if add_noise and noise_pos == "post":
                z = z + self._noise_std() * torch.randn_like(z)
        else:
            update = (1 - self.alpha) * z_prev + self.alpha * pre
            if add_noise and noise_pos == "post":
                update = update + self._noise_std() * torch.randn_like(update)
            z = self.rate_map(update) if mode == "post_blend" else update
        return z

    # ---------------- hard contract ----------------
    def recurrence(self, x_t, z_prev, *, inputs=None):
        """Single Euler step. x_t: (B, input_dim), z_prev: (B, M) -> z_t: (B, M).

        In "rate" mode the recurrent matrix reads the firing rate rate_map(z);
        otherwise it reads z. See _euler_step for the full update.
        """
        mode = self.config.nonlinearity_mode
        rec_in = self.rate_map(z_prev) if mode == "rate" else z_prev
        return self._euler_step(rec_in, z_prev, x_t)

    def readout(self, z_t):
        """Readout from ``rate_map(z)`` in "rate" mode (family deviation from
        CTRNN's readout-from-state), from the state otherwise."""
        r = self.rate_map(z_t) if self.config.nonlinearity_mode == "rate" else z_t
        return F.linear(r, self._output_weight(), self.readout_layer.bias)


@register_model("stp_rnn")
class StpRNNModel(GainRNNModel):
    """RNN with short-term plasticity as a dynamic per-neuron gain.

    Latent state is the concatenation ``[h, syn_x, syn_u]`` (3M); the static
    gain/bias are frozen at identity by default and the effective presynaptic
    gain is the dynamic factor ``syn_x * syn_u`` (Tsodyks-Markram rate form,
    cell-specific). The neuromodulator ``stp_alpha`` (runtime buffer, never
    trained) scales U -> U_eff = clamp(stp_alpha * U, 0, 1) and can be set per
    trial via :meth:`set_stp_alpha` (Zhou & Buonomano 2024).

    Analysis fallback: ``recurrence`` also accepts an (B, M) state, treating
    syn_x = syn_u = 1 and returning (B, M). Fixed-point / linearization tools
    therefore analyze the frozen-efficacy M-dim map, not the full 3M system —
    interpret fixed points accordingly.

    Family defaults reproduce the notebook-11 model (Masse et al. 2019 style);
    see StpRNNConfig. Note the parity details: ``input2h.bias`` is zeroed and
    frozen (the reference model has no input bias), and the gamma init matches
    the reference weight distributions.
    """

    config_class = StpRNNConfig

    def __init__(self, config: StpRNNConfig) -> None:
        if config.dt is None:
            raise ValueError(
                "stp_rnn requires a physical dt (ms) for the STP dynamics "
                "(dt_sec = dt/1000 and alpha_x = dt/tau_x); pass dt explicitly "
                "instead of only alpha."
            )
        super().__init__(config)
        M = config.latent_dim

        stp = self._resolve_stp_params(config, M)
        self.stp_tau_x = nn.Parameter(stp["stp_tau_x"])
        self.stp_tau_u = nn.Parameter(stp["stp_tau_u"])
        self.stp_U = nn.Parameter(stp["stp_U"])
        self.register_buffer("stp_alpha", _to_param_vector(config.stp_alpha, M, "stp_alpha"))

        # notebook-11 parity: the reference model has no input bias.
        with torch.no_grad():
            self.input2h.bias.zero_()
        self.input2h.bias.requires_grad = False

        if config.init_method == "gamma":
            self._init_weights_gamma()
            # Re-apply masks after overwriting weights (SERNN orthogonal-init precedent).
            self._apply_masks_to_weights()

        # 3rd application: stp parameters now exist.
        self.apply_freeze_config()

    # ---------------- freeze plumbing ----------------
    def _freeze_groups(self) -> dict[str, list[str]]:
        groups = super()._freeze_groups()
        groups.update({"stp": [r"^stp_tau_x$", r"^stp_tau_u$", r"^stp_U$"]})
        return groups

    def apply_freeze_config(self) -> list[str]:
        """Apply base/family freeze flags plus the STP-specific freeze_stp flag."""
        frozen = list(super().apply_freeze_config())
        if getattr(self.config, "freeze_stp", False):
            frozen += self.freeze_parameters(groups="stp")
        return frozen

    # ---------------- STP parameter resolution / init ----------------
    @staticmethod
    def _resolve_stp_params(config: StpRNNConfig, M: int) -> dict[str, torch.Tensor]:
        """Resolve per-neuron (tau_x, tau_u, U).

        Any explicit array field is used directly (scalars broadcast); only when
        all three are scalars does ``stp_init`` decide. "random" mode shares one
        generator so tau_x / tau_u / U are sampled independently.
        """
        raws = {"stp_tau_x": config.stp_tau_x, "stp_tau_u": config.stp_tau_u, "stp_U": config.stp_U}
        if any(isinstance(v, (list, tuple, np.ndarray, torch.Tensor)) for v in raws.values()):
            return {k: _to_param_vector(v, M, k) for k, v in raws.items()}
        if config.stp_init == "alternating":
            is_fac = torch.arange(M) % 2 == 0

            def alt(fac: float, dep: float) -> torch.Tensor:
                """(M,) vector: ``fac`` on facilitating units, ``dep`` on depressing ones."""
                return torch.where(
                    is_fac, torch.full((M,), float(fac)), torch.full((M,), float(dep))
                )

            return {
                "stp_tau_x": alt(config.tau_x_fac, config.tau_x_dep),
                "stp_tau_u": alt(config.tau_u_fac, config.tau_u_dep),
                "stp_U": alt(config.U_fac, config.U_dep),
            }
        if config.stp_init == "random":
            gen = torch.Generator()
            if config.stp_seed is not None:
                gen.manual_seed(config.stp_seed)
            out = {}
            for kind, (mean, std, lo, hi) in {
                "stp_tau_x": (config.stp_tau_mean, config.stp_tau_std, config.stp_tau_min, config.stp_tau_max),
                "stp_tau_u": (config.stp_tau_mean, config.stp_tau_std, config.stp_tau_min, config.stp_tau_max),
                "stp_U": (config.stp_U_mean, config.stp_U_std, config.stp_U_min, config.stp_U_max),
            }.items():
                out[kind] = torch.empty(M).normal_(mean, std, generator=gen).clamp_(lo, hi)
            return out
        return {k: torch.full((M,), float(v)) for k, v in raws.items()}

    def _init_weights_gamma(self) -> None:
        """Notebook-11 gamma initialization (Masse et al. 2019 reference)."""
        cfg = self.config
        M = cfg.latent_dim
        e_size = int(round(M * cfg.ei_ratio))
        rng = np.random.default_rng(cfg.init_seed)
        with torch.no_grad():
            w_rnn = np.zeros((M, M), dtype=np.float32)
            w_rnn[:, :e_size] = rng.gamma(cfg.gamma_shape_exc, cfg.gamma_scale, size=(M, e_size))
            w_rnn[:, e_size:] = rng.gamma(cfg.gamma_shape_inh, cfg.gamma_scale, size=(M, M - e_size))
            np.fill_diagonal(w_rnn, 0.0)  # no self-connections
            self.h2h.weight.copy_(torch.from_numpy(w_rnn))
            w_in = rng.gamma(0.2, cfg.gamma_scale, size=(M, cfg.input_dim)).astype(np.float32)
            self.input2h.weight.copy_(torch.from_numpy(w_in))
            w_out = np.zeros((cfg.output_dim, M), dtype=np.float32)
            w_out[:, :e_size] = rng.gamma(cfg.gamma_shape_exc, cfg.gamma_scale, size=(cfg.output_dim, e_size))
            self.readout_layer.weight.copy_(torch.from_numpy(w_out))

    # ---------------- STP dynamics ----------------
    @property
    def dt_sec(self) -> float:
        """Time step in seconds (STP release terms assume rates in spikes/s)."""
        return self.config.dt / 1000.0

    def _effective_U(self) -> torch.Tensor:
        """Neuromodulated baseline release probability clamp(alpha * U, 0, 1)."""
        return (self.stp_alpha * self.stp_U.clamp(0.0, 1.0)).clamp(0.0, 1.0)

    def _stp_step(self, r: torch.Tensor, syn_x: torch.Tensor, syn_u: torch.Tensor):
        """Advance STP variables one step; old values update simultaneously.

        Mongillo et al. 2008 / notebook 11 (dt in ms, dt_sec = dt/1000):
            x' = x + (dt/tau_x)(1 - x) - dt_sec * u * x * r
            u' = u + (dt/tau_u)(U_eff - u) + dt_sec * U_eff * (1 - u) * r
        Both clamped to [0, 1].
        """
        tau_x = self.stp_tau_x.clamp_min(1e-3)
        tau_u = self.stp_tau_u.clamp_min(1e-3)
        U_eff = self._effective_U()
        dt = self.config.dt
        x_new = syn_x + (dt / tau_x) * (1.0 - syn_x) - self.dt_sec * syn_u * syn_x * r
        u_new = syn_u + (dt / tau_u) * (U_eff - syn_u) + self.dt_sec * U_eff * (1.0 - syn_u) * r
        return x_new.clamp(0.0, 1.0), u_new.clamp(0.0, 1.0)

    def set_stp_alpha(self, value: Any) -> None:
        """Set the neuromodulator vector (trial-level cue; scalar or (M,))."""
        v = _to_param_vector(value, self.config.latent_dim, "stp_alpha")
        if bool((v < 0).any()):
            raise ValueError("stp_alpha must be non-negative")
        with torch.no_grad():
            self.stp_alpha.copy_(v.to(self.stp_alpha.device))

    def get_stp_alpha(self) -> torch.Tensor:
        """Current neuromodulator vector (detached copy)."""
        return self.stp_alpha.detach().clone()

    @staticmethod
    def synaptic_efficacy(extras: dict[str, torch.Tensor]) -> torch.Tensor:
        """Effective dynamic gain trajectory syn_x * syn_u from forward extras."""
        return extras["syn_x"] * extras["syn_u"]

    # ---------------- state handling ----------------
    def init_state(self, batch_size: int, device="cpu") -> torch.Tensor:
        """Steady-state initial condition [h0, 1, U_eff] of shape (B, 3M).

        syn_u starts at U_eff (the STP steady state), not 1 — starting at 1
        destroys the low-baseline utilization that lets synaptic efficacy
        encode recent activity (notebook-11 key fix).
        """
        h = self.h0.to(device).expand(batch_size, -1)
        sx = torch.ones(batch_size, self.config.latent_dim, device=device)
        su = self._effective_U().detach().to(device).expand(batch_size, -1)
        return torch.cat([h, sx, su], dim=-1)

    # ---------------- hard contract ----------------
    def recurrence(self, x_t, z_prev, *, inputs=None):
        """Single-step transition.

        z_prev (B, 3M) = [h, syn_x, syn_u] -> (B, 3M). Fallback for analysis
        tools: z_prev (B, M) treats syn_x = syn_u = 1 and returns (B, M).
        """
        M = self.config.latent_dim
        full = z_prev.shape[-1] == 3 * M
        if full:
            h, syn_x, syn_u = z_prev.split(M, dim=-1)
        elif z_prev.shape[-1] == M:
            h = z_prev
            syn_x = torch.ones_like(h)
            syn_u = torch.ones_like(h)
        else:
            raise ValueError(
                f"z_prev has unexpected dim {z_prev.shape[-1]}; expected {M} or {3 * M}"
            )

        mode = self.config.nonlinearity_mode
        r_pre = self.rate_map(h) if mode == "rate" else h
        syn_x_new, syn_u_new = self._stp_step(r_pre, syn_x, syn_u)
        rec_in = r_pre * syn_x_new * syn_u_new
        h_new = self._euler_step(rec_in, h, x_t)
        if full:
            return torch.cat([h_new, syn_x_new, syn_u_new], dim=-1)
        return h_new

    def readout(self, z_t: torch.Tensor) -> torch.Tensor:
        """Readout from the h part of the state. (B, 3M) or (B, M) -> (B, output_dim)."""
        M = self.config.latent_dim
        h = z_t[..., :M] if z_t.shape[-1] == 3 * M else z_t
        return super().readout(h)

    def forward(self, inputs: torch.Tensor | None = None, *,
                initial_state: torch.Tensor | None = None,
                n_steps: int | None = None,
                return_states: bool = True) -> DynamicsModelOutput:
        """Rollout returning states = h only (B, T, M); syn_x / syn_u in extras.

        ``initial_state`` must be the full (B, 3M) state — an M-dim state would
        silently produce empty synaptic trajectories, so it raises instead.
        """
        if inputs is not None:
            assert inputs.dim() == 3, "inputs must be (batch, T, input_dim)"
            batch_size, T = inputs.shape[0], inputs.shape[1]
            device = inputs.device
        else:
            assert n_steps is not None, "n_steps must be provided for autonomous rollout"
            assert initial_state is not None, "initial_state must be provided for autonomous rollout"
            batch_size, T, device = initial_state.shape[0], n_steps, initial_state.device

        z = initial_state if initial_state is not None else self.init_state(batch_size, device)
        M = self.config.latent_dim
        if z.shape[-1] != 3 * M:
            raise ValueError(
                f"initial_state must have last dim {3 * M} ([h, syn_x, syn_u]); "
                f"got {z.shape[-1]}. Use model.init_state(batch_size) to build one."
            )

        states_h, states_x, states_u, outputs = [], [], [], []
        for t in range(T):
            x_t = inputs[:, t] if inputs is not None else None
            z = self.recurrence(x_t, z, inputs=inputs)
            states_h.append(z[..., :M])
            states_x.append(z[..., M:2 * M])
            states_u.append(z[..., 2 * M:])
            outputs.append(self.readout(z))

        outputs_t = torch.stack(outputs, dim=1)
        states_h_t = torch.stack(states_h, dim=1) if return_states else None
        extras = {
            "syn_x": torch.stack(states_x, dim=1),
            "syn_u": torch.stack(states_u, dim=1),
        }
        return DynamicsModelOutput(outputs=outputs_t, states=states_h_t, extras=extras)

    def forward_with_dropout(
        self,
        inputs: torch.Tensor,
        *,
        dropout_rate: float = 0.0,
        dropout_sampling: str = "uniform",
        dropout_beta: float = 1.0,
        participation: torch.Tensor | None = None,
        initial_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Dropout on the h component only (synaptic variables are never dropped).

        The base implementation would broadcast an (M,) mask against the (3M,)
        state; this override applies the mask to the h segment and returns
        h-only states, keeping the Trainer's dropout path usable.
        """
        if dropout_rate <= 0:
            out = self.forward(inputs, initial_state=initial_state, return_states=True)
            return out.states, out.outputs, out.states, out.outputs

        assert inputs.dim() == 3, "inputs must be (batch, T, input_dim)"
        batch_size, T = inputs.shape[0], inputs.shape[1]
        device = inputs.device
        M = self.config.latent_dim

        z0 = initial_state if initial_state is not None else self.init_state(batch_size, device)
        mask = self._sample_dropout_mask(M, dropout_rate, dropout_sampling,
                                         dropout_beta, participation, device)
        scale = 1.0 / (1.0 - dropout_rate)

        def rollout(apply_dropout: bool):
            """One T-step rollout; returns (h trajectory (B,T,M), readouts (B,T,O))."""
            z = z0.clone()
            hs, ys = [], []
            for t in range(T):
                z = self.recurrence(inputs[:, t], z, inputs=inputs)
                if apply_dropout:
                    h = z[..., :M] * mask * scale
                    z = torch.cat([h, z[..., M:]], dim=-1)
                hs.append(z[..., :M])
                ys.append(self.readout(z))
            return torch.stack(hs, dim=1), torch.stack(ys, dim=1)

        states_clean, outputs_clean = rollout(False)
        states_dropped, outputs_dropped = rollout(True)
        return states_clean, outputs_clean, states_dropped, outputs_dropped
