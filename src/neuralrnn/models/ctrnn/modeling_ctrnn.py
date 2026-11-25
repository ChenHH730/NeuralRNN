"""CTRNN family model implementations (Paradigm A reference implementation).

Default rate-state update (pre_activation):
    r(t+dt) = r(t) + (dt/tau)[-r(t) + f(W_r r + W_x x + b)]
            = (1-alpha) r + alpha * f(pre_activation)

Song 2016's pycog instead integrates currents with nonlinearity_mode="rate"
and reads firing rates. EIRNNModel uses that form by default, with configurable
input, recurrent and output sign constraints and structural masks.

This file demonstrates the standard "Contract A (model adapter)" pattern for other
models to copy:
  - Inherit NeuralDynamicsModel, set config_class, register with @register_model
  - __init__ reads parameters only from config
  - Implement recurrence / readout (hard contract)
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ...activations import get_activation
from ...modeling_utils import NeuralDynamicsModel
from ...auto.modeling_auto import register_model
from .configuration_ctrnn import CTRNNConfig, EIRNNConfig


@register_model("ctrnn")
class CTRNNModel(NeuralDynamicsModel):
    """Continuous-time RNN with Euler discretization (Paradigm A reference implementation).

    Single-step update (z: latent state, x: external input, f: activation, α = config.alpha):

        pre = W_in @ x_t + W_rec @ u + b           (u = f(z) if nonlinearity_mode="rate" else z)
        z_t = (1-α)·z + α·f(pre)                    (pre_activation; post_blend applies f after
                                                     the blend, rate applies no f on the update)

    Shapes: x_t (B, input_dim) -> z_t (B, latent_dim); readout z_t -> y_t (B, output_dim).
    With dt=None (α=1) this reduces to the discrete-time vanilla RNN.
    """
    config_class = CTRNNConfig

    def __init__(self, config: CTRNNConfig) -> None:
        super().__init__(config)
        M = config.latent_dim
        self.alpha = config.alpha
        self.act = get_activation(config.activation)

        self.input2h = nn.Linear(config.input_dim, M)
        self.h2h = nn.Linear(M, M)
        self.readout_layer = nn.Linear(M, config.output_dim)

        if config.trainable_h0:
            self.h0 = nn.Parameter(torch.zeros(M))
        else:
            self.register_buffer("h0", torch.zeros(M))

        # Dale constraints (EI variant): fixed sign mask + non-negative weight magnitudes; see EI_RNN.ipynb.
        if config.dale:
            if getattr(config, "dale_signs", None) is not None:
                # Per-unit sign vector (e.g. per-area E/I splits); validated in config.
                sign = torch.tensor(config.dale_signs, dtype=torch.float32)
            else:
                n_exc = int(round(M * config.ei_ratio))
                sign = torch.ones(M)
                sign[n_exc:] = -1.0
            self.register_buffer("dale_mask", torch.diag(sign))  # (M,M)
        else:
            self.dale_mask = None

        self.apply_freeze_config()

    def _freeze_groups(self) -> dict[str, list[str]]:
        return {
            "input": [r"^input2h\."],
            "recurrent": [r"^h2h\."],
            "output": [r"^readout_layer\."],
            "h0": [r"^h0$"],
        }

    def init_state(self, batch_size, device="cpu"):
        """Initial state z_0 = h0 broadcast to (batch_size, latent_dim)."""
        return self.h0.to(device).expand(batch_size, -1).contiguous()

    def _recurrent_weight(self) -> torch.Tensor:
        W = self.h2h.weight
        if self.dale_mask is not None:
            # Enforce column signs consistent with Dale's law: |W| @ sign-diag
            W = W.abs() @ self.dale_mask
        return W

    # ---------------- hard contract ----------------
    def recurrence(self, x_t, z_prev, *, inputs=None):
        """Single Euler step. x_t: (B, input_dim), z_prev: (B, M) -> z_t: (B, M).

        Recurrent noise (sigma_rec > 0) is added to the pre-activation in
        training mode by default; evaluate_noise enables it in eval mode.
        See docs/DYNAMICS_CONTRACT.md for placement and variance.
        """
        W = self._recurrent_weight()
        mode = self.config.nonlinearity_mode
        # "rate": the recurrent matrix reads the firing rate r = f(z); otherwise it reads z.
        rec_in = self.act(z_prev) if mode == "rate" else z_prev
        pre = self.input2h(x_t) + torch.nn.functional.linear(rec_in, W, self.h2h.bias)
        return self._euler_update(pre, z_prev, self.act)

    def readout(self, z_t):
        """Linear readout. z_t: (B, M) -> y_t: (B, output_dim)."""
        return self.readout_layer(z_t)

@register_model("ei_rnn")
class EIRNNModel(CTRNNModel):
    """E-I dynamics with pycog current/rate semantics and configurable constraints.

    Rate mode stores currents and exposes firing rates to observation heads.
    The other two modes store activity and read it directly. Effective weights
    enforce signs and connectivity during every call, including training.
    """
    config_class = EIRNNConfig

    def __init__(self, config: EIRNNConfig) -> None:
        super().__init__(config)
        n = config.latent_dim
        signs = (torch.tensor(config.dale_signs, dtype=self.h2h.weight.dtype)
                 if config.dale_signs is not None else torch.tensor(
                     [1.] * int(n * config.ei_ratio) + [-1.] * (n - int(n * config.ei_ratio))))
        self.register_buffer("ei_signs", signs)
        self.register_buffer("excitatory_indices", (signs > 0).nonzero().flatten())
        self.e_size = self.excitatory_indices.numel()
        self.i_size = n - self.e_size
        if config.readout_e_only and self.e_size == 0:
            raise ValueError("readout_e_only requires at least one excitatory unit")
        self.dale_mask = torch.diag(signs) if config.dale else None
        self.input2h = nn.Linear(config.input_dim, n, bias=config.input_bias)
        self.h2h = nn.Linear(n, n, bias=config.recurrent_bias)
        self.readout_layer = nn.Linear(self.e_size if config.readout_e_only else n,
                                       config.output_dim, bias=config.output_bias)
        for name, shape in (("in_mask", (config.input_dim, n)),
                            ("rec_mask", (n, n)), ("out_mask", (n, config.output_dim))):
            value = getattr(config, name)
            mask = torch.ones(shape) if value is None else torch.as_tensor(value, dtype=torch.get_default_dtype())
            if tuple(mask.shape) != shape or not ((mask == 0) | (mask == 1)).all():
                raise ValueError(f"{name} must be a binary mask of shape {shape}")
            self.register_buffer(name, mask.clone())
        if config.no_self_connections:
            self.rec_mask.fill_diagonal_(0)
        self._ei_init_weights()
        self.apply_freeze_config()

    def _positive(self, weight):
        return weight.relu() if self.config.ei_positive_func == "rectify" else weight.abs()

    def _input_weight(self):
        weight = self.input2h.weight
        if self.config.positive_input_weights:
            weight = self._positive(weight)
        return weight * self.in_mask.T

    def _recurrent_weight(self):
        weight = self.h2h.weight
        if self.config.dale:
            weight = self._positive(weight) * self.ei_signs
        return weight * self.rec_mask

    def _output_weight(self):
        weight = self.readout_layer.weight
        signs, mask = self.ei_signs, self.out_mask
        if self.config.readout_e_only:
            signs = signs.index_select(0, self.excitatory_indices)
            mask = mask.index_select(0, self.excitatory_indices)
        if self.config.positive_output_weights:
            weight = self._positive(weight)
            if self.config.dale:
                weight = weight * signs
        return weight * mask.T

    def _ei_init_weights(self):
        cfg = self.config
        with torch.no_grad():
            if cfg.init_method == "pycog":
                self.input2h.weight.uniform_(-.1, .1)
                self.readout_layer.weight.uniform_(-.1, .1)
                weight = torch.distributions.Gamma(2., 20.).sample(self.h2h.weight.shape)
                if cfg.rec_mask is None and cfg.dale and self.e_size and self.i_size:
                    # pycog trainer's default mean-balanced, row-normalized Crec.
                    scale = self.rec_mask.clone()
                    e = self.ei_signs > 0
                    exc = scale[:, e].sum(1)
                    inh = scale[:, ~e].sum(1)
                    scale[:, ~e] *= (exc / inh.clamp_min(1)).unsqueeze(1)
                    scale /= scale.norm(dim=1, keepdim=True).clamp_min(1e-12)
                    weight *= scale
                self.h2h.weight.copy_(weight)
            elif cfg.init_method == "gamma":
                self.h2h.weight.copy_(torch.distributions.Gamma(4., 4.).sample(self.h2h.weight.shape))
            elif self.e_size and self.i_size:
                self.h2h.weight[:, self.excitatory_indices] *= self.i_size / self.e_size
            # Start constrained magnitudes above zero; rectify still gates raw
            # parameters that become negative during optimization, as in pycog.
            if cfg.positive_input_weights:
                self.input2h.weight.abs_()
            if cfg.dale:
                self.h2h.weight.abs_()
            if cfg.positive_output_weights:
                self.readout_layer.weight.abs_()
            self.h2h.weight.mul_(self.rec_mask)
            if cfg.spectral_radius is not None:
                radius = torch.linalg.eigvals(self._recurrent_weight()).abs().max()
                if radius <= torch.finfo(radius.dtype).eps:
                    raise ValueError("Cannot scale a zero-radius recurrent graph; set spectral_radius=None")
                self.h2h.weight.mul_(cfg.spectral_radius / radius)
            if cfg.init_method == "pycog":
                for layer in (self.input2h, self.h2h, self.readout_layer):
                    if layer.bias is not None:
                        layer.bias.zero_()
            self.h0.fill_(cfg.h0_init)

    def recurrence(self, x_t, z_prev, *, inputs=None):
        rate = self.readout_features(z_prev)
        pre = (torch.nn.functional.linear(x_t, self._input_weight(), self.input2h.bias)
               + torch.nn.functional.linear(rate, self._recurrent_weight(), self.h2h.bias))
        return self._euler_update(pre, z_prev, self.act)

    def readout_features(self, z_t: torch.Tensor) -> torch.Tensor:
        """Full activity vector; in rate mode returned states are signed currents."""
        return self.act(z_t) if self.config.nonlinearity_mode == "rate" else z_t

    def readout(self, z_t: torch.Tensor) -> torch.Tensor:
        features = self.readout_features(z_t)
        if self.config.readout_e_only:
            features = features.index_select(-1, self.excitatory_indices)
        return torch.nn.functional.linear(features, self._output_weight(), self.readout_layer.bias)
