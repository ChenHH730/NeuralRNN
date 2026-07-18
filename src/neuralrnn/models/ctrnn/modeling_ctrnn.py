"""CTRNN family model implementations (Paradigm A reference implementation).

Ported from Song 2016's CTRNN:
    r(t+dt) = r(t) + (dt/tau)[-r(t) + f(W_r r + W_x x + b)]
            = (1-alpha) r + alpha * f(pre_activation)

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
        return self.h0.to(device).expand(batch_size, -1).contiguous()

    def _recurrent_weight(self) -> torch.Tensor:
        W = self.h2h.weight
        if self.dale_mask is not None:
            # Enforce column signs consistent with Dale's law: |W| @ sign-diag
            W = W.abs() @ self.dale_mask
        return W

    # ---------------- hard contract ----------------
    def recurrence(self, x_t, z_prev, *, inputs=None):
        W = self._recurrent_weight()
        mode = self.config.nonlinearity_mode
        # "rate": the recurrent matrix reads the firing rate r = f(z); otherwise it reads z.
        rec_in = self.act(z_prev) if mode == "rate" else z_prev
        pre = self.input2h(x_t) + torch.nn.functional.linear(rec_in, W, self.h2h.bias)
        if self.config.sigma_rec > 0 and self.training:
            if self.config.noise_alpha_scaling:
                # Reference formula: sqrt(2 * alpha * sigma^2) * N(0,1)
                noise_std = (2 * self.alpha * self.config.sigma_rec ** 2) ** 0.5
            else:
                # Default: sigma * N(0,1)
                noise_std = self.config.sigma_rec
            pre = pre + noise_std * torch.randn_like(pre)
        if mode == "post_blend":
            # nn-brain original formula: f((1-α)z + α·pre)
            z = self.act((1 - self.alpha) * z_prev + self.alpha * pre)
        elif mode == "rate":
            # Classic firing-rate form: z' = (1-α)z + α·(W@r + B@x + b); no f on the update
            z = (1 - self.alpha) * z_prev + self.alpha * pre
        else:
            # Standard Euler discretization: (1-α)z + α·f(pre)
            z = (1 - self.alpha) * z_prev + self.alpha * self.act(pre)
        return z

    def readout(self, z_t):
        return self.readout_layer(z_t)

@register_model("ei_rnn")
class EIRNNModel(CTRNNModel):
    """Excitatory-Inhibitory RNN with Dale's principle.

    Key differences from vanilla CTRNN:
    - Dale's constraint: E units have non-negative outgoing weights, I units non-positive.
    - Readout restricted to excitatory units only (long-range projections are excitatory).
    - Proper EI initialization with E/I balance (E weights scaled by I/E ratio).

    Reference:
        Song, H.F., Yang, G.R. and Wang, X.J., 2016.
        Training excitatory-inhibitory recurrent neural networks
        for cognitive tasks: a simple and flexible framework.
        PLoS computational biology, 12(2).
    """
    config_class = EIRNNConfig

    def __init__(self, config: EIRNNConfig) -> None:
        super().__init__(config)
        self.e_size = int(round(config.latent_dim * config.ei_ratio))
        self.i_size = config.latent_dim - self.e_size

        # Override readout to only use excitatory units if configured
        if config.readout_e_only:
            self.readout_layer = nn.Linear(self.e_size, config.output_dim)

        # Apply EI-specific initialization
        self._ei_init_weights()

    def _ei_init_weights(self) -> None:
        """Initialize weights with EI balance (E weights scaled by I/E ratio).

        This matches the reference implementation where E columns of the recurrent
        weight matrix are scaled by (e_size / i_size) to balance excitatory and
        inhibitory inputs to each unit.
        """
        e_size = self.e_size
        i_size = self.i_size
        if i_size == 0:
            return

        with torch.no_grad():
            # Scale E columns of recurrent weight by E-I ratio
            # This balances E and I contributions: each E unit's weight is scaled down
            # because there are more E units than I units
            self.h2h.weight[:, :e_size] /= (e_size / i_size)

    def readout(self, z_t: torch.Tensor) -> torch.Tensor:
        """Readout from excitatory units only (long-range projections are excitatory)."""
        if self.config.readout_e_only:
            return self.readout_layer(z_t[:, :self.e_size])
        return self.readout_layer(z_t)
