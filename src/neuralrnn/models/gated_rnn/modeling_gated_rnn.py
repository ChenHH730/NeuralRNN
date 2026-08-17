"""Gated RNN models — single-layer GRU / LSTM with raw-input recurrence.

``gated_rnn`` is the general gated-recurrence family:
    input (input_dim) -> GRU/LSTM(input_dim, latent_dim) -> readout -> output

    NO extra input projection layer — the recurrent cell takes raw input
    directly, unlike families such as ctrnn/plrnn that use an input layer.

``tiny_rnn`` (:class:`TinyRNNModel`) is the GRU specialization for behavioral
fitting (Ji-An, Benna & Mattar, Nature 2025), adding diagonal readout
(``readout_FC=False``), the ``output_h0`` T+1 alignment convention, and an L1
hook on the recurrent weights.

The recurrent module is registered under a type-dependent attribute name
(``self.gru`` for GRU, ``self.lstm`` for LSTM) so that checkpoints saved by
the legacy ``models/tiny_rnn`` module (state-dict keys ``gru.*``) keep loading
unchanged. Access it generically via the :attr:`GatedRNNModel.rnn` property.

LSTM state convention (follows the stp_rnn packed-state precedent):
``init_state`` returns the packed state ``[h, c]`` of width ``2*M``;
``recurrence`` accepts either the packed ``(B, 2M)`` state or a plain
``(B, M)`` hidden state (cell state falls back to zeros, for analysis tools),
and returns the packed ``(B, 2M)`` state. ``forward`` uses ``nn.LSTM``
directly and reports ``states`` as the hidden sequence ``(B, T, M)``.

Reference:
    Ji-An, L., Benna, M.K. & Mattar, M.G. (2025).
    Discovering cognitive strategies with tiny recurrent neural networks.
    Nature. https://doi.org/10.1038/s41586-025-09142-4
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ...modeling_utils import NeuralDynamicsModel, DynamicsModelOutput
from ...auto.modeling_auto import register_model
from .configuration_gated_rnn import GatedRNNConfig, TinyRNNConfig


@register_model("gated_rnn")
class GatedRNNModel(NeuralDynamicsModel):
    """Single-layer gated RNN (GRU or LSTM) with linear readout.

    The recurrent cell receives raw input directly (no input projection).
    Implements the hard contract (``recurrence`` / ``readout`` / ``init_state``)
    with manually written gate equations, so analysis tools (fixed points,
    vector fields) work step-by-step; ``forward`` uses the efficient
    ``nn.GRU`` / ``nn.LSTM`` full-sequence path.

    Supports:
        - GRU or LSTM cells (config.rnn_type)
        - Trainable initial hidden state (and cell state for LSTM)
        - float32 / float64 weights (config.dtype)
        - L1 regularization hook on recurrent weights (:meth:`get_l1_loss`)
        - Optional diagonal readout / output_h0 via the tiny_rnn subclass
          config fields (inert on plain gated_rnn configs)
    """

    config_class = GatedRNNConfig

    _RNN_CLASSES = {"GRU": nn.GRU, "LSTM": nn.LSTM}

    def __init__(self, config: GatedRNNConfig) -> None:
        super().__init__(config)
        M = config.latent_dim

        # Recurrent layer — takes raw input directly (no input projection).
        # Registered under the type-dependent name ("gru"/"lstm") so legacy
        # tiny_rnn checkpoints with `gru.*` keys keep loading unchanged.
        rnn_cls = self._RNN_CLASSES[config.rnn_type]
        setattr(self, config.rnn_type.lower(),
                rnn_cls(config.input_dim, M, batch_first=True))

        # Readout layer (fully-connected; the tiny_rnn subclass may swap in a
        # diagonal readout when config.readout_FC=False)
        self.readout_layer = nn.Linear(M, config.output_dim)

        # Initial hidden state (and cell state for LSTM)
        if config.trainable_h0:
            self.h0 = nn.Parameter(torch.zeros(M))
        else:
            self.register_buffer("h0", torch.zeros(M))
        if config.rnn_type == "LSTM":
            if config.trainable_c0:
                self.c0 = nn.Parameter(torch.zeros(M))
            else:
                self.register_buffer("c0", torch.zeros(M))

        # Cast the whole model to the requested dtype (the original tinyRNN
        # code uses .double() / float64). This must happen after all submodules
        # and buffers have been registered.
        self.to(config.torch_dtype)
        self.apply_freeze_config()

    @property
    def rnn(self) -> nn.Module:
        """The recurrent module (``self.gru`` or ``self.lstm``)."""
        return getattr(self, self.config.rnn_type.lower())

    def _freeze_groups(self) -> dict[str, list[str]]:
        name = self.config.rnn_type.lower()
        return {
            "input": [rf"^{name}\.weight_ih_l0$", rf"^{name}\.bias_ih_l0$"],
            "recurrent": [rf"^{name}\.weight_hh_l0$", rf"^{name}\.bias_hh_l0$"],
            "output": [r"^readout_layer\.", r"^readout_coef$"],
            "h0": [r"^h0$", r"^c0$"],
        }

    def init_state(self, batch_size: int, device: str | torch.device = "cpu") -> torch.Tensor:
        """Initial state z_0: (B, M) for GRU; packed [h0, c0] (B, 2M) for LSTM."""
        h0 = self.h0.to(device).expand(batch_size, -1)
        if self.config.rnn_type == "LSTM":
            c0 = self.c0.to(device).expand(batch_size, -1)
            return torch.cat([h0, c0], dim=-1).contiguous()
        return h0.contiguous()

    # ==================== Hard contract ====================
    def recurrence(self, x_t: torch.Tensor, z_prev: torch.Tensor,
                   *, inputs: torch.Tensor | None = None) -> torch.Tensor:
        """Single-step gated transition (manual implementation using rnn weights).

        Args:
            x_t: (B, input_dim) — raw input at the current step
            z_prev: (B, M) for GRU; for LSTM either packed [h, c] (B, 2M) or a
                plain hidden state (B, M) (cell state falls back to zeros)

        Returns:
            z_t: (B, M) for GRU; packed [h, c] (B, 2M) for LSTM
        """
        if self.config.rnn_type == "GRU":
            return self._gru_step(x_t, z_prev)
        return self._lstm_step(x_t, z_prev)

    def _gru_step(self, x_t: torch.Tensor, z_prev: torch.Tensor) -> torch.Tensor:
        """GRU cell equations (PyTorch gate order: reset, update, new):
            r = sigmoid(W_ir @ x + b_ir + W_hr @ h + b_hr)
            z = sigmoid(W_iz @ x + b_iz + W_hz @ h + b_hz)
            n = tanh(W_in @ x + b_in + r * (W_hn @ h + b_hn))
            h_new = (1 - z) * n + z * h
        """
        # For GRU(input_dim, M): weight_ih is (3*M, input_dim), weight_hh is (3*M, M)
        W_ih = self.rnn.weight_ih_l0
        W_hh = self.rnn.weight_hh_l0
        b_ih = self.rnn.bias_ih_l0
        b_hh = self.rnn.bias_hh_l0
        M = z_prev.shape[-1]

        W_ir, W_iz, W_in = W_ih.split(M, dim=0)
        W_hr, W_hz, W_hn = W_hh.split(M, dim=0)
        b_ir, b_iz, b_in = b_ih.split(M, dim=0)
        b_hr, b_hz, b_hn = b_hh.split(M, dim=0)

        # Gate computations (x_t is raw input, NOT projected)
        r = torch.sigmoid(x_t @ W_ir.t() + b_ir + z_prev @ W_hr.t() + b_hr)
        z = torch.sigmoid(x_t @ W_iz.t() + b_iz + z_prev @ W_hz.t() + b_hz)
        n = torch.tanh(x_t @ W_in.t() + b_in + r * (z_prev @ W_hn.t() + b_hn))
        z_t = (1 - z) * n + z * z_prev
        return z_t

    def _lstm_step(self, x_t: torch.Tensor, z_prev: torch.Tensor) -> torch.Tensor:
        """LSTM cell equations (PyTorch gate order: input, forget, cell, output):
            i = sigmoid(W_ii @ x + b_ii + W_hi @ h + b_hi)
            f = sigmoid(W_if @ x + b_if + W_hf @ h + b_hf)
            g = tanh(W_ig @ x + b_ig + W_hg @ h + b_hg)
            o = sigmoid(W_io @ x + b_io + W_ho @ h + b_ho)
            c_new = f * c + i * g
            h_new = o * tanh(c_new)
        Returns the packed state [h_new, c_new] (B, 2M).
        """
        M = self.config.latent_dim
        if z_prev.shape[-1] == 2 * M:
            h_prev, c_prev = z_prev[..., :M], z_prev[..., M:]
        else:
            h_prev = z_prev
            c_prev = torch.zeros_like(h_prev)

        # For LSTM(input_dim, M): weight_ih is (4*M, input_dim), weight_hh is (4*M, M)
        W_ih = self.rnn.weight_ih_l0
        W_hh = self.rnn.weight_hh_l0
        b_ih = self.rnn.bias_ih_l0
        b_hh = self.rnn.bias_hh_l0

        W_ii, W_if, W_ig, W_io = W_ih.split(M, dim=0)
        W_hi, W_hf, W_hg, W_ho = W_hh.split(M, dim=0)
        b_ii, b_if, b_ig, b_io = b_ih.split(M, dim=0)
        b_hi, b_hf, b_hg, b_ho = b_hh.split(M, dim=0)

        i = torch.sigmoid(x_t @ W_ii.t() + b_ii + h_prev @ W_hi.t() + b_hi)
        f = torch.sigmoid(x_t @ W_if.t() + b_if + h_prev @ W_hf.t() + b_hf)
        g = torch.tanh(x_t @ W_ig.t() + b_ig + h_prev @ W_hg.t() + b_hg)
        o = torch.sigmoid(x_t @ W_io.t() + b_io + h_prev @ W_ho.t() + b_ho)
        c_new = f * c_prev + i * g
        h_new = o * torch.tanh(c_new)
        return torch.cat([h_new, c_new], dim=-1)

    def readout(self, z_t: torch.Tensor) -> torch.Tensor:
        """Readout: hidden state -> output.

        Args:
            z_t: (B, M) — or the packed LSTM state (B, 2M), whose h part is used

        Returns:
            y_t: (B, output_dim)
        """
        M = self.config.latent_dim
        if self.config.rnn_type == "LSTM" and z_t.shape[-1] == 2 * M:
            z_t = z_t[..., :M]
        if getattr(self.config, "readout_FC", True):
            return self.readout_layer(z_t)
        return self.readout_coef * z_t

    # ==================== Efficient full-sequence forward ====================
    def forward(self, inputs: torch.Tensor | None = None, *,
                initial_state: torch.Tensor | None = None,
                n_steps: int | None = None,
                return_states: bool = True) -> DynamicsModelOutput:
        """Full-sequence forward pass using nn.GRU / nn.LSTM for efficiency.

        - The cell takes raw input directly (no input projection)
        - If config.output_h0=True (tiny_rnn), prepend the initial hidden state
          to the state/output sequence (T+1 length)

        Args:
            inputs: (B, T, input_dim) — batch-first input sequence (required)
            initial_state: optional initial state — (B, M), or packed (B, 2M)
                for LSTM
            n_steps: unused (kept for interface compatibility)
            return_states: if True, return the hidden sequence (B, T, M)

        Returns:
            DynamicsModelOutput with outputs and states
        """
        if inputs is None:
            raise ValueError(f"{type(self).__name__} requires an input sequence")
        B, T, _ = inputs.shape
        device = inputs.device
        M = self.config.latent_dim

        z0 = initial_state if initial_state is not None else self.init_state(B, device)

        if self.config.rnn_type == "GRU":
            rnn_out, _ = self.rnn(inputs, z0.unsqueeze(0))  # (B, T, M)
            h0_seq = z0
        else:
            if z0.shape[-1] == 2 * M:
                h0, c0 = z0[..., :M], z0[..., M:]
            else:
                h0, c0 = z0, torch.zeros_like(z0)
            rnn_out, _ = self.rnn(inputs, (h0.unsqueeze(0), c0.unsqueeze(0)))
            h0_seq = h0

        # output_h0 (tiny_rnn): prepend initial hidden state to the sequence
        # This matches the original: rnn_out = torch.cat((h0_expand, rnn_out), 0)
        if getattr(self.config, "output_h0", False):
            rnn_out = torch.cat((h0_seq.unsqueeze(1), rnn_out), dim=1)  # (B, T+1, M)

        # Readout
        if getattr(self.config, "readout_FC", True):
            outputs = self.readout_layer(rnn_out)
        else:
            outputs = self.readout_coef * rnn_out

        return DynamicsModelOutput(
            outputs=outputs,
            states=rnn_out if return_states else None,
        )

    def get_l1_loss(self) -> torch.Tensor:
        """L1 norm of the recurrent weights (weight_hh), for regularized
        objectives such as BehavioralObjective."""
        return self.rnn.weight_hh_l0.abs().sum()


@register_model("tiny_rnn")
class TinyRNNModel(GatedRNNModel):
    """Tiny GRU RNN for behavioral prediction (1-4 hidden units).

    GRU specialization of :class:`GatedRNNModel` matching the original tinyRNN
    RNNnet architecture exactly:
        input (input_dim) -> GRU(input_dim, hidden_dim) -> readout -> logits

    Adds the paper-specific options on top of the gated_rnn base:
        - Fully-connected or diagonal readout (config.readout_FC)
        - output_h0: prepend h0 to the output sequence (T+1 length)
        - L1 regularization on recurrent weights (via config.l1_weight,
          consumed by BehavioralObjective)

    For the PRL task: input is [action, stage2, reward] (3 features),
    output is logits over actions (2 classes).
    """

    config_class = TinyRNNConfig

    def __init__(self, config: TinyRNNConfig) -> None:
        if config.rnn_type != "GRU":
            raise ValueError(f"Unsupported rnn_type: {config.rnn_type}. "
                             f"Supported: 'GRU'")
        super().__init__(config)

        if not config.readout_FC:
            # Diagonal readout (inverse temperature scaling): replace the FC
            # readout created by the base class.
            assert config.latent_dim == config.output_dim, (
                f"Diagonal readout requires latent_dim == output_dim, "
                f"got {config.latent_dim} vs {config.output_dim}"
            )
            del self.readout_layer
            self.readout_coef = nn.Parameter(
                torch.ones(1, dtype=config.torch_dtype))
            # Re-apply freeze flags so they cover the swapped readout
            # (freeze-layering invariant: idempotent, polymorphic).
            self.apply_freeze_config()
