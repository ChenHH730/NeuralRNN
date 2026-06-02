"""Tiny RNN model — small GRU/RNN for behavioral fitting (1-4 units).

Ported from tinyRNN (Ji-An, Benna & Mattar, Nature 2025).
Uses standard PyTorch nn.GRU for the recurrent layer.

Architecture (matches original RNNnet exactly):
    input (input_dim) -> GRU(input_dim, latent_dim) -> readout -> logits

    NO extra input projection layer — GRU takes raw input directly.
    This is different from other NeuralRNN models that use input_layer.

Key design choices:
  - Batch-first contract: input (B, T, input_dim), output (B, T, output_dim)
  - Internally converts to seq_len-first for nn.GRU (matching original)
  - Implements GRU cell step manually in recurrence() for analysis compatibility
  - Supports fully-connected or diagonal readout
  - Supports trainable initial hidden state
  - Supports output_h0: prepend initial hidden state to output (T+1 length)

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
from .configuration_tiny_rnn import TinyRNNConfig


@register_model("tiny_rnn")
class TinyRNNModel(NeuralDynamicsModel):
    """Tiny GRU RNN for behavioral prediction (1-4 hidden units).

    Matches the original tinyRNN RNNnet architecture exactly:
        input (input_dim) -> GRU(input_dim, hidden_dim) -> readout -> logits

    The GRU receives raw input directly (no input projection layer).
    For PRL task: input is [action, stage2, reward] (3 features),
    output is logits over actions (2 classes).

    Supports:
        - Standard GRU (nn.GRU)
        - Fully-connected or diagonal readout
        - Trainable initial hidden state
        - output_h0: prepend h0 to output sequence (T+1 length)
        - L1 regularization on recurrent weights (via config.l1_weight)
    """

    config_class = TinyRNNConfig

    def __init__(self, config: TinyRNNConfig) -> None:
        super().__init__(config)
        M = config.latent_dim

        # Recurrent layer — takes raw input directly (no input projection)
        # This matches original: nn.GRU(input_dim, hidden_dim)
        if config.rnn_type == "GRU":
            self.gru = nn.GRU(config.input_dim, M, batch_first=True)
        else:
            raise ValueError(f"Unsupported rnn_type: {config.rnn_type}. "
                             f"Supported: 'GRU'")

        # Readout layer
        if config.readout_FC:
            self.readout_layer = nn.Linear(M, config.output_dim)
        else:
            # Diagonal readout (inverse temperature scaling)
            assert M == config.output_dim, (
                f"Diagonal readout requires latent_dim == output_dim, "
                f"got {M} vs {config.output_dim}"
            )
            self.readout_coef = nn.Parameter(torch.ones(1))

        # Initial hidden state
        if config.trainable_h0:
            self.h0 = nn.Parameter(torch.zeros(M))
        else:
            self.register_buffer("h0", torch.zeros(M))

        self._readout_FC = config.readout_FC
        self._output_h0 = config.output_h0

    def init_state(self, batch_size: int, device: str | torch.device = "cpu") -> torch.Tensor:
        """Initial hidden state z_0: (B, M)."""
        return self.h0.to(device).expand(batch_size, -1).contiguous()

    # ==================== Hard contract ====================
    def recurrence(self, x_t: torch.Tensor, z_prev: torch.Tensor,
                   *, inputs: torch.Tensor | None = None) -> torch.Tensor:
        """Single-step GRU transition (manual implementation using gru weights).

        Matches original RNNnet: GRU takes raw input directly, no projection.

        Implements the GRU cell equations:
            r = sigmoid(W_ir @ x + b_ir + W_hr @ h + b_hr)
            z = sigmoid(W_iz @ x + b_iz + W_hz @ h + b_hz)
            n = tanh(W_in @ x + b_in + r * (W_hn @ h + b_hn))
            h_new = (1 - z) * n + z * h

        Args:
            x_t: (B, input_dim) — raw input at current trial
            z_prev: (B, M) — previous hidden state

        Returns:
            z_t: (B, M) — new hidden state
        """
        # Extract GRU weights
        # For GRU(input_dim, M): weight_ih is (3*M, input_dim), weight_hh is (3*M, M)
        W_ih = self.gru.weight_ih_l0  # (3*M, input_dim)
        W_hh = self.gru.weight_hh_l0  # (3*M, M)
        b_ih = self.gru.bias_ih_l0    # (3*M,)
        b_hh = self.gru.bias_hh_l0    # (3*M,)
        M = z_prev.shape[-1]

        # Split into r, z, n gates (order matches PyTorch GRU: reset, input, new)
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

    def readout(self, z_t: torch.Tensor) -> torch.Tensor:
        """Readout: hidden state -> action logits.

        Args:
            z_t: (B, M)

        Returns:
            y_t: (B, output_dim) — action logits
        """
        if self._readout_FC:
            return self.readout_layer(z_t)
        else:
            return self.readout_coef * z_t

    # ==================== Efficient full-sequence forward ====================
    def forward(self, inputs: torch.Tensor | None = None, *,
                initial_state: torch.Tensor | None = None,
                n_steps: int | None = None,
                return_states: bool = True) -> DynamicsModelOutput:
        """Full-sequence forward pass using nn.GRU for efficiency.

        Matches original RNNnet behavior:
        - GRU takes raw input directly (no input projection)
        - If output_h0=True, prepend initial hidden state to output (T+1 length)

        Args:
            inputs: (B, T, input_dim) — batch-first input sequence
            initial_state: (B, M) — optional initial state
            n_steps: unused (kept for interface compatibility)
            return_states: if True, return hidden states

        Returns:
            DynamicsModelOutput with outputs and states
        """
        if inputs is not None:
            B, T, _ = inputs.shape
            device = inputs.device
        else:
            raise ValueError("TinyRNNModel requires input sequence")

        z0 = initial_state if initial_state is not None else self.init_state(B, device)

        # GRU forward: input is raw (B, T, input_dim), no projection
        # z0 needs to be (1, B, M) for nn.GRU
        rnn_out, _ = self.gru(inputs, z0.unsqueeze(0))  # (B, T, M)

        # output_h0: prepend initial hidden state to output sequence
        # This matches original: rnn_out = torch.cat((h0_expand, rnn_out), 0)
        if self._output_h0:
            rnn_out = torch.cat((z0.unsqueeze(1), rnn_out), dim=1)  # (B, T+1, M)

        # Readout
        if self._readout_FC:
            outputs = self.readout_layer(rnn_out)
        else:
            outputs = self.readout_coef * rnn_out

        return DynamicsModelOutput(
            outputs=outputs,
            states=rnn_out if return_states else None,
        )

    def get_l1_loss(self) -> torch.Tensor:
        """Compute L1 regularization loss on recurrent weights.

        Returns:
            Scalar tensor with L1 norm of recurrent weights.
        """
        if self.config.rnn_type == "GRU":
            # L1 on recurrent weights (weight_hh) only
            return self.gru.weight_hh_l0.abs().sum()
        return torch.tensor(0.0)
