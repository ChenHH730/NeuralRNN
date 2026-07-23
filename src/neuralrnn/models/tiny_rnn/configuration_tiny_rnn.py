"""Tiny RNN configuration.

Reference:
    Ji-An, L., Benna, M.K. & Mattar, M.G. (2025).
    Discovering cognitive strategies with tiny recurrent neural networks.
    Nature. https://doi.org/10.1038/s41586-025-09142-4
"""
from __future__ import annotations

import torch

from ...configuration_utils import NeuralRNNConfig


class TinyRNNConfig(NeuralRNNConfig):
    """Configuration for tiny GRU/RNN behavioral models (1-4 units).

    Maps to the original tinyRNN project's base_config fields:
        rnn_type -> rnn_type
        hidden_dim -> latent_dim
        input_dim -> input_dim (default 3: [action, stage2, reward])
        output_dim -> output_dim (default 2: binary choice)
        readout_FC -> readout_FC
        trainable_h0 -> trainable_h0
        l1_weight -> l1_weight (L1 regularization on recurrent weights)

    Args:
        input_dim:  Input feature dimension (3 for [action, stage2, reward])
        latent_dim: Hidden state dimension (1-4 typically)
        output_dim: Output dimension (2 for binary choice)
        rnn_type:   RNN architecture ('GRU' or 'SGRU')
        readout_FC: If True, use fully-connected readout; if False, diagonal
        trainable_h0: If True, initial hidden state is a learned parameter
        l1_weight:  L1 regularization coefficient on recurrent weights
        dtype:      Model weight dtype, "float32" or "float64". The original
                    tinyRNN code uses float64 (``.double()``). Default "float32".
    """

    model_type = "tiny_rnn"

    def __init__(
        self,
        input_dim: int = 3,
        latent_dim: int = 2,
        output_dim: int = 2,
        rnn_type: str = "GRU",
        readout_FC: bool = True,
        trainable_h0: bool = False,
        output_h0: bool = False,
        l1_weight: float = 1e-5,
        dtype: str = "float32",
        **kwargs,
    ) -> None:
        super().__init__(input_dim=input_dim, latent_dim=latent_dim,
                         output_dim=output_dim, **kwargs)
        self.rnn_type = rnn_type
        self.readout_FC = readout_FC
        self.trainable_h0 = trainable_h0
        self.output_h0 = output_h0
        self.l1_weight = l1_weight
        self.dtype = dtype
        self._validate_dtype()

    def _validate_dtype(self):
        if self.dtype not in ("float32", "float64"):
            raise ValueError(f"TinyRNNConfig.dtype must be 'float32' or 'float64', got {self.dtype}")

    @property
    def torch_dtype(self):
        """The torch dtype matching ``self.dtype`` ("float32"/"float64")."""
        return torch.float32 if self.dtype == "float32" else torch.float64
