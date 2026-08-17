"""Gated RNN configuration (GRU / LSTM) and the Tiny RNN specialization.

``gated_rnn`` is the general gated-recurrence family: a single-layer
``nn.GRU`` or ``nn.LSTM`` taking raw inputs directly (no input projection),
plus a linear readout.

``tiny_rnn`` is the GRU specialization used for behavioral fitting in
Ji-An, Benna & Mattar (2025), adding the paper-specific options
``readout_FC`` (diagonal readout), ``output_h0`` (T+1 output alignment) and
``l1_weight`` (L1 penalty on recurrent weights, consumed by
:class:`~neuralrnn.train.objectives.behavioral.BehavioralObjective`).

Reference:
    Ji-An, L., Benna, M.K. & Mattar, M.G. (2025).
    Discovering cognitive strategies with tiny recurrent neural networks.
    Nature. https://doi.org/10.1038/s41586-025-09142-4
"""
from __future__ import annotations

import torch

from ...configuration_utils import NeuralRNNConfig

SUPPORTED_RNN_TYPES = ("GRU", "LSTM")


class GatedRNNConfig(NeuralRNNConfig):
    """Configuration for gated RNN models (GRU or LSTM).

    Args:
        input_dim:    Input feature dimension.
        latent_dim:   Hidden state dimension M.
        output_dim:   Output dimension.
        rnn_type:     Gated recurrent cell type, "GRU" or "LSTM".
        trainable_h0: If True, the initial hidden state is a learned parameter.
        trainable_c0: If True, the initial LSTM cell state is a learned
                      parameter (only valid for ``rnn_type="LSTM"``).
        dtype:        Model weight dtype, "float32" or "float64".
    """

    model_type = "gated_rnn"

    def __init__(
        self,
        input_dim: int = 1,
        latent_dim: int = 32,
        output_dim: int = 1,
        rnn_type: str = "GRU",
        trainable_h0: bool = False,
        trainable_c0: bool = False,
        dtype: str = "float32",
        **kwargs,
    ) -> None:
        rnn_type = str(rnn_type).upper()
        if rnn_type not in SUPPORTED_RNN_TYPES:
            raise ValueError(
                f"GatedRNNConfig.rnn_type must be one of {SUPPORTED_RNN_TYPES}, "
                f"got {rnn_type!r}"
            )
        if trainable_c0 and rnn_type != "LSTM":
            raise ValueError(
                "GatedRNNConfig.trainable_c0 is only valid for rnn_type='LSTM', "
                f"got rnn_type={rnn_type!r}"
            )
        super().__init__(input_dim=input_dim, latent_dim=latent_dim,
                         output_dim=output_dim, **kwargs)
        self.rnn_type = rnn_type
        self.trainable_h0 = trainable_h0
        self.trainable_c0 = trainable_c0
        self.dtype = dtype
        self._validate_dtype()

    def _validate_dtype(self):
        if self.dtype not in ("float32", "float64"):
            raise ValueError(
                f"{type(self).__name__}.dtype must be 'float32' or 'float64', "
                f"got {self.dtype}"
            )

    @property
    def torch_dtype(self):
        """The torch dtype matching ``self.dtype`` ("float32"/"float64")."""
        return torch.float32 if self.dtype == "float32" else torch.float64


class TinyRNNConfig(GatedRNNConfig):
    """Configuration for tiny GRU behavioral models (1-4 units).

    GRU specialization of :class:`GatedRNNConfig` for the tinyRNN paper
    (Ji-An, Benna & Mattar, Nature 2025). Maps to the original tinyRNN
    project's base_config fields:

        rnn_type -> rnn_type (only "GRU" is supported by TinyRNNModel)
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
        rnn_type:   RNN architecture; only "GRU" is supported
        readout_FC: If True, use fully-connected readout; if False, diagonal
        trainable_h0: If True, initial hidden state is a learned parameter
        output_h0:  If True, prepend the h0 readout to the output sequence
                    (T+1 length), matching the original tinyRNN convention
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
                         output_dim=output_dim, rnn_type=rnn_type,
                         trainable_h0=trainable_h0, dtype=dtype, **kwargs)
        self.readout_FC = readout_FC
        self.output_h0 = output_h0
        self.l1_weight = l1_weight
