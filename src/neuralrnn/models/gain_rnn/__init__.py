"""Gain RNN family: per-neuron gain/bias rate maps (gain_rnn) and short-term
plasticity as a dynamic gain parameterization (stp_rnn)."""
from .configuration_gain_rnn import (
    GainRNNConfig,
    StpRNNConfig,
    SUPPORTED_GAIN_POSITIONS,
    SUPPORTED_NOISE_POSITIONS,
    SUPPORTED_STP_INIT,
    validate_gain_position,
    validate_noise_position,
    validate_stp_init,
)
from .modeling_gain_rnn import GainRNNModel, StpRNNModel, make_stp_masks

__all__ = [
    "GainRNNConfig",
    "GainRNNModel",
    "StpRNNConfig",
    "StpRNNModel",
    "SUPPORTED_GAIN_POSITIONS",
    "SUPPORTED_NOISE_POSITIONS",
    "SUPPORTED_STP_INIT",
    "validate_gain_position",
    "validate_noise_position",
    "validate_stp_init",
    "make_stp_masks",
]
