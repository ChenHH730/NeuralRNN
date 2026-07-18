"""Multi-area RNN family: cascaded areas with structured inter-area connectivity.

Ported for Kleinman et al. 2025 (eLife, "The information bottleneck as a
principle underlying multi-area cortical representations during decision-making").

The family is a thin Contract-A adapter over ConstrainedRNNModel: all dynamics
(Euler CTRNN, Dale constraints, nonlinearity modes, freeze) are inherited; this
family only generates the block-structured masks (rec/in/out) and per-area Dale
sign vectors from a small set of area-level hyperparameters.
"""
from .configuration_multiarea_rnn import MultiAreaRNNConfig
from .masks import build_multiarea_masks, area_slices, area_ei_indices
from .modeling_multiarea_rnn import MultiAreaRNNModel, rescale_effective_spectral_radius

__all__ = [
    "MultiAreaRNNConfig",
    "MultiAreaRNNModel",
    "build_multiarea_masks",
    "area_slices",
    "area_ei_indices",
    "rescale_effective_spectral_radius",
]
