"""Latent circuit model — low-dimensional recurrent circuit inference.

Ported from Langdon & Engel (2025), Nature Neuroscience.
"Latent circuit inference from heterogeneous neural responses during cognitive tasks."
"""
from .configuration_latent_circuit import LatentCircuitConfig
from .modeling_latent_circuit import LatentCircuitModel

__all__ = ["LatentCircuitConfig", "LatentCircuitModel"]
