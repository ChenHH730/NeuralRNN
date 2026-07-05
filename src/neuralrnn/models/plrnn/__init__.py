"""PLRNN family (Paradigm B reference implementation): shallow / dendritic / almost-linear PLRNN."""
from .configuration_plrnn import ShallowPLRNNConfig, DendPLRNNConfig, ALRNNConfig
from .modeling_plrnn import ShallowPLRNNModel, DendPLRNNModel, ALRNNModel

__all__ = [
    "ShallowPLRNNConfig", "DendPLRNNConfig", "ALRNNConfig",
    "ShallowPLRNNModel", "DendPLRNNModel", "ALRNNModel",
]
