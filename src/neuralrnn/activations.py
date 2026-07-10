"""Centralized activation factory for NeuralRNN.

All built-in models should select their nonlinearities through
:func:`get_activation` so that activation names are consistent across the
framework and new activations only need to be registered in one place.
"""
from __future__ import annotations

from functools import partial
from typing import Callable

import torch
import torch.nn.functional as F

ActivationFn = Callable[[torch.Tensor], torch.Tensor]

# Non-parameterized activations.  We deliberately use the direct PyTorch
# implementations (e.g. ``torch.relu``, ``F.softplus``) so that the default
# behavior is byte-for-byte identical to the per-model _ACT dictionaries that
# existed before this module was introduced.
_BASE_ACTIVATIONS: dict[str, ActivationFn] = {
    "relu": torch.relu,
    "tanh": torch.tanh,
    "sigmoid": torch.sigmoid,
    "selu": torch.selu,
    "gelu": F.gelu,
    "silu": F.silu,
    "swish": F.silu,  # alias
}


def get_activation(name: str, **kwargs) -> ActivationFn:
    """Return a callable activation function by name.

    Supported names:

    * ``"relu"`` -> :func:`torch.relu`
    * ``"tanh"`` -> :func:`torch.tanh`
    * ``"sigmoid"`` -> :func:`torch.sigmoid`
    * ``"selu"`` -> :func:`torch.selu`
    * ``"gelu"`` -> :func:`torch.nn.functional.gelu`
    * ``"silu"`` / ``"swish"`` -> :func:`torch.nn.functional.silu`
    * ``"softplus"`` (kwarg ``beta``) -> :func:`torch.nn.functional.softplus`
    * ``"leaky_relu"`` / ``"leakyrelu"`` (kwarg ``negative_slope``) ->
      :func:`torch.nn.functional.leaky_relu`
    * ``"elu"`` (kwarg ``alpha``) -> :func:`torch.nn.functional.elu`

    Args:
        name: Activation name. Case-insensitive and ``-``/``_`` neutral.
        **kwargs: Activation-specific parameters (e.g. ``beta`` for softplus).

    Returns:
        A callable ``f(x) -> Tensor``.

    Raises:
        ValueError: If ``name`` is unknown or unexpected kwargs are passed.
    """
    name = name.lower().replace("-", "_")

    if name in ("leakyrelu", "leaky_relu"):
        negative_slope = kwargs.pop("negative_slope", kwargs.pop("alpha", 1e-2))
        return partial(F.leaky_relu, negative_slope=negative_slope)

    if name == "softplus":
        beta = kwargs.pop("beta", 1.0)
        return partial(F.softplus, beta=beta)

    if name == "elu":
        alpha = kwargs.pop("alpha", 1.0)
        return partial(F.elu, alpha=alpha)

    if kwargs:
        raise ValueError(
            f"Activation '{name}' does not accept kwargs {set(kwargs)!r}."
        )

    if name not in _BASE_ACTIVATIONS:
        raise ValueError(
            f"Unknown activation '{name}'. "
            f"Supported: {sorted(_BASE_ACTIVATIONS)} plus "
            "softplus, leaky_relu/leakyrelu, elu."
        )

    return _BASE_ACTIVATIONS[name]


SUPPORTED_ACTIVATIONS: tuple[str, ...] = tuple(
    sorted(_BASE_ACTIVATIONS) + ["softplus", "leaky_relu", "leakyrelu", "elu"]
)
