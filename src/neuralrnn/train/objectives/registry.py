"""Objective registry and factory.

Mirrors the ``AutoConfig`` / ``AutoModel`` pattern: every built-in objective is
registered with a string name and can be instantiated through ``build_objective``.
Custom objectives can be registered with the ``@register_objective`` decorator.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import Objective

OBJECTIVE_REGISTRY: dict[str, type["Objective"]] = {}


def register_objective(name: str):
    """Class decorator that registers an Objective subclass under ``name``."""
    def decorator(cls: type["Objective"]) -> type["Objective"]:
        """Register ``cls`` under ``name`` (idempotent for the same class)."""
        existing = OBJECTIVE_REGISTRY.get(name)
        if existing is not None:
            # Idempotent re-registration: tools that re-execute modules
            # (importlib.reload, lazydocs' load_module) recreate the same
            # class object; registering it again is a no-op, not a conflict.
            if (existing.__name__ == cls.__name__
                    and existing.__module__ == cls.__module__):
                return cls
            raise ValueError(
                f"Objective name '{name}' is already registered to "
                f"{OBJECTIVE_REGISTRY[name].__name__}"
            )
        OBJECTIVE_REGISTRY[name] = cls
        return cls
    return decorator


def build_objective(name_or_instance: str | Objective | dict | None = None,
                    **kwargs) -> Objective | None:
    """Build an Objective from a registry name or pass an existing instance through.

    Args:
        name_or_instance: one of
            - ``None``: returns ``None`` (useful for optional objectives).
            - an ``Objective`` instance: returned unchanged.
            - a ``str`` registry key: the corresponding class is instantiated.
            - a ``dict`` with key ``"name"``; remaining keys are constructor kwargs.
        **kwargs: additional constructor kwargs when ``name_or_instance`` is a string.

    Returns:
        An Objective instance, or None if the input was None.

    Raises:
        ValueError: if the name is not registered.
        TypeError: if the input type is not supported.
    """
    if name_or_instance is None:
        return None
    if isinstance(name_or_instance, dict):
        kwargs = {**name_or_instance, **kwargs}
        name_or_instance = kwargs.pop("name")
    if hasattr(name_or_instance, "compute_loss"):
        # Already an Objective instance
        return name_or_instance
    if isinstance(name_or_instance, str):
        if name_or_instance not in OBJECTIVE_REGISTRY:
            _ensure_builtin_objectives_loaded()
        if name_or_instance not in OBJECTIVE_REGISTRY:
            available = sorted(OBJECTIVE_REGISTRY)
            raise ValueError(
                f"Unknown objective '{name_or_instance}'. "
                f"Available objectives: {available}"
            )
        return OBJECTIVE_REGISTRY[name_or_instance](**kwargs)

    raise TypeError(
        f"Cannot build objective from {type(name_or_instance).__name__}. "
        "Expected str, Objective instance, dict, or None."
    )


class AutoObjective:
    """Lightweight factory for building objectives by name.

    Mirrors ``AutoConfig.for_model`` / ``AutoModel.from_config``. The actual
    registry logic lives in ``build_objective``; this class provides a familiar
    entry point for config-driven training scripts.

        from neuralrnn import AutoObjective
        obj = AutoObjective.from_name("supervised", task_type="classification")
    """

    @staticmethod
    def from_name(name: str, **kwargs) -> "Objective":
        """Build an Objective from its registry name."""
        return build_objective(name, **kwargs)


def _ensure_builtin_objectives_loaded():
    """Lazy import built-in objective modules so their decorators run."""
    # Importing these modules triggers the @register_objective decorators.
    from . import (
        supervised,
        teacher_forcing,
        behavioral,
        variational,
        reconstruction,
        constrained,
        regularized_supervised,
    )
    # Silence unused-import warnings by referencing the modules.
    _ = (
        supervised,
        teacher_forcing,
        behavioral,
        variational,
        reconstruction,
        constrained,
        regularized_supervised,
    )
