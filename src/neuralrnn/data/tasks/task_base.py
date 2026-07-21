"""Unified base class for built-in cognitive tasks.

Every task in this package is a subclass of ``Task`` (see docs/DATA_REFACTOR.md
for the design). Subclasses declare:

    name:           canonical registry name
    aliases:        deprecated names that still resolve to this task
    input_dim / output_dim
    default_dt:     ms per time step (None for step-indexed tasks)
    deprecated_kwargs: old -> new parameter-name mapping (accepted with a
                    DeprecationWarning via ``from_kwargs``)

and implement ``generate_trials()`` returning::

    inputs:     (n_trials, T, input_dim)  torch tensor
    targets:    (n_trials, T, output_dim) torch tensor
    mask:       (n_trials, T, output_dim) torch tensor (1 = loss active)
    conditions: list of per-trial dicts

Condition-dict schema (unified, additive over legacy keys):
    epochs:   {phase: (start_step, end_step)} — per trial
    n_steps:  true trial length in steps (before any padding)
    is_catch: bool
Legacy keys (coherence/context/choice/...) are kept unchanged.
"""
from __future__ import annotations

import warnings
from abc import ABC, abstractmethod

import numpy as np


class Task(ABC):
    """Abstract base for built-in cognitive tasks (see module docstring)."""

    name: str = ""
    aliases: tuple = ()
    input_dim: int = 0
    output_dim: int = 0
    default_dt: float | None = None
    deprecated_kwargs: dict = {}

    @abstractmethod
    def generate_trials(self):
        """Generate trials -> (inputs, targets, mask, conditions)."""

    @classmethod
    def from_kwargs(cls, **kwargs) -> "Task":
        """Construct from user kwargs, mapping deprecated parameter names."""
        for old, new in cls.deprecated_kwargs.items():
            if old in kwargs:
                if new in kwargs:
                    raise TypeError(
                        f"Task '{cls.name}': got both '{old}' and '{new}'; use '{new}' only."
                    )
                warnings.warn(
                    f"Task '{cls.name}': parameter '{old}' is deprecated, use '{new}' instead.",
                    DeprecationWarning,
                    stacklevel=3,
                )
                kwargs[new] = kwargs.pop(old)
        return cls(**kwargs)

    def _seed_np(self) -> None:
        """Seed the legacy numpy RNG (preserves pre-refactor call sequences)."""
        if self.seed is not None:
            np.random.seed(self.seed)

    @staticmethod
    def _cond(epochs: dict, n_steps: int, is_catch: bool = False, **extra) -> dict:
        """Build a condition dict with the unified schema (legacy keys in extra)."""
        return {
            "epochs": epochs,
            "n_steps": int(n_steps),
            "is_catch": bool(is_catch),
            **extra,
        }
