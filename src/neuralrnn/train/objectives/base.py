"""Objective interface -- the key decoupling point unifying the two paradigms.

Design motivation (ARCHITECTURE §4): the model only describes the "dynamical system + readout"
and does **not** carry its own loss. Paradigm differences (task optimization vs. dynamics
reconstruction vs. behavior fitting vs. variational inference) are all encapsulated in Objectives:

    Objective.compute_loss(model, batch) -> (loss, logs)

Trainer is fully generic -- it only calls model.forward / objective.compute_loss and is agnostic
to the concrete paradigm. To add a new paper's training paradigm, usually only a new Objective
subclass is needed (Contract C, see PORTING_GUIDE).
"""
from __future__ import annotations

import torch

from ...modeling_utils import NeuralDynamicsModel


class Objective:
    """Base class for all objectives. Subclasses implement compute_loss.

    Objectives can be registered by name using
    ``@neuralrnn.train.objectives.register_objective("name")`` and then built
    with ``build_objective("name", **kwargs)``.
    """

    def compute_loss(self, model: NeuralDynamicsModel,
                     batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        """Return (scalar loss, log dict). Trainer backpropagates loss and records logs."""
        raise NotImplementedError

    # Optional: objectives that support curriculum forcing annealing override this (Trainer calls it each step)
    def set_forcing(self, alpha: float) -> None:  # noqa: D401
        """Set the forcing strength (no-op by default; e.g. GTF alpha annealing)."""
        pass
