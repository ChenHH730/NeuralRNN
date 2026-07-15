"""Supervised objective (Paradigm A: task-optimized RNN).

Corresponds to training in RNN_DynamicalSystemAnalysis.ipynb / EI_RNN: feed the batch inputs to
the model for a full rollout, and compute the loss between readout outputs and targets.

- Classification tasks (neurogym decision / memory): targets are (B,T) class indices, use CrossEntropy;
  outputs (B,T,C) are reshaped to (B*T,C).
- Regression tasks: targets are (B,T,output_dim), use MSE.
Optional mask (B,T) counts loss only at valid time steps.
"""
from __future__ import annotations

import torch

from .base import Objective
from .registry import register_objective
from ..losses import masked_cross_entropy, masked_mse, accuracy_classification
from ...modeling_utils import NeuralDynamicsModel


@register_objective("supervised")
class SupervisedObjective(Objective):
    def __init__(self, task_type: str = "classification"):
        assert task_type in ("classification", "regression")
        self.task_type = task_type

    def compute_loss(self, model: NeuralDynamicsModel, batch):
        out = model(batch["inputs"])           # DynamicsModelOutput
        y = out.outputs                         # (B,T,output_dim)
        target = batch["targets"]
        mask = batch.get("mask")

        if self.task_type == "classification":
            loss = masked_cross_entropy(y, target, mask)
            with torch.no_grad():
                acc = accuracy_classification(y, target, mask).item()
            return loss, {"loss": loss.item(), "acc": acc}

        # regression
        if target.dim() == 2:
            target = target.unsqueeze(-1)
        loss = masked_mse(y, target, mask)
        return loss, {"loss": loss.item()}
