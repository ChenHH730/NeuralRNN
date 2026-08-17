"""Supervised objective (Paradigm A: task-optimized RNN).

Corresponds to training in RNN_DynamicalSystemAnalysis.ipynb / EI_RNN: feed the batch inputs to
the model for a full rollout, and compute the loss between readout outputs and targets.

- Classification tasks (neurogym decision / memory): targets are (B,T) class indices, use CrossEntropy;
  outputs (B,T,C) are reshaped to (B*T,C).
- Regression tasks: targets are (B,T,output_dim), use MSE.
Optional mask (B,T) counts loss only at valid time steps.

Models with ``output_h0=True`` (e.g. tiny_rnn) emit one extra leading output step
(readout of the initial state); when the output length is one greater than the target
length, the leading step is dropped automatically (``y[:, :-1]``) to align with targets.
"""
from __future__ import annotations

import torch

from .base import Objective
from .registry import register_objective
from ..losses import masked_cross_entropy, masked_mse, accuracy_classification
from ...modeling_utils import NeuralDynamicsModel


@register_objective("supervised")
class SupervisedObjective(Objective):
    """Supervised task loss (Paradigm A). See module docstring for batch shapes."""

    def __init__(self, task_type: str = "classification"):
        """task_type: "classification" (masked CE, integer targets (B,T)) or
        "regression" (masked MSE, targets (B,T,O))."""
        assert task_type in ("classification", "regression"), \
            f"task_type must be 'classification' or 'regression', got {task_type!r}"
        self.task_type = task_type

    def compute_loss(self, model: NeuralDynamicsModel, batch):
        """Batch keys: "inputs" (B,T,K), "targets" ((B,T) or (B,T,O)),
        optional "mask" (B,T). Returns (loss, {"loss", ["acc"]})."""
        out = model(batch["inputs"])           # DynamicsModelOutput
        y = out.outputs                         # (B,T,output_dim)
        target = batch["targets"]
        mask = batch.get("mask")

        # output_h0 alignment: outputs include the readout of the initial
        # hidden state; drop the leading step to match target length.
        if getattr(model.config, "output_h0", False) and y.shape[1] == target.shape[1] + 1:
            y = y[:, :-1]

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
