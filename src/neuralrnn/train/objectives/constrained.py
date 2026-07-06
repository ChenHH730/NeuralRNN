"""Constrained supervised objective.

Wraps SupervisedObjective and adds any structural regularizer exposed by the model
via ``model.constraint_loss()``. This is useful for seRNN spatial-embedding
regularization while keeping the base supervised loss unchanged.
"""
from __future__ import annotations

from .supervised import SupervisedObjective


class ConstrainedSupervisedObjective(SupervisedObjective):
    """Supervised loss + optional model constraint regularizer.

    Args:
        task_type: "classification" or "regression" (passed to SupervisedObjective).
        constraint_weight: Coefficient multiplying ``model.constraint_loss()``.
            If the model does not define ``constraint_loss``, only the task loss is used.
    """

    def __init__(self, task_type: str = "classification", constraint_weight: float = 0.0):
        super().__init__(task_type=task_type)
        self.constraint_weight = constraint_weight

    def compute_loss(self, model, batch):
        task_loss, logs = super().compute_loss(model, batch)
        if self.constraint_weight > 0 and hasattr(model, "constraint_loss"):
            reg = model.constraint_loss()
            loss = task_loss + self.constraint_weight * reg
            logs["loss"] = loss.item()
            logs["task_loss"] = task_loss.item()
            logs["constraint_loss"] = reg.item()
            return loss, logs
        return task_loss, logs
