"""Behavior-fitting objective (Tiny RNN paradigm).

Corresponds to 01-fitting-generated-data.ipynb: use a small GRU to fit subjects' trial-by-trial
choices in bandit-like tasks, predicting the log-odds of the next action with CrossEntropy.
Often combined with nested cross-validation (see train/cv.py and PORTING_GUIDE recipe 7).

This is a thin compatibility subclass of :class:`SupervisedObjective` (classification):
the base class already provides the masked cross-entropy loss and the ``output_h0``
alignment; this subclass only adds the tinyRNN-style L1 penalty on recurrent weights
and the ``nll`` log entry.

Standard batch (ARCHITECTURE §3.1 behavior):
    {"inputs": (B,T,input_dim) encoded history (action/reward...),
     "targets": (B,T) next-action class, "mask": (B,T)|None}
"""
from __future__ import annotations

from .supervised import SupervisedObjective
from .registry import register_objective
from ...modeling_utils import NeuralDynamicsModel


@register_objective("behavioral")
class BehavioralObjective(SupervisedObjective):
    """Negative log-likelihood of the next action, plus optional L1 on recurrent weights.

    Inherits the masked cross-entropy loss and ``output_h0`` alignment (leading
    output step dropped when the model emits T+1 outputs) from
    :class:`SupervisedObjective`.

    Args:
        task_type: passed to SupervisedObjective (default "classification").
        l1_weight: L1 coefficient on recurrent weights. If None (default), the
            value is read from ``model.config.l1_weight`` (tiny_rnn convention).
            An explicit value overrides the config; 0 disables the penalty.
    """

    def __init__(self, task_type: str = "classification",
                 l1_weight: float | None = None):
        super().__init__(task_type=task_type)
        self.l1_weight = l1_weight

    def compute_loss(self, model: NeuralDynamicsModel, batch):
        """Batch keys: "inputs" (B,T,K), "targets" (B,T) action indices,
        optional "mask" (B,T). Returns (loss, {"loss", "acc"/..., "nll", ["l1"]})."""
        loss, logs = super().compute_loss(model, batch)
        if self.task_type == "classification":
            logs["nll"] = logs["loss"]  # task NLL before the L1 term

        # Optional L1 regularization on recurrent weights (tiny_rnn).
        l1_weight = self.l1_weight
        if l1_weight is None:
            l1_weight = getattr(model.config, "l1_weight", 0.0)
        if l1_weight > 0 and hasattr(model, "get_l1_loss"):
            l1 = model.get_l1_loss()
            loss = loss + l1_weight * l1
            logs["l1"] = l1.item()
            logs["loss"] = loss.item()

        return loss, logs
