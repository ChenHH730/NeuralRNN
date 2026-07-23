"""Behavior-fitting objective (Tiny RNN paradigm).

Corresponds to 01-fitting-generated-data.ipynb: use a small GRU to fit subjects' trial-by-trial
choices in bandit-like tasks, predicting the log-odds of the next action with CrossEntropy.
Differs from the supervised objective in input / target semantics (behavioral sequences) and is
often combined with nested cross-validation (see train/cv.py and PORTING_GUIDE recipe 7).

Standard batch (ARCHITECTURE §3.1 behavior):
    {"inputs": (B,T,input_dim) encoded history (action/reward...),
     "targets": (B,T) next-action class, "mask": (B,T)|None}
"""
from __future__ import annotations

import torch

from .base import Objective
from .registry import register_objective
from ..losses import masked_nll
from ...modeling_utils import NeuralDynamicsModel


@register_objective("behavioral")
class BehavioralObjective(Objective):
    """Negative log-likelihood of the next action. Readout outputs action logits.

    Supports tiny_rnn's ``output_h0=True`` config: when the model output length is one greater
    than target length, automatically take ``logits[:, :-1]`` to align with target
    (matching the original project's ``scores[:-1]``).
    If ``l1_weight`` exists in config and the model provides ``get_l1_loss()``, the L1 term is added to loss.
    """

    def compute_loss(self, model: NeuralDynamicsModel, batch):
        """Batch keys: "inputs" (B,T,K), "targets" (B,T) action indices,
        optional "mask" (B,T). Returns (loss, {"loss", "nll", ["l1"]})."""
        out = model(batch["inputs"])
        logits = out.outputs                 # (B, T or T+1, n_actions)
        target = batch["targets"].long()     # (B, T)
        mask = batch.get("mask")

        # Handle output_h0=True: outputs include readout of initial hidden state.
        output_h0 = getattr(model.config, "output_h0", False)
        if output_h0 and logits.shape[1] == target.shape[1] + 1:
            logits = logits[:, :-1]

        loss = masked_nll(logits, target, mask)
        logs = {"loss": loss.item(), "nll": loss.item()}

        # Optional L1 regularization on recurrent weights (tiny_rnn).
        l1_weight = getattr(model.config, "l1_weight", 0.0)
        if l1_weight > 0 and hasattr(model, "get_l1_loss"):
            l1 = model.get_l1_loss()
            loss = loss + l1_weight * l1
            logs["l1"] = l1.item()
            logs["loss"] = loss.item()

        return loss, logs
