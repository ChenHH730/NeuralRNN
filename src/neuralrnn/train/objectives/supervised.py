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
import torch.nn.functional as F

from .base import Objective
from ...modeling_utils import NeuralDynamicsModel


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
            B, T, C = y.shape
            logits = y.reshape(B * T, C)
            tgt = target.reshape(B * T).long()
            loss_per = F.cross_entropy(logits, tgt, reduction="none")  # (B*T,)
            if mask is not None:
                m = mask.float()
                # Handle both mask formats: (B,T) or (B,T,output_dim)
                if m.dim() == 3:
                    m = m[..., 0]  # (B,T,output_dim) -> (B,T), take first channel
                m = m.reshape(B * T)
                loss = (loss_per * m).sum() / m.sum().clamp_min(1.0)
            else:
                loss = loss_per.mean()
            with torch.no_grad():
                pred = logits.argmax(-1)
                acc = (pred == tgt).float().mean().item()
            return loss, {"loss": loss.item(), "acc": acc}

        # regression
        if target.dim() == 2:
            target = target.unsqueeze(-1)
        err = (y - target) ** 2
        if mask is not None:
            m = mask.float()
            # Handle both mask formats: (B,T) or (B,T,output_dim)
            if m.dim() == 2:
                m = m.unsqueeze(-1)  # (B,T) -> (B,T,1), broadcasts with err
            # else: m already has shape (B,T,output_dim), matches err directly
            loss = (err * m).sum() / m.sum().clamp_min(1.0)
        else:
            loss = err.mean()
        return loss, {"loss": loss.item()}
