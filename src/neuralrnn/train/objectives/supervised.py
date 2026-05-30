"""监督目标（范式 A：任务优化 RNN）。

对应 RNN_DynamicalSystemAnalysis.ipynb / EI_RNN 的训练：把 batch 的 inputs 喂给
模型整段 rollout，readout 输出与 targets 做损失。

- 分类任务（neurogym 决策/记忆）：targets 为 (B,T) 类别索引，用 CrossEntropy；
  输出 (B,T,C) 展平到 (B*T,C)。
- 回归任务：targets 为 (B,T,output_dim)，用 MSE。
可选 mask (B,T) 仅在有效时间步计损失。
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
                m = mask.reshape(B * T).float()
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
            m = mask.unsqueeze(-1).float()
            loss = (err * m).sum() / m.sum().clamp_min(1.0)
        else:
            loss = err.mean()
        return loss, {"loss": loss.item()}
