"""行为拟合目标（Tiny RNN 范式）。

对应 01-fitting-generated-data.ipynb：用小 GRU 拟合被试在 bandit 等任务上的逐试次
选择，预测下一步动作的对数几率，做 CrossEntropy。与监督目标的区别在于输入/目标语义
（行为序列）以及常配合嵌套交叉验证（见 train/cv.py 与 PORTING_GUIDE 配方7）。

标准 batch（ARCHITECTURE §3.1 行为）：
    {"inputs": (B,T,input_dim) 编码的历史(动作/奖励...),
     "targets": (B,T) 下一步动作类别, "mask": (B,T)|None}
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import Objective
from ...modeling_utils import NeuralDynamicsModel


class BehavioralObjective(Objective):
    """预测下一步动作的负对数似然。readout 输出动作 logits。"""

    def compute_loss(self, model: NeuralDynamicsModel, batch):
        out = model(batch["inputs"])
        logits = out.outputs                 # (B,T,n_actions)
        target = batch["targets"].long()     # (B,T)
        mask = batch.get("mask")

        B, T, C = logits.shape
        nll = F.cross_entropy(logits.reshape(B * T, C),
                              target.reshape(B * T), reduction="none")
        if mask is not None:
            m = mask.reshape(B * T).float()
            loss = (nll * m).sum() / m.sum().clamp_min(1.0)
        else:
            loss = nll.mean()
        return loss, {"loss": loss.item(), "nll": loss.item()}
