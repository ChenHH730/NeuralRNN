"""目标函数（Objective）接口 —— 两范式统一的关键解耦点。

设计动机（ARCHITECTURE §4）：模型只描述"动力系统 + 读出"，**不**自带损失。
范式差异（任务优化 vs 动力学重构 vs 行为拟合 vs 变分推断）全部封装在 Objective 里：

    Objective.compute_loss(model, batch) -> (loss, logs)

Trainer 完全通用——它只调用 model.forward / objective.compute_loss，对具体范式无感。
要纳入一篇新论文的训练范式，通常只需写一个新的 Objective 子类（契约 C，见 PORTING_GUIDE）。
"""
from __future__ import annotations

import torch

from ...modeling_utils import NeuralDynamicsModel


class Objective:
    """所有目标函数的基类。子类实现 compute_loss。"""

    def compute_loss(self, model: NeuralDynamicsModel,
                     batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        """返回 (标量 loss, 日志 dict)。Trainer 对 loss 反传，对 logs 做记录。"""
        raise NotImplementedError

    # 可选：支持课程式 forcing 退火的目标覆盖此方法（Trainer 每步调用）
    def set_forcing(self, alpha: float) -> None:  # noqa: D401
        pass
