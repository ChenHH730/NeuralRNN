"""损失函数库 — 可复用的损失项与评估指标。

多数训练损失已封装进 train/objectives（与范式绑定）。此处放置纯损失函数和
指标计算工具，供 notebooks 和训练循环直接使用，不需要依赖 Trainer 框架。

已实现：
- loss_mse: 带掩码的逐试次 MSE 损失（范式 A 回归任务）
- accuracy_general: 基于符号的二选决策准确率（范式 A 决策任务）
"""
from .loss_functions import loss_mse, accuracy_general

__all__ = ["loss_mse", "accuracy_general"]
