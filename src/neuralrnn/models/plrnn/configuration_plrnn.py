"""PLRNN 系列配置（分段线性 RNN，用于动力学重构 DSR）。

参考实现：移植自 Durstewitz lab 的 CNS2023_tutorial.ipynb（shallowPLRNN）。
作为"范式 B（动力学重构）"的契约 A 抄写模板，并演示解析 Jacobian 模型的配置。
dend_plrnn / alrnn 共享本包，按各自论文改 recurrence/jacobian。
"""
from __future__ import annotations

from ...configuration_utils import NeuralRNNConfig


class ShallowPLRNNConfig(NeuralRNNConfig):
    """shallowPLRNN 状态方程：
        z_t = A z_{t-1} + W1 ReLU(W2 z_{t-1} + h2) + h1 + C s_t

    Args:
        latent_dim: 潜维度 M（DSR 中通常等于观测维度 N）
        hidden_dim: 隐维度 L（控制表达力的超参）
        input_dim:  外部输入维度 K；0 表示自治系统（省略 C s_t）
        output_dim: 读出维度；默认 None → 取 latent_dim（DSR identity 读出）
        autonomous: 是否自治（无外部输入）。默认 None → 由 input_dim==0 推断；
                    显式置 True 时会强制 input_dim=0 以保持一致。
        observation: 观测模型；"identity" 表示直接观测潜状态（x_t = z_t）
    """

    model_type = "shallow_plrnn"

    def __init__(
        self,
        latent_dim: int = 3,
        hidden_dim: int = 50,
        input_dim: int = 0,
        output_dim: int | None = None,
        observation: str = "identity",
        autonomous: bool | None = None,
        **kwargs,
    ) -> None:
        kwargs.pop("activation", None)          # PLRNN 恒为 ReLU 结构，避免与下方固定值冲突
        if autonomous is None:
            autonomous = (input_dim == 0)
        if autonomous:
            input_dim = 0                       # 自治 ⟺ 无外部输入，保持不变量一致
        if output_dim is None:
            output_dim = latent_dim             # DSR 读出通常恒等
        super().__init__(input_dim=input_dim, latent_dim=latent_dim,
                         output_dim=output_dim, activation="relu", **kwargs)
        self.hidden_dim = hidden_dim
        self.observation = observation
        self.autonomous = bool(autonomous)      # 存为属性（含进 config.json，可正确回读）


class DendPLRNNConfig(ShallowPLRNNConfig):
    """dendritic PLRNN（基函数展开）。占位：移植 dendPLRNN-main 时补充其专属字段
    （如基函数个数 B、阈值参数等）。"""
    model_type = "dend_plrnn"

    def __init__(self, n_bases: int = 20, **kwargs):
        super().__init__(**kwargs)
        self.n_bases = n_bases


class ALRNNConfig(ShallowPLRNNConfig):
    """almost-linear RNN。占位：移植 ALRNN-DSR-main 时补充（如线性单元数）。"""
    model_type = "alrnn"

    def __init__(self, n_linear: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.n_linear = n_linear
