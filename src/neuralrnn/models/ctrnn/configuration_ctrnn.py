"""CTRNN 系列配置（连续时间 RNN，含 vanilla / EI 变体）。

参考实现：移植自 nn-brain 的 RNN+DynamicalSystemAnalysis.ipynb / EI_RNN.ipynb。
作为"范式 A（任务优化 RNN）"的契约 A 抄写模板。
"""
from __future__ import annotations

from ...configuration_utils import NeuralRNNConfig


class CTRNNConfig(NeuralRNNConfig):
    """连续时间 RNN：τ dr/dt = -r + f(W_r r + W_x x + b)，欧拉离散步长 dt。

    Args:
        input_dim:  输入维度
        latent_dim: 隐单元数 M
        output_dim: 读出维度（任务类别数等）
        dt:         离散步长；alpha = dt/tau
        tau:        时间常数
        activation: 非线性（relu / tanh / softplus）
        dale:       是否施加 Dale 约束（兴奋/抑制分离），EI 变体置 True
        ei_ratio:   兴奋单元占比（dale=True 时生效）
        trainable_h0: 初值是否可训练
        sigma_rec:  递归噪声标准差（0 关闭）
        relu_after_blend: True = f((1-α)z + α·pre)（nn-brain 原始公式）；
                          False = (1-α)z + α·f(pre)（标准 Euler 离散化，默认）
    """

    model_type = "ctrnn"

    def __init__(
        self,
        input_dim: int = 3,
        latent_dim: int = 64,
        output_dim: int = 3,
        dt: float | None = 100.0,
        tau: float = 100.0,
        activation: str = "relu",
        dale: bool = False,
        ei_ratio: float = 0.8,
        trainable_h0: bool = False,
        sigma_rec: float = 0.0,
        relu_after_blend: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(input_dim=input_dim, latent_dim=latent_dim,
                         output_dim=output_dim, dt=dt, activation=activation, **kwargs)
        self.tau = tau
        self.dale = dale
        self.ei_ratio = ei_ratio
        self.trainable_h0 = trainable_h0
        self.sigma_rec = sigma_rec
        self.relu_after_blend = relu_after_blend


class VanillaRNNConfig(CTRNNConfig):
    """离散 vanilla RNN（dt=None 等价 alpha=1）。"""
    model_type = "vanilla_rnn"

    def __init__(self, **kwargs):
        kwargs.setdefault("dt", None)
        super().__init__(**kwargs)


class EIRNNConfig(CTRNNConfig):
    """兴奋-抑制 RNN（Dale 约束默认开启）。"""
    model_type = "ei_rnn"

    def __init__(self, **kwargs):
        kwargs.setdefault("dale", True)
        super().__init__(**kwargs)
