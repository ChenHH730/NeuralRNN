"""CTRNN 系列模型实现（范式 A 参考实现）。

移植自 nn-brain 的 CTRNN：
    r(t+dt) = r(t) + (dt/tau)[-r(t) + f(W_r r + W_x x + b)]
            = (1-alpha) r + alpha * f(pre_activation)

本文件演示"契约 A（模型适配器）"的标准写法，供其它模型移植抄写：
  - 继承 NeuralDynamicsModel，设 config_class，@register_model 注册
  - __init__ 只从 config 读参数
  - 实现 recurrence / readout（硬契约）
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ...modeling_utils import NeuralDynamicsModel
from ...auto.modeling_auto import register_model
from .configuration_ctrnn import CTRNNConfig, VanillaRNNConfig, EIRNNConfig

_ACT = {"relu": torch.relu, "tanh": torch.tanh, "softplus": torch.nn.functional.softplus}


@register_model("ctrnn")
class CTRNNModel(NeuralDynamicsModel):
    config_class = CTRNNConfig

    def __init__(self, config: CTRNNConfig) -> None:
        super().__init__(config)
        M = config.latent_dim
        self.alpha = 1.0 if config.dt is None else config.dt / config.tau
        self.act = _ACT[config.activation]

        self.input2h = nn.Linear(config.input_dim, M)
        self.h2h = nn.Linear(M, M)
        self.readout_layer = nn.Linear(M, config.output_dim)

        if config.trainable_h0:
            self.h0 = nn.Parameter(torch.zeros(M))
        else:
            self.register_buffer("h0", torch.zeros(M))

        # Dale 约束（EI 变体）：用固定符号掩码 + 非负权重幅度，详见 EI_RNN.ipynb。
        if config.dale:
            n_exc = int(round(M * config.ei_ratio))
            sign = torch.ones(M)
            sign[n_exc:] = -1.0
            self.register_buffer("dale_mask", torch.diag(sign))  # (M,M)
        else:
            self.dale_mask = None

    def init_state(self, batch_size, device="cpu"):
        return self.h0.to(device).expand(batch_size, -1).contiguous()

    def _recurrent_weight(self) -> torch.Tensor:
        W = self.h2h.weight
        if self.dale_mask is not None:
            # 强制列符号符合 Dale 律：|W| @ sign-diag
            W = W.abs() @ self.dale_mask
        return W

    # ---------------- 硬契约 ----------------
    def recurrence(self, x_t, z_prev, *, inputs=None):
        W = self._recurrent_weight()
        pre = self.input2h(x_t) + torch.nn.functional.linear(z_prev, W, self.h2h.bias)
        if self.config.sigma_rec > 0 and self.training:
            pre = pre + self.config.sigma_rec * torch.randn_like(pre)
        if self.config.relu_after_blend:
            # nn-brain 原始公式: f((1-α)z + α·pre)
            z = self.act((1 - self.alpha) * z_prev + self.alpha * pre)
        else:
            # 标准 Euler 离散化: (1-α)z + α·f(pre)
            z = (1 - self.alpha) * z_prev + self.alpha * self.act(pre)
        return z

    def readout(self, z_t):
        return self.readout_layer(z_t)


@register_model("vanilla_rnn")
class VanillaRNNModel(CTRNNModel):
    config_class = VanillaRNNConfig


@register_model("ei_rnn")
class EIRNNModel(CTRNNModel):
    config_class = EIRNNConfig
