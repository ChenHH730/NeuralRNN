"""PLRNN 系列模型实现（范式 B 参考实现，演示解析 Jacobian / 解析不动点能力）。

移植自 CNS2023_tutorial.ipynb 的 shallowPLRNN：
    z_t = A ⊙ z_{t-1} + W1 ReLU(W2 z_{t-1} + h2) + h1 (+ C s_t)
其解析 Jacobian：
    J(z) = diag(A) + W1 diag(1[W2 z + h2 > 0]) W2
分段线性结构使不动点/ k-cycle 可解析求解（见 analysis/fixed_points.py 解析后端）。

本文件演示"契约 A + 解析能力"的标准写法：
  - supports_analytic_fixed_points = True
  - 实现解析 jacobian（与基类自动微分对拍应一致）
  - 暴露 (A, W1, W2, h1, h2) 供分析层的解析不动点算法使用
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.init import uniform_

from ...modeling_utils import NeuralDynamicsModel
from ...auto.modeling_auto import register_model
from .configuration_plrnn import ShallowPLRNNConfig, DendPLRNNConfig, ALRNNConfig


@register_model("shallow_plrnn")
class ShallowPLRNNModel(NeuralDynamicsModel):
    config_class = ShallowPLRNNConfig

    def __init__(self, config: ShallowPLRNNConfig) -> None:
        super().__init__(config)
        M, L, K = config.latent_dim, config.hidden_dim, config.input_dim
        r1, r2 = 1.0 / (L ** 0.5), 1.0 / (M ** 0.5)
        self.W1 = nn.Parameter(uniform_(torch.empty(M, L), -r1, r1))
        self.W2 = nn.Parameter(uniform_(torch.empty(L, M), -r2, r2))
        self.A = nn.Parameter(uniform_(torch.empty(M), a=0.5, b=0.9))  # 对角
        self.h2 = nn.Parameter(uniform_(torch.empty(L), -r1, r1))
        self.h1 = nn.Parameter(torch.zeros(M))
        if config.autonomous:
            self.register_parameter("C", None)
        else:
            r3 = 1.0 / (K ** 0.5)
            self.C = nn.Parameter(uniform_(torch.empty(M, K), -r3, r3))

    # ---------------- 硬契约 ----------------
    def recurrence(self, x_t, z_prev, *, inputs=None):
        # z_prev:(B,M) -> z_t:(B,M)，与原 shallowPLRNN.forward 数值一致
        z = self.A * z_prev + torch.relu(z_prev @ self.W2.T + self.h2) @ self.W1.T + self.h1
        if self.C is not None and x_t is not None:
            z = z + x_t @ self.C.T
        return z

    def readout(self, z_t):
        # observation == "identity"：直接观测潜状态（DSR 标准设定）
        return z_t

    # ---------------- 解析分析支持 ----------------
    @property
    def supports_analytic_fixed_points(self) -> bool:
        return True

    def jacobian(self, z: torch.Tensor, *, inputs=None) -> torch.Tensor:
        """解析 Jacobian：diag(A) + W1 diag(1[W2 z + h2 > 0]) W2。z:(M,) -> (M,M)。
        与基类自动微分结果应 allclose（ReLU 边界除外）。"""
        d = (self.W2 @ z > -self.h2).float()           # (L,) 指示向量
        return torch.diag(self.A) + self.W1 @ torch.diag(d) @ self.W2

    def analytic_parameters(self) -> dict[str, torch.Tensor]:
        """暴露解析不动点算法（scy_fi）所需参数，全部转 numpy 友好的 detached tensor。
        analysis/fixed_points.py 的解析后端从这里取 (A_diag, W1, W2, h1, h2)。"""
        return {
            "A": torch.diag(self.A).detach(),     # (M,M) 对角化，匹配原 main(np.diag(A),...) 期望
            "W1": self.W1.detach(),
            "W2": self.W2.detach(),
            "h1": self.h1.detach(),
            "h2": self.h2.detach(),
        }


@register_model("dend_plrnn")
class DendPLRNNModel(ShallowPLRNNModel):
    """占位：移植 dendPLRNN-main 时覆盖 recurrence/jacobian 为其基函数展开形式。"""
    config_class = DendPLRNNConfig
    # TODO(移植): 按 dendPLRNN 论文重写 recurrence 与解析 jacobian。


@register_model("alrnn")
class ALRNNModel(ShallowPLRNNModel):
    """占位：移植 ALRNN-DSR-main 时覆盖为 almost-linear 形式。"""
    config_class = ALRNNConfig
    # TODO(移植): 按 ALRNN 论文重写 recurrence 与解析 jacobian。
