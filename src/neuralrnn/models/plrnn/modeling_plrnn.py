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
        self.apply_freeze_config()

    def _freeze_groups(self) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = {
            "recurrent": [r"^A$", r"^W1\.", r"^W2\.", r"^h1$", r"^h2$"],
            "output": [],  # identity readout: no trainable output parameters
            "h0": [],      # default zero initial state, not trainable
        }
        if self.C is not None:
            groups["input"] = [r"^C\."]
        else:
            groups["input"] = []
        return groups

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

    def analytic_parameters(self, task_input: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """暴露解析不动点算法（scy_fi）所需参数，全部转 numpy 友好的 detached tensor。
        analysis/fixed_points.py 的解析后端从这里取 (A_diag, W1, W2, h1, h2)。

        若提供常值 task_input，则将其折叠到有效偏置 ``h1_eff = h1 + C @ task_input``
        中，使非自治系统的解析求解 reuse 自治 SCYFI 求解器。
        """
        h1_eff = self.h1
        if task_input is not None and self.C is not None:
            s = torch.as_tensor(task_input, dtype=self.C.dtype, device=self.C.device)
            if s.dim() > 1:
                s = s.squeeze(0)
            h1_eff = h1_eff + self.C @ s
        return {
            "A": torch.diag(self.A).detach(),     # (M,M) 对角化，匹配原 main(np.diag(A),...) 期望
            "W1": self.W1.detach(),
            "W2": self.W2.detach(),
            "h1": h1_eff.detach(),
            "h2": self.h2.detach(),
        }


@register_model("dend_plrnn")
class DendPLRNNModel(NeuralDynamicsModel):
    """dendritic PLRNN with linear spline basis expansion.

    State equation:
        z_t = A z_{t-1}
              + W [sum_b alpha_b ReLU(z_{t-1} - H_b)]^T
              + h + C s_t

    The term in brackets is a (B, M) matrix computed by broadcasting; the
    recurrence follows the BPTT reference implementation in
    dendPLRNN/BPTT_TF/bptt/PLRNN_model.py.
    """
    config_class = DendPLRNNConfig

    def __init__(self, config: DendPLRNNConfig) -> None:
        super().__init__(config)
        M, B, K = config.latent_dim, config.n_bases, config.input_dim
        r = 1.0 / (M ** 0.5)
        self.W = nn.Parameter(uniform_(torch.empty(M, M), -r, r))
        self.A = nn.Parameter(uniform_(torch.empty(M), a=0.5, b=0.9))
        self.h = nn.Parameter(torch.zeros(M))
        self.H = nn.Parameter(uniform_(torch.empty(M, B), -r, r))
        self.alphas = nn.Parameter(uniform_(torch.empty(B), -r, r))
        if config.autonomous:
            self.register_parameter("C", None)
        else:
            r3 = 1.0 / (K ** 0.5)
            self.C = nn.Parameter(uniform_(torch.empty(M, K), -r3, r3))
        self.clip_range = config.clip_range
        self.use_clipping = config.use_clipping
        self.apply_freeze_config()

    def _freeze_groups(self) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = {
            "recurrent": [r"^A$", r"^W\.", r"^h$", r"^H\.", r"^alphas$"],
            "output": [],
            "h0": [],
        }
        if self.C is not None:
            groups["input"] = [r"^C\."]
        else:
            groups["input"] = []
        return groups

    def _basis(self, z: torch.Tensor) -> torch.Tensor:
        """Compute basis expansion sum_b alpha_b ReLU(z - H_b).

        z: (B, M) -> (B, M)
        """
        z_ = z.unsqueeze(-1)          # (B, M, 1)
        H_ = self.H.unsqueeze(0)      # (1, M, B)
        a_ = self.alphas.view(1, 1, -1)  # (1, 1, B)
        return (a_ * torch.relu(z_ - H_)).sum(dim=-1)

    def recurrence(self, x_t, z_prev, *, inputs=None):
        z = self.A * z_prev + self._basis(z_prev) @ self.W.t() + self.h
        if self.C is not None and x_t is not None:
            z = z + x_t @ self.C.t()
        if self.clip_range is not None:
            z = torch.clamp(z, -self.clip_range, self.clip_range)
        return z

    def readout(self, z_t):
        return z_t

    @property
    def supports_analytic_fixed_points(self) -> bool:
        return True

    def jacobian(self, z: torch.Tensor, *, inputs=None) -> torch.Tensor:
        """Analytic Jacobian for dendPLRNN.

        z: (M,) -> (M, M)
        """
        # active indicator per unit and basis: (M, B)
        d = (z.unsqueeze(-1) > self.H).float()
        # effective slope per unit: (M,)
        alpha_d = (self.alphas * d).sum(dim=-1)
        return torch.diag(self.A) + self.W @ torch.diag(alpha_d)

    def analytic_parameters(self, task_input: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Expose effective shallowPLRNN parameters for the analytic solver.

        Equivalent shallowPLRNN:
            z = A z + W1 ReLU(W2 z + h2) + h1
        with hidden_dim = M * B,
            W2 = [I; I; ...; I]          (M*B, M)
            h2 = [-H_1; -H_2; ...; -H_B] (M*B,)
            W1 = [alpha_1 W, ..., alpha_B W] (M, M*B)
            A  = diag(A), h1 = h

        If a constant task_input is provided, it is folded into the effective bias:
            h1_eff = h + C @ task_input.
        """
        M, B = self.config.latent_dim, self.config.n_bases
        A_eff = torch.diag(self.A).detach()
        W1 = torch.zeros(M, M * B, dtype=self.W.dtype, device=self.W.device)
        W2 = torch.zeros(M * B, M, dtype=self.W.dtype, device=self.W.device)
        h2 = torch.zeros(M * B, dtype=self.h.dtype, device=self.h.device)
        for b in range(B):
            W1[:, b * M:(b + 1) * M] = self.alphas[b].item() * self.W
            W2[b * M:(b + 1) * M, :] = torch.eye(M, dtype=W2.dtype, device=W2.device)
            h2[b * M:(b + 1) * M] = -self.H[:, b]

        h1_eff = self.h
        if task_input is not None and self.C is not None:
            s = torch.as_tensor(task_input, dtype=self.C.dtype, device=self.C.device)
            if s.dim() > 1:
                s = s.squeeze(0)
            h1_eff = h1_eff + self.C @ s

        return {
            "A": A_eff,
            "W1": W1.detach(),
            "W2": W2.detach(),
            "h1": h1_eff.detach(),
            "h2": h2.detach(),
        }


@register_model("alrnn")
class ALRNNModel(NeuralDynamicsModel):
    """Almost-linear RNN (ALRNN).

    State equation:
        z_t = A z_{t-1} + W Phi*(z_{t-1}) + h + C s_t
        Phi*(z) = [z_1, ..., z_{M-P}, ReLU(z_{M-P+1}), ..., ReLU(z_M)]

    Only the last P = latent_dim - n_linear units are ReLU; the rest are linear.
    """
    config_class = ALRNNConfig

    def __init__(self, config: ALRNNConfig) -> None:
        super().__init__(config)
        M, K = config.latent_dim, config.input_dim
        P = M - config.n_linear
        if P <= 0:
            raise ValueError(
                f"ALRNN requires n_linear < latent_dim, got n_linear={config.n_linear}, "
                f"latent_dim={M}"
            )
        r = 1.0 / (M ** 0.5)
        self.W = nn.Parameter(uniform_(torch.empty(M, M), -r, r))
        self.A = nn.Parameter(uniform_(torch.empty(M), a=0.5, b=0.9))
        self.h = nn.Parameter(torch.zeros(M))
        if config.autonomous:
            self.register_parameter("C", None)
        else:
            r3 = 1.0 / (K ** 0.5)
            self.C = nn.Parameter(uniform_(torch.empty(M, K), -r3, r3))
        self.clip_range = config.clip_range
        self.use_clipping = config.use_clipping
        self.apply_freeze_config()

    def _freeze_groups(self) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = {
            "recurrent": [r"^A$", r"^W\.", r"^h$"],
            "output": [],
            "h0": [],
        }
        if self.C is not None:
            groups["input"] = [r"^C\."]
        else:
            groups["input"] = []
        return groups

    @property
    def n_linear(self) -> int:
        return self.config.n_linear

    @property
    def n_relu(self) -> int:
        return self.config.latent_dim - self.config.n_linear

    def _phi_star(self, z: torch.Tensor) -> torch.Tensor:
        """Almost-linear activation: keep first M-P units linear, ReLU the rest."""
        P = self.n_relu
        if P == 0:
            return z
        linear_part = z[..., :-P] if P > 0 else z
        nonlinear_part = torch.relu(z[..., -P:])
        return torch.cat([linear_part, nonlinear_part], dim=-1)

    def recurrence(self, x_t, z_prev, *, inputs=None):
        z = self.A * z_prev + self._phi_star(z_prev) @ self.W.t() + self.h
        if self.C is not None and x_t is not None:
            z = z + x_t @ self.C.t()
        if self.clip_range is not None:
            z = torch.clamp(z, -self.clip_range, self.clip_range)
        return z

    def readout(self, z_t):
        return z_t

    @property
    def supports_analytic_fixed_points(self) -> bool:
        return True

    def jacobian(self, z: torch.Tensor, *, inputs=None) -> torch.Tensor:
        """Analytic Jacobian for ALRNN.

        z: (M,) -> (M, M)
        """
        P = self.n_relu
        mask = torch.cat([
            torch.ones(self.n_linear, dtype=z.dtype, device=z.device),
            (z[-P:] > 0).float() if P > 0 else torch.empty(0, dtype=z.dtype, device=z.device),
        ])
        return torch.diag(self.A) + self.W @ torch.diag(mask)

    def analytic_parameters(self, task_input: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Expose effective shallowPLRNN parameters for the analytic solver.

        The ALRNN recurrence can be written as:
            z = (diag(A) + W[:, :M-P] @ S_linear) z + W[:, M-P:] ReLU(S_relu z) + h
        where S_linear selects the first M-P rows and S_relu selects the last P rows.
        This is a shallowPLRNN with hidden_dim = P, W2 = S_relu, h2 = 0, and a full
        effective A matrix.

        If a constant task_input is provided, it is folded into the effective bias:
            h_eff = h + C @ task_input.
        """
        M, P = self.config.latent_dim, self.n_relu
        S_linear = torch.eye(M, dtype=self.W.dtype, device=self.W.device)[:self.n_linear, :]
        A_eff = torch.diag(self.A) + self.W[:, :self.n_linear] @ S_linear
        W1 = self.W[:, self.n_linear:]
        W2 = torch.zeros(P, M, dtype=self.W.dtype, device=self.W.device)
        W2[:, self.n_linear:] = torch.eye(P, dtype=W2.dtype, device=W2.device)
        h2 = torch.zeros(P, dtype=self.h.dtype, device=self.h.device)

        h_eff = self.h
        if task_input is not None and self.C is not None:
            s = torch.as_tensor(task_input, dtype=self.C.dtype, device=self.C.device)
            if s.dim() > 1:
                s = s.squeeze(0)
            h_eff = h_eff + self.C @ s

        return {
            "A": A_eff.detach(),
            "W1": W1.detach(),
            "W2": W2.detach(),
            "h1": h_eff.detach(),
            "h2": h2.detach(),
        }
