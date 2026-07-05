"""广义教师强制目标 GTF（范式 B：动力学重构）。

移植自 CNS2023_tutorial.ipynb 的 predict_sequence_using_gtf：

    z_0      = F(z_prev=x_obs[0], s[0])
    z_forced = alpha * x_obs[t] + (1 - alpha) * z_pred        # 广义教师强制
    z_t      = F(z_prev=z_forced, s[t])
    loss     = MSE( readout(Z), targets )

forcing 强度 alpha ∈ [0,1]：alpha=1 退化为纯 teacher forcing，alpha=0 为自由
运行；DSR 常用较小 alpha（如 0.1）做"稀疏强制"以稳定混沌系统的训练。

关键改写（对齐框架契约，见 PORTING_GUIDE 配方2 / 契约C）：
  - 原代码 `model(z_prev, s)` 是单步转移 → 这里调用 `model.recurrence(x_t=s, z_prev=...)`；
  - 原代码假设潜维 = 观测维（identity readout）并整段 blend；这里推广为"只强制
    观测子空间的前 obs_dim 维"，当 latent_dim == obs_dim 时与原实现完全一致。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import Objective
from ...modeling_utils import NeuralDynamicsModel


def generalized_teacher_forcing(z_pred: torch.Tensor, z_obs: torch.Tensor,
                                alpha: float) -> torch.Tensor:
    """z = alpha * z_obs + (1 - alpha) * z_pred（逐元素混合）。"""
    return alpha * z_obs + (1.0 - alpha) * z_pred


class TeacherForcingObjective(Objective):
    def __init__(self, alpha: float = 0.1, forcing_interval: int | None = None):
        """
        Args:
            alpha: Teacher forcing blending strength in [0, 1].
            forcing_interval: If None (default), apply GTF at every step.
                If an integer tau > 0, apply forcing only when t % tau == 0,
                matching sparse teacher forcing used by ALRNN-DSR.
        """
        self.alpha = float(alpha)
        if forcing_interval is not None and forcing_interval <= 0:
            raise ValueError("forcing_interval must be None or a positive integer")
        self.forcing_interval = forcing_interval

    def set_forcing(self, alpha: float) -> None:
        self.alpha = float(alpha)

    def _force(self, z_pred: torch.Tensor, x_obs_t: torch.Tensor) -> torch.Tensor:
        """把观测注入预测潜状态。latent_dim == obs_dim 时整段混合；否则只混前 obs_dim 维。"""
        M = z_pred.shape[-1]
        N = x_obs_t.shape[-1]
        if M == N:
            return generalized_teacher_forcing(z_pred, x_obs_t, self.alpha)
        forced = z_pred.clone()
        forced[..., :N] = generalized_teacher_forcing(z_pred[..., :N], x_obs_t, self.alpha)
        return forced

    def compute_loss(self, model: NeuralDynamicsModel, batch):
        X = batch["inputs"]                 # (B,T,N) 观测（=潜轨迹，DSR identity）
        Y = batch["targets"]                # (B,T,N) 右移一位的观测
        S = batch.get("external_inputs")    # (B,T,K) 或 None
        B, T, N = X.shape

        s0 = S[:, 0] if S is not None else None
        M = model.config.latent_dim
        device = next(model.parameters()).device
        # 当潜维 M == 观测维 N 时，直接用首观测初始化前一状态；
        # 当 M != N 时，先用 model.init_state 初始化，再对前 N 维做 teacher forcing。
        if M == N:
            z = X[:, 0].to(device)
        else:
            z = model.init_state(B, device)
            z = self._force(z, X[:, 0])
        z = model.recurrence(s0, z)
        preds = [z]
        for t in range(1, T):
            apply_force = (
                self.forcing_interval is None or t % self.forcing_interval == 0
            )
            if apply_force:
                z_forced = self._force(z, X[:, t])
            else:
                z_forced = z
            s_t = S[:, t] if S is not None else None
            z = model.recurrence(s_t, z_forced)
            preds.append(z)

        Z = torch.stack(preds, dim=1)        # (B,T,M)
        Yhat = model.readout(Z)              # identity 时即 Z
        # 当潜维 M 与观测维 N 不一致时，只对前 N 维计算损失
        if Yhat.shape[-1] != Y.shape[-1]:
            Yhat = Yhat[..., :Y.shape[-1]]
        loss = F.mse_loss(Yhat, Y)
        return loss, {"loss": loss.item(), "alpha": self.alpha}
