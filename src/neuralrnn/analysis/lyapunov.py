"""最大 Lyapunov 指数（混沌判据）。

移植自 CNS2023_tutorial.ipynb 的 max_lyapunov_exponent：沿轨迹累乘 Jacobian，
周期 QR 重正交，累加 log|R[0,0]| / T。λ_max > 0 表征混沌（Lorenz63 ≈ 0.9）。

模型无关：只用 model.generate（自由 rollout）与 model.jacobian（契约）。
"""
from __future__ import annotations

import torch

from ..modeling_utils import NeuralDynamicsModel


@torch.no_grad()
def max_lyapunov_exponent(model: NeuralDynamicsModel, z1: torch.Tensor, T: int = 10000,
                          T_trans: int = 1000, ons: int = 1) -> float:
    """z1:(M,) 初值。先演化 T_trans 步弃暂态，再沿 T 步累计最大指数。"""
    model.eval()
    M = model.config.latent_dim
    device = z1.device

    # 演化暂态：用 generate 得到末态（generate 返回 (1,T+1,M)）
    z = z1.unsqueeze(0)
    traj = model.generate(z, T_trans)
    z = traj[:, -1]                          # (1,M)

    lyap = 0.0
    Q = torch.eye(M, device=device)
    for t in range(T):
        z = model.recurrence(None, z)        # 自治单步 (1,M)
        J = model.jacobian(z.squeeze(0))     # (M,M)
        Q = J @ Q
        if t % ons == 0:
            Q, R = torch.linalg.qr(Q)
            lyap += torch.log(torch.abs(R[0, 0])).item()
    return lyap / T
