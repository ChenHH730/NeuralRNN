"""向量场 / 速度场分析。

在状态空间的某 2D 平面（常取 PCA 前两主成分张成的平面）上采网格，计算每点的
单步位移 F(z) − z，得到 quiver 所需的向量场，并标注速度场范数（接近 0 的低速区
即不动点/慢流形的候选）。模型无关：只用 recurrence 契约。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..modeling_utils import NeuralDynamicsModel


@dataclass
class VectorField:
    grid_pc: np.ndarray        # 网格点（平面坐标）(G, 2)
    velocity_pc: np.ndarray    # 投影回平面的速度向量 (G, 2)
    speed: np.ndarray          # ‖F(z)−z‖ 全维范数 (G,)


@torch.no_grad()
def compute_vector_field(model: NeuralDynamicsModel, basis: np.ndarray, mean: np.ndarray,
                         *, task_input: torch.Tensor | None = None,
                         extent=(-3.0, 3.0), n_grid: int = 20) -> VectorField:
    """在由 basis(2×M) + mean(M,) 定义的平面上计算向量场。

    basis：两行为平面的两个方向（如 PCA components_[:2]）；mean：平面原点（数据均值）。
    把平面坐标 (a,b) 映射到全维 z = mean + a·basis[0] + b·basis[1]，算 F(z)−z，
    再投影回平面（点乘 basis）得到可画的二维向量。
    """
    model.eval()
    device = next(model.parameters()).device
    lo, hi = extent
    xs = np.linspace(lo, hi, n_grid)
    ys = np.linspace(lo, hi, n_grid)
    aa, bb = np.meshgrid(xs, ys)
    coords = np.stack([aa.ravel(), bb.ravel()], axis=1)          # (G,2)

    B = torch.as_tensor(basis, dtype=torch.float32, device=device)   # (2,M)
    mu = torch.as_tensor(mean, dtype=torch.float32, device=device)   # (M,)
    C = torch.as_tensor(coords, dtype=torch.float32, device=device)  # (G,2)
    Z = mu + C @ B                                                 # (G,M)

    xin = None if task_input is None else task_input.to(device).unsqueeze(0).expand(Z.shape[0], -1)
    F = model.recurrence(xin, Z)
    dz = F - Z                                                     # (G,M)
    vel_pc = (dz @ B.T).cpu().numpy()                              # (G,2) 投影回平面
    speed = dz.norm(dim=-1).cpu().numpy()

    # Reshape to (n_grid, n_grid, ...) for convenient 2D indexing in plotting
    grid_2d = coords.reshape(n_grid, n_grid, 2)
    vel_2d = vel_pc.reshape(n_grid, n_grid, 2)
    speed_2d = speed.reshape(n_grid, n_grid)
    return VectorField(grid_pc=grid_2d, velocity_pc=vel_2d, speed=speed_2d)
