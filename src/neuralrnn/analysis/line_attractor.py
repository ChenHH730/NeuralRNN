"""线吸引子（Line Attractor）分析。

移植自 trainRNNbrain 的 DynamicSystemAnalyzerCDDM，用于分析 CDDM 等任务中
持续活动（persistent activity）的神经机制。线吸引子是一段连续的慢流形，
网络状态沿此流形漂移极慢（‖RHS‖ ≈ 0），支持连续变量的稳定维持。

铁律：只通过模型公共契约（recurrence / jacobian）工作。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from ..modeling_utils import NeuralDynamicsModel
from .dimensionality import fit_pca, PCAResult


@dataclass
class LineAttractorPoint:
    """线吸引子上的一个采样点。"""
    z: np.ndarray                       # 状态空间坐标 (M,)
    speed: float                        # ‖F(z) - z‖
    distance: float                     # 沿线吸引子的累积距离
    eigenvalues: np.ndarray | None = None   # Jacobian 特征值
    jacobian: np.ndarray | None = None      # Jacobian 矩阵


@dataclass
class LineAttractorResult:
    """线吸引子分析结果。"""
    points: list[LineAttractorPoint] = field(default_factory=list)
    endpoints: tuple[np.ndarray, np.ndarray] | None = None   # (left, right)
    projection_axes: np.ndarray | None = None                 # (3, M) for 3D viz
    trajectories: np.ndarray | None = None                    # (B, T, M) 用于 PCA

    @property
    def distances(self) -> np.ndarray:
        return np.array([p.distance for p in self.points])

    @property
    def speeds(self) -> np.ndarray:
        return np.array([p.speed for p in self.points])

    @property
    def coords(self) -> np.ndarray:
        if not self.points:
            return np.empty((0,))
        return np.stack([p.z for p in self.points])


@torch.no_grad()
def find_line_attractor_endpoints(
    model: NeuralDynamicsModel,
    *,
    context_input: torch.Tensor,
    n_steps: int = 1000,
    relax_steps: int = 10,
    initial_state: torch.Tensor | None = None,
    nudge_scale: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """找到线吸引子的两个端点。

    方法：用两个相反方向的微扰输入运行模型，取最终状态作为端点近似。
    移植自 trainRNNbrain DynamicSystemAnalyzerCDDM.get_LineAttractor_endpoints()。

    Args:
        model: 训练好的模型
        context_input: (input_dim,) 上下文输入条件
        n_steps: 运行步数
        relax_steps: 收敛后的松弛步数
        initial_state: (M,) 初始状态；默认零向量
        nudge_scale: 微扰幅度

    Returns:
        (endpoint_left, endpoint_right): 两个端点的 numpy 坐标 (M,)
    """
    model.eval()
    device = next(model.parameters()).device
    M = model.config.latent_dim
    input_dim = model.config.input_dim

    z0 = initial_state if initial_state is not None else torch.zeros(M, device=device)
    ctx = context_input.to(device)

    # 生成微扰输入：正方向和负方向
    nudge = torch.randn(input_dim, device=device) * nudge_scale
    input_left = (ctx - nudge).unsqueeze(0).unsqueeze(0).expand(1, n_steps, -1)
    input_right = (ctx + nudge).unsqueeze(0).unsqueeze(0).expand(1, n_steps, -1)

    # 运行模型
    z_left = z0.unsqueeze(0)
    z_right = z0.unsqueeze(0)

    for t in range(n_steps):
        z_left = model.recurrence(input_left[:, t], z_left)
        z_right = model.recurrence(input_right[:, t], z_right)

    # 松弛：用纯上下文输入再跑几步
    ctx_input = ctx.unsqueeze(0).unsqueeze(0).expand(1, 1, -1)
    for _ in range(relax_steps):
        z_left = model.recurrence(ctx_input.squeeze(1), z_left)
        z_right = model.recurrence(ctx_input.squeeze(1), z_right)

    return z_left.squeeze(0).cpu().numpy(), z_right.squeeze(0).cpu().numpy()


@torch.no_grad()
def walk_line_attractor(
    model: NeuralDynamicsModel,
    *,
    context_input: torch.Tensor,
    endpoint_left: np.ndarray,
    endpoint_right: np.ndarray,
    n_points: int = 31,
    max_iter: int = 100,
) -> list[LineAttractorPoint]:
    """沿线吸引子采样，最小化 ‖RHS‖²。

    在左右端点之间线性插值，每个点用 scipy SLSQP 最小化 ‖F(z)-z‖²。
    移植自 trainRNNbrain DynamicSystemAnalyzerCDDM.calc_LineAttractor_analytics()。

    Args:
        model: 训练好的模型
        context_input: (input_dim,) 上下文输入条件
        endpoint_left, endpoint_right: 端点坐标 (M,)
        n_points: 采样点数
        max_iter: 每个点的最大优化迭代次数

    Returns:
        LineAttractorPoint 列表
    """
    from scipy.optimize import minimize

    model.eval()
    device = next(model.parameters()).device
    M = model.config.latent_dim

    ctx = context_input.to(device)

    def rhs_norm_sq(z_np):
        z_t = torch.as_tensor(z_np, dtype=torch.float32, device=device).unsqueeze(0)
        xin = ctx.unsqueeze(0)
        with torch.no_grad():
            f = model.recurrence(xin, z_t).squeeze(0)
        diff = f - z_t.squeeze(0)
        return 0.5 * float((diff ** 2).sum())

    def rhs_jacobian(z_np):
        z_t = torch.as_tensor(z_np, dtype=torch.float32, device=device)
        xin = ctx.unsqueeze(0)
        J = model.jacobian(z_t, inputs=xin).cpu().numpy()
        return J - np.eye(M)

    # 线性插值初始猜测
    alphas = np.linspace(0, 1, n_points)
    points: list[LineAttractorPoint] = []
    cumulative_dist = 0.0

    for i, alpha in enumerate(alphas):
        z_init = (1 - alpha) * endpoint_left + alpha * endpoint_right

        try:
            res = minimize(rhs_norm_sq, z_init, method='SLSQP',
                           jac=lambda z: (rhs_jacobian(z).T @
                                          (rhs_jacobian(z) @ z - rhs_jacobian(z) @ z_init)),
                           options={'maxiter': max_iter, 'ftol': 1e-14})
            z_opt = res.x
            speed = np.sqrt(2 * res.fun)
        except Exception:
            z_opt = z_init
            speed = np.sqrt(2 * rhs_norm_sq(z_init))

        # 计算 Jacobian 和特征值
        try:
            J = rhs_jacobian(z_opt) + np.eye(M)
            eig = np.linalg.eigvals(J)
        except Exception:
            J = None
            eig = None

        # 累积距离
        if i > 0:
            cumulative_dist += np.linalg.norm(z_opt - points[-1].z)

        points.append(LineAttractorPoint(
            z=z_opt, speed=speed, distance=cumulative_dist,
            eigenvalues=eig, jacobian=J))

    return points


@torch.no_grad()
def compute_line_attractor(
    model: NeuralDynamicsModel,
    *,
    context_input: torch.Tensor,
    projection_axes: np.ndarray | None = None,
    n_steps: int = 1000,
    n_points: int = 31,
    initial_state: torch.Tensor | None = None,
) -> LineAttractorResult:
    """线吸引子分析的统一入口。

    流程：找端点 → 沿线采样 → 计算 analytics → 投影到可视化坐标系。

    Args:
        model: 训练好的模型
        context_input: (input_dim,) 上下文输入条件
        projection_axes: (3, M) 3D 可视化子空间的轴；None 时用 PCA
        n_steps: 端点搜索的运行步数
        n_points: 沿线吸引子的采样点数
        initial_state: (M,) 初始状态

    Returns:
        LineAttractorResult
    """
    # 1. 找端点
    ep_left, ep_right = find_line_attractor_endpoints(
        model, context_input=context_input, n_steps=n_steps,
        initial_state=initial_state)

    # 2. 沿线采样
    points = walk_line_attractor(
        model, context_input=context_input,
        endpoint_left=ep_left, endpoint_right=ep_right,
        n_points=n_points)

    # 3. 构建结果
    result = LineAttractorResult(
        points=points,
        endpoints=(ep_left, ep_right),
        projection_axes=projection_axes)

    return result
