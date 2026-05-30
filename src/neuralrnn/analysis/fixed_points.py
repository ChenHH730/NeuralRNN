"""不动点 / k-环分析（双后端）。

铁律（PORTING_GUIDE 契约D）：分析层只通过模型公共契约工作
（recurrence / jacobian / supports_analytic_fixed_points / analytic_parameters），
**绝不** import 任何具体模型类。这样任意满足契约的模型都能被同一套分析器分析。

两个后端：
  1) 数值后端 NumericFixedPointFinder —— 移植自 RNN_DynamicalSystemAnalysis.ipynb：
     并行初始化一批候选状态，用 Adam 最小化 ‖F(z) − z‖²（速度场范数），筛速度阈值并去重。
     适用于任意模型（含连续 CTRNN 的离散步）。
  2) 解析后端 AnalyticPLRNNFixedPointFinder —— 移植自 CNS2023_tutorial.ipynb 的
     scy_fi / main：利用 PLRNN 的分段线性结构精确求解不动点与 k-环及其特征值。
     仅当 model.supports_analytic_fixed_points 且实现 analytic_parameters() 时可用。

统一入口 find_fixed_points(model, ...) 自动按能力择优（解析优先，回退数值）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from ..modeling_utils import NeuralDynamicsModel


@dataclass
class FixedPoint:
    z: np.ndarray                       # 不动点坐标 (M,)
    speed: float                        # ‖F(z) − z‖（数值后端）；解析后端为 0
    eigenvalues: np.ndarray | None = None  # Jacobian 特征值
    is_stable: bool | None = None       # 离散系统：max|eig| < 1
    order: int = 1                      # 1=不动点；k>1=k 环（解析后端）
    cycle: np.ndarray | None = None     # k 环的全部点 (order, M)


@dataclass
class FixedPointSet:
    points: list[FixedPoint] = field(default_factory=list)

    def coords(self) -> np.ndarray:
        return np.stack([p.z for p in self.points]) if self.points else np.empty((0,))

    def __len__(self): return len(self.points)
    def __iter__(self): return iter(self.points)


# =========================================================================
# 数值后端（梯度法，模型无关）
# =========================================================================
class NumericFixedPointFinder:
    """通过最小化速度场范数 ‖F(z) − z‖² 搜索不动点（nn-brain 风格）。"""

    def __init__(self, n_candidates: int = 64, n_iters: int = 10000, lr: float = 1e-3,
                 speed_tol: float = 1e-1, dedup_tol: float = 1e-2,
                 init_scale: float = 3.0, init_positive: bool = True):
        self.n_candidates = n_candidates
        self.n_iters = n_iters
        self.lr = lr
        self.speed_tol = speed_tol
        self.dedup_tol = dedup_tol
        self.init_scale = init_scale
        self.init_positive = init_positive

    @torch.no_grad()
    def _speed(self, model, z, task_input):
        f = model.recurrence(task_input, z)
        return (f - z).norm(dim=-1)

    def find(self, model: NeuralDynamicsModel, *, task_input: torch.Tensor | None = None,
             init_states: torch.Tensor | None = None) -> FixedPointSet:
        """task_input:(input_dim,) 搜索时固定的输入条件（如决策任务 0-coherence 均值输入）。"""
        was_training = model.training
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        M = model.config.latent_dim
        device = next(model.parameters()).device

        if init_states is None:
            base = torch.rand(self.n_candidates, M, device=device) * self.init_scale
            z = base if self.init_positive else (base - self.init_scale / 2)
        else:
            z = init_states.clone().to(device)
        z = z.detach().requires_grad_(True)

        if task_input is not None:
            xin = task_input.to(device).unsqueeze(0).expand(z.shape[0], -1)
        else:
            xin = None

        opt = torch.optim.Adam([z], lr=self.lr)
        for _ in range(self.n_iters):
            opt.zero_grad()
            f = model.recurrence(xin, z)
            loss = ((f - z) ** 2).mean()
            loss.backward()
            opt.step()

        # 速度筛选（按速度排序，优先保留最佳候选）
        speeds = self._speed(model, z.detach(), xin)
        order = speeds.argsort()
        z_sorted = z.detach()[order]
        speeds_sorted = speeds[order]
        keep = speeds_sorted < self.speed_tol
        cand = z_sorted[keep].cpu().numpy()
        cand_speed = speeds_sorted[keep].cpu().numpy()

        # Fallback: if nothing passes the filter, keep the best candidate
        if len(cand) == 0:
            best_idx = speeds.argmin()
            cand = z.detach()[best_idx:best_idx+1].cpu().numpy()
            cand_speed = speeds[best_idx:best_idx+1].cpu().numpy()

        # 去重 + 求特征值/稳定性（用模型 jacobian 契约）
        fps = FixedPointSet()
        for c, sp in zip(cand, cand_speed):
            if any(np.linalg.norm(c - p.z) < self.dedup_tol for p in fps.points):
                continue
            J = model.jacobian(torch.as_tensor(c, dtype=torch.float32, device=device),
                               inputs=(xin if xin is not None else None)).cpu().numpy()
            eig = np.linalg.eigvals(J)
            fps.points.append(FixedPoint(
                z=c, speed=float(sp), eigenvalues=eig,
                is_stable=bool(np.max(np.abs(eig)) < 1.0)))

        if was_training:
            model.train()
        return fps


# =========================================================================
# 解析后端（PLRNN：scy_fi / main，精确求 FP + k 环）
# =========================================================================
def _construct_relu_matrix(number_quadrant: int, dim: int) -> np.ndarray:
    bits = format(number_quadrant, f"0{dim}b")[::-1]
    return np.diag(np.array([bool(int(b)) for b in bits]))


def _relu_matrix_list(dim: int, order: int) -> np.ndarray:
    out = np.empty((dim, dim, order))
    for i in range(order):
        n = int(np.floor(np.random.rand() * (2 ** dim)))
        out[:, :, i] = _construct_relu_matrix(n, dim)
    return out


def _get_factors(A, W1, W2, D_list, order):
    factor_z = np.eye(A.shape[0])
    factor_h1 = np.eye(A.shape[0])
    factor_h2 = W1.dot(D_list[:, :, 0])
    for i in range(order - 1):
        factor_z = (A + W1.dot(D_list[:, :, i]).dot(W2)).dot(factor_z)
        factor_h1 = (A + W1.dot(D_list[:, :, i + 1]).dot(W2)).dot(factor_h1) + np.eye(A.shape[0])
        factor_h2 = (A + W1.dot(D_list[:, :, i + 1]).dot(W2)).dot(factor_h2) + W1.dot(D_list[:, :, i + 1])
    factor_z = (A + W1.dot(D_list[:, :, order - 1]).dot(W2)).dot(factor_z)
    return factor_z, factor_h1, factor_h2


def _cycle_point_candidate(A, W1, W2, h1, h2, D_list, order):
    z_f, h1_f, h2_f = _get_factors(A, W1, W2, D_list, order)
    try:
        inv = np.linalg.inv(np.eye(A.shape[0]) - z_f)
        return inv.dot(h1_f.dot(h1) + h2_f.dot(h2))
    except np.linalg.LinAlgError:
        return None


def _latent_step(z, A, W1, W2, h1, h2):
    return A.dot(z) + W1.dot(np.maximum(W2.dot(z) + h2, 0)) + h1


def _latent_series(steps, A, W1, W2, h1, h2, dz, z0):
    z = z0 if z0 is not None else np.random.randn(dz)
    traj = [z]
    for _ in range(1, steps):
        z = _latent_step(z, A, W1, W2, h1, h2)
        traj.append(z)
    return traj


def _get_eigvals(A, W1, W2, D_list, order):
    # A 为 (M,M) 对角阵；与 CNS2023 原实现一致地取 np.diag(A) 后逐步累乘。
    e = np.eye(A.shape[0])
    for i in range(order):
        e = (np.diag(A) + W1.dot(D_list[:, :, i]).dot(W2)).dot(e)
    return np.linalg.eigvals(e)


def _scy_fi(A, W1, W2, h1, h2, order, found_lower, outer_it=300, inner_it=100):
    """启发式精确求解 order 阶环（移植自 CNS2023 scy_fi）。A 为 (M,M) 对角阵。"""
    hidden_dim = h2.shape[0]
    latent_dim = h1.shape[0]
    cycles, eigvals = [], []
    i = -1
    while i < outer_it:
        i += 1
        D = _relu_matrix_list(hidden_dim, order)
        diff = 1
        c = 0
        while c < inner_it:
            c += 1
            zc = _cycle_point_candidate(A, W1, W2, h1, h2, D, order)
            if zc is None:
                D = _relu_matrix_list(hidden_dim, order)
                continue
            traj = _latent_series(order, A, W1, W2, h1, h2, latent_dim, z0=zc)
            traj_D = np.empty((hidden_dim, hidden_dim, order))
            for j in range(order):
                traj_D[:, :, j] = np.diag((W2.dot(traj[j]) + h2) > 0)
            for j in range(order):
                diff = np.sum(np.abs(traj_D[:, :, j] - D[:, :, j]))
                if diff != 0:
                    break
                if found_lower and np.round(traj[0], 2) in np.round(
                        np.array(found_lower).flatten(), 2):
                    diff = 1
                    break
            if diff == 0 and not np.any(np.isin(np.round(traj[0], 2),
                                                np.round(cycles, 2) if cycles else np.array([]))):
                e = _get_eigvals(A, W1, W2, D, order)
                cycles.append(traj)
                eigvals.append(e)
                i = 0
                c = 0
            D = _relu_matrix_list(hidden_dim, order) if np.array_equal(D, traj_D) else traj_D
    return cycles, eigvals


class AnalyticPLRNNFixedPointFinder:
    """PLRNN 解析后端：精确枚举 1..max_order 阶环及特征值（CNS2023 main）。"""

    def __init__(self, max_order: int = 1, outer_it: int = 300, inner_it: int = 100):
        self.max_order = max_order
        self.outer_it = outer_it
        self.inner_it = inner_it

    def find(self, model: NeuralDynamicsModel) -> FixedPointSet:
        if not model.supports_analytic_fixed_points:
            raise RuntimeError(f"{type(model).__name__} 不支持解析不动点；请用数值后端。")
        p = model.analytic_parameters()        # 仅依赖契约暴露的参数
        A = p["A"].cpu().numpy()               # (M,M) 对角
        W1 = p["W1"].cpu().numpy()
        W2 = p["W2"].cpu().numpy()
        h1 = p["h1"].cpu().numpy()
        h2 = p["h2"].cpu().numpy()

        fps = FixedPointSet()
        found_lower = []
        for order in range(1, self.max_order + 1):
            cycles, eigvals = _scy_fi(A, W1, W2, h1, h2, order, found_lower,
                                      self.outer_it, self.inner_it)
            found_lower.append(cycles)
            for traj, e in zip(cycles, eigvals):
                traj = np.asarray(traj)
                fps.points.append(FixedPoint(
                    z=traj[0], speed=0.0, eigenvalues=e,
                    is_stable=bool(np.max(np.abs(e)) < 1.0),
                    order=order, cycle=traj if order > 1 else None))
        return fps


# =========================================================================
# 统一入口
# =========================================================================
def find_fixed_points(model: NeuralDynamicsModel, *, backend: str = "auto",
                      task_input: torch.Tensor | None = None,
                      max_order: int = 1, **kwargs) -> FixedPointSet:
    """自动择优：解析优先（若模型支持），否则数值。

    backend: "auto" / "numeric" / "analytic"。
    task_input: 数值后端的输入条件；max_order: 解析后端搜索的最高环阶。
    """
    if backend == "analytic" or (backend == "auto" and model.supports_analytic_fixed_points):
        return AnalyticPLRNNFixedPointFinder(max_order=max_order, **kwargs).find(model)
    return NumericFixedPointFinder(**kwargs).find(model, task_input=task_input)
