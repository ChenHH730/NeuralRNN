"""不动点局部线性化与分类。

围绕模型的 jacobian 契约，提供：在某状态处取 Jacobian、特征分解、稳定性分类、
主特征方向（用于画线吸引子/慢流形方向，见 RNN_DynamicalSystemAnalysis.ipynb 中
对 jac 做特征分解并取最大特征值方向画线吸引子）。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..modeling_utils import NeuralDynamicsModel


@dataclass
class LinearizationResult:
    jacobian: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    is_stable: bool                      # 离散系统 max|eig| < 1
    n_unstable: int                      # |eig| ≥ 1 的个数（不稳定方向数）


def linearize(model: NeuralDynamicsModel, z, *, task_input: torch.Tensor | None = None
              ) -> LinearizationResult:
    """在状态 z 处线性化。z: (M,) tensor/ndarray。"""
    device = next(model.parameters()).device
    z = torch.as_tensor(np.asarray(z), dtype=torch.float32, device=device)
    xin = None if task_input is None else task_input.to(device).unsqueeze(0)
    J = model.jacobian(z, inputs=xin).detach().cpu().numpy()
    eigval, eigvec = np.linalg.eig(J)
    mods = np.abs(eigval)
    return LinearizationResult(
        jacobian=J, eigenvalues=eigval, eigenvectors=eigvec,
        is_stable=bool(mods.max() < 1.0), n_unstable=int((mods >= 1.0).sum()))


def dominant_direction(lin: LinearizationResult) -> np.ndarray:
    """最大特征值对应的实部方向（画线吸引子/慢流形用）。"""
    i = int(np.argmax(np.abs(lin.eigenvalues)))
    return np.real(lin.eigenvectors[:, i])


def classify_fixed_point(lin: LinearizationResult) -> str:
    """粗分类：stable / saddle(k) / unstable（离散系统按 |eig| 与 1 比较）。"""
    if lin.is_stable:
        return "stable"
    if lin.n_unstable == lin.eigenvalues.shape[0]:
        return "unstable"
    return f"saddle({lin.n_unstable})"
