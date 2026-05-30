"""降维分析（PCA 等）与轨迹收集工具。

对应 RNN_DynamicalSystemAnalysis.ipynb 中对网络活动做 PCA、把不动点/轨迹投影到
PC 平面可视化的流程。提供：收集模型在一批输入下的潜轨迹、拟合 PCA、把任意点
（轨迹/不动点/向量场网格）投影到同一坐标系，确保多种分析叠加在一致的低维空间里。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..modeling_utils import NeuralDynamicsModel


@dataclass
class PCAResult:
    components: np.ndarray     # (n_components, M)
    mean: np.ndarray           # (M,)
    explained_variance_ratio: np.ndarray

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        return (X - self.mean) @ self.components.T

    def inverse_transform(self, Y: np.ndarray) -> np.ndarray:
        return np.asarray(Y) @ self.components + self.mean


def fit_pca(X: np.ndarray, n_components: int = 2) -> PCAResult:
    """对 (N, M) 状态矩阵做 PCA（SVD 实现，无 sklearn 依赖）。"""
    X = np.asarray(X, dtype=np.float64)
    mean = X.mean(0)
    Xc = X - mean
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = (S ** 2) / (X.shape[0] - 1)
    return PCAResult(components=Vt[:n_components],
                     mean=mean,
                     explained_variance_ratio=(var / var.sum())[:n_components])


@torch.no_grad()
def collect_states(model: NeuralDynamicsModel, dataset, n_batches: int = 1) -> np.ndarray:
    """跑若干 batch，收集潜轨迹并展平成 (N_points, M)，供 PCA/向量场使用。"""
    model.eval()
    chunks = []
    for _ in range(n_batches):
        batch = dataset.sample_batch()
        out = model(batch["inputs"])
        chunks.append(out.states.reshape(-1, out.states.shape[-1]).cpu().numpy())
    return np.concatenate(chunks, axis=0)
