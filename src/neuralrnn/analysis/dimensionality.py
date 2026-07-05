"""Dimensionality-reduction analysis (PCA, etc.) and trajectory collection utilities.

Corresponds to the RNN_DynamicalSystemAnalysis.ipynb workflow of performing PCA on network activity and
projecting fixed points / trajectories onto the PC plane for visualization. Provides: collecting latent
trajectories of a model under a batch of inputs, fitting PCA, and projecting arbitrary points
(trajectories / fixed points / vector-field grids) into the same coordinate system, ensuring that multiple
analyses are overlaid in a consistent low-dimensional space.
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
    """Perform PCA on an (N, M) state matrix (SVD implementation, no sklearn dependency)."""
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
    """Run several batches, collect latent trajectories, and flatten them into (N_points, M) for PCA /
    vector-field use."""
    model.eval()
    chunks = []
    for _ in range(n_batches):
        batch = dataset.sample_batch()
        out = model(batch["inputs"])
        chunks.append(out.states.reshape(-1, out.states.shape[-1]).cpu().numpy())
    return np.concatenate(chunks, axis=0)
