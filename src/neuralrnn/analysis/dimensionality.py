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
    """PCA fit result: principal axes + mean, with (inverse) projection helpers."""

    components: np.ndarray     # (n_components, M)
    mean: np.ndarray           # (M,)
    explained_variance_ratio: np.ndarray

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project points onto the PCs. X: (N, M) -> (N, n_components)."""
        X = np.asarray(X)
        return (X - self.mean) @ self.components.T

    def inverse_transform(self, Y: np.ndarray) -> np.ndarray:
        """Map PC coordinates back to state space. Y: (N, n_components) -> (N, M)."""
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


def effective_dimensionality(
    X: np.ndarray,
    variance_threshold: float = 0.95,
) -> int:
    """Effective dimensionality: number of PCs needed to explain threshold fraction of variance.

    Args:
        X: (N, M) state matrix.
        variance_threshold: Fraction of total variance to explain (default 0.95).

    Returns:
        Integer number of principal components.
    """
    X = np.asarray(X, dtype=np.float64)
    Xc = X - X.mean(0)
    _, S, _ = np.linalg.svd(Xc, full_matrices=False)
    var = (S ** 2) / (X.shape[0] - 1)
    cumvar = np.cumsum(var) / var.sum()
    n = int(np.searchsorted(cumvar, variance_threshold)) + 1
    return min(n, X.shape[1])


@torch.no_grad()
def collect_states(model: NeuralDynamicsModel, dataset, n_batches: int = 1, *,
                   full_state: bool = False, recurrent_noise: bool | None = None,
                   noise_generator=None) -> np.ndarray:
    """Run several batches, collect latent trajectories, and flatten them into (N_points, M) for PCA /
    vector-field use."""
    chunks = []
    with model.evaluation_context(noise=recurrent_noise, generator=noise_generator):
        for _ in range(n_batches):
            batch = dataset.sample_batch()
            states = (model.full_state_trajectory(batch["inputs"]) if full_state
                      else model(batch["inputs"]).states)
            chunks.append(states.reshape(-1, states.shape[-1]).cpu().numpy())
    return np.concatenate(chunks, axis=0)
