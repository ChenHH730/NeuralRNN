"""Fixed-point local linearization and classification.

Centered on the model's jacobian contract, this module provides: Jacobian at a state, eigendecomposition,
stability classification, and the dominant eigen-direction (used for drawing line attractors / slow-manifold
directions; see RNN_DynamicalSystemAnalysis.ipynb, where the Jacobian is decomposed and the largest-eigenvalue
direction is used to draw line attractors).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..modeling_utils import NeuralDynamicsModel
from ._context import deterministic_analysis
from ._stability import spectral_stability


@dataclass
class LinearizationResult:
    """Jacobian eigensystem at a state, with stability summary."""

    jacobian: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    is_stable: bool | None               # None for near-neutral / unknown spectra
    n_unstable: int                      # Number of |eig| > 1 + stability_tol
    n_neutral: int = 0
    stability: str = 'unknown'
    spectral_radius: float | None = None
    stability_tol: float = 1e-6


@deterministic_analysis
def linearize(model: NeuralDynamicsModel, z, *, task_input: torch.Tensor | None = None,
              stability_tol: float = 1e-6
              ) -> LinearizationResult:
    """Linearize at state z. z: (M,) tensor/ndarray or (1, M) (auto-squeezed)."""
    spectral_stability(None, stability_tol)
    device = next(model.parameters()).device
    z = torch.as_tensor(z, dtype=next(model.parameters()).dtype, device=device).detach()
    # Ensure z is 1-D (M,) — handle (1, M) from e.g. fp.z.unsqueeze(0)
    if z.dim() > 1:
        z = z.squeeze(0)
    model.validate_state(z)
    xin = None if task_input is None else torch.as_tensor(task_input, device=device, dtype=z.dtype).reshape(1, -1)
    J = model.jacobian(z, inputs=xin).detach().cpu().numpy()
    eigval, eigvec = np.linalg.eig(J)
    label, stable, unstable, neutral, radius = spectral_stability(eigval, stability_tol)
    return LinearizationResult(
        jacobian=J, eigenvalues=eigval, eigenvectors=eigvec,
        is_stable=stable, n_unstable=unstable, n_neutral=neutral,
        stability=label, spectral_radius=radius, stability_tol=stability_tol)


def dominant_direction(lin: LinearizationResult) -> np.ndarray:
    """Real direction corresponding to the eigenvalue with largest magnitude (used for drawing line attractors / slow manifolds)."""
    i = int(np.argmax(np.abs(lin.eigenvalues)))
    return np.real(lin.eigenvectors[:, i])


def classify_fixed_point(lin: LinearizationResult) -> str:
    """Classify the discrete spectrum; do not force a near-neutral verdict."""
    if lin.is_stable:
        return "stable"
    if lin.is_stable is None:
        return 'near_neutral' if lin.n_neutral else 'unknown'
    if lin.n_unstable == lin.eigenvalues.shape[0]:
        return "unstable"
    return f"saddle({lin.n_unstable})"
