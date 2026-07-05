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


@dataclass
class LinearizationResult:
    jacobian: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    is_stable: bool                      # Discrete system: max|eig| < 1
    n_unstable: int                      # Number of |eig| ≥ 1 (number of unstable directions)


def linearize(model: NeuralDynamicsModel, z, *, task_input: torch.Tensor | None = None
              ) -> LinearizationResult:
    """Linearize at state z. z: (M,) tensor/ndarray or (1, M) (auto-squeezed)."""
    device = next(model.parameters()).device
    z = torch.as_tensor(np.asarray(z), dtype=torch.float32, device=device)
    # Ensure z is 1-D (M,) — handle (1, M) from e.g. fp.z.unsqueeze(0)
    if z.dim() > 1:
        z = z.squeeze(0)
    xin = None if task_input is None else task_input.to(device).unsqueeze(0)
    J = model.jacobian(z, inputs=xin).detach().cpu().numpy()
    eigval, eigvec = np.linalg.eig(J)
    mods = np.abs(eigval)
    return LinearizationResult(
        jacobian=J, eigenvalues=eigval, eigenvectors=eigvec,
        is_stable=bool(mods.max() < 1.0), n_unstable=int((mods >= 1.0).sum()))


def dominant_direction(lin: LinearizationResult) -> np.ndarray:
    """Real direction corresponding to the eigenvalue with largest magnitude (used for drawing line attractors / slow manifolds)."""
    i = int(np.argmax(np.abs(lin.eigenvalues)))
    return np.real(lin.eigenvectors[:, i])


def classify_fixed_point(lin: LinearizationResult) -> str:
    """Coarse classification: stable / saddle(k) / unstable (for discrete systems, compare |eig| with 1)."""
    if lin.is_stable:
        return "stable"
    if lin.n_unstable == lin.eigenvalues.shape[0]:
        return "unstable"
    return f"saddle({lin.n_unstable})"
