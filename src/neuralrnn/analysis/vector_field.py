"""Vector field / velocity field analysis.

Sample a grid on a 2D plane in state space (usually the plane spanned by the first two PCA components),
compute the single-step displacement F(z) − z at each grid point to obtain the vector field for a quiver
plot, and record the velocity-field norm (low-speed regions near 0 are candidates for fixed points /
slow manifolds). Model-agnostic: only uses the recurrence contract.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..modeling_utils import NeuralDynamicsModel


@dataclass
class VectorField:
    """Vector field sampled on a 2D plane (see compute_vector_field)."""

    grid_pc: np.ndarray        # Grid points (plane coordinates) (G, 2)
    velocity_pc: np.ndarray    # Velocity vectors projected back to the plane (G, 2)
    speed: np.ndarray          # Full-dimensional norm of F(z)−z (G,)


@torch.no_grad()
def compute_vector_field(model: NeuralDynamicsModel, basis: np.ndarray, mean: np.ndarray,
                         *, task_input: torch.Tensor | None = None,
                         extent=(-3.0, 3.0), n_grid: int = 20) -> VectorField:
    """Compute the vector field on the plane defined by basis (2×M) + mean (M,).

    basis: two rows giving the two directions of the plane (e.g., PCA components_[:2]);
    mean: origin of the plane (data mean).
    Map plane coordinates (a,b) to the full-dimensional z = mean + a·basis[0] + b·basis[1],
    compute F(z)−z, and project back onto the plane (dot with basis) to obtain a plottable 2D vector.
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
    vel_pc = (dz @ B.T).cpu().numpy()                              # (G,2) project back to the plane
    speed = dz.norm(dim=-1).cpu().numpy()

    # Reshape to (n_grid, n_grid, ...) for convenient 2D indexing in plotting
    grid_2d = coords.reshape(n_grid, n_grid, 2)
    vel_2d = vel_pc.reshape(n_grid, n_grid, 2)
    speed_2d = speed.reshape(n_grid, n_grid)
    return VectorField(grid_pc=grid_2d, velocity_pc=vel_2d, speed=speed_2d)
