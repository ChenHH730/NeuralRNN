"""Stable/unstable manifold tracing for PLRNNs (DetectingManifolds-style).

This module implements Algorithm 1 from Eisenmann et al. (2025),
"Detecting Invariant Manifolds in ReLU-Based RNNs", for the shallowPLRNN
family. It operates on the effective shallowPLRNN parameters returned by
``model.analytic_parameters(task_input)``, so it works for shallowPLRNN,
dendPLRNN, and ALRNN once they expose the (A, W1, W2, h1, h2) form.

For a constant task input, the model's analytic_parameters() folds C*s into
h1, so the manifold tracer sees an autonomous PLRNN map.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class ManifoldSegment:
    """One manifold segment in a single linear region."""
    points: np.ndarray                      # (n_points, M) support points
    center: np.ndarray                      # (M,) anchor point
    basis: np.ndarray                       # (manifold_dim, M) orthonormal basis
    eigenvalues: np.ndarray                 # (manifold_dim,) local eigenvalues
    region_id: int | None = None            # ReLU region identifier
    is_stable: bool = True                  # True = stable, False = unstable


@dataclass
class ManifoldTrace:
    """Collection of manifold segments traced across regions."""
    segments: list[ManifoldSegment] = field(default_factory=list)
    fixed_point: np.ndarray | None = None
    is_stable: bool = True

    def all_points(self) -> np.ndarray:
        """Return all support points as (n_total, M)."""
        if not self.segments:
            return np.empty((0,))
        return np.concatenate([s.points for s in self.segments], axis=0)


def _relu_region(z: np.ndarray, W2: np.ndarray, h2: np.ndarray) -> np.ndarray:
    """Return ReLU activation pattern diag(W2 z + h2 > 0)."""
    return ((W2 @ z + h2) > 0).astype(np.float64)


def _region_id(z: np.ndarray, W2: np.ndarray, h2: np.ndarray) -> int:
    """Integer region identifier from activation pattern."""
    pattern = ((W2 @ z + h2) > 0).astype(np.int64)
    exponents = 2 ** np.arange(pattern.shape[0])
    return int(pattern @ exponents)


def _latent_step(z: np.ndarray, A: np.ndarray, W1: np.ndarray, W2: np.ndarray,
                 h1: np.ndarray, h2: np.ndarray) -> np.ndarray:
    """One forward PLRNN step."""
    return A @ z + W1 @ np.maximum(W2 @ z + h2, 0) + h1


def _latent_step_backward(z_next: np.ndarray, A: np.ndarray, W1: np.ndarray,
                          W2: np.ndarray, h1: np.ndarray, h2: np.ndarray,
                          D: np.ndarray) -> np.ndarray | None:
    """One backward step using a given ReLU pattern D.

    Solves z_{t-1} = (A + W1 D W2)^{-1} (z_t - h1 - W1 D h2).
    Returns None if the matrix is singular.
    """
    M = A.shape[0]
    D_mat = np.diag(D)
    L = A + W1 @ D_mat @ W2
    try:
        z_prev = np.linalg.solve(L, z_next - h1 - W1 @ D_mat @ h2)
    except np.linalg.LinAlgError:
        return None
    return z_prev


def _backward_step_consistent(z_next: np.ndarray, A: np.ndarray, W1: np.ndarray,
                              W2: np.ndarray, h1: np.ndarray, h2: np.ndarray,
                              D_init: np.ndarray, max_flips: int = 3) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Try to invert the PLRNN map consistently.

    First use D_init. If the resulting z_prev does not reproduce D_init,
    try the candidate's own D. If that fails, try bitflips of increasing
    Hamming distance up to max_flips.
    """
    D_pool = [D_init]

    # candidate's own D
    z_candidate = _latent_step_backward(z_next, A, W1, W2, h1, h2, D_init)
    if z_candidate is not None:
        D_cand = _relu_region(z_candidate, W2, h2)
        if np.allclose(D_cand, D_init):
            return z_candidate, D_init
        D_pool.append(D_cand)

    # bitflips of increasing Hamming distance
    d_len = D_init.shape[0]
    base_pattern = (D_init > 0).astype(bool)
    for k in range(1, max_flips + 1):
        for indices in _combination_iter(range(d_len), k):
            trial = base_pattern.copy()
            for idx in indices:
                trial[idx] = not trial[idx]
            D_trial = trial.astype(np.float64)
            z_trial = _latent_step_backward(z_next, A, W1, W2, h1, h2, D_trial)
            if z_trial is None:
                continue
            D_check = _relu_region(z_trial, W2, h2)
            if np.allclose(D_check, D_trial):
                return z_trial, D_trial

    return None, None


def _combination_iter(items, k):
    """Simple combination generator (no itertools dependency)."""
    items = list(items)
    if k == 0:
        yield []
        return
    if k > len(items):
        return
    if k == len(items):
        yield items
        return
    if k == 1:
        for item in items:
            yield [item]
        return
    for i, first in enumerate(items):
        for rest in _combination_iter(items[i + 1:], k - 1):
            yield [first] + rest


def _jacobian_at(z: np.ndarray, A: np.ndarray, W1: np.ndarray, W2: np.ndarray,
                 h2: np.ndarray) -> np.ndarray:
    """Local Jacobian at z."""
    d = ((W2 @ z + h2) > 0).astype(np.float64)
    return A + W1 @ np.diag(d) @ W2


def _local_manifold_basis(z: np.ndarray, A: np.ndarray, W1: np.ndarray,
                          W2: np.ndarray, h2: np.ndarray, stable: bool = True,
                          tol: float = 1e-10) -> tuple[np.ndarray, np.ndarray, int]:
    """Compute the local stable or unstable manifold basis at z.

    Returns:
        basis: (manifold_dim, M) orthonormal basis vectors (rows)
        eigvals: (manifold_dim,) eigenvalues of selected directions
        manifold_dim: dimension of the manifold
    """
    J = _jacobian_at(z, A, W1, W2, h2)
    e, v = np.linalg.eig(J)
    abs_e = np.abs(e)

    if stable:
        mask = abs_e < (1.0 - tol)
    else:
        mask = abs_e > (1.0 + tol)

    # Filter out near-zero directions (numerical noise)
    idx = np.where(mask)[0]
    if len(idx) == 0:
        # Fallback: include the closest direction even if it barely crosses threshold
        if stable:
            idx = np.array([np.argmin(abs_e)])
        else:
            idx = np.array([np.argmax(abs_e)])

    selected_vals = e[idx]
    selected_vecs = v[:, idx].real

    # Orthonormalize
    basis, _ = np.linalg.qr(selected_vecs.T.astype(np.float64).T)
    # QR returns orthonormal columns; transpose to (dim, M)
    basis = basis.T
    return basis, selected_vals, basis.shape[0]


def _initialize_on_manifold(z: np.ndarray, basis: np.ndarray, n_points: int = 500,
                            factor: float = 0.1, W2: np.ndarray | None = None,
                            h2: np.ndarray | None = None) -> np.ndarray:
    """Sample points along the manifold basis while staying in the same ReLU region.

    Args:
        z: base point (M,)
        basis: (manifold_dim, M) basis vectors
        n_points: number of points to collect
        factor: perturbation scale
        W2, h2: if provided, enforce same ReLU region as z
    """
    dim = basis.shape[0]
    collected = []
    attempts = 0
    max_attempts = n_points * 20

    orthant = None
    if W2 is not None and h2 is not None:
        orthant = _relu_region(z, W2, h2)

    while len(collected) < n_points and attempts < max_attempts:
        attempts += 1
        coeffs = (np.random.rand(dim) * 2 - 1) * factor
        if dim == 1:
            # for 1D manifold, sample along both directions more evenly
            coeffs = np.random.choice([-1, 1], size=1) * np.random.rand(1) * factor
        pt = z + coeffs @ basis

        if orthant is not None:
            pt_orthant = _relu_region(pt, W2, h2)
            if not np.allclose(pt_orthant, orthant):
                continue
        collected.append(pt)

    if not collected:
        return z.reshape(1, -1)
    return np.stack(collected, axis=0)


def _propagate_until_new_region(points: np.ndarray, A: np.ndarray, W1: np.ndarray,
                                W2: np.ndarray, h1: np.ndarray, h2: np.ndarray,
                                stable: bool = True, max_steps: int = 100,
                                min_points: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """Propagate points forward (unstable) or backward (stable) until they enter a new region.

    Returns:
        points: array of points that crossed into new regions
        region_ids: region identifier for each point in the new region
    """
    if points.ndim == 1:
        points = points.reshape(1, -1)

    initial_regions = np.array([_region_id(p, W2, h2) for p in points])
    current = points.copy()
    active = np.ones(len(current), dtype=bool)
    final_region = initial_regions.copy()

    for _ in range(max_steps):
        if not active.any():
            break

        new_pts = []
        for i in np.where(active)[0]:
            if stable:
                D = _relu_region(current[i], W2, h2)
                pt_new, _ = _backward_step_consistent(
                    current[i], A, W1, W2, h1, h2, D, max_flips=2
                )
            else:
                pt_new = _latent_step(current[i], A, W1, W2, h1, h2)

            if pt_new is None:
                active[i] = False
                new_pts.append(None)
            else:
                new_pts.append(pt_new)

        # Update positions for active points and check region change
        for i, idx in enumerate(np.where(active)[0]):
            pt_new = new_pts[i]
            if pt_new is None:
                active[idx] = False
                continue
            current[idx] = pt_new
            region_new = _region_id(pt_new, W2, h2)
            if region_new != initial_regions[idx]:
                final_region[idx] = region_new
                active[idx] = False

    # Keep only points that actually crossed
    crossed = final_region != initial_regions
    kept = current[crossed]
    regions = final_region[crossed]

    if len(kept) < min_points:
        # fallback: keep all active points even if they didn't cross
        kept = current[~active]
        regions = final_region[~active]

    return kept, regions


def _fit_manifold_segment(points: np.ndarray, variance_threshold: float = 0.95) -> tuple[np.ndarray, np.ndarray, int]:
    """Fit a planar manifold segment by PCA.

    Returns:
        center: (M,)
        basis: (manifold_dim, M)
        explained_variance: cumulative variance ratios
    """
    X = np.asarray(points)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    center = X.mean(axis=0)
    Xc = X - center
    u, s, vt = np.linalg.svd(Xc, full_matrices=False)
    total = (s ** 2).sum()
    if total < 1e-12:
        return center, np.eye(X.shape[1])[:1], np.array([1.0])
    var_ratio = (s ** 2) / total
    cumvar = np.cumsum(var_ratio)
    dim = int(np.searchsorted(cumvar, variance_threshold)) + 1
    basis = vt[:dim]
    return center, basis, cumvar[:dim]


class PLRNNManifoldTracer:
    """Trace stable or unstable manifolds of a PLRNN fixed point across ReLU regions.

    Works on the effective shallowPLRNN parameters. Use the model's
    ``analytic_parameters(task_input)`` to obtain them.
    """

    def __init__(self, max_iter: int = 10, n_samples: int = 500,
                 factor: float = 0.1, propagation_steps: int = 100,
                 variance_threshold: float = 0.95, seed: int | None = None):
        self.max_iter = max_iter
        self.n_samples = n_samples
        self.factor = factor
        self.propagation_steps = propagation_steps
        self.variance_threshold = variance_threshold
        if seed is not None:
            np.random.seed(seed)

    def trace(self, A: np.ndarray, W1: np.ndarray, W2: np.ndarray,
              h1: np.ndarray, h2: np.ndarray, z0: np.ndarray,
              stable: bool = True) -> ManifoldTrace:
        """Trace the manifold of a fixed point z0 across regions.

        Args:
            A, W1, W2, h1, h2: effective shallowPLRNN parameters.
            z0: fixed point coordinate (M,).
            stable: True for stable manifold, False for unstable manifold.

        Returns:
            ManifoldTrace containing segments per visited region.
        """
        z0 = np.asarray(z0, dtype=np.float64)
        basis, eigvals, dim = _local_manifold_basis(z0, A, W1, W2, h2, stable=stable)

        result = ManifoldTrace(fixed_point=z0, is_stable=stable)
        initial_region = _region_id(z0, W2, h2)
        result.segments.append(ManifoldSegment(
            points=z0.reshape(1, -1),
            center=z0,
            basis=basis,
            eigenvalues=eigvals,
            region_id=initial_region,
            is_stable=stable,
        ))

        # Seed points in the local manifold
        seed_points = _initialize_on_manifold(
            z0, basis, n_points=self.n_samples, factor=self.factor,
            W2=W2, h2=h2
        )
        if len(seed_points) < 2:
            return result

        visited_regions = {initial_region}
        current_points = seed_points

        for iteration in range(self.max_iter):
            if len(current_points) < 10:
                break

            propagated, regions = _propagate_until_new_region(
                current_points, A, W1, W2, h1, h2,
                stable=stable, max_steps=self.propagation_steps
            )

            if len(propagated) == 0:
                break

            # Group by new region
            unique_regions = np.unique(regions)
            next_seed_points = []
            for region in unique_regions:
                mask = regions == region
                pts = propagated[mask]
                if len(pts) < 5:
                    continue
                if region in visited_regions:
                    # Already visited: add points to existing segment's support set
                    for seg in result.segments:
                        if seg.region_id == region:
                            seg.points = np.concatenate([seg.points, pts], axis=0)
                            # Re-fit basis
                            center, basis_new, _ = _fit_manifold_segment(
                                seg.points, self.variance_threshold
                            )
                            seg.center = center
                            seg.basis = basis_new
                            break
                    next_seed_points.append(pts)
                    continue

                # New region: fit a new segment
                visited_regions.add(region)
                center, basis_new, _ = _fit_manifold_segment(pts, self.variance_threshold)
                result.segments.append(ManifoldSegment(
                    points=pts,
                    center=center,
                    basis=basis_new,
                    eigenvalues=np.empty(basis_new.shape[0]),
                    region_id=region,
                    is_stable=stable,
                ))

                # Sample new seed points in this segment to continue propagation
                new_seeds = _initialize_on_manifold(
                    center, basis_new, n_points=max(20, self.n_samples // 10),
                    factor=self.factor * 2, W2=W2, h2=h2
                )
                next_seed_points.append(new_seeds)

            if not next_seed_points:
                break
            current_points = np.concatenate(next_seed_points, axis=0)

        return result


def compute_manifold(model, fixed_point: np.ndarray,
                     task_input: torch.Tensor | None = None,
                     stable: bool = True, **tracer_kwargs) -> ManifoldTrace:
    """Convenience wrapper to trace a manifold of a PLRNN fixed point.

    Args:
        model: PLRNN model with supports_analytic_fixed_points=True.
        fixed_point: (M,) fixed point coordinate.
        task_input: optional constant external input folded into the bias.
        stable: True for stable manifold, False for unstable manifold.
        **tracer_kwargs: passed to PLRNNManifoldTracer.

    Returns:
        ManifoldTrace.
    """
    if not getattr(model, "supports_analytic_fixed_points", False):
        raise RuntimeError(
            f"{type(model).__name__} does not support analytic fixed points / manifolds."
        )
    params = model.analytic_parameters(task_input=task_input)
    required = {"A", "W1", "W2", "h1", "h2"}
    if not required.issubset(params):
        raise RuntimeError(
            f"analytic_parameters() must return {required}, got {set(params.keys())}"
        )

    A = params["A"].detach().cpu().numpy().astype(np.float64)
    W1 = params["W1"].detach().cpu().numpy().astype(np.float64)
    W2 = params["W2"].detach().cpu().numpy().astype(np.float64)
    h1 = params["h1"].detach().cpu().numpy().astype(np.float64)
    h2 = params["h2"].detach().cpu().numpy().astype(np.float64)

    tracer = PLRNNManifoldTracer(**tracer_kwargs)
    return tracer.trace(A, W1, W2, h1, h2, np.asarray(fixed_point), stable=stable)
