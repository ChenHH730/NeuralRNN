"""Demixed PCA (dPCA) and inter-area axis-alignment analyses.

Implements the marginalization machinery of Kobak et al. (2016, "Demixed
principal component analysis of neural population data", eLife) in a compact
form aimed at task-trained RNNs: trial-averaged condition means are decomposed
into marginalizations (one per subset of task variables, plus the
condition-independent term), each marginalization's variance fraction is
reported, and per-marginalization axes are extracted via SVD. Also provides
the axis-alignment utilities used by Kleinman et al. (2025) to relate
representational axes to the singular vectors of inter-area weight matrices
(potent/null space projections, Fig. 4c/f).

This module is pure numpy and only consumes arrays plus the per-trial
condition dicts returned by the data layer (Contract D).

References:
    - Kobak, Brendel, Constantinidis, Feierstein, Kepecs, Mainen, Qi,
      Romo, Uchida & Machens (2016). "Demixed principal component analysis
      of neural population data." eLife 5:e10989.
      Official implementation: https://github.com/wielandbrendel/dPCA
      (vendored reference copy at reference_project/analysis/dPCA).
    - Brendel, Romo & Machens (2011). "Demixed principal component
      analysis." NIPS 24 (marginalization definition).

Simplification vs. the official package: the official dPCA jointly
optimizes encoder/decoder matrices per marginalization under a
trial-noise-regularized loss; here axes are taken from the plain SVD of
each (trial-count-weighted) marginalized condition-mean matrix, which is
the variance-maximizing encoder when noise covariance is ignored. This is
adequate for noise-free RNN state trajectories and matches the usage in
Kleinman et al. (2025).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np


@dataclass
class DPCAResult:
    """Result of fit_dpca.

    axes:           marginalization name -> (n_axes, M) array of unit axes.
    variance_ratio: marginalization name -> fraction of total (trial-count
        weighted, condition-mean) variance captured by that marginalization.
    variables:      task-variable names used, in order.
    n_axes:         axes kept per marginalization.
    """
    axes: dict[str, np.ndarray]
    variance_ratio: dict[str, float]
    variables: tuple[str, ...]
    n_axes: int = 1

    def transform(self, states: np.ndarray, marginalization: str) -> np.ndarray:
        """Project (..., M) states onto this marginalization's axes -> (..., n_axes)."""
        return np.asarray(states) @ self.axes[marginalization].T


def _condition_mean_tensor(
    states: np.ndarray, label_arrays: list[np.ndarray]
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    """Average trials into a (c_1, ..., c_k, T, M) condition-mean tensor.

    Returns the tensor, the list of unique labels per variable, and the
    per-condition-cell trial counts (c_1, ..., c_k).
    """
    uniques, inverses = [], []
    for labels in label_arrays:
        u, inv = np.unique(labels, return_inverse=True)
        uniques.append(u)
        inverses.append(inv)
    shape = tuple(len(u) for u in uniques)
    T, M = states.shape[1], states.shape[2]
    sums = np.zeros(shape + (T, M), dtype=np.float64)
    counts = np.zeros(shape, dtype=np.float64)
    for trial_idx, key in enumerate(zip(*inverses)):
        sums[key] += states[trial_idx]
        counts[key] += 1
    counts_safe = np.maximum(counts, 1)
    cm = sums / counts_safe[..., None, None]
    return cm, uniques, counts


def _marginalization(cm: np.ndarray, subset: tuple[int, ...]) -> np.ndarray:
    """Inclusion-exclusion marginalization of a condition-mean tensor.

    cm: (c_1, ..., c_k, T, M), centered. subset: variable indices defining the
    marginalization (empty tuple = condition-independent term).
    """
    k = cm.ndim - 2
    out = np.zeros_like(cm)
    n = len(subset)
    for r in range(n + 1):
        for sub in combinations(subset, r):
            sign = (-1) ** (n - r)
            mean_axes = tuple(i for i in range(k) if i not in sub)
            if mean_axes:
                term = cm.mean(axis=mean_axes, keepdims=True)
                term = np.broadcast_to(term, cm.shape)
            else:
                term = cm
            out = out + sign * term
    return out


def _marginalization_name(variables: tuple[str, ...], subset: tuple[int, ...]) -> str:
    if not subset:
        return "condition_independent"
    return "x".join(variables[i] for i in subset)


def fit_dpca(
    states: np.ndarray,
    conditions: list[dict],
    variables: tuple[str, ...] = ("direction", "color"),
    n_axes: int = 1,
) -> DPCAResult:
    """Demixed PCA over trial-aligned RNN states.

    Args:
        states: (n_trials, T, M) array (e.g. one area's activity on eval-mode
            trials; trials must be time-aligned).
        conditions: per-trial condition dicts (framework data-layer format).
            Each dict must contain every key in ``variables``; non-catch trials
            only — filter beforehand if the task has catch trials.
        variables: task variables to demix, e.g. ("direction", "color",
            "target_config").
        n_axes: number of axes (components) kept per marginalization; 1
            matches Kleinman et al. (2025) for RNN data.

    Returns:
        DPCAResult with unit axes and variance fractions per marginalization
        ("condition_independent", each variable, and all interactions).
    """
    states = np.asarray(states, dtype=np.float64)
    n_trials, T, M = states.shape
    assert len(conditions) == n_trials, "conditions must match trials"

    label_arrays = [np.asarray([c[v] for c in conditions]) for v in variables]

    # Center per neuron across trials AND time, matching the official dPCA
    # package (Kobak et al. 2016; https://github.com/wielandbrendel/dPCA,
    # dPCA.py::_marginalize). Centering per time step instead would zero out
    # the condition-independent marginalization.
    Xc = states - states.mean(axis=(0, 1), keepdims=True)

    cm, uniques, counts = _condition_mean_tensor(Xc, label_arrays)
    k = len(variables)

    axes, variance_ratio = {}, {}
    # Total variance: trial-count weighted condition-mean power.
    total_var = float((counts[..., None, None] * cm ** 2).sum())
    for r in range(0, k + 1):
        for subset in combinations(range(k), r):
            name = _marginalization_name(variables, subset)
            marg = _marginalization(cm, subset)
            var = float((counts[..., None, None] * marg ** 2).sum())
            variance_ratio[name] = var / total_var if total_var > 0 else 0.0
            # Axes via SVD of the (conditions * T, M) marginalized matrix.
            flat = marg.reshape(-1, M)
            # Weight rows by sqrt(trial count) so axes match the variance metric.
            w = np.sqrt(counts).reshape(-1)
            flat_w = flat.reshape(-1, T, M) * w[:, None, None]
            _, _, Vt = np.linalg.svd(flat_w.reshape(-1, M), full_matrices=False)
            axes[name] = Vt[:n_axes]
    return DPCAResult(axes=axes, variance_ratio=variance_ratio,
                      variables=tuple(variables), n_axes=n_axes)


def axis_overlap_matrix(axes_a: np.ndarray, axes_b: np.ndarray) -> np.ndarray:
    """Dot products between two sets of unit axes (|.| <= 1).

    Kleinman et al. (2025) Fig. 4b "partial orthogonalization": dot products
    between dPCA axes of different variables (e.g. direction x color).
    """
    a = np.atleast_2d(np.asarray(axes_a, dtype=np.float64))
    b = np.atleast_2d(np.asarray(axes_b, dtype=np.float64))
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return a @ b.T


def axis_svd_alignment(
    axis: np.ndarray,
    W: np.ndarray,
    n_random: int = 100,
    seed: int = 0,
) -> dict:
    """Alignment of a representational axis with the right singular vectors of W.

    Used to test whether an axis (e.g. the direction axis of a source area)
    preferentially aligns with the top readout dimensions of an inter-area
    matrix W21/W32 (Kleinman et al. 2025, Fig. 4f).

    Args:
        axis: (N_src,) unit axis in the source-area (E-subspace) coordinates.
        W: (N_tgt, N_src) inter-area weight submatrix.
        n_random: number of random unit vectors for the baseline.
        seed: RNG seed.

    Returns:
        dict with ``alignments`` (|cos| with each right singular vector),
        ``singular_values``, ``random_mean`` / ``random_std`` (baseline |cos|
        distribution of random vectors against the same singular vectors).
    """
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    U, S, Vt = np.linalg.svd(np.asarray(W, dtype=np.float64), full_matrices=False)
    alignments = np.abs(Vt @ axis)
    rng = np.random.default_rng(seed)
    rand = rng.normal(size=(n_random, axis.size))
    rand /= np.linalg.norm(rand, axis=1, keepdims=True)
    rand_align = np.abs(rand @ Vt.T)  # (n_random, n_singular)
    return {
        "alignments": alignments,
        "singular_values": S,
        "random_mean": rand_align.mean(axis=0),
        "random_std": rand_align.std(axis=0),
    }


def potent_null_projection(
    axis: np.ndarray,
    W: np.ndarray,
    rank: int,
) -> dict:
    """Split an axis into potent- and null-space components of W.

    The potent space is spanned by the top-``rank`` right singular vectors of
    W (Kleinman et al. 2025, Fig. 4c: projection of color/direction axes onto
    the potent/null space of the within-area recurrent matrix).

    Returns:
        dict with squared-norm fractions ``potent_frac`` and ``null_frac``
        (they sum to ~1 when rank <= N_src).
    """
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    _, _, Vt = np.linalg.svd(np.asarray(W, dtype=np.float64), full_matrices=False)
    V_pot = Vt[:rank]
    proj = V_pot @ axis
    potent_frac = float(proj @ proj)
    return {"potent_frac": potent_frac, "null_frac": 1.0 - potent_frac}
