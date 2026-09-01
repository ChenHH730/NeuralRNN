"""Canonical correlation analysis (CCA) for aligning neural state spaces.

Aligns hidden-unit activity recorded under several *views* — experimental
conditions, sessions, days, or sub-populations — into one common latent
space, so that trajectories can be compared across views in shared
coordinates. The pipeline ("PCA -> multi-view CCA") is the standard
representational-alignment method used for long-term stability and
cross-condition comparisons (Gallego et al. 2018, Nat. Neurosci.):

1. per view: z-score the units, PCA, keep the components explaining
   ``var_threshold`` of the variance -> loading matrix (H, k_i);
2. truncate all views to ``min(k_i)`` components (this requires the units
   to be matched across views, e.g. the same RNN probed under different
   conditions);
3. multi-view regularized CCA (SUMCORR criterion, generalized eigenvalue
   solution) finds per-view canonical weights maximizing the sum of
   pairwise correlations of the canonical variates;
4. ``CCA.transform`` projects new activity as z-score -> loadings -> CCA
   weights.

Relation to ``sklearn.cross_decomposition.CCA``: sklearn implements the
classical TWO-view CCA (via SVD of the whitened cross-covariance). This
module implements the MULTI-view (multiset) SUMCORR generalization; for
two views and small ``reg`` the canonical correlations agree with sklearn's
(verified numerically), while ``n_views >= 3`` is not supported by sklearn.
Use sklearn for a quick two-matrix canonical-correlation analysis; use this
module when you need the full PCA-preprocessing pipeline, more than two
views, or a reusable projector for held-out activity (``CCA.transform``).

Note: ``pca_loadings`` re-implements z-scored PCA instead of reusing
``analysis.dimensionality.fit_pca`` because the CCA pipeline needs the
per-unit mean/scale to project held-out activity, and uses the R ``scale``
convention (sample std, ddof=1).
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import eigh


def _zscore(X: np.ndarray, ddof: int = 1):
    """Column-wise z-score (sample std, ddof=1)."""
    mean = X.mean(axis=0)
    std = X.std(axis=0, ddof=ddof)
    std = np.where(std > 0, std, 1.0)
    return (X - mean) / std, mean, std


def pca_loadings(Z: np.ndarray, var_threshold: float = 0.95):
    """Z-score + PCA; keep components explaining ``var_threshold`` of variance.

    Args:
        Z: (T, H) activity matrix (time x units).

    Returns:
        loadings (H, k), mean (H,), scale (H,), n_components
    """
    Zs, mean, scale = _zscore(Z)
    # PCA via SVD of the centered/scaled data
    U, S, Vt = np.linalg.svd(Zs, full_matrices=False)
    var = S ** 2
    frac = np.cumsum(var) / var.sum()
    k = int(np.searchsorted(frac, var_threshold) + 1)
    loadings = Vt[:k].T                                # (H, k)
    return loadings, mean, scale, k


def multiset_cca(views: list[np.ndarray], n_components: int = 3, reg: float = 0.0):
    """Regularized multi-view CCA (SUMCORR) on row-matched matrices.

    Args:
        views: list of (n, k_i) matrices with matched rows (e.g. per-view
            PCA loadings of the same units, or simultaneous recordings).
        n_components: number of canonical components to return.
        reg: ridge regularization added to within-view covariances.

    Returns:
        List of per-view weight matrices (k_i, n_components).
    """
    n_views = len(views)
    views = [np.asarray(V, dtype=float) for V in views]
    n = views[0].shape[0]
    assert all(V.shape[0] == n for V in views), "views must have matched rows"
    views = [V - V.mean(axis=0) for V in views]
    ks = [V.shape[1] for V in views]

    # Block cross-covariance R and block-diagonal within-covariance D.
    dims = np.cumsum([0] + ks)
    p = dims[-1]
    R = np.zeros((p, p))
    D = np.zeros((p, p))
    for i in range(n_views):
        for j in range(n_views):
            C = views[i].T @ views[j] / (n - 1)
            R[dims[i]:dims[i + 1], dims[j]:dims[j + 1]] = C
        D[dims[i]:dims[i + 1], dims[i]:dims[i + 1]] = (
            views[i].T @ views[i] / (n - 1) + reg * np.eye(ks[i]))

    # Solve R a = lambda D a (D is symmetric positive definite for reg > 0;
    # add a tiny jitter for reg = 0 with collinear views).
    evals, evecs = eigh(R, D + 1e-12 * np.eye(p))
    order = np.argsort(evals)[::-1]
    n_components = min(n_components, p)
    weights = []
    for i in range(n_views):
        W = evecs[dims[i]:dims[i + 1], order[:n_components]]
        # Normalize columns to unit within-view variance.
        norms = np.sqrt(np.sum((views[i] @ W) ** 2, axis=0) / (n - 1))
        weights.append(W / np.maximum(norms, 1e-12))
    return weights


class CCA:
    """Fitted PCA -> multi-view CCA alignment across views (see module docstring).

    Attributes:
        n_components: number of canonical dimensions.
        view_params: per-view dict {mean, scale, loadings, cca_weights}.
    """

    def __init__(self, view_params: list[dict], n_components: int):
        self.view_params = view_params
        self.n_components = n_components

    def transform(self, Z: np.ndarray, view: int) -> np.ndarray:
        """Project (T, H) activity of view ``view`` into the common space."""
        p = self.view_params[view]
        Zs = (np.asarray(Z, dtype=float) - p["mean"]) / p["scale"]
        return Zs @ (p["loadings"] @ p["cca_weights"])


def fit_cca(activities: list[np.ndarray], var_threshold: float = 0.95,
            n_components: int = 3, reg: float = 0.0) -> CCA:
    """Fit the PCA -> multi-view CCA pipeline on per-view activity.

    Args:
        activities: list of (T_i, H) activity matrices, one per view
            (condition/session/...); the H units must be matched across
            views, time axes need not match.
        var_threshold: variance threshold for the per-view PCA step.
        n_components: canonical dimensions (truncated to the smallest view).
        reg: CCA ridge regularization (0 = unregularized).

    Returns:
        CCA; use ``.transform(Z, view)`` to project new activity.
    """
    params = []
    for Z in activities:
        loadings, mean, scale, k = pca_loadings(Z, var_threshold)
        params.append({"mean": mean, "scale": scale, "loadings": loadings})
    k_min = min(p["loadings"].shape[1] for p in params)
    k_min = max(1, min(k_min, n_components))
    loadings = [p["loadings"][:, :k_min] for p in params]
    weights = multiset_cca(loadings, n_components=k_min, reg=reg)
    for p, W in zip(params, weights):
        p["loadings"] = p["loadings"][:, :k_min]
        p["cca_weights"] = W
    return CCA(params, k_min)
