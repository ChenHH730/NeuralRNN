"""Linear algebra and trajectory utilities for neural dynamics analysis.

All functions are model-agnostic: they operate on numpy arrays or torch
tensors and never import specific model classes.  Ported from the
Dubreuil et al. (2022) / Valente et al. (2022) LowrankRNN reference
codebase (helpers.py).
"""
from __future__ import annotations

import numpy as np
import torch


# ═══════════════════════════════════════════════════════════════════════
# Activation derivatives
# ═══════════════════════════════════════════════════════════════════════

def phi_prime(x):
    """Derivative of tanh: 1 - tanh^2(x).

    Used in gain analysis to measure how sensitive each neuron is
    to input changes at activation level x.

    Args:
        x: numpy array, any shape

    Returns:
        numpy array, same shape as x
    """
    return 1 - np.tanh(x) ** 2


# ═══════════════════════════════════════════════════════════════════════
# Orthogonalization
# ═══════════════════════════════════════════════════════════════════════

def gram_schmidt(vecs):
    """Classical Gram-Schmidt orthogonalization.

    Args:
        vecs: list of 1-D numpy arrays, each shape (M,)

    Returns:
        list of 1-D numpy arrays — orthonormalized basis vectors.
        Zero-norm vectors are returned as-is (their subspace component is zero).
    """
    ortho = []
    for v in vecs:
        v = np.asarray(v, dtype=np.float64).copy()
        for u in ortho:
            v -= u * (u @ v)
        norm = np.linalg.norm(v)
        if norm > 1e-12:
            v /= norm
        ortho.append(v)
    return ortho


def gram_schmidt_pt(mat):
    """Orthogonalize rows of a matrix via Gram-Schmidt (in-place, PyTorch).

    Args:
        mat: (K, M) torch.Tensor — modified in-place.  Row 0 is kept,
             subsequent rows are orthogonalized against previous rows.

    Notes:
        This is the PyTorch in-place version used for basis orthogonalization
        in SupportLowRankRNN.  Use gram_schmidt() for numpy arrays.
    """
    mat[0] = mat[0] / torch.norm(mat[0])
    for i in range(1, mat.shape[0]):
        mat[i] = mat[i] - (mat[:i].t() @ mat[:i] @ mat[i])
        mat[i] = mat[i] / torch.norm(mat[i])


# ═══════════════════════════════════════════════════════════════════════
# Matrix factorization
# ═══════════════════════════════════════════════════════════════════════

def gram_factorization(G):
    """Factorize a Gramian / covariance matrix G into basis vectors.

    Computes eigendecomposition G = V diag(w) V^T, then returns
    X = V diag(sqrt(max(w, 0))) so that X @ X^T ≈ G.

    Used in clustering.to_support_net() to draw new population
    connectivity vectors from fitted Gaussian covariances.

    Args:
        G: (d, d) symmetric positive-semidefinite numpy array

    Returns:
        (d, d) numpy array X such that X @ X^T ≈ G
    """
    w, v = np.linalg.eigh(G)
    x = v * np.sqrt(np.maximum(w, 0))
    return x


# ═══════════════════════════════════════════════════════════════════════
# Vector overlap / correlation
# ═══════════════════════════════════════════════════════════════════════

def overlap_matrix(vecs):
    """Compute pairwise overlap (inner product / N) matrix of vectors.

    Args:
        vecs: list of K vectors, each shape (N,)

    Returns:
        (K, K) numpy array.  Entry (i, j) = vecs[i] @ vecs[j] / N.
        Diagonal entries are 0.
    """
    N = len(vecs[0])
    K = len(vecs)
    ov = np.zeros((K, K))
    for i in range(K):
        for j in range(i + 1, K):
            ov[i, j] = vecs[i] @ vecs[j] / N
        ov[i, i] = 0
    for i in range(K):
        for j in range(0, i):
            ov[i, j] = ov[j, i]
    return ov


def corrvecs(v, w):
    """Cosine similarity (correlation) between two vectors.

    Args:
        v: (N,) numpy array
        w: (N,) numpy array

    Returns:
        float — cosine of the angle between v and w
    """
    v, w = np.asarray(v, dtype=np.float64), np.asarray(w, dtype=np.float64)
    return v @ w / (np.linalg.norm(v) * np.linalg.norm(w))


def project(v, subspace_vecs):
    """Project vector v onto the subspace spanned by subspace_vecs.

    The subspace_vecs are first orthonormalized via Gram-Schmidt,
    then v is projected.

    Args:
        v: (N,) numpy array
        subspace_vecs: list of (N,) numpy arrays (will be orthonormalized)

    Returns:
        (N,) numpy array — projection of v onto span(subspace_vecs)
    """
    v = np.asarray(v, dtype=np.float64)
    ortho = gram_schmidt(subspace_vecs)
    proj = np.zeros_like(v)
    for u in ortho:
        u = np.asarray(u, dtype=np.float64)
        proj += u * (u @ v)
    return proj


def angle_vectors(v, w):
    """Angle (in radians) between two vectors.

    Args:
        v: (N,) numpy array
        w: (N,) numpy array

    Returns:
        float — angle in [0, pi] radians
    """
    v = np.asarray(v, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    cos = np.dot(v, w) / (np.linalg.norm(v) * np.linalg.norm(w))
    return np.arccos(np.clip(cos, -1, 1))


def angle_vec_subsp(v, vecs):
    """Angle (in radians) between vector v and the subspace spanned by vecs.

    Args:
        v: (N,) numpy array
        vecs: list of (N,) numpy arrays (spanning the subspace)

    Returns:
        float — angle in [0, pi/2] radians
    """
    v_proj = project(v, vecs)
    return angle_vectors(v, v_proj)


# ═══════════════════════════════════════════════════════════════════════
# Trajectory reshaping
# ═══════════════════════════════════════════════════════════════════════

def flatten_trajectory(X):
    """Flatten trial-structured trajectory to (n_trials * n_time, dim).

    Args:
        X: (n_trials, n_time, dim) numpy array

    Returns:
        (n_trials * n_time, dim) numpy array
    """
    if len(X.shape) == 3:
        n_neurons = X.shape[-1]
        X = X.transpose((2, 0, 1)).reshape((n_neurons, -1)).T
    return X


def unflatten_trajectory(X_flat, n_trials):
    """Reverse of flatten_trajectory.

    Args:
        X_flat: (n_trials * n_time, dim) numpy array
        n_trials: int

    Returns:
        (n_trials, n_time, dim) numpy array
    """
    n_neurons = X_flat.shape[1]
    X = X_flat.T.reshape((n_neurons, n_trials, -1)).transpose((1, 2, 0))
    return X


# ═══════════════════════════════════════════════════════════════════════
# Device utility
# ═══════════════════════════════════════════════════════════════════════

def map_device(tensors, net):
    """Move tensor(s) to the same device as a network.

    Args:
        tensors: torch.Tensor, or list/tuple/dict of tensors
        net: nn.Module — device is inferred from first parameter

    Returns:
        Same structure as tensors, on net's device
    """
    # Get the device from the first parameter of the network
    device = next(net.parameters()).device
    if isinstance(tensors, torch.Tensor):
        return tensors.to(device)
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(map_device(t, net) for t in tensors)
    if isinstance(tensors, dict):
        return {k: map_device(v, net) for k, v in tensors.items()}
    return tensors
