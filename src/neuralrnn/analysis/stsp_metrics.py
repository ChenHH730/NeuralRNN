"""Dynamical reconstruction evaluation metrics: D_stsp and D_H (from PLRNN family).

D_stsp (state-space divergence): compares the distributional divergence between generated and real
trajectories in state space.
    - binning version: histogram + Laplace smoothing + KL
    - gmm version: Monte-Carlo KL with Gaussian mixtures (more stable in high dimensions)
D_H (power-spectrum distance): dimension-wise mean Hellinger distance between power spectra,
characterizing temporal structure.

Both are implemented purely in numpy / torch without extra dependencies and can be used directly for
training evaluation or paper reproduction.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d


# ============================ D_H: power-spectrum Hellinger ============================
def _smoothed_power_spectrum(x: np.ndarray, smoothing: float) -> np.ndarray:
    x_ = (x - x.mean()) / x.std()
    ps = np.abs(np.fft.rfft(x_)) ** 2 * 2 / len(x_)
    ps = gaussian_filter1d(ps, smoothing)
    return ps / ps.sum()


def hellinger_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Hellinger distance between two normalized distributions p, q (same shape)."""
    return float(np.sqrt(1 - np.sum(np.sqrt(p * q))))


def power_spectrum_error(X: np.ndarray, X_gen: np.ndarray, smoothing: float = 20.0) -> float:
    """D_H: mean dimension-wise Hellinger distance between power spectra. X, X_gen: (T, N)."""
    X = np.asarray(X)
    X_gen = np.asarray(X_gen)
    dists = []
    for i in range(X.shape[1]):
        ps = _smoothed_power_spectrum(X[:, i], smoothing)
        ps_gen = _smoothed_power_spectrum(X_gen[:, i], smoothing)
        dists.append(hellinger_distance(ps, ps_gen))
    return float(np.mean(dists))


# ============================ D_stsp: state-space divergence ============================
def _calc_histogram(x: torch.Tensor, n_bins: int, min_, max_) -> torch.Tensor:
    dim_x = x.shape[1]
    coords = (n_bins * (x - min_) / (max_ - min_)).long()
    inlier = (coords > 0).all(1) * (coords < n_bins).all(1)
    coords = coords[inlier]
    size_ = tuple(n_bins for _ in range(dim_x))
    vals = torch.ones(coords.shape[0])
    return torch.sparse_coo_tensor(coords.t(), vals, size=size_).to_dense()


def _pdf_laplace(hist: torch.Tensor, n_bins: int, alpha: float = 1e-5):
    if hist.sum() == 0:
        return None
    dim_x = len(hist.shape)
    return (hist + alpha) / (hist.sum() + alpha * n_bins ** dim_x)


def _kl(p1, p2):
    if p1 is None or p2 is None:
        return torch.tensor(float("nan"))
    return (p1 * torch.log(p1 / p2)).sum()


def state_space_divergence_binning(x_gen, x_true, n_bins: int = 30) -> float:
    """D_stsp (binning version): preferred for low dimensions (≲4). x_*: (T,N)."""
    xt = torch.as_tensor(np.asarray(x_true), dtype=torch.float32)
    xg = torch.as_tensor(np.asarray(x_gen), dtype=torch.float32)
    mn, mx = xt.min(0).values, xt.max(0).values
    p_true = _pdf_laplace(_calc_histogram(xt, n_bins, mn, mx), n_bins)
    p_gen = _pdf_laplace(_calc_histogram(xg, n_bins, mn, mx), n_bins)
    return float(_kl(p_true, p_gen).item())


def _gmm_likelihood(z, mu, std):
    T = mu.shape[0]
    mu = mu.reshape((1, T, -1))
    vec = (z - mu).float()
    prec = torch.diag_embed(1 / (std ** 2)).float()
    prec_vec = torch.einsum("zij,azj->azi", prec, vec)
    exponent = torch.einsum("abc,abc->ab", vec, prec_vec)
    sqrt_det = torch.prod(std, dim=1)
    lik = torch.exp(-0.5 * exponent) / sqrt_det
    return lik.sum(dim=1) / T


def state_space_divergence_gmm(x_gen, x_true, scaling: float = 1.0,
                               max_used: int = 10000, mc_n: int = 1000) -> float:
    """D_stsp (GMM Monte-Carlo version): more stable in high dimensions. x_*: (T,N)."""
    X_true = torch.as_tensor(np.asarray(x_true), dtype=torch.float32)
    X_gen = torch.as_tensor(np.asarray(x_gen), dtype=torch.float32)
    T = min(X_true.shape[0], max_used)
    mu_true, mu_gen = X_true[:T], X_gen[:T]
    std_true = torch.sqrt(torch.ones(X_true.shape[-1]).repeat(T, 1) * scaling)
    std_gen = torch.sqrt(torch.ones(X_gen.shape[-1]).repeat(T, 1) * scaling)
    t = torch.randint(0, mu_true.shape[0], (mc_n,))
    z = (mu_true[t] + std_true[t] * torch.randn(mu_true[t].shape)).reshape((mc_n, 1, -1))
    prior = _gmm_likelihood(z, mu_gen, std_gen)
    posterior = _gmm_likelihood(z, mu_true, std_true)
    nz = prior != 0
    prior, posterior = prior[nz], posterior[nz]
    kl = torch.mean(torch.log(posterior + 1e-8) - torch.log(prior + 1e-8), dim=0)
    return float(kl.item())


def state_space_divergence(x_gen, x_true, method: str = "auto", **kwargs) -> float:
    """Unified entry point: use binning when dimension ≤ 4, otherwise gmm (can be overridden with method)."""
    dim = np.asarray(x_true).shape[-1]
    if method == "binning" or (method == "auto" and dim <= 4):
        return state_space_divergence_binning(x_gen, x_true,
                                              n_bins=kwargs.get("n_bins", 30))
    return state_space_divergence_gmm(x_gen, x_true, **kwargs)
