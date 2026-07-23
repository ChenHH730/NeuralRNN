"""Sequentiality analysis for task-trained RNNs.

Implements the Sequentiality Index (SI) from Orhan & Ma (2019) and helpers
for sorting neurons by peak time and analyzing recurrent weight profiles
sorted by peak order.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.stats import entropy as scipy_entropy


def compute_sequentiality_index(
    states: np.ndarray,
    threshold: float = 0.1,
    window: int = 5,
    n_bins: int = 20,
) -> float:
    """Compute Orhan & Ma's Sequentiality Index.

    Args:
        states: Array of shape (N_trials, T, M) or (N_points, M). If 3D,
            the SI is computed per trial and averaged.
        threshold: Minimum mean activity for a unit to be included.
        window: Half-width (in time steps) around the peak for the ridge window.
            The paper uses a total window of 5 time steps; here window=5 means
            5 steps on each side (total 11) by default. Set window=2 to match
            the paper's "window size of 5" if interpreted as total width.
        n_bins: Number of bins for the peak-time histogram.

    Returns:
        Mean sequentiality index across trials (or a single value for 2D input).
    """
    states = np.asarray(states)
    if states.ndim == 2:
        states = states[np.newaxis, :, :]
    elif states.ndim != 3:
        raise ValueError(f"states must be 2D or 3D, got shape {states.shape}")

    n_trials, t_steps, _ = states.shape
    si_per_trial = []
    for b in range(n_trials):
        hid = states[b]
        selected = np.nonzero(hid.mean(axis=0) > threshold)[0]
        if len(selected) == 0:
            si_per_trial.append(np.nan)
            continue
        hid = hid[:, selected]
        peak_times = np.argmax(hid, axis=0)
        hist, _ = np.histogram(peak_times, bins=n_bins, range=(0, t_steps))
        hist = hist + 0.1  # pseudocount
        entr = scipy_entropy(hist)

        half = window // 2
        log_ratios = []
        for n in range(len(selected)):
            pt = peak_times[n]
            start = max(0, pt - half)
            end = min(t_steps, pt + half + 1)
            ridge = hid[start:end, n].mean()
            background = np.concatenate([hid[:start, n], hid[end:, n]]).mean()
            if ridge > 0 and background > 0:
                log_ratios.append(np.log(ridge) - np.log(background))
            else:
                log_ratios.append(np.nan)
        mean_log_ratio = np.nanmean(log_ratios)
        if np.isnan(mean_log_ratio):
            si_per_trial.append(np.nan)
        else:
            si_per_trial.append(mean_log_ratio + entr)
    return float(np.nanmean(si_per_trial))


def sort_neurons_by_peak_time(states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return peak times and sorting indices for each unit.

    Args:
        states: (T, M) or (N_trials, T, M). If 3D, peak times are computed from
            the mean across trials.

    Returns:
        peak_times: (M,) array of integer peak times.
        sort_idx:   (M,) array sorting units by peak time.
    """
    states = np.asarray(states)
    if states.ndim == 3:
        states = states.mean(axis=0)
    peak_times = np.argmax(states, axis=0)
    sort_idx = np.argsort(peak_times)
    return peak_times, sort_idx


def weight_profile_by_peak_order(
    weight: np.ndarray,
    peak_times: np.ndarray,
    max_lag: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute mean/sd recurrent weights as a function of peak-time order difference.

    Args:
        weight: (M, M) recurrent weight matrix (row = post, col = pre).
        peak_times: (M,) peak times used to sort neurons.
        max_lag: Maximum order difference to report. Default M//2.

    Returns:
        lags: (L,) integer order differences (post - pre).
        means: (L,) mean weight at each lag.
        stds: (L,) standard deviation at each lag.
    """
    weight = np.asarray(weight)
    peak_times = np.asarray(peak_times)
    M = weight.shape[0]
    assert weight.shape == (M, M), \
        f"weight must be square, got shape {weight.shape}"
    assert peak_times.shape == (M,), \
        f"peak_times must have shape ({M},), got {peak_times.shape}"

    sort_idx = np.argsort(peak_times)
    inv_order = np.argsort(sort_idx)
    W_sorted = weight[np.ix_(sort_idx, sort_idx)]

    if max_lag is None:
        max_lag = M // 2
    lags = np.arange(-max_lag, max_lag + 1)
    means = np.zeros(len(lags), dtype=float)
    stds = np.zeros(len(lags), dtype=float)
    for k, lag in enumerate(lags):
        vals = []
        for i in range(M):
            j = i + lag
            if 0 <= j < M and i != j:
                vals.append(W_sorted[i, j])
        if len(vals) > 0:
            means[k] = np.mean(vals)
            stds[k] = np.std(vals)
        else:
            means[k] = np.nan
            stds[k] = np.nan
    return lags, means, stds


def split_ei_weight_submatrices(
    weight: np.ndarray,
    ei_mask: Sequence[int] | np.ndarray,
) -> dict[str, np.ndarray]:
    """Split recurrent weight matrix into E/I submatrices.

    Args:
        weight: (M, M) recurrent weight matrix.
        ei_mask: (M,) array of +1 (excitatory) and -1 (inhibitory) signs.

    Returns:
        Dictionary with keys 'EE', 'EI', 'IE', 'II' giving the four submatrices.
        Keys are read as post -> pre (e.g. 'IE' = inhibitory post, excitatory pre).
    """
    weight = np.asarray(weight)
    ei_mask = np.asarray(ei_mask)
    M = weight.shape[0]
    assert ei_mask.shape == (M,), \
        f"ei_mask must have shape ({M},), got {ei_mask.shape}"
    e_idx = np.where(ei_mask > 0)[0]
    i_idx = np.where(ei_mask < 0)[0]
    return {
        "EE": weight[np.ix_(e_idx, e_idx)],
        "EI": weight[np.ix_(e_idx, i_idx)],
        "IE": weight[np.ix_(i_idx, e_idx)],
        "II": weight[np.ix_(i_idx, i_idx)],
    }
