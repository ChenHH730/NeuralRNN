"""Mask construction for the multi-area RNN family.

Weight convention (framework-wide): ``h2h.weight[i, j]`` is the connection
from unit j to unit i, so in ``rec_mask`` rows are targets and columns are
sources. Inter-area blocks are therefore placed at
``rec_mask[target_area_slice, source_area_slice]``.

Inter-area connections originate only from excitatory (E) units of the source
area (long-range cortical projections are excitatory); feedforward targets are
sampled at ``ff_ee_density`` (E targets) and ``ff_ei_density`` (I targets),
feedback targets at ``fb_density`` / ``fb_ei_density``. Signs are enforced by
the Dale mechanism in the model, not by the mask.
"""
from __future__ import annotations

import numpy as np


def area_slices(area_sizes: list[int]) -> list[slice]:
    """Return the unit-index slice of each area within the hidden state."""
    slices = []
    start = 0
    for n in area_sizes:
        slices.append(slice(start, start + n))
        start += n
    return slices


def area_ei_indices(
    area_sizes: list[int], ei_ratio: float
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """Per-area excitatory/inhibitory split.

    Returns:
        dale_signs: (M,) array of +1.0 (E) / -1.0 (I). Within each area the
            first ``round(n * ei_ratio)`` units are E, the rest I.
        e_indices:  list of per-area E-unit index arrays.
        i_indices:  list of per-area I-unit index arrays.
    """
    dale_signs = np.zeros(int(sum(area_sizes)), dtype=np.float32)
    e_indices, i_indices = [], []
    for sl, n in zip(area_slices(area_sizes), area_sizes):
        n_e = int(round(n * ei_ratio))
        idx = np.arange(sl.start, sl.stop)
        e_idx, i_idx = idx[:n_e], idx[n_e:]
        dale_signs[e_idx] = 1.0
        dale_signs[i_idx] = -1.0
        e_indices.append(e_idx)
        i_indices.append(i_idx)
    return dale_signs, e_indices, i_indices


def build_multiarea_masks(
    area_sizes: list[int],
    input_dim: int,
    output_dim: int,
    ei_ratio: float = 0.8,
    intra_density: float = 1.0,
    ff_ee_density: float = 0.10,
    ff_ei_density: float = 0.02,
    fb_density: float = 0.05,
    fb_ei_density: float = 0.0,
    input_areas: tuple | list = (0,),
    input_e_only: bool = False,
    output_area: int = -1,
    output_e_only: bool = True,
    allow_self_connections: bool = True,
    mask_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build (rec_mask, in_mask, out_mask, dale_signs) for a multi-area RNN.

    Shapes follow ConstrainedRNNConfig: rec_mask (M, M), in_mask (input_dim, M),
    out_mask (M, output_dim), dale_signs (M,).
    """
    area_sizes = [int(s) for s in area_sizes]
    n_areas = len(area_sizes)
    M = int(sum(area_sizes))
    rng = np.random.default_rng(mask_seed)
    slices = area_slices(area_sizes)
    dale_signs, e_indices, i_indices = area_ei_indices(area_sizes, ei_ratio)

    # --- recurrent mask: dense intra-area blocks ---
    rec_mask = np.zeros((M, M), dtype=np.float32)
    for sl in slices:
        block = (rng.random((sl.stop - sl.start, sl.stop - sl.start)) < intra_density)
        rec_mask[sl, sl] = block.astype(np.float32)

    # --- inter-area blocks: source restricted to E units of the source area ---
    for src in range(n_areas):
        for tgt in range(n_areas):
            if src == tgt:
                continue
            if tgt == src + 1:  # feedforward
                d_ee, d_ei = ff_ee_density, ff_ei_density
            elif tgt == src - 1:  # feedback
                d_ee, d_ei = fb_density, fb_ei_density
            else:  # no skipping connections by default
                continue
            if d_ee > 0:
                conn = rng.random((len(e_indices[tgt]), len(e_indices[src]))) < d_ee
                rec_mask[np.ix_(e_indices[tgt], e_indices[src])] = conn.astype(np.float32)
            if d_ei > 0 and len(i_indices[tgt]) > 0:
                conn = rng.random((len(i_indices[tgt]), len(e_indices[src]))) < d_ei
                rec_mask[np.ix_(i_indices[tgt], e_indices[src])] = conn.astype(np.float32)

    if not allow_self_connections:
        np.fill_diagonal(rec_mask, 0.0)

    # --- input mask: (input_dim, M); only input areas receive external input ---
    in_mask = np.zeros((input_dim, M), dtype=np.float32)
    for a in input_areas:
        a = int(a) % n_areas
        targets = e_indices[a] if input_e_only else np.arange(slices[a].start, slices[a].stop)
        in_mask[:, targets] = 1.0

    # --- output mask: (M, output_dim); readout only from the output area ---
    out_mask = np.zeros((M, output_dim), dtype=np.float32)
    out_area = int(output_area) % n_areas
    readout_units = (
        e_indices[out_area]
        if output_e_only
        else np.arange(slices[out_area].start, slices[out_area].stop)
    )
    out_mask[readout_units, :] = 1.0

    return rec_mask, in_mask, out_mask, dale_signs
