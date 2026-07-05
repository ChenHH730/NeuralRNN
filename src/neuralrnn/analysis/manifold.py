"""Manifold / trajectory-geometry wrappers (MARBLE, neuralflow) — placeholder skeleton.

Design decision (ARCHITECTURE §3.3 / §5): MARBLE and neuralflow are **not** RNN models;
they are analysis methods operating on "trajectory / vector-field data", so they belong in analysis/
rather than the model zoo. They are optional heavy dependencies (pip install 'neuralrnn[manifold]').

Unified convention: the upstream code obtains trajectories and velocities (position + velocity pairs)
using model.generate / collect_states; this module feeds them into the MARBLE / neuralflow APIs to
produce embeddings / distances / comparison results. See PORTING_GUIDE recipe 6 (MARBLE) and recipe 8
(neuralflow) for wiring details.
"""
from __future__ import annotations

import numpy as np


def trajectories_to_pos_vel(traj: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert a trajectory (B,T,M) or (T,M) into the (position, velocity) pair required by MARBLE.
    velocity[t] = x[t+1] − x[t], and positions are the first T−1 steps."""
    arr = np.asarray(traj)
    if arr.ndim == 3:
        arr = arr.reshape(-1, arr.shape[-1])
    pos = arr[:-1]
    vel = np.diff(arr, axis=0)
    return pos, vel


def marble_embedding(pos: np.ndarray, vel: np.ndarray, **marble_kwargs):
    """Learn an unsupervised manifold embedding from (pos, vel) with MARBLE.

    Implement when porting (recipe 6):
        from MARBLE import construct_dataset, net
        data = construct_dataset(pos, features=vel)
        model = net(data, **marble_kwargs); model.fit()
        return model.transform(data)
    """
    raise NotImplementedError(
        "MARBLE manifold embedding: follow PORTING_GUIDE recipe 6 to integrate MARBLE, and "
        "pip install 'neuralrnn[manifold]'."
    )


def neuralflow_analysis(spike_data, **kwargs):
    """neuralflow (continuous-time latent flow field) analysis entry point. See recipe 8 for porting."""
    raise NotImplementedError(
        "neuralflow analysis: follow PORTING_GUIDE recipe 8 to integrate neuralflow."
    )
