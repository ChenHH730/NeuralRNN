"""Parametric working memory task.

2 inputs (cos(theta), sin(theta)), 2 outputs (same angle).
Stimulus presented briefly, then a long delay before report.

Ported from Langdon & Engel (2025) reference implementation.
"""
import numpy as np
import torch


def generate_trials(n_trials=100, n_t=75):
    """Create trials for the parametric working memory task.

    Args:
        n_trials: Total number of trials (each with a random angle).
        n_t: Number of time steps per trial.

    Returns:
        inputs: (N, n_t, 2) tensor.
        targets: (N, n_t, 2) tensor.
        mask: (N, n_t, 2) tensor — loss mask (0 during delay).
        conditions: list of dicts.
    """
    stim_on = int(round(n_t * 0.1))
    stim_off = int(round(n_t * 0.4))
    dec_on = int(round(n_t * 0.75))
    dec_off = n_t

    inputs = []
    targets = []
    conditions = []
    for i in range(n_trials):
        theta = np.round(np.random.uniform(0, 2 * np.pi), 3)
        conditions.append({"theta": theta})

        # Input: cos/sin during stimulus period
        input_stream = np.zeros([n_t, 2])
        input_stream[stim_on - 1:stim_off, 0] = np.cos(theta)
        input_stream[stim_on - 1:stim_off, 1] = np.sin(theta)

        # Target: same cos/sin during decision period
        target_stream = np.zeros([n_t, 2])
        target_stream[dec_on - 1:dec_off, 0] = np.cos(theta)
        target_stream[dec_on - 1:dec_off, 1] = np.sin(theta)

        inputs.append(input_stream)
        targets.append(target_stream)

    inputs = np.stack(inputs, 0)
    targets = np.stack(targets, 0)

    inputs = torch.tensor(inputs).float()
    targets = torch.tensor(targets).float()

    # Mask: 1 during stimulus and decision, 0 during delay
    mask = torch.ones_like(targets)
    mask[:, stim_off:dec_on, :] = 0

    return inputs, targets, mask, conditions
