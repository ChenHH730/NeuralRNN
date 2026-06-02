"""Two-alternative forced choice (2AFC) task.

2 inputs (motion right/left), 2 outputs (choice right/left).
Simple motion discrimination with no context.

Ported from Langdon & Engel (2025) reference implementation.
"""
import numpy as np
import torch


def generate_trials(n_trials=25, n_t=75):
    """Create trials for the 2AFC task.

    Args:
        n_trials: Number of trials per coherence level.
        n_t: Number of time steps per trial.

    Returns:
        inputs: (N, n_t, 2) tensor.
        targets: (N, n_t, 2) tensor.
        mask: (N, n_t, 2) tensor — loss mask (1 during decision only).
        conditions: list of dicts.
    """
    cohs = [-1.0, -0.25, -0.125, -0.0625, 0.0, 0.0625, 0.125, 0.25, 1.0]

    stim_on = int(round(n_t * 0.4))
    stim_off = n_t
    dec_on = int(round(n_t * 0.75))
    dec_off = n_t

    inputs = []
    targets = []
    conditions = []
    for motion_coh in cohs:
        for i in range(n_trials):
            correct_choice = 1 if motion_coh > 0 else -1
            conditions.append({
                "motion_coh": motion_coh,
                "correct_choice": correct_choice,
            })

            motion_r = (1 + motion_coh) / 2
            motion_l = 1 - motion_r

            input_stream = np.zeros([n_t, 2])
            input_stream[stim_on - 1:stim_off, 0] = motion_r
            input_stream[stim_on - 1:stim_off, 1] = motion_l

            target_stream = 0.2 * np.ones([n_t, 2])
            if motion_coh > 0:
                target_stream[dec_on - 1:dec_off, 0] = 1.2
                target_stream[dec_on - 1:dec_off, 1] = 0.2
            else:
                target_stream[dec_on - 1:dec_off, 0] = 0.2
                target_stream[dec_on - 1:dec_off, 1] = 1.2

            inputs.append(input_stream)
            targets.append(target_stream)

    inputs = np.stack(inputs, 0)
    targets = np.stack(targets, 0)

    perm = np.random.permutation(len(inputs))
    inputs = torch.tensor(inputs[perm, :, :]).float()
    targets = torch.tensor(targets[perm, :, :]).float()
    conditions = [conditions[index] for index in perm]

    # Mask: 1 during decision period only
    mask = torch.ones_like(targets)
    mask[:, :dec_on, :] = 0

    return inputs, targets, mask, conditions
