"""Mante Short task — simplified context-dependent decision-making.

6 inputs (2 context, 2 motion, 2 color), 2 outputs.
No noise, no baseline. Context cue on from start. Fixed coherences.

Ported from Langdon & Engel (2025) reference implementation.
"""
import numpy as np
import torch


def generate_trials(n_trials=25, n_t=125):
    """Create trials for the Mante Short task.

    Args:
        n_trials: Number of trials per condition combination.
        n_t: Number of time steps per trial.

    Returns:
        inputs: (N, n_t, 6) tensor.
        targets: (N, n_t, 2) tensor.
        mask: (N, n_t, 2) tensor — loss mask (0 during delay steps 40-79).
        conditions: list of dicts.
    """
    cohs = [-1.0, -0.25, -0.125, -0.0625, 0.0, 0.0625, 0.125, 0.25, 1.0]

    cue_on = 0
    cue_off = n_t
    stim_on = 40
    stim_off = n_t
    dec_on = 80
    dec_off = n_t

    inputs = []
    targets = []
    conditions = []
    for context in ["motion", "color"]:
        for motion_coh in cohs:
            for color_coh in cohs:
                for i in range(n_trials):
                    correct_choice = 1 if (
                        (context == "motion" and motion_coh > 0) or
                        (context == "color" and color_coh > 0)
                    ) else -1
                    conditions.append({
                        "context": context,
                        "motion_coh": motion_coh,
                        "color_coh": color_coh,
                        "correct_choice": correct_choice,
                    })

                    motion_r = (1 + motion_coh) / 2
                    motion_l = 1 - motion_r
                    color_r = (1 + color_coh) / 2
                    color_l = 1 - color_r

                    cue_input = np.zeros([n_t, 6])
                    if context == "motion":
                        cue_input[cue_on:cue_off, 0] = 1.2
                    else:
                        cue_input[cue_on:cue_off, 1] = 1.2

                    motion_input = np.zeros([n_t, 6])
                    motion_input[stim_on - 1:stim_off, 2] = motion_r
                    motion_input[stim_on - 1:stim_off, 3] = motion_l

                    color_input = np.zeros([n_t, 6])
                    color_input[stim_on - 1:stim_off, 4] = color_r
                    color_input[stim_on - 1:stim_off, 5] = color_l

                    # No noise, no baseline
                    input_stream = np.maximum(cue_input + motion_input + color_input, 0)

                    target_stream = 0.2 * np.ones([n_t, 2])
                    if (context == "motion" and motion_coh > 0) or (context == "color" and color_coh > 0):
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

    # Mask: zero out delay period (steps 40-79)
    mask = torch.ones_like(targets)
    mask[:, 40:80, :] = 0

    return inputs, targets, mask, conditions
