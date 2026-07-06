"""Mante et al. context-dependent decision-making task.

6 inputs (2 context, 2 motion, 2 color), 2 outputs.
Trial structure: context cue -> delay -> stimulus -> decision.
The training mask is active only during the decision period.

Reference: Mante et al. (2013), Nature.
Ported from Langdon & Engel (2025) reference implementation.
"""
import numpy as np
import torch


def generate_input_target_stream(
    context, motion_coh, color_coh, baseline, alpha, sigma_in,
    n_t, cue_on, cue_off, stim_on, stim_off, dec_on, dec_off,
):
    """Generate input and target sequence for a single trial."""
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

    noise = np.sqrt(2 / alpha * sigma_in * sigma_in) * np.random.multivariate_normal(
        [0, 0, 0, 0, 0, 0], np.eye(6), n_t
    )
    baseline_arr = baseline * np.ones([n_t, 6])
    input_stream = np.maximum(baseline_arr + cue_input + motion_input + color_input + noise, 0)

    target_stream = 0.2 * np.ones([n_t, 2])
    if (context == "motion" and motion_coh > 0) or (context == "color" and color_coh > 0):
        target_stream[dec_on - 1:dec_off, 0] = 1.2
        target_stream[dec_on - 1:dec_off, 1] = 0.2
    else:
        target_stream[dec_on - 1:dec_off, 0] = 0.2
        target_stream[dec_on - 1:dec_off, 1] = 1.2

    return input_stream, target_stream


def generate_trials(n_trials=25, alpha=0.2, sigma_in=0.01, baseline=0.2, n_coh=6, n_t=75):
    """Create trials for the Mante task.

    Trial structure: context cue (steps ~7-24) -> delay -> stimulus
    (steps ~30-74) -> decision (steps ~55-74). The context cue precedes the
    stimulus, matching the standard Mante et al. (2013) design. The training
    mask is active only during the decision period.

    Returns:
        inputs: (N, n_t, 6) tensor.
        targets: (N, n_t, 2) tensor — full-length target sequence.
        mask: (N, n_t, 2) float tensor — 1 during decision period, 0 otherwise.
        conditions: list of dicts.
    """
    cohs = np.linspace(-0.2, 0.2, n_coh)

    cue_on = int(round(n_t * 0.1))
    cue_off = int(round(n_t * 0.33))
    stim_on = int(round(n_t * 0.4))
    stim_off = int(round(n_t))
    dec_on = int(round(n_t * 0.75))
    dec_off = int(round(n_t))

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
                    inp, tgt = generate_input_target_stream(
                        context, motion_coh, color_coh, baseline, alpha, sigma_in,
                        n_t, cue_on, cue_off, stim_on, stim_off, dec_on, dec_off,
                    )
                    inputs.append(inp)
                    targets.append(tgt)

    inputs = np.stack(inputs, 0)
    targets = np.stack(targets, 0)

    perm = np.random.permutation(len(inputs))
    inputs = torch.tensor(inputs[perm, :, :]).float()
    targets = torch.tensor(targets[perm, :, :]).float()
    conditions = [conditions[index] for index in perm]

    # Decision-period mask (1 during decision, 0 otherwise)
    mask = torch.zeros_like(targets)
    mask[:, dec_on - 1:dec_off, :] = 1.0

    return inputs, targets, mask, conditions
