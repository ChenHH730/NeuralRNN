"""Siegel & Miller context-dependent decision-making task.

6 inputs (2 context cues, 2 motion channels, 2 color channels), 2 outputs.
Trial structure: context cue -> delay -> stimulus -> decision.

Reference: Siegel et al. (2015), Nature Neuroscience.
Ported from Langdon & Engel (2025) reference implementation.
"""
import numpy as np
import torch


def generate_input_target_stream(
    context, motion_coh, color_coh, baseline, alpha, sigma_in,
    n_t, cue_on, cue_off, stim_on, stim_off, dec_on, dec_off,
):
    """Generate input and target sequence for a single trial.

    Args:
        context: "motion" or "color".
        motion_coh: Motion coherence value.
        color_coh: Color coherence value.
        baseline: Baseline input level.
        alpha: Time constant parameter (dt/tau).
        sigma_in: Input noise standard deviation.
        n_t: Number of time steps.
        cue_on, cue_off: Context cue timing.
        stim_on, stim_off: Stimulus timing.
        dec_on, dec_off: Decision period timing.

    Returns:
        input_stream: (n_t, 6) array.
        target_stream: (n_t, 2) array.
    """
    # Transform coherence to signal
    motion_r = (1 + motion_coh) / 2
    motion_l = 1 - motion_r
    color_r = (1 + color_coh) / 2
    color_l = 1 - color_r

    # Cue input stream
    cue_input = np.zeros([n_t, 6])
    if context == "motion":
        cue_input[cue_on:cue_off, 0] = 1.2
    else:
        cue_input[cue_on:cue_off, 1] = 1.2

    # Motion input stream
    motion_input = np.zeros([n_t, 6])
    motion_input[stim_on - 1:stim_off, 2] = motion_r
    motion_input[stim_on - 1:stim_off, 3] = motion_l

    # Color input stream
    color_input = np.zeros([n_t, 6])
    color_input[stim_on - 1:stim_off, 4] = color_r
    color_input[stim_on - 1:stim_off, 5] = color_l

    # Noise and baseline signal
    noise = np.sqrt(2 / alpha * sigma_in * sigma_in) * np.random.multivariate_normal(
        [0, 0, 0, 0, 0, 0], np.eye(6), n_t
    )
    baseline_arr = baseline * np.ones([n_t, 6])

    # Input stream is rectified sum of baseline, task and noise signals
    input_stream = np.maximum(baseline_arr + cue_input + motion_input + color_input + noise, 0)

    # Target stream
    target_stream = 0.2 * np.ones([n_t, 2])
    if (context == "motion" and motion_coh > 0) or (context == "color" and color_coh > 0):
        target_stream[dec_on - 1:dec_off, 0] = 1.2
        target_stream[dec_on - 1:dec_off, 1] = 0.2
    else:
        target_stream[dec_on - 1:dec_off, 0] = 0.2
        target_stream[dec_on - 1:dec_off, 1] = 1.2

    return input_stream, target_stream


def generate_trials(n_trials=25, alpha=0.2, sigma_in=0.01, baseline=0.2, n_coh=6, n_t=75):
    """Create a set of trials for the context-dependent decision-making task.

    Args:
        n_trials: Number of trials per condition combination.
        alpha: Time constant parameter (dt/tau).
        sigma_in: Input noise standard deviation.
        baseline: Baseline input level.
        n_coh: Number of coherence levels.
        n_t: Number of time steps per trial.

    Returns:
        inputs: (N, n_t, 6) tensor — task inputs.
        targets: (N, n_t, 2) tensor — target outputs.
        mask: (N, n_t, 2) tensor — loss mask (1 during decision, 0 otherwise).
        conditions: list of dicts — trial conditions.
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

    # Mask: 1 during decision period, 0 otherwise
    mask = torch.ones_like(targets)
    mask[:, :dec_on, :] = 0

    return inputs, targets, mask, conditions
