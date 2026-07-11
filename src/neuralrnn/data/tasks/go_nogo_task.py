"""Go/NoGo task generator for Paradigm A.

A single scalar input value is presented throughout the trial. Upon presentation
of a Go cue at the end of the trial, the network must output 1 if the input
value is above 0.5, 0 if below 0.5, and 0.5 if exactly 0.5.

Task structure is taken from Tolmachev & Engel (2025),
"Single-unit activations confer inductive biases for emergent circuit solutions
to cognitive tasks".
"""
from __future__ import annotations

import numpy as np
import torch


def generate_trials(
    n_steps: int = 60,
    stim_on: int = 0,
    stim_off: int = 60,
    cue_on: int = 30,
    cue_off: int = 60,
    n_values: int = 11,
    input_dim: int = 3,
    output_dim: int = 1,
    mask_periods: tuple[tuple[int, int], ...] = ((10, 30), (40, 60)),
    seed: int | None = None,
):
    """Generate Go/NoGo trials.

    Args:
        n_steps: Trial length in time steps.
        stim_on: Start of stimulus period.
        stim_off: End of stimulus period.
        cue_on: Start of Go cue.
        cue_off: End of Go cue.
        n_values: Number of input values uniformly spaced in [0, 1].
        input_dim: Number of input channels (3: value, Go cue, bias).
        output_dim: Number of output channels (1).
        mask_periods: Time intervals where the loss is evaluated.
        seed: Optional random seed (kept for API consistency; task is deterministic).

    Returns:
        inputs: (n_trials, n_steps, input_dim) tensor.
        targets: (n_trials, n_steps, output_dim) tensor.
        mask: (n_trials, n_steps, output_dim) tensor — 1 at evaluated time steps.
        conditions: list of dicts with keys ``input_value`` and ``output_value``.
    """
    if seed is not None:
        np.random.seed(seed)

    values = np.linspace(0, 1, n_values)
    n_trials = len(values)

    inputs = np.zeros((n_trials, n_steps, input_dim), dtype=np.float32)
    targets = np.zeros((n_trials, n_steps, output_dim), dtype=np.float32)
    conditions = []

    for i, value in enumerate(values):
        inputs[i, stim_on:stim_off, 0] = value
        inputs[i, cue_on:cue_off, 1] = 1.0
        inputs[i, :, 2] = 1.0

        if value < 0.5:
            output_value = 0.0
        elif value > 0.5:
            output_value = 1.0
        else:
            output_value = 0.5

        targets[i, cue_on:cue_off, 0] = output_value
        conditions.append({"input_value": float(value), "output_value": float(output_value)})

    mask = torch.zeros((n_trials, n_steps, output_dim), dtype=torch.float32)
    for start, end in mask_periods:
        mask[:, start:end, :] = 1.0

    return (
        torch.from_numpy(inputs).float(),
        torch.from_numpy(targets).float(),
        mask,
        conditions,
    )
