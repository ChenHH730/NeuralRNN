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

from .task_base import Task


class GoNogoTask(Task):
    """Go/NoGo task (unified Task interface; deterministic)."""

    name = "go_nogo"
    input_dim = 3
    output_dim = 1
    default_dt = None

    def __init__(self, *, n_steps=60, stim_on=0, stim_off=60, cue_on=30, cue_off=60,
                 n_values=11, n_reps=1, input_dim=3, output_dim=1,
                 mask_periods=((10, 30), (40, 60)), seed=None):
        self.n_steps = n_steps
        self.stim_on = stim_on
        self.stim_off = stim_off
        self.cue_on = cue_on
        self.cue_off = cue_off
        self.n_values = n_values
        self.n_reps = n_reps  # repetitions per value (total trials = n_values * n_reps)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mask_periods = tuple(mask_periods)
        self.seed = seed

    @property
    def n_trials(self):
        return int(self.n_values * self.n_reps)

    def generate_trials(self):
        """Generate Go/NoGo trials -> (inputs, targets, mask, conditions)."""
        self._seed_np()  # task is deterministic; kept for API consistency

        values = np.linspace(0, 1, self.n_values)
        n_trials = self.n_trials
        n_steps = self.n_steps

        inputs = np.zeros((n_trials, n_steps, self.input_dim), dtype=np.float32)
        targets = np.zeros((n_trials, n_steps, self.output_dim), dtype=np.float32)
        conditions = []

        epochs = {
            "stimulus": (self.stim_on, self.stim_off),
            "cue": (self.cue_on, self.cue_off),
        }

        i = 0
        for _ in range(self.n_reps):
            for value in values:
                inputs[i, self.stim_on:self.stim_off, 0] = value
                inputs[i, self.cue_on:self.cue_off, 1] = 1.0
                inputs[i, :, 2] = 1.0

                if value < 0.5:
                    output_value = 0.0
                elif value > 0.5:
                    output_value = 1.0
                else:
                    output_value = 0.5

                targets[i, self.cue_on:self.cue_off, 0] = output_value
                conditions.append(self._cond(
                    epochs, n_steps, False,
                    input_value=float(value), output_value=float(output_value),
                ))
                i += 1

        mask = torch.zeros((n_trials, n_steps, self.output_dim), dtype=torch.float32)
        for start, end in self.mask_periods:
            mask[:, start:end, :] = 1.0

        return (
            torch.from_numpy(inputs).float(),
            torch.from_numpy(targets).float(),
            mask,
            conditions,
        )


def generate_trials(**kwargs):
    """Backward-compatible shim: GoNogoTask(**kwargs).generate_trials()."""
    return GoNogoTask.from_kwargs(**kwargs).generate_trials()
