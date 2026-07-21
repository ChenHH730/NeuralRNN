"""Parametric working memory task (circular angle).

A continuous-angle working-memory task. A brief stimulus encodes an angle on the
unit circle; after a delay the network must reproduce the same angle.

Task family: parametric working memory.
Inputs:  2 channels [cos(theta), sin(theta)] during stimulus.
Targets: 2 channels [cos(theta), sin(theta)] during decision.

Also called the circular working-memory or angle-memory task.

References:
    Langdon & Engel (2025) reference implementation.
"""
import numpy as np
import torch

from .task_base import Task


class WMAngleTask(Task):
    """Circular-angle parametric working-memory task (unified interface)."""

    name = "wm_angle"
    aliases = ("parametric_wm",)
    input_dim = 2
    output_dim = 2
    default_dt = None

    def __init__(self, n_trials=100, *, n_t=75, sigma_in=0.0, seed=None):
        self.n_trials = n_trials
        self.n_t = n_t
        self.sigma_in = sigma_in  # i.i.d. input noise std (0 = noiseless, the default)
        self.seed = seed

    def generate_trials(self):
        """Create trials for the parametric working-memory task.

        Returns:
            inputs: (N, n_t, 2) tensor.
            targets: (N, n_t, 2) tensor.
            mask: (N, n_t, 2) tensor — 1 during stimulus and decision, 0 during delay.
            conditions: list of dicts with key ``theta`` (+ unified epochs/n_steps/is_catch).
        """
        self._seed_np()
        n_t = self.n_t
        stim_on = int(round(n_t * 0.1))
        stim_off = int(round(n_t * 0.4))
        dec_on = int(round(n_t * 0.75))
        dec_off = n_t

        epochs = {
            "stimulus": (stim_on - 1, stim_off),
            "delay": (stim_off, dec_on - 1),
            "decision": (dec_on - 1, dec_off),
        }

        inputs = []
        targets = []
        conditions = []
        for _ in range(self.n_trials):
            theta = np.round(np.random.uniform(0, 2 * np.pi), 3)
            conditions.append(self._cond(epochs, n_t, False, theta=float(theta)))

            # Input: cos/sin during stimulus period
            input_stream = np.zeros([n_t, 2])
            input_stream[stim_on - 1:stim_off, 0] = np.cos(theta)
            input_stream[stim_on - 1:stim_off, 1] = np.sin(theta)
            if self.sigma_in > 0:
                input_stream = input_stream + self.sigma_in * np.random.randn(n_t, 2)

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


def generate_trials(**kwargs):
    """Backward-compatible shim: WMAngleTask(**kwargs).generate_trials()."""
    return WMAngleTask.from_kwargs(**kwargs).generate_trials()
