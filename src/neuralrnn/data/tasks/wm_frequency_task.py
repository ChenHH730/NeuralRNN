"""Romo parametric working-memory task.

Two vibrotactile stimuli at different frequencies (f1, f2) are presented
sequentially with a variable delay. The network must report which frequency
was higher. This is a classic parametric working-memory / comparison task.

Task family: parametric working memory / delayed comparison.
Inputs:  1 channel (normalized frequency value).
Targets: 1 channel (signed comparison (f1 - f2) / span).

Timing (ms, at default dt=20): fixation=100, stim1=100, delay=500-1000,
stim2=100, decision=100.

References:
    Romo et al. (1999), Nature.
    Dubreuil et al. (2022), Nature Neuroscience.
"""
import numpy as np
import torch

from .task_base import Task

DELTA_T = 20.0
FIXATION_DURATION = 100
STIM1_DURATION = 100
DELAY_MIN = 500
DELAY_MAX = 1000
STIM2_DURATION = 100
DECISION_DURATION = 100
STD_DEFAULT = 0.01

# Frequency pairs used in the task
FDIFFS = [-24, -16, -8, 8, 16, 24]
FPAIRS = [(base, base + fdiff) for fdiff in FDIFFS
          for base in range(max(10 - fdiff, 10), min(34, 34 - fdiff) + 1)]
FMAX = max([max(*p) for p in FPAIRS])
FMIN = min([min(*p) for p in FPAIRS])
FMIDDLE = (FMAX + FMIN) / 2.0
FSPAN = FMAX - FMIN


class WMFrequencyTask(Task):
    """Romo frequency-comparison working-memory task (unified interface)."""

    name = "wm_frequency"
    aliases = ("romo",)
    input_dim = 1
    output_dim = 1
    default_dt = DELTA_T
    deprecated_kwargs = {
        "num_trials": "n_trials",
        "std": "sigma_in",
        "fraction_catch_trials": "catch_fraction",
    }

    def __init__(self, n_trials=1000, *, sigma_in=STD_DEFAULT, fpairs=None,
                 catch_fraction=0.0, delay_discrete=None, seed=None, dt=DELTA_T):
        self.n_trials = n_trials
        self.sigma_in = sigma_in
        self.fpairs = FPAIRS if fpairs is None else fpairs
        self.catch_fraction = catch_fraction
        self.delay_discrete = delay_discrete
        self.seed = seed
        self.dt = dt
        # Discrete timing (previously module-level globals via _setup())
        self.fixation_discrete = int(FIXATION_DURATION / dt)
        self.stim1_discrete = int(STIM1_DURATION / dt)
        self.stim1_end = self.fixation_discrete + self.stim1_discrete
        self.stim2_discrete = int(STIM2_DURATION / dt)
        self.decision_discrete = int(DECISION_DURATION / dt)
        self.min_delay_discrete = int(DELAY_MIN / dt)
        self.max_delay_discrete = int(DELAY_MAX / dt)
        self.total_duration = (self.fixation_discrete + self.stim1_discrete
                               + self.max_delay_discrete + self.stim2_discrete
                               + self.decision_discrete)

    def generate_trials(self):
        """Generate trials -> (inputs, targets, mask, conditions)."""
        self._seed_np()
        n = self.n_trials
        fpairs = self.fpairs

        inputs = self.sigma_in * torch.randn((n, self.total_duration, 1))
        targets = torch.zeros((n, self.total_duration, 1), dtype=torch.float32)
        mask = torch.zeros((n, self.total_duration, 1), dtype=torch.float32)
        conditions = []

        for i in range(n):
            if np.random.rand() > self.catch_fraction:
                f1, f2 = fpairs[np.random.randint(0, len(fpairs))]

                if self.delay_discrete is None:
                    delay = np.random.randint(self.min_delay_discrete, self.max_delay_discrete + 1)
                else:
                    delay = self.delay_discrete

                stim2_begin = self.stim1_end + delay
                stim2_end = stim2_begin + self.stim2_discrete
                decision_end = stim2_end + self.decision_discrete

                # Normalize frequencies to [-0.5, 0.5] range
                inputs[i, self.fixation_discrete:self.stim1_end] += (f1 - FMIDDLE) / FSPAN
                inputs[i, stim2_begin:stim2_end] += (f2 - FMIDDLE) / FSPAN
                targets[i, stim2_end:decision_end, 0] = (f1 - f2) / FSPAN
                mask[i, stim2_end:decision_end, 0] = 1.0

                epochs = {
                    "fixation": (0, self.fixation_discrete),
                    "stim1": (self.fixation_discrete, self.stim1_end),
                    "delay": (self.stim1_end, stim2_begin),
                    "stim2": (stim2_begin, stim2_end),
                    "decision": (stim2_end, decision_end),
                }
                conditions.append(self._cond(
                    epochs, self.total_duration, False,
                    f1=f1, f2=f2, delay=delay,
                ))
            else:
                conditions.append(self._cond(
                    {"fixation": (0, self.fixation_discrete)}, self.total_duration, True,
                    f1=0.0, f2=0.0, delay=0,
                ))

        return inputs, targets, mask, conditions


def generate_trials(**kwargs):
    """Backward-compatible shim: WMFrequencyTask(**kwargs).generate_trials()."""
    return WMFrequencyTask.from_kwargs(**kwargs).generate_trials()
