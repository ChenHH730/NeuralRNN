"""Random Dot Motion (RDM) perceptual decision-making task.

A single noisy stimulus with variable coherence is presented. The network
must report the sign of the coherence (left vs right). This is a classic
perceptual decision-making / evidence-accumulation task, also called a
single-coherence two-alternative forced-choice task.

Task family: perceptual decision making / evidence accumulation.
Inputs:  1 channel (coherence signal + noise).
Targets: 1 channel (choice direction: +1 or -1).

Timing (ms, at default dt=20): fixation=100, stimulus=800, delay=100, decision=20.

References:
    Dubreuil et al. (2022), Nature Neuroscience.
    Valente et al. (2022), NeurIPS.
"""
import numpy as np
import torch

from .task_base import Task

# Task constants (defaults; dt is an instance parameter since the refactor)
DELTA_T = 20.0
FIXATION_DURATION = 100
STIMULUS_DURATION = 800
DELAY_DURATION = 100
DECISION_DURATION = 20
SCALE = 0.1
STD_DEFAULT = 0.1


class RDMTask(Task):
    """RDM perceptual decision-making task (unified Task interface)."""

    name = "rdm"
    aliases = ("two_afc",)
    input_dim = 1
    output_dim = 1
    default_dt = DELTA_T
    deprecated_kwargs = {
        "num_trials": "n_trials",
        "std": "sigma_in",
        "fraction_catch_trials": "catch_fraction",
    }

    def __init__(self, n_trials=1000, *, coherences=None, catch_fraction=0.0,
                 sigma_in=STD_DEFAULT, seed=None, dt=DELTA_T):
        self.n_trials = n_trials
        self.coherences = [-4, -2, -1, 1, 2, 4] if coherences is None else coherences
        self.catch_fraction = catch_fraction
        self.sigma_in = sigma_in
        self.seed = seed
        self.dt = dt
        # Discrete timing (previously module-level globals via _setup())
        self.fixation_discrete = int(FIXATION_DURATION / dt)
        self.stimulus_discrete = int(STIMULUS_DURATION / dt)
        self.stim_end = self.fixation_discrete + self.stimulus_discrete
        self.delay_discrete = int(DELAY_DURATION / dt)
        self.response_begin = self.stim_end + self.delay_discrete
        self.decision_discrete = int(DECISION_DURATION / dt)
        self.total_duration = (self.fixation_discrete + self.stimulus_discrete
                               + self.delay_discrete + self.decision_discrete)

    def generate_trials(self):
        """Generate RDM task trials -> (inputs, targets, mask, conditions)."""
        self._seed_np()
        n = self.n_trials
        coherences = self.coherences

        inputs = self.sigma_in * torch.randn((n, self.total_duration, 1), dtype=torch.float32)
        targets = torch.zeros((n, self.total_duration, 1), dtype=torch.float32)
        mask = torch.zeros((n, self.total_duration, 1), dtype=torch.float32)
        conditions = []

        epochs = {
            "fixation": (0, self.fixation_discrete),
            "stimulus": (self.fixation_discrete, self.stim_end),
            "delay": (self.stim_end, self.response_begin),
            "decision": (self.response_begin, self.total_duration),
        }

        for i in range(n):
            is_catch = np.random.rand() <= self.catch_fraction
            if not is_catch:
                coh = coherences[np.random.randint(0, len(coherences))]
                inputs[i, self.fixation_discrete:self.stim_end, 0] += coh * SCALE
                targets[i, self.response_begin:, 0] = 1.0 if coh > 0 else -1.0
            else:
                coh = 0.0
            mask[i, self.response_begin:, 0] = 1.0
            conditions.append(self._cond(
                epochs, self.total_duration, is_catch,
                coherence=coh,
                correct_choice=(0 if is_catch else (1 if coh > 0 else -1)),
            ))

        return inputs, targets, mask, conditions


def generate_trials(**kwargs):
    """Backward-compatible shim: RDMTask(**kwargs).generate_trials().

    Deprecated parameter names (num_trials/std/fraction_catch_trials) are
    accepted with a DeprecationWarning.
    """
    return RDMTask.from_kwargs(**kwargs).generate_trials()
