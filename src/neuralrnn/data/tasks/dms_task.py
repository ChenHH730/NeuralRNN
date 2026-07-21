"""Discrete delayed match-to-sample (DMS) task.

Two stimuli (A or B) are presented sequentially with a variable delay.
The network must report whether the two stimuli match (A-A, B-B) or
differ (A-B, B-A). This is the discrete-symbol version used in the
low-rank RNN literature; for a continuous-coherence variant see
:mod:`dms_continuous_task`.

Task family: delayed match-to-sample / working memory.
Inputs:  2 channels (one-hot encoding of A vs B).
Targets: 1 channel (match=+1 / different=-1).

Timing (ms, at default dt=20): fixation=100, stim1=500-500, delay=500-3000,
stim2=500-500, decision=1000 (per-trial durations are sampled in the ranges).

References:
    Dubreuil et al. (2022), Nature Neuroscience.
    Valente et al. (2022), NeurIPS.
"""
import numpy as np
import torch

from .task_base import Task

DELTA_T = 20.0
TAU = 100.0
ALPHA = DELTA_T / TAU
STD_DEFAULT = 0.03

FIXATION_DURATION = 100
STIM1_DURATION_MIN = 500
STIM1_DURATION_MAX = 500
DELAY_MIN = 500
DELAY_MAX = 3000
STIM2_DURATION_MIN = 500
STIM2_DURATION_MAX = 500
DECISION_DURATION = 1000


class DMSTask(Task):
    """Discrete delayed match-to-sample task (unified Task interface)."""

    name = "dms"
    input_dim = 2
    output_dim = 1
    default_dt = DELTA_T
    deprecated_kwargs = {
        "num_trials": "n_trials",
        "std": "sigma_in",
        "fraction_catch_trials": "catch_fraction",
    }

    def __init__(self, n_trials=1000, *, trial_type=None, catch_fraction=0.0,
                 sigma_in=STD_DEFAULT, seed=None, dt=DELTA_T):
        self.n_trials = n_trials
        self.trial_type = trial_type
        self.catch_fraction = catch_fraction
        self.sigma_in = sigma_in
        self.seed = seed
        self.dt = dt
        # Discrete timing (previously module-level globals via _setup())
        self.fixation_discrete = int(FIXATION_DURATION / dt)
        self.stim1_min_discrete = int(STIM1_DURATION_MIN / dt)
        self.stim1_max_discrete = int(STIM1_DURATION_MAX / dt)
        self.stim2_min_discrete = int(STIM2_DURATION_MIN / dt)
        self.stim2_max_discrete = int(STIM2_DURATION_MAX / dt)
        self.decision_discrete = int(DECISION_DURATION / dt)
        self.delay_min_discrete = int(DELAY_MIN / dt)
        self.delay_max_discrete = int(DELAY_MAX / dt)
        self.total_duration = (self.fixation_discrete + self.stim1_max_discrete
                               + self.delay_max_discrete + self.stim2_max_discrete
                               + self.decision_discrete)

    def generate_trials(self):
        """Generate trials -> (inputs, targets, mask, conditions)."""
        self._seed_np()
        n = self.n_trials
        trial_types = ['A-A', 'A-B', 'B-A', 'B-B']

        inputs = self.sigma_in * torch.randn((n, self.total_duration, 2))
        targets = torch.zeros((n, self.total_duration, 1))
        mask = torch.zeros((n, self.total_duration, 1))
        conditions = []

        for i in range(n):
            if np.random.rand() > self.catch_fraction:
                if self.trial_type is None:
                    cur_type = trial_types[np.random.randint(0, 4)]
                else:
                    cur_type = self.trial_type

                # Determine stimuli and correct answer
                if cur_type == 'A-A':
                    input1, input2, choice = 1, 1, 1
                elif cur_type == 'A-B':
                    input1, input2, choice = 1, 0, -1
                elif cur_type == 'B-A':
                    input1, input2, choice = 0, 1, -1
                else:  # 'B-B'
                    input1, input2, choice = 0, 0, 1

                # Random durations
                delay_dur = np.random.randint(self.delay_min_discrete, self.delay_max_discrete + 1)
                stim1_dur = np.random.randint(self.stim1_min_discrete, self.stim1_max_discrete + 1)
                stim2_dur = np.random.randint(self.stim2_min_discrete, self.stim2_max_discrete + 1)

                stim1_begin = self.fixation_discrete
                stim1_end = stim1_begin + stim1_dur
                stim2_begin = stim1_end + delay_dur
                stim2_end = stim2_begin + stim2_dur
                decision_begin = stim2_end
                decision_end = decision_begin + self.decision_discrete

                # Input: one-hot for stimulus A (channel 0) and B (channel 1)
                inputs[i, stim1_begin:stim1_end, 0] += input1
                inputs[i, stim1_begin:stim1_end, 1] += 1 - input1
                inputs[i, stim2_begin:stim2_end, 0] += input2
                inputs[i, stim2_begin:stim2_end, 1] += 1 - input2

                targets[i, decision_begin:decision_end] = choice
                mask[i, decision_begin:decision_end] = 1.0

                epochs = {
                    "fixation": (0, stim1_begin),
                    "stim1": (stim1_begin, stim1_end),
                    "delay": (stim1_end, stim2_begin),
                    "stim2": (stim2_begin, stim2_end),
                    "decision": (decision_begin, decision_end),
                }
                conditions.append(self._cond(
                    epochs, self.total_duration, False,
                    trial_type=cur_type,
                    input1=input1,
                    input2=input2,
                    choice=choice,
                    stim1_dur=stim1_dur,
                    stim2_dur=stim2_dur,
                    delay_dur=delay_dur,
                ))
            else:
                conditions.append(self._cond(
                    {"fixation": (0, self.fixation_discrete)}, self.total_duration, True,
                    trial_type="catch",
                ))

        return inputs, targets, mask, conditions


def generate_trials(**kwargs):
    """Backward-compatible shim: DMSTask(**kwargs).generate_trials()."""
    return DMSTask.from_kwargs(**kwargs).generate_trials()
