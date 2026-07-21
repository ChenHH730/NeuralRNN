"""Raposo multisensory decision-making task.

Two stimuli (visual and auditory) are presented with variable coherences.
A context cue indicates which modality to attend to (visual, auditory, or both).
This is the multisensory / context-dependent decision task used in the
low-rank RNN literature.

Task family: multisensory / context-dependent perceptual decision making.
Inputs:  4 channels [visual_stim, auditory_stim, visual_ctx, auditory_ctx].
Targets: 1 channel (choice sign).

Timing (ms, at default dt=20): fixation=100, ctx_pre=350, stimulus=800,
delay=100, decision=20.

References:
    Raposo et al. (2014), Nature Neuroscience.
    Dubreuil et al. (2022), Nature Neuroscience.
"""
import numpy as np
import torch

from .task_base import Task

DELTA_T = 20.0
FIXATION_DURATION = 100
CTX_ONLY_PRE_DURATION = 350
STIMULUS_DURATION = 800
DELAY_DURATION = 100
DECISION_DURATION = 20
SCALE = 0.1
SCALE_CTX = 0.1
STD_DEFAULT = 0.1


class RaposoTask(Task):
    """Raposo multisensory decision-making task (unified Task interface)."""

    name = "raposo"
    input_dim = 4
    output_dim = 1
    default_dt = DELTA_T
    deprecated_kwargs = {
        "num_trials": "n_trials",
        "std": "sigma_in",
        "fraction_catch_trials": "catch_fraction",
    }

    def __init__(self, n_trials=1000, *, coherences=None, catch_fraction=0.0,
                 context=None, sigma_in=STD_DEFAULT, seed=None, dt=DELTA_T):
        self.n_trials = n_trials
        self.coherences = [-4, -2, -1, 1, 2, 4] if coherences is None else coherences
        self.catch_fraction = catch_fraction
        self.context = context
        self.sigma_in = sigma_in
        self.seed = seed
        self.dt = dt
        # Discrete timing (previously module-level globals via _setup())
        self.fixation_discrete = int(FIXATION_DURATION / dt)
        self.ctx_pre_discrete = int(CTX_ONLY_PRE_DURATION / dt)
        self.stimulus_discrete = int(STIMULUS_DURATION / dt)
        self.delay_discrete = int(DELAY_DURATION / dt)
        self.decision_discrete = int(DECISION_DURATION / dt)
        self.stim_begin = self.fixation_discrete + self.ctx_pre_discrete
        self.stim_end = self.stim_begin + self.stimulus_discrete
        self.response_begin = self.stim_end + self.delay_discrete
        self.total_duration = (self.fixation_discrete + self.stimulus_discrete
                               + self.delay_discrete + self.ctx_pre_discrete
                               + self.decision_discrete)

    def generate_trials(self):
        """Generate trials -> (inputs, targets, mask, conditions)."""
        self._seed_np()
        n = self.n_trials
        coherences = self.coherences
        coherences_pos = [c for c in coherences if c >= 0]
        coherences_neg = [c for c in coherences if c < 0]

        inputs_sensory = self.sigma_in * torch.randn(
            (n, self.total_duration, 2), dtype=torch.float32)
        inputs_context = torch.zeros((n, self.total_duration, 2))
        inputs = torch.cat([inputs_sensory, inputs_context], dim=2)
        targets = torch.zeros((n, self.total_duration, 1), dtype=torch.float32)
        mask = torch.zeros((n, self.total_duration, 1), dtype=torch.float32)
        conditions = []

        epochs = {
            "fixation": (0, self.fixation_discrete),
            "ctx_pre": (self.fixation_discrete, self.stim_begin),
            "stimulus": (self.stim_begin, self.stim_end),
            "delay": (self.stim_end, self.response_begin),
            "decision": (self.response_begin, self.total_duration),
        }

        for i in range(n):
            if np.random.rand() > self.catch_fraction:
                choice = np.random.choice([-1.0, 1.0])
                if len(coherences_pos) == 0:
                    choice = -1.0
                elif len(coherences_neg) == 0:
                    choice = 1.0

                if self.context is None:
                    ctx = np.random.randint(-1, 2)  # -1, 0, or 1
                else:
                    ctx = self.context

                # Visual channel (channel 0)
                if ctx in [1, 0]:
                    if choice > 0:
                        coh = coherences_pos[np.random.randint(0, len(coherences_pos))]
                    else:
                        coh = coherences_neg[np.random.randint(0, len(coherences_neg))]
                    inputs[i, self.stim_begin:self.stim_end, 0] += coh * SCALE
                    inputs[i, self.fixation_discrete:self.stim_end, 2] = 1.0 * SCALE_CTX

                # Auditory channel (channel 1)
                if ctx in [-1, 0]:
                    if choice > 0:
                        coh = coherences_pos[np.random.randint(0, len(coherences_pos))]
                    else:
                        coh = coherences_neg[np.random.randint(0, len(coherences_neg))]
                    inputs[i, self.stim_begin:self.stim_end, 1] += coh * SCALE
                    inputs[i, self.fixation_discrete:self.stim_end, 3] = 1.0 * SCALE_CTX

                targets[i, self.response_begin:, 0] = choice
                conditions.append(self._cond(
                    epochs, self.total_duration, False,
                    context=ctx, choice=choice,
                ))
            else:
                conditions.append(self._cond(
                    epochs, self.total_duration, True,
                    context=0, choice=0.0,
                ))

            mask[i, self.response_begin:, 0] = 1.0

        return inputs, targets, mask, conditions


def generate_trials(**kwargs):
    """Backward-compatible shim: RaposoTask(**kwargs).generate_trials()."""
    return RaposoTask.from_kwargs(**kwargs).generate_trials()
