"""Mante context-dependent decision-making task — low-rank RNN convention.

Two stimuli (color and motion) are presented simultaneously, with a context
cue indicating which to attend to. This is the same cognitive paradigm as
:mod:`mante_task`, but using the low-rank RNN convention (4 inputs, 1 scalar
output, millisecond timing) from Dubreuil et al. (2022) / Valente et al. (2022).
Registered as ``mante2`` (formerly ``lr_mante``, kept as a deprecated alias).

Task family: context-dependent perceptual decision making.
Inputs:  4 channels [color_stim, motion_stim, color_ctx, motion_ctx].
Targets: 1 channel (decision sign: +1 or -1).

Timing (ms, at default dt=20): fixation=100, ctx_pre=350, stimulus=800,
delay=100, decision=20.

References:
    Mante et al. (2013), Nature.
    Dubreuil et al. (2022), Nature Neuroscience.
    Valente et al. (2022), NeurIPS.
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


def _setup():
    """Compute module-level discrete timing (backward compatibility).

    The Mante2Task class computes per-instance timing and does NOT read these
    globals; they exist so legacy code importing ``total_duration`` /
    ``stim_begin`` / ... from this module (formerly ``lr_mante_task``) keeps
    working. ``SCALE`` / ``SCALE_CTX`` ARE read by generate_trials at call
    time, so mutating them (as notebook 07 does with SCALE_CTX) still works.
    """
    global fixation_discrete, ctx_pre_discrete, stimulus_discrete
    global delay_discrete, decision_discrete, total_duration
    global stim_begin, stim_end, response_begin
    fixation_discrete = int(FIXATION_DURATION / DELTA_T)
    ctx_pre_discrete = int(CTX_ONLY_PRE_DURATION / DELTA_T)
    stimulus_discrete = int(STIMULUS_DURATION / DELTA_T)
    delay_discrete = int(DELAY_DURATION / DELTA_T)
    decision_discrete = int(DECISION_DURATION / DELTA_T)
    stim_begin = fixation_discrete + ctx_pre_discrete
    stim_end = stim_begin + stimulus_discrete
    response_begin = stim_end + delay_discrete
    total_duration = (fixation_discrete + stimulus_discrete +
                      delay_discrete + ctx_pre_discrete + decision_discrete)


_setup()


class Mante2Task(Task):
    """Mante context-dependent DM, low-rank convention (unified interface)."""

    name = "mante2"
    aliases = ("lr_mante",)
    input_dim = 4
    output_dim = 1
    default_dt = DELTA_T
    deprecated_kwargs = {
        "num_trials": "n_trials",
        "std": "sigma_in",
        "fraction_catch_trials": "catch_fraction",
    }

    def __init__(self, n_trials=1000, *, coherences=None, sigma_in=STD_DEFAULT,
                 catch_fraction=0.0, coh_color=None, coh_motion=None,
                 context=None, seed=None, dt=DELTA_T):
        self.n_trials = n_trials
        self.coherences = [-4, -2, -1, 1, 2, 4] if coherences is None else coherences
        self.sigma_in = sigma_in
        self.catch_fraction = catch_fraction
        self.coh_color = coh_color
        self.coh_motion = coh_motion
        self.context = context
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
                if self.coh_color is None:
                    c_color = coherences[np.random.randint(0, len(coherences))]
                else:
                    c_color = self.coh_color
                if self.coh_motion is None:
                    c_motion = coherences[np.random.randint(0, len(coherences))]
                else:
                    c_motion = self.coh_motion
                if self.context is None:
                    ctx = np.random.randint(1, 3)  # 1 or 2
                else:
                    ctx = self.context

                inputs[i, self.stim_begin:self.stim_end, 0] += c_color * SCALE
                inputs[i, self.stim_begin:self.stim_end, 1] += c_motion * SCALE

                if ctx == 1:
                    inputs[i, self.fixation_discrete:self.response_begin, 2] = 1.0 * SCALE_CTX
                    targets[i, self.response_begin:, 0] = 1.0 if c_color > 0 else -1.0
                else:  # ctx == 2
                    inputs[i, self.fixation_discrete:self.response_begin, 3] = 1.0 * SCALE_CTX
                    targets[i, self.response_begin:, 0] = 1.0 if c_motion > 0 else -1.0

                conditions.append(self._cond(
                    epochs, self.total_duration, False,
                    context=ctx,
                    coh_color=c_color,
                    coh_motion=c_motion,
                    target=targets[i, self.response_begin:, 0].mean().item(),
                ))
            else:
                conditions.append(self._cond(epochs, self.total_duration, True))

            mask[i, self.response_begin:, 0] = 1.0

        return inputs, targets, mask, conditions


def generate_trials(**kwargs):
    """Backward-compatible shim: Mante2Task(**kwargs).generate_trials()."""
    return Mante2Task.from_kwargs(**kwargs).generate_trials()
