"""Mante et al. (2013) context-dependent task — Langdon & Engel (2025) convention.

Same cognitive paradigm as :mod:`mante_task`, but using the input/output
convention of the latent-circuit paper (Langdon & Engel 2025, Methods), as
used by the Fig. 4 example for both task optimization and PFC reconstruction:
150 steps of 10 ms, the context cue on for the whole trial, sensory channels
(1 +/- coherence)/2 from step 75, and 0/1 choice targets supervised only in
the last 25 steps. The default coherence grid has 11 levels including zero and
+/-1/2/4%, so psychometric functions are sampled in the graded regime; sensory
channels carry per-step Gaussian noise (rectified at zero), and trials with
zero context-relevant coherence are excluded from the loss (mask left at zero)
but kept for psychometric analyses.

Task family: context-dependent perceptual decision making.
Inputs:  6 channels [motion_ctx, color_ctx, motion_r, motion_l, color_r, color_l].
Targets: 2 channels [choice_right, choice_left], 0/1 in the decision window.

Note: trials are generated on a condition grid (context x motion_coh x
color_coh) without shuffling; ``n_reps`` repetitions per grid cell, so the
total trial count is ``n_reps * 2 * n_coh_motion * n_coh_color``.

References:
    Mante et al. (2013), "Context-dependent computation by recurrent dynamics
        in prefrontal cortex", Nature.
    Langdon & Engel (2025), latent-circuit task convention.
"""
import numpy as np
import torch

from .task_base import Task


DEFAULT_COHS = (-0.2, -0.1, -0.04, -0.02, -0.01, 0.0, 0.01, 0.02, 0.04, 0.1, 0.2)


class ManteLCTask(Task):
    """Mante context-dependent DM, Langdon & Engel convention (unified interface)."""

    name = "mante_lc"
    input_dim = 6
    output_dim = 2
    default_dt = 10.0  # ms per step (150 steps = 1.5 s)

    def __init__(self, n_reps=25, *, cohs=None, cohs_motion=None, cohs_color=None,
                 sigma_in=0.05, alpha=0.2, n_t=150, stim_on=75, dec_on=125, seed=None):
        if not (0 <= stim_on < dec_on <= n_t):
            raise ValueError("require 0 <= stim_on < dec_on <= n_t")
        self.n_reps = n_reps
        self.cohs = cohs
        self.cohs_motion = cohs_motion
        self.cohs_color = cohs_color
        self.sigma_in = sigma_in
        self.alpha = alpha
        self.n_t = n_t
        self.stim_on = stim_on
        self.dec_on = dec_on
        self.seed = seed

    def _grids(self):
        """(motion coherences, color coherences) as plain Python floats."""
        base = self.cohs if self.cohs is not None else DEFAULT_COHS
        motion = self.cohs_motion if self.cohs_motion is not None else base
        color = self.cohs_color if self.cohs_color is not None else base
        return [float(v) for v in motion], [float(v) for v in color]

    @property
    def n_trials(self):
        """Total trial count (n_reps x 2 contexts x n_coh_motion x n_coh_color)."""
        motion, color = self._grids()
        return int(self.n_reps * 2 * len(motion) * len(color))

    def generate_trials(self):
        """Create trials for the Mante / Langdon & Engel convention task.

        Trial structure: context cue (1.2, whole trial, channel 0 = motion /
        channel 1 = color); sensory channels (1 +/- coherence)/2 from
        ``stim_on`` with per-step Gaussian noise (std ``sigma_in * sqrt(2/alpha)``,
        rectified at zero); 0/1 choice targets supervised in steps
        ``dec_on``..``n_t``. Trials whose context-relevant coherence is zero
        have no unique correct choice: their mask stays zero (excluded from
        the loss) and ``correct_choice`` is 0.

        Returns:
            inputs: (N, n_t, 6) tensor.
            targets: (N, n_t, 2) tensor.
            mask: (N, n_t, 2) float tensor — 1 in the decision window of
                well-defined trials, 0 otherwise.
            conditions: list of dicts with keys ``context``, ``motion_coh``,
                ``color_coh``, ``correct_choice`` (+ unified epochs/n_steps/is_catch).
        """
        cohs_motion, cohs_color = self._grids()
        rng = np.random.default_rng(self.seed)
        noise_std = self.sigma_in * np.sqrt(2 / self.alpha)
        n_t, stim_on, dec_on = self.n_t, self.stim_on, self.dec_on

        conds = [(ctx, m, c) for ctx in (1, -1) for m in cohs_motion for c in cohs_color]
        n_trials = len(conds) * self.n_reps
        inputs = np.zeros((n_trials, n_t, 6), 'float32')
        targets = np.zeros((n_trials, n_t, 2), 'float32')
        mask = np.zeros((n_trials, n_t, 2), 'float32')
        epochs = {"context": (0, n_t), "stimulus": (stim_on, n_t), "decision": (dec_on, n_t)}

        conditions = []
        k = 0
        for ctx, motion_coh, color_coh in conds:
            relevant = motion_coh if ctx == 1 else color_coh
            for _ in range(self.n_reps):
                inputs[k, :, 0] = 1.2 if ctx == 1 else 0.0
                inputs[k, :, 1] = 0.0 if ctx == 1 else 1.2
                inputs[k, stim_on:, 2] = (1 + motion_coh) / 2
                inputs[k, stim_on:, 3] = (1 - motion_coh) / 2
                inputs[k, stim_on:, 4] = (1 + color_coh) / 2
                inputs[k, stim_on:, 5] = (1 - color_coh) / 2
                if self.sigma_in > 0:
                    inputs[k, stim_on:, 2:] += noise_std * rng.standard_normal(
                        (n_t - stim_on, 4))
                    np.maximum(inputs[k, stim_on:, 2:], 0.0, out=inputs[k, stim_on:, 2:])
                if relevant != 0:                        # supervised decision window
                    targets[k, dec_on:, 0 if relevant > 0 else 1] = 1.0
                    mask[k, dec_on:, :] = 1.0
                conditions.append(self._cond(
                    epochs, n_t, False,
                    context='motion' if ctx == 1 else 'color',
                    motion_coh=motion_coh,
                    color_coh=color_coh,
                    correct_choice=int(np.sign(relevant)),
                ))
                k += 1

        return (torch.from_numpy(inputs), torch.from_numpy(targets),
                torch.from_numpy(mask), conditions)


def generate_trials(**kwargs):
    """Backward-compatible shim: ManteLCTask(**kwargs).generate_trials()."""
    return ManteLCTask.from_kwargs(**kwargs).generate_trials()
