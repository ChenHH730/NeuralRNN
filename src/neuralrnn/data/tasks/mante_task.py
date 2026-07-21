"""Mante et al. (2013) context-dependent decision-making task.

Also commonly referred to as the Siegel-Miller task (Siegel et al., 2015) when
used in the latent-circuit literature. Both names describe the same paradigm:
a context cue instructs the network to attend to either motion or color evidence
and report the sign of the attended coherence.

Task family: context-dependent perceptual decision making.
Inputs:  6 channels [motion_ctx, color_ctx, motion_r, motion_l, color_r, color_l].
Targets: 2 channels [choice_right, choice_left] active during decision.

Note: trials are generated on a condition grid (context x motion_coh x
color_coh); ``n_reps`` is the number of repetitions per grid cell, so the
total trial count is ``n_reps * 2 * n_coh ** 2``.

References:
    Mante et al. (2013), "Context-dependent computation by recurrent dynamics
        in prefrontal cortex", Nature.
    Siegel et al. (2015), "Neural signatures of categorization in temporal
        lobe neurons", Nature Neuroscience.
    Langdon & Engel (2025) reference implementation.
"""
import numpy as np
import torch

from .task_base import Task


CHANNEL_ORDER = ["motion_ctx", "color_ctx", "motion_r", "motion_l", "color_r", "color_l"]
OUTPUT_ORDER = ["choice_r", "choice_l"]


def _generate_single_trial(
    context, motion_coh, color_coh, baseline, alpha, sigma_in,
    n_t, cue_on, cue_off, stim_on, stim_off, dec_on, dec_off,
):
    """Generate input and target sequence for a single trial."""
    motion_r = (1 + motion_coh) / 2
    motion_l = 1 - motion_r
    color_r = (1 + color_coh) / 2
    color_l = 1 - color_r

    cue_input = np.zeros([n_t, 6])
    if context == "motion":
        cue_input[cue_on:cue_off, 0] = 1.2
    else:
        cue_input[cue_on:cue_off, 1] = 1.2

    motion_input = np.zeros([n_t, 6])
    motion_input[stim_on - 1:stim_off, 2] = motion_r
    motion_input[stim_on - 1:stim_off, 3] = motion_l

    color_input = np.zeros([n_t, 6])
    color_input[stim_on - 1:stim_off, 4] = color_r
    color_input[stim_on - 1:stim_off, 5] = color_l

    noise = np.sqrt(2 / alpha * sigma_in * sigma_in) * np.random.multivariate_normal(
        [0, 0, 0, 0, 0, 0], np.eye(6), n_t
    )
    baseline_arr = baseline * np.ones([n_t, 6])
    input_stream = np.maximum(baseline_arr + cue_input + motion_input + color_input + noise, 0)

    target_stream = 0.2 * np.ones([n_t, 2])
    if (context == "motion" and motion_coh > 0) or (context == "color" and color_coh > 0):
        target_stream[dec_on - 1:dec_off, 0] = 1.2
        target_stream[dec_on - 1:dec_off, 1] = 0.2
    else:
        target_stream[dec_on - 1:dec_off, 0] = 0.2
        target_stream[dec_on - 1:dec_off, 1] = 1.2

    return input_stream, target_stream


class ManteTask(Task):
    """Mante / Siegel-Miller context-dependent DM task (unified interface)."""

    name = "mante"
    aliases = ("siegel_miller",)
    input_dim = 6
    output_dim = 2
    default_dt = None  # step-indexed (n_t); alpha = dt/tau is a task parameter
    deprecated_kwargs = {"n_trials": "n_reps"}  # pre-refactor n_trials = per-condition reps

    def __init__(self, n_reps=25, *, alpha=0.2, sigma_in=0.01, baseline=0.2,
                 n_coh=6, cohs=None, n_t=75, seed=None):
        self.n_reps = n_reps
        self.alpha = alpha
        self.sigma_in = sigma_in
        self.baseline = baseline
        self.n_coh = n_coh
        self.cohs = cohs
        self.n_t = n_t
        self.seed = seed

    @property
    def n_trials(self):
        """Total trial count (n_reps x 2 contexts x n_coh^2)."""
        n_coh = len(self.cohs) if self.cohs is not None else self.n_coh
        return int(self.n_reps * 2 * n_coh * n_coh)

    def generate_trials(self):
        """Create trials for the Mante / Siegel-Miller task.

        Trial structure: context cue (steps ~7-24) -> delay -> stimulus
        (steps ~30-74) -> decision (steps ~55-74). The context cue precedes the
        stimulus, matching the standard Mante et al. (2013) design. The training
        mask is active only during the decision period.

        Returns:
            inputs: (N, n_t, 6) tensor.
            targets: (N, n_t, 2) tensor — full-length target sequence.
            mask: (N, n_t, 2) float tensor — 1 during decision period, 0 otherwise.
            conditions: list of dicts with keys ``context``, ``motion_coh``,
                ``color_coh``, ``correct_choice`` (+ unified epochs/n_steps/is_catch).
        """
        self._seed_np()
        alpha, sigma_in, baseline, n_t = self.alpha, self.sigma_in, self.baseline, self.n_t
        if self.cohs is None:
            cohs = np.linspace(-0.2, 0.2, self.n_coh)
        else:
            cohs = np.asarray(self.cohs)

        cue_on = int(round(n_t * 0.1))
        cue_off = int(round(n_t * 0.33))
        stim_on = int(round(n_t * 0.4))
        stim_off = int(round(n_t))
        dec_on = int(round(n_t * 0.75))
        dec_off = int(round(n_t))

        epochs = {
            "cue": (cue_on, cue_off),
            "stimulus": (stim_on - 1, stim_off),
            "decision": (dec_on - 1, dec_off),
        }

        inputs = []
        targets = []
        conditions = []
        for context in ["motion", "color"]:
            for motion_coh in cohs:
                for color_coh in cohs:
                    for _ in range(self.n_reps):
                        correct_choice = 1 if (
                            (context == "motion" and motion_coh > 0) or
                            (context == "color" and color_coh > 0)
                        ) else -1
                        conditions.append(self._cond(
                            epochs, n_t, False,
                            context=context,
                            motion_coh=float(motion_coh),
                            color_coh=float(color_coh),
                            correct_choice=correct_choice,
                        ))
                        inp, tgt = _generate_single_trial(
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

        # Decision-period mask (1 during decision, 0 otherwise)
        mask = torch.zeros_like(targets)
        mask[:, dec_on - 1:dec_off, :] = 1.0

        return inputs, targets, mask, conditions


def generate_trials(**kwargs):
    """Backward-compatible shim: ManteTask(**kwargs).generate_trials().

    Note: the pre-refactor ``n_trials`` parameter (per-condition repetitions)
    is now ``n_reps``; it is still accepted with a DeprecationWarning.
    """
    return ManteTask.from_kwargs(**kwargs).generate_trials()
