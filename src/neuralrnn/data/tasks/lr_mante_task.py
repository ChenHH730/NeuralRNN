"""Mante task for low-rank RNNs (context-dependent decision-making).

Two stimuli (color and motion) are presented simultaneously, with a context
cue indicating which to attend to. 4 input channels + optional noise.

Timing (ms): fixation=100, ctx_pre=350, stimulus=800, delay=100, decision=20
Input:  4 channels [color_stim, motion_stim, color_ctx, motion_ctx]
Output: 1 channel (decision sign)

This version is the one used in Dubreuil et al. (2022) and Valente et al. (2022).
It differs from the latent circuit Mante task which uses 6 inputs and 2 outputs.

Reference:
    Mante et al. (2013), Nature.
    Dubreuil et al. (2022), Nature Neuroscience.
"""
import numpy as np
import torch

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
    """Compute discrete timing."""
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


def generate_trials(
    num_trials: int = 1000,
    coherences: list | None = None,
    std: float = STD_DEFAULT,
    fraction_catch_trials: float = 0.0,
    coh_color: int | None = None,
    coh_motion: int | None = None,
    context: int | None = None,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list]:
    """Generate Mante context-dependent decision-making trials.

    Args:
        num_trials: Number of trials.
        coherences: List of coherence values (default: [-4, -2, -1, 1, 2, 4]).
        std: Input noise standard deviation.
        fraction_catch_trials: Fraction of catch trials.
        coh_color: Fixed color coherence (None for random).
        coh_motion: Fixed motion coherence (None for random).
        context: Fixed context (1=color, 2=motion, None=random).
        seed: Random seed.

    Returns:
        inputs:  (N, total_duration, 4) tensor
        targets: (N, total_duration, 1) tensor
        mask:    (N, total_duration, 1) tensor
        conditions: list of dicts
    """
    if seed is not None:
        np.random.seed(seed)

    if coherences is None:
        coherences = [-4, -2, -1, 1, 2, 4]

    inputs_sensory = std * torch.randn((num_trials, total_duration, 2), dtype=torch.float32)
    inputs_context = torch.zeros((num_trials, total_duration, 2))
    inputs = torch.cat([inputs_sensory, inputs_context], dim=2)
    targets = torch.zeros((num_trials, total_duration, 1), dtype=torch.float32)
    mask = torch.zeros((num_trials, total_duration, 1), dtype=torch.float32)
    conditions = []

    for i in range(num_trials):
        if np.random.rand() > fraction_catch_trials:
            if coh_color is None:
                c_color = coherences[np.random.randint(0, len(coherences))]
            else:
                c_color = coh_color
            if coh_motion is None:
                c_motion = coherences[np.random.randint(0, len(coherences))]
            else:
                c_motion = coh_motion
            if context is None:
                ctx = np.random.randint(1, 3)  # 1 or 2
            else:
                ctx = context

            inputs[i, stim_begin:stim_end, 0] += c_color * SCALE
            inputs[i, stim_begin:stim_end, 1] += c_motion * SCALE

            if ctx == 1:
                inputs[i, fixation_discrete:response_begin, 2] = 1.0 * SCALE_CTX
                targets[i, response_begin:, 0] = 1.0 if c_color > 0 else -1.0
            else:  # ctx == 2
                inputs[i, fixation_discrete:response_begin, 3] = 1.0 * SCALE_CTX
                targets[i, response_begin:, 0] = 1.0 if c_motion > 0 else -1.0

            conditions.append({
                "context": ctx,
                "coh_color": c_color,
                "coh_motion": c_motion,
                "target": targets[i, response_begin:, 0].mean().item(),
            })
        else:
            conditions.append({"is_catch": True})

        mask[i, response_begin:, 0] = 1.0

    return inputs, targets, mask, conditions
