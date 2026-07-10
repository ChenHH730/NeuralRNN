"""Raposo multisensory decision-making task.

Two stimuli (visual and auditory) are presented with variable coherences.
A context cue indicates which modality to attend to (visual, auditory, or both).
This is the multisensory / context-dependent decision task used in the
low-rank RNN literature.

Task family: multisensory / context-dependent perceptual decision making.
Inputs:  4 channels [visual_stim, auditory_stim, visual_ctx, auditory_ctx].
Targets: 1 channel (choice sign).

Timing (ms): fixation=100, ctx_pre=350, stimulus=800, delay=100, decision=20.

References:
    Raposo et al. (2014), Nature Neuroscience.
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
    fraction_catch_trials: float = 0.0,
    context: int | None = None,
    std: float = STD_DEFAULT,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list]:
    """Generate Raposo multisensory decision trials.

    Args:
        num_trials: Number of trials.
        coherences: List of coherence values (default: [-4, -2, -1, 1, 2, 4]).
        fraction_catch_trials: Fraction of catch trials (no decision target).
        context: Fixed context (1=visual, -1=auditory, 0=both, None=random).
        std: Input noise standard deviation.
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
    coherences_pos = [c for c in coherences if c >= 0]
    coherences_neg = [c for c in coherences if c < 0]

    inputs_sensory = std * torch.randn((num_trials, total_duration, 2), dtype=torch.float32)
    inputs_context = torch.zeros((num_trials, total_duration, 2))
    inputs = torch.cat([inputs_sensory, inputs_context], dim=2)
    targets = torch.zeros((num_trials, total_duration, 1), dtype=torch.float32)
    mask = torch.zeros((num_trials, total_duration, 1), dtype=torch.float32)
    conditions = []

    for i in range(num_trials):
        if np.random.rand() > fraction_catch_trials:
            choice = np.random.choice([-1.0, 1.0])
            if len(coherences_pos) == 0:
                choice = -1.0
            elif len(coherences_neg) == 0:
                choice = 1.0

            if context is None:
                ctx = np.random.randint(-1, 2)  # -1, 0, or 1
            else:
                ctx = context

            # Visual channel (channel 0)
            if ctx in [1, 0]:
                if choice > 0:
                    coh = coherences_pos[np.random.randint(0, len(coherences_pos))]
                else:
                    coh = coherences_neg[np.random.randint(0, len(coherences_neg))]
                inputs[i, stim_begin:stim_end, 0] += coh * SCALE
                inputs[i, fixation_discrete:stim_end, 2] = 1.0 * SCALE_CTX

            # Auditory channel (channel 1)
            if ctx in [-1, 0]:
                if choice > 0:
                    coh = coherences_pos[np.random.randint(0, len(coherences_pos))]
                else:
                    coh = coherences_neg[np.random.randint(0, len(coherences_neg))]
                inputs[i, stim_begin:stim_end, 1] += coh * SCALE
                inputs[i, fixation_discrete:stim_end, 3] = 1.0 * SCALE_CTX

            targets[i, response_begin:, 0] = choice
            conditions.append({"context": ctx, "choice": choice})
        else:
            conditions.append({"context": 0, "choice": 0.0, "is_catch": True})

        mask[i, response_begin:, 0] = 1.0

    return inputs, targets, mask, conditions
