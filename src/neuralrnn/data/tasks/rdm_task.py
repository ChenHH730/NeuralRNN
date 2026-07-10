"""Random Dot Motion (RDM) perceptual decision-making task.

A single noisy stimulus with variable coherence is presented. The network
must report the sign of the coherence (left vs right). This is a classic
perceptual decision-making / evidence-accumulation task, also called a
single-coherence two-alternative forced-choice task.

Task family: perceptual decision making / evidence accumulation.
Inputs:  1 channel (coherence signal + noise).
Targets: 1 channel (choice direction: +1 or -1).

Timing (ms): fixation=100, stimulus=800, delay=100, decision=20.

References:
    Dubreuil et al. (2022), Nature Neuroscience.
    Valente et al. (2022), NeurIPS.
"""
import numpy as np
import torch

# Task constants
DELTA_T = 20.0
FIXATION_DURATION = 100
STIMULUS_DURATION = 800
DELAY_DURATION = 100
DECISION_DURATION = 20
SCALE = 0.1
STD_DEFAULT = 0.1


def _setup():
    """Compute discrete timing from continuous parameters."""
    global fixation_discrete, stimulus_discrete, delay_discrete, decision_discrete
    global stim_end, response_begin, total_duration
    fixation_discrete = int(FIXATION_DURATION / DELTA_T)
    stimulus_discrete = int(STIMULUS_DURATION / DELTA_T)
    stim_end = fixation_discrete + stimulus_discrete
    delay_discrete = int(DELAY_DURATION / DELTA_T)
    response_begin = stim_end + delay_discrete
    decision_discrete = int(DECISION_DURATION / DELTA_T)
    total_duration = (fixation_discrete + stimulus_discrete +
                      delay_discrete + decision_discrete)


_setup()


def generate_trials(
    num_trials: int = 1000,
    coherences: list | None = None,
    fraction_catch_trials: float = 0.0,
    std: float = STD_DEFAULT,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list]:
    """Generate RDM task trials.

    Args:
        num_trials: Number of trials to generate.
        coherences: List of coherence values (default: [-4, -2, -1, 1, 2, 4]).
        fraction_catch_trials: Fraction of catch trials (no decision target).
        std: Standard deviation of input noise.
        seed: Random seed for reproducibility.

    Returns:
        inputs:  (N, total_duration, 1) tensor — stimulus input
        targets: (N, total_duration, 1) tensor — decision target
        mask:    (N, total_duration, 1) tensor — loss mask
        conditions: list of dicts with trial metadata
    """
    if seed is not None:
        np.random.seed(seed)

    if coherences is None:
        coherences = [-4, -2, -1, 1, 2, 4]

    inputs = std * torch.randn((num_trials, total_duration, 1), dtype=torch.float32)
    targets = torch.zeros((num_trials, total_duration, 1), dtype=torch.float32)
    mask = torch.zeros((num_trials, total_duration, 1), dtype=torch.float32)
    conditions = []

    for i in range(num_trials):
        if np.random.rand() > fraction_catch_trials:
            coh = coherences[np.random.randint(0, len(coherences))]
            inputs[i, fixation_discrete:stim_end, 0] += coh * SCALE
            targets[i, response_begin:, 0] = 1.0 if coh > 0 else -1.0
        mask[i, response_begin:, 0] = 1.0
        conditions.append({
            "coherence": coh if np.random.rand() > fraction_catch_trials else 0.0,
            "is_catch": bool(np.random.rand() <= fraction_catch_trials),
        })

    return inputs, targets, mask, conditions
