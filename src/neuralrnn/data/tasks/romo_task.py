"""Parametric Working Memory (Romo) task.

Two stimuli at different frequencies (f1, f2) are presented sequentially with
a variable delay. The network must report which frequency was higher.

Timing (ms): fixation=100, stim1=100, delay=500-1000, stim2=100, decision=100
Input:  1 channel (frequency value)
Output: 1 channel (comparison decision)

Reference:
    Romo et al. (1999), Nature.
    Dubreuil et al. (2022), Nature Neuroscience.
"""
import numpy as np
import torch

DELTA_T = 20.0
FIXATION_DURATION = 100
STIM1_DURATION = 100
DELAY_MIN = 500
DELAY_MAX = 1000
STIM2_DURATION = 100
DECISION_DURATION = 100

# Frequency pairs used in the task
FDIFFS = [-24, -16, -8, 8, 16, 24]
FPAIRS = [(base, base + fdiff) for fdiff in FDIFFS
          for base in range(max(10 - fdiff, 10), min(34, 34 - fdiff) + 1)]
FMAX = max([max(*p) for p in FPAIRS])
FMIN = min([min(*p) for p in FPAIRS])
FMIDDLE = (FMAX + FMIN) / 2.0
FSPAN = FMAX - FMIN


def _setup():
    """Compute discrete timing variables."""
    global fixation_discrete, stim1_discrete, stim1_end
    global stim2_discrete, decision_discrete
    global min_delay_discrete, max_delay_discrete, total_duration

    fixation_discrete = int(FIXATION_DURATION / DELTA_T)
    stim1_discrete = int(STIM1_DURATION / DELTA_T)
    stim1_end = fixation_discrete + stim1_discrete
    stim2_discrete = int(STIM2_DURATION / DELTA_T)
    decision_discrete = int(DECISION_DURATION / DELTA_T)
    min_delay_discrete = int(DELAY_MIN / DELTA_T)
    max_delay_discrete = int(DELAY_MAX / DELTA_T)
    total_duration = (fixation_discrete + stim1_discrete +
                      max_delay_discrete + stim2_discrete + decision_discrete)


_setup()


def generate_trials(
    num_trials: int = 1000,
    std: float = 0.01,
    fpairs: list | None = None,
    fraction_catch_trials: float = 0.0,
    delay_discrete: int | None = None,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list]:
    """Generate parametric working memory trials.

    Args:
        num_trials: Number of trials.
        std: Input noise standard deviation.
        fpairs: List of (f1, f2) frequency pairs. Defaults to all valid pairs.
        fraction_catch_trials: Fraction of catch trials.
        delay_discrete: Fixed delay duration in discrete steps (default: random).
        seed: Random seed.

    Returns:
        inputs:  (N, total_duration, 1) tensor
        targets: (N, total_duration, 1) tensor
        mask:    (N, total_duration, 1) tensor
        conditions: list of dicts
    """
    if seed is not None:
        np.random.seed(seed)

    if fpairs is None:
        fpairs = FPAIRS

    inputs = std * torch.randn((num_trials, total_duration, 1))
    targets = torch.zeros((num_trials, total_duration, 1), dtype=torch.float32)
    mask = torch.zeros((num_trials, total_duration, 1), dtype=torch.float32)
    conditions = []

    for i in range(num_trials):
        if np.random.rand() > fraction_catch_trials:
            f1, f2 = fpairs[np.random.randint(0, len(fpairs))]

            if delay_discrete is None:
                delay = np.random.randint(min_delay_discrete, max_delay_discrete + 1)
            else:
                delay = delay_discrete

            stim2_begin = stim1_end + delay
            stim2_end = stim2_begin + stim2_discrete
            decision_end = stim2_end + decision_discrete

            # Normalize frequencies to [-0.5, 0.5] range
            inputs[i, fixation_discrete:stim1_end] += (f1 - FMIDDLE) / FSPAN
            inputs[i, stim2_begin:stim2_end] += (f2 - FMIDDLE) / FSPAN
            targets[i, stim2_end:decision_end, 0] = (f1 - f2) / FSPAN
            mask[i, stim2_end:decision_end, 0] = 1.0

            conditions.append({"f1": f1, "f2": f2, "delay": delay})
        else:
            conditions.append({"f1": 0.0, "f2": 0.0, "delay": 0, "is_catch": True})

    return inputs, targets, mask, conditions
