"""Delayed Match-to-Sample (DMS) task.

Two stimuli (A or B) are presented sequentially with a variable delay.
The network must report whether the two stimuli match (A-A, B-B) or
differ (A-B, B-A).

Timing (ms): fixation=100, stim1=500, delay=500-3000, stim2=500, decision=1000
Input:  2 channels (one-hot encoding of A vs B)
Output: 1 channel (match/different)

Reference:
    Dubreuil et al. (2022), Nature Neuroscience.
    Valente et al. (2022), NeurIPS.
"""
import numpy as np
import torch

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


def _setup():
    """Compute discrete timing."""
    global fixation_discrete, stim1_min_discrete, stim1_max_discrete
    global stim2_min_discrete, stim2_max_discrete, decision_discrete
    global delay_min_discrete, delay_max_discrete, total_duration

    fixation_discrete = int(FIXATION_DURATION / DELTA_T)
    stim1_min_discrete = int(STIM1_DURATION_MIN / DELTA_T)
    stim1_max_discrete = int(STIM1_DURATION_MAX / DELTA_T)
    stim2_min_discrete = int(STIM2_DURATION_MIN / DELTA_T)
    stim2_max_discrete = int(STIM2_DURATION_MAX / DELTA_T)
    decision_discrete = int(DECISION_DURATION / DELTA_T)
    delay_min_discrete = int(DELAY_MIN / DELTA_T)
    delay_max_discrete = int(DELAY_MAX / DELTA_T)
    total_duration = (fixation_discrete + stim1_max_discrete +
                      delay_max_discrete + stim2_max_discrete + decision_discrete)


_setup()


def generate_trials(
    num_trials: int = 1000,
    trial_type: str | None = None,
    fraction_catch_trials: float = 0.0,
    std: float = STD_DEFAULT,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list]:
    """Generate DMS task trials.

    Args:
        num_trials: Number of trials.
        trial_type: Fixed trial type ('A-A', 'A-B', 'B-A', 'B-B') or None for mixed.
        fraction_catch_trials: Fraction of catch trials.
        std: Input noise standard deviation.
        seed: Random seed.

    Returns:
        inputs:  (N, total_duration, 2) tensor
        targets: (N, total_duration, 1) tensor
        mask:    (N, total_duration, 1) tensor
        conditions: list of dicts
    """
    if seed is not None:
        np.random.seed(seed)

    trial_types = ['A-A', 'A-B', 'B-A', 'B-B']

    inputs = std * torch.randn((num_trials, total_duration, 2))
    targets = torch.zeros((num_trials, total_duration, 1))
    mask = torch.zeros((num_trials, total_duration, 1))
    conditions = []

    for i in range(num_trials):
        if np.random.rand() > fraction_catch_trials:
            if trial_type is None:
                cur_type = trial_types[np.random.randint(0, 4)]
            else:
                cur_type = trial_type

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
            delay_dur = np.random.randint(delay_min_discrete, delay_max_discrete + 1)
            stim1_dur = np.random.randint(stim1_min_discrete, stim1_max_discrete + 1)
            stim2_dur = np.random.randint(stim2_min_discrete, stim2_max_discrete + 1)

            stim1_begin = fixation_discrete
            stim1_end = stim1_begin + stim1_dur
            stim2_begin = stim1_end + delay_dur
            stim2_end = stim2_begin + stim2_dur
            decision_begin = stim2_end
            decision_end = decision_begin + decision_discrete

            # Input: one-hot for stimulus A (channel 0) and B (channel 1)
            inputs[i, stim1_begin:stim1_end, 0] += input1
            inputs[i, stim1_begin:stim1_end, 1] += 1 - input1
            inputs[i, stim2_begin:stim2_end, 0] += input2
            inputs[i, stim2_begin:stim2_end, 1] += 1 - input2

            targets[i, decision_begin:decision_end] = choice
            mask[i, decision_begin:decision_end] = 1.0

            conditions.append({
                "trial_type": cur_type,
                "input1": input1,
                "input2": input2,
                "choice": choice,
                "stim1_dur": stim1_dur,
                "stim2_dur": stim2_dur,
                "delay_dur": delay_dur,
            })
        else:
            conditions.append({"trial_type": "catch", "is_catch": True})

    return inputs, targets, mask, conditions
