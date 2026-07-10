"""Continuous delayed match-to-sample (DMS) task.

Also called DMS-continuous to distinguish it from the discrete-symbol DMS task
(:mod:`dms_task`). Two sequential noisy stimuli are presented; the network must
report whether they have the same sign. This version uses continuous coherences
and two output channels.

Task family: delayed match-to-sample / working memory.
Inputs:  4 channels [test_r, test_l, sample_r, sample_l].
Targets: 2 channels [match, non-match] active during decision.

References:
    Langdon & Engel (2025) reference implementation.
"""
import numpy as np
import torch


def _generate_single_trial(
    test_coh, sample_coh, baseline, alpha, sigma_in,
    n_t, test_on, test_off, sample_on, sample_off, dec_on, dec_off,
):
    """Generate input and target sequence for a single trial."""
    test_r = (1 + test_coh) / 2
    test_l = 1 - test_r
    sample_r = (1 + sample_coh) / 2
    sample_l = 1 - sample_r

    input_stream = np.zeros([n_t, 4])
    input_stream[test_on:test_off, 0] = test_r
    input_stream[test_on:test_off, 1] = test_l
    input_stream[sample_on - 1:sample_off, 2] = sample_r
    input_stream[sample_on - 1:sample_off, 3] = sample_l

    noise = np.sqrt(2 / alpha * sigma_in * sigma_in) * np.random.multivariate_normal(
        [0, 0, 0, 0], np.eye(4), n_t
    )
    baseline_arr = baseline * np.ones([n_t, 4])
    input_stream = np.maximum(baseline_arr + input_stream + noise, 0)

    # Match: same sign of coherence
    match = (test_coh > 0 and sample_coh > 0) or (test_coh < 0 and sample_coh < 0)
    target_stream = 0.2 * np.ones([n_t, 2])
    if match:
        target_stream[dec_on - 1:dec_off, 0] = 1.2
        target_stream[dec_on - 1:dec_off, 1] = 0.2
    else:
        target_stream[dec_on - 1:dec_off, 0] = 0.2
        target_stream[dec_on - 1:dec_off, 1] = 1.2

    return input_stream, target_stream


def generate_trials(n_trials=25, alpha=0.2, sigma_in=0.01, baseline=0.2, n_coh=6, n_t=75):
    """Create trials for the continuous delayed match-to-sample task.

    Returns full-length targets and a boolean mask, matching the standard
    NeuralRNN cognitive-task format.

    Args:
        n_trials: Number of trials per condition combination.
        alpha: Time constant parameter (dt/tau).
        sigma_in: Input noise standard deviation.
        baseline: Baseline input level.
        n_coh: Number of coherence levels.
        n_t: Number of time steps per trial.

    Returns:
        inputs: (N, n_t, 4) tensor.
        targets: (N, n_t, 2) tensor — full-length targets.
        mask: (N, n_t, 2) float tensor — 1 during pre-sample and decision, 0 during delay.
        conditions: list of dicts with keys ``test_coh``, ``sample_coh``,
            ``match``, ``correct_choice``.
    """
    cohs = np.linspace(-0.2, 0.2, n_coh)

    test_on = 0
    test_off = int(round(n_t * 0.47))
    sample_on = int(round(n_t * 0.53))
    sample_off = n_t
    dec_on = int(round(n_t * 0.73))
    dec_off = n_t

    inputs = []
    targets = []
    conditions = []
    for test_coh in cohs:
        for sample_coh in cohs:
            for _ in range(n_trials):
                match = (test_coh > 0 and sample_coh > 0) or (test_coh < 0 and sample_coh < 0)
                conditions.append({
                    "test_coh": float(test_coh),
                    "sample_coh": float(sample_coh),
                    "match": match,
                    "correct_choice": 1 if match else -1,
                })
                inp, tgt = _generate_single_trial(
                    test_coh, sample_coh, baseline, alpha, sigma_in,
                    n_t, test_on, test_off, sample_on, sample_off, dec_on, dec_off,
                )
                inputs.append(inp)
                targets.append(tgt)

    inputs = np.stack(inputs, 0)
    targets = np.stack(targets, 0)

    perm = np.random.permutation(len(inputs))
    inputs = torch.tensor(inputs[perm, :, :]).float()
    targets = torch.tensor(targets[perm, :, :]).float()
    conditions = [conditions[index] for index in perm]

    # Mask: pre-sample + decision period (delay between sample_off and dec_on is ignored)
    mask = torch.zeros_like(targets)
    mask[:, :test_off, :] = 1.0
    mask[:, dec_on - 1:dec_off, :] = 1.0

    return inputs, targets, mask, conditions
