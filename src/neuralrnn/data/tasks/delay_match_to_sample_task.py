"""Delayed match-to-sample task.

4 inputs (test right/left, sample right/left), 2 outputs.
Test stimulus presented first, then sample after a gap. Match = same sign.

Ported from Langdon & Engel (2025) reference implementation.
"""
import numpy as np
import torch


def generate_input_target_stream(
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
    """Create trials for the delayed match-to-sample task.

    Note: Returns targets pre-sliced by a training_mask.

    Returns:
        inputs: (N, n_t, 4) tensor.
        targets: (N, len(training_mask), 2) tensor — pre-sliced targets.
        mask: numpy index array — training_mask indices.
        conditions: list of dicts.
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
            for i in range(n_trials):
                match = (test_coh > 0 and sample_coh > 0) or (test_coh < 0 and sample_coh < 0)
                conditions.append({
                    "test_coh": test_coh,
                    "sample_coh": sample_coh,
                    "match": match,
                    "correct_choice": 1 if match else -1,
                })
                inp, tgt = generate_input_target_stream(
                    test_coh, sample_coh, baseline, alpha, sigma_in,
                    n_t, test_on, test_off, sample_on, sample_off, dec_on, dec_off,
                )
                inputs.append(inp)
                targets.append(tgt)

    inputs = np.stack(inputs, 0)
    targets = np.stack(targets, 0)

    perm = np.random.permutation(len(inputs))
    inputs = torch.tensor(inputs[perm, :, :]).float()
    conditions = [conditions[index] for index in perm]

    # Training mask: pre-sample + decision period
    training_mask = np.append(range(sample_on - 1), range(dec_on - 1, dec_off - 1))
    targets = torch.tensor(targets[:, training_mask, :]).float()

    return inputs, targets, training_mask, conditions
