"""Reusable loss functions and metrics for cognitive-task RNN training.

All functions accept either raw torch.Tensor or DynamicsModelOutput
(which delegates arithmetic ops to .outputs via its __sub__/__mul__/etc.
methods). This means you can pass model.forward() return values directly
without extracting .outputs first.
"""
from __future__ import annotations

import torch


def loss_mse(output, target, mask):
    """Masked mean squared error loss.

    Compatible with both raw tensors and DynamicsModelOutput.
    Computes (output - target)^2 elementwise, applies the mask,
    and normalizes per-trial.

    Args:
        output: (B, T, O) tensor or DynamicsModelOutput
        target: (B, T, O) tensor
        mask:   (B, T, 1) or (B, T, O) float/bool tensor —
                True where loss should be counted

    Returns:
        scalar torch.Tensor — mean MSE loss across trials
    """
    loss_tensor = (mask * (target - output)).pow(2).mean(dim=-1)
    loss_by_trial = loss_tensor.sum(dim=-1) / mask[:, :, 0].sum(dim=-1).clamp_min(1.0)
    return loss_by_trial.mean()


def accuracy_general(output, targets, mask):
    """Sign-based accuracy for binary decision tasks.

    Only considers trials where targets are non-zero (valid trials).
    The decision is the sign of the masked mean output / target over
    the valid (masked) timesteps.

    Args:
        output:  (B, T, O) tensor or DynamicsModelOutput
        targets: (B, T, O) tensor
        mask:    (B, T, 1) or (B, T, O) float/bool tensor

    Returns:
        scalar torch.Tensor — fraction of correct decisions,
        or NaN if no valid trials
    """
    good_trials = (targets != 0).any(dim=1).squeeze()
    if good_trials.sum() == 0:
        return torch.tensor(float('nan'))
    target_decisions = torch.sign(
        (targets[good_trials, :, :] * mask[good_trials, :, :]).mean(dim=1).squeeze()
    )
    decisions = torch.sign(
        (output[good_trials, :, :] * mask[good_trials, :, :]).mean(dim=1).squeeze()
    )
    return (target_decisions == decisions).type(torch.float32).mean()
