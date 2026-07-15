"""Reusable metric functions for RNN evaluation.

Metrics differ from losses in that they are not backpropagated; they are only
logged or reported. All functions accept raw tensors or DynamicsModelOutput.
"""
from __future__ import annotations

import torch

from ...modeling_utils import DynamicsModelOutput


def _extract_outputs(output):
    if isinstance(output, DynamicsModelOutput):
        if output.outputs is None:
            raise ValueError("DynamicsModelOutput has no outputs tensor")
        return output.outputs
    return output


def accuracy_classification(logits, targets, mask=None):
    """Standard argmax classification accuracy, optionally masked along (B,T).

    Args:
        logits: (B, T, C) tensor or DynamicsModelOutput.
        targets: (B, T) long tensor.
        mask: optional (B, T) float/bool tensor.

    Returns:
        scalar Tensor — fraction correct, or NaN if no valid positions.
    """
    logits = _extract_outputs(logits)
    pred = logits.argmax(dim=-1)
    correct = (pred == targets).float()
    if mask is None:
        if correct.numel() == 0:
            return torch.tensor(float("nan"), device=correct.device)
        return correct.mean()

    m = mask.float()
    if m.numel() == 0 or m.sum() == 0:
        return torch.tensor(float("nan"), device=correct.device)
    return (correct * m).sum() / m.sum()


def accuracy_general(output, targets, mask):
    """Sign-based accuracy for binary decision tasks.

    Only considers trials where targets are non-zero (valid trials).
    The decision is the sign of the masked mean output / target over
    the valid (masked) timesteps.

    Args:
        output:  (B, T, O) tensor or DynamicsModelOutput.
        targets: (B, T, O) tensor.
        mask:    (B, T, 1) or (B, T, O) float/bool tensor.

    Returns:
        scalar torch.Tensor — fraction of correct decisions,
        or NaN if no valid trials.
    """
    output = _extract_outputs(output)
    good_trials = (targets != 0).any(dim=(1, 2), keepdim=False).squeeze()
    if good_trials.sum() == 0:
        return torch.tensor(float("nan"), device=output.device)

    target_decisions = torch.sign(
        (targets[good_trials, :, :] * mask[good_trials, :, :]).mean(dim=1).squeeze()
    )
    decisions = torch.sign(
        (output[good_trials, :, :] * mask[good_trials, :, :]).mean(dim=1).squeeze()
    )
    return (target_decisions == decisions).type(torch.float32).mean()
