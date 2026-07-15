"""Reusable masked loss functions for RNN training.

All functions accept either raw ``torch.Tensor`` or ``DynamicsModelOutput``
(which delegates arithmetic ops to ``.outputs``). This means you can pass
``model.forward()`` return values directly without extracting ``.outputs`` first.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ...modeling_utils import DynamicsModelOutput


def _extract_outputs(output):
    """Return the underlying outputs tensor if ``output`` is a DynamicsModelOutput."""
    if isinstance(output, DynamicsModelOutput):
        if output.outputs is None:
            raise ValueError("DynamicsModelOutput has no outputs tensor")
        return output.outputs
    return output


def masked_mse(output, target, mask=None, reduction="per_trial"):
    """Masked mean squared error loss.

    Args:
        output: (B, T, O) tensor or DynamicsModelOutput.
        target: (B, T, O) tensor.
        mask:   (B, T) or (B, T, 1) or (B, T, O) float/bool tensor.
                If None, all positions are counted.
        reduction: "per_trial" (default) computes the MSE independently for each
                trial and returns the mean across trials; "global" computes a
                single MSE by dividing total squared error by total mask weight.
                "global" matches the convention used in several reference
                notebooks (e.g. latent-circuit and flexible-multitask training).

    Returns:
        scalar torch.Tensor.
    """
    if reduction not in ("per_trial", "global"):
        raise ValueError(f"masked_mse reduction must be 'per_trial' or 'global', got {reduction!r}")

    y = _extract_outputs(output)
    err = (y - target) ** 2
    if mask is None:
        return err.mean()

    m = mask.float()
    # Broadcast mask to match err shape (B, T, O)
    while m.ndim < err.ndim:
        m = m.unsqueeze(-1)

    if reduction == "global":
        return (err * m).sum() / m.sum().clamp_min(1.0)

    loss_per_trial = (err * m).sum(dim=(1, 2)) / m.sum(dim=(1, 2)).clamp_min(1.0)
    return loss_per_trial.mean()


def masked_cross_entropy(logits, targets, mask=None):
    """Masked cross-entropy for time-series classification.

    Args:
        logits: (B, T, C) tensor or DynamicsModelOutput.
        targets: (B, T) long tensor of class indices.
        mask:   (B, T) or (B, T, 1) float/bool tensor. If None, all positions counted.

    Returns:
        scalar torch.Tensor.
    """
    logits = _extract_outputs(logits)
    B, T, C = logits.shape
    logits_flat = logits.reshape(B * T, C)
    targets_flat = targets.reshape(B * T).long()
    ce = F.cross_entropy(logits_flat, targets_flat, reduction="none")
    if mask is None:
        return ce.mean()

    m = mask.float().reshape(B * T)
    return (ce * m).sum() / m.sum().clamp_min(1.0)


def masked_nll(logits, targets, mask=None):
    """Alias for ``masked_cross_entropy``."""
    return masked_cross_entropy(logits, targets, mask)


# Backward-compatible alias used by older notebooks and external code.
loss_mse = masked_mse
