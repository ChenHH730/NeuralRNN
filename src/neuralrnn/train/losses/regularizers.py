"""Reusable regularizer terms for RNN training.

All functions are stateless and return a scalar ``torch.Tensor``.
They operate on raw tensors / models and can be composed inside custom
objectives without importing specific model classes.
"""
from __future__ import annotations

import re
import warnings

import torch
import torch.nn as nn


def activity_l2(states: torch.Tensor, mask: torch.Tensor | None = None, reduction: str = "per_trial") -> torch.Tensor:
    """Mean squared activity penalty ``E[h^2]``.

    Args:
        states: (B, T, M) hidden-state tensor.
        mask:   optional (B, T) or (B, T, 1) float/bool mask.
        reduction: "per_trial" (default) computes the masked mean per trial and
            returns the mean across trials; "global" ignores the mask and
            returns the global mean over all elements.  "global" matches the
            convention used in notebooks that regularize mean firing rate
            independently of the loss mask.

    Returns:
        scalar Tensor.
    """
    if reduction not in ("per_trial", "global"):
        raise ValueError(f"activity_l2 reduction must be 'per_trial' or 'global', got {reduction!r}")

    if mask is None or reduction == "global":
        return (states ** 2).mean()

    m = mask.float()
    if m.dim() == states.dim() - 1:
        m = m.unsqueeze(-1)                       # (B, T) -> (B, T, 1)
    elif m.dim() == states.dim():
        m = m[..., :1]                            # (B, T, O) -> (B, T, 1)
    m = m.expand_as(states)                       # broadcast feature dimension
    loss_per_trial = ((states ** 2) * m).sum(dim=list(range(1, states.ndim))) / \
                     m.sum(dim=list(range(1, states.ndim))).clamp_min(1.0)
    return loss_per_trial.mean()


def weight_l2(model: nn.Module, patterns: list[str] | None = None, reduction: str = "mean") -> torch.Tensor:
    """Squared L2 penalty over matched parameters.

    Args:
        model: a torch module.
        patterns: optional list of regex patterns on parameter names.
            If None, all trainable parameters are included.
        reduction: "mean" (default) returns the mean squared value over all
            matched parameters; "sum" returns the raw sum of squares.  "sum"
            matches the convention used in some reference notebooks where the
            regularizer coefficient is applied directly to ``sum(p**2)``.

    Returns:
        scalar Tensor (zero if no parameters match).
    """
    if reduction not in ("mean", "sum"):
        raise ValueError(f"weight_l2 reduction must be 'mean' or 'sum', got {reduction!r}")

    total = torch.tensor(0.0, dtype=torch.float32)
    count = 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if patterns is not None and not any(re.search(pat, name) for pat in patterns):
            continue
        total = total + (p ** 2).sum()
        count += p.numel()

    if count == 0:
        return torch.tensor(0.0, device=next(model.parameters()).device)
    if reduction == "sum":
        return total
    return total / count


def weight_l1(model: nn.Module, patterns: list[str] | None = None) -> torch.Tensor:
    """Mean L1 penalty over matched parameters.

    Args:
        model: a torch module.
        patterns: optional list of regex patterns on parameter names.

    Returns:
        scalar Tensor (zero if no parameters match).
    """
    total = torch.tensor(0.0, dtype=torch.float32)
    count = 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if patterns is not None and not any(re.search(pat, name) for pat in patterns):
            continue
        total = total + p.abs().sum()
        count += p.numel()

    if count == 0:
        return torch.tensor(0.0, device=next(model.parameters()).device)
    return total / count


def orthogonality_penalty(
    input_weight: torch.Tensor,
    output_weight: torch.Tensor,
    normalize_columns: bool = True,
) -> torch.Tensor:
    """Input/output orthogonality penalty used in latent-circuit / CDDM training.

    Computes ``||B^T B - diag(B^T B)||_2`` where
    ``B = normalize([W_in | W_out^T])``.

    Args:
        input_weight: (input_dim, M) or (M, input_dim) tensor.
        output_weight: (output_dim, M) or (M, output_dim) tensor.
        normalize_columns: if True, L2-normalize each column of B before the penalty.

    Returns:
        scalar Tensor.
    """
    # Normalize shapes: W_in -> (M, input_dim), W_out -> (M, output_dim)
    if input_weight.shape[0] == output_weight.shape[-1]:
        # input_weight is likely (M, input_dim)
        w_in = input_weight
    else:
        w_in = input_weight.t()

    if output_weight.shape[-1] == w_in.shape[0]:
        # output_weight is (output_dim, M); transpose -> (M, output_dim)
        w_out = output_weight.t()
    else:
        w_out = output_weight

    b = torch.cat([w_in, w_out], dim=1)
    if normalize_columns:
        norm = b.norm(dim=0, keepdim=True).clamp_min(1e-8)
        b = b / norm

    gram = b.t() @ b
    diag = torch.diag(torch.diag(gram))
    return torch.norm(gram - diag, p=2)


def model_orthogonality_penalty(
    model: nn.Module,
    input_name: str = "input2h",
    output_name: str = "readout_layer",
    normalize_columns: bool = True,
) -> torch.Tensor:
    """Convenience wrapper that extracts weights from a model by attribute names.

    If the attributes are missing, returns 0 and emits a warning once.
    """
    try:
        input_weight = getattr(model, input_name).weight
        output_weight = getattr(model, output_name).weight
    except AttributeError:
        warnings.warn(
            f"model_orthogonality_penalty could not find '{input_name}' or '{output_name}' "
            "on the model; returning 0. Provide explicit weight matrices to "
            "orthogonality_penalty() if you want this regularizer.",
            stacklevel=2,
        )
        # Try to infer device from model parameters
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        return torch.tensor(0.0, device=device)

    return orthogonality_penalty(input_weight, output_weight, normalize_columns)
