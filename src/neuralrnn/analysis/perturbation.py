"""Perturbation analysis for latent circuit models.

Translates perturbations of latent circuit connections to rank-one perturbations
of the high-dimensional RNN connectivity:
    delta_{ij} → q_i * q_j^T

where q_i is the i-th row of Q (embedding matrix).

Reference: Langdon & Engel (2025), Nature Neuroscience.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import numpy as np


@dataclass
class PerturbationSpec:
    """Specification of a latent circuit perturbation.

    Attributes:
        i: Row index in w_rec (target node).
        j: Column index in w_rec (source node).
        delta: Perturbation amount (negative = weaken connection).
        label: Human-readable description.
    """
    i: int
    j: int
    delta: float
    label: str = ""


@dataclass
class PerturbationResult:
    """Result of a single perturbation experiment.

    Attributes:
        spec: The perturbation specification.
        rnn_perturbation: The rank-one perturbation applied to W_rec (N x N).
        behavior_before: Psychometric data before perturbation.
        behavior_after: Psychometric data after perturbation.
    """
    spec: PerturbationSpec
    rnn_perturbation: np.ndarray
    behavior_before: dict
    behavior_after: dict


def latent_to_rnn_perturbation(spec: PerturbationSpec, Q: np.ndarray) -> np.ndarray:
    """Convert a latent circuit perturbation to an RNN connectivity perturbation.

    delta_{ij} in w_rec maps to: delta * outer(Q[j], Q[i]) in W_rec

    This is because w_rec = Q @ W_rec @ Q^T, so perturbing w_rec[i,j] by delta
    is equivalent to perturbing W_rec by delta * q_i^T @ q_j (rank-one).

    Args:
        spec: Perturbation specification (i, j, delta).
        Q: Embedding matrix (n x N).

    Returns:
        RNN perturbation matrix (N x N).
    """
    # Q has shape (n, N), q_i = Q[i] has shape (N,)
    q_i = Q[spec.i]  # (N,)
    q_j = Q[spec.j]  # (N,)
    return spec.delta * np.outer(q_j, q_i)


def apply_perturbation(model, perturbation: np.ndarray) -> None:
    """Apply a perturbation to the RNN's recurrent weights.

    Args:
        model: RNN model with h2h layer.
        perturbation: (N, N) perturbation to add to W_rec.
    """
    with torch.no_grad():
        perturbation_t = torch.tensor(perturbation, dtype=model.h2h.weight.dtype)
        model.h2h.weight.add_(perturbation_t)


def compute_choice(model, inputs: torch.Tensor) -> np.ndarray:
    """Compute network choice for each trial.

    Choice = ReLU(output[:, -1, 0] - output[:, -1, 1])
    Positive = right choice, zero = left choice.

    Args:
        model: Trained RNN or latent circuit model.
        inputs: (N, T, input_dim) task inputs.

    Returns:
        choices: (N,) array of choice values.
    """
    model.eval()
    with torch.no_grad():
        out = model(inputs)
        outputs = out.outputs  # (N, T, output_dim)
        # Choice at last time step
        choices = torch.relu(outputs[:, -1, 0] - outputs[:, -1, 1])
    return choices.cpu().numpy()


def analyze_perturbation(
    rnn_model,
    latent_model,
    task_dataset,
    perturbation: PerturbationSpec,
) -> PerturbationResult:
    """Analyze the behavioral effect of a latent circuit perturbation.

    1. Compute baseline behavior (psychometric curve)
    2. Translate latent perturbation to RNN perturbation
    3. Apply perturbation to RNN
    4. Compute perturbed behavior
    5. Restore original RNN weights

    Args:
        rnn_model: Trained high-dimensional RNN model.
        latent_model: Trained latent circuit model.
        task_dataset: CognitiveTaskDataset with task data and conditions.
        perturbation: Perturbation specification.

    Returns:
        PerturbationResult with before/after behavior.
    """
    from .psychometric import compute_psychometric

    # Get all trial data
    data = task_dataset.get_all_trials()
    inputs = data["inputs"]
    conditions = task_dataset.conditions

    # Get Q
    Q = latent_model.embedding_matrix.detach().cpu().numpy()

    # Compute baseline behavior
    behavior_before = compute_psychometric(rnn_model, inputs, conditions)

    # Compute RNN perturbation
    rnn_pert = latent_to_rnn_perturbation(perturbation, Q)

    # Save original weights
    original_weights = rnn_model.h2h.weight.data.clone()

    # Apply perturbation
    apply_perturbation(rnn_model, rnn_pert)

    # Compute perturbed behavior
    behavior_after = compute_psychometric(rnn_model, inputs, conditions)

    # Restore original weights
    rnn_model.h2h.weight.data.copy_(original_weights)

    return PerturbationResult(
        spec=perturbation,
        rnn_perturbation=rnn_pert,
        behavior_before=behavior_before,
        behavior_after=behavior_after,
    )
