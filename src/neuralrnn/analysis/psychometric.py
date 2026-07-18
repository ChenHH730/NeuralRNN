"""Psychometric function analysis for cognitive task models.

Computes and fits psychometric curves (probability of right choice vs coherence)
separately for different task contexts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import numpy as np
from scipy.optimize import curve_fit


def _sigmoid(x, alpha, beta):
    """Logistic function: 1 / (1 + exp(-(x - alpha) / beta))."""
    return 1.0 / (1.0 + np.exp(-(x - alpha) / beta))


@dataclass
class PsychometricCurve:
    """A single psychometric curve.

    Attributes:
        coherences: Coherence values.
        prob_right: Probability of right choice at each coherence.
        sigmoid_params: Fitted (alpha, beta) parameters, or None if not fitted.
        label: Label for this curve (e.g., "motion_context_motion_coh").
    """
    coherences: np.ndarray
    prob_right: np.ndarray
    sigmoid_params: tuple | None = None
    label: str = ""


@dataclass
class PsychometricResult:
    """Full psychometric analysis result.

    Attributes:
        curves: Dict mapping (context, variable) -> PsychometricCurve.
        choices: Raw choice values for all trials.
        conditions: Trial conditions.
    """
    curves: dict[tuple[str, str], PsychometricCurve]
    choices: np.ndarray
    conditions: list


def compute_psychometric(
    model,
    inputs: torch.Tensor,
    conditions: list,
) -> dict[str, Any]:
    """Compute psychometric curves from model behavior.

    Groups trials by context and coherence, computes probability of right choice,
    and fits sigmoid functions.

    Args:
        model: Trained RNN or latent circuit model.
        inputs: (N, T, input_dim) task inputs.
        conditions: list of dicts with trial conditions.

    Returns:
        dict with keys:
            "curves": dict mapping (context, variable) -> PsychometricCurve
            "choices": (N,) raw choice values
    """
    model.eval()
    with torch.no_grad():
        out = model(inputs)
        outputs = out.outputs  # (N, T, output_dim)
        # Choice: ReLU(output[t=-1, 0] - output[t=-1, 1])
        choices = torch.relu(outputs[:, -1, 0] - outputs[:, -1, 1]).cpu().numpy()

    # Build DataFrame-like structure
    contexts = [c.get("context", None) for c in conditions]
    motion_cohs = [c.get("motion_coh", None) for c in conditions]
    color_cohs = [c.get("color_coh", None) for c in conditions]

    curves = {}

    # Get unique contexts
    unique_contexts = sorted(set(c for c in contexts if c is not None))

    for ctx in unique_contexts:
        ctx_mask = np.array([c == ctx for c in contexts])

        # Motion coherence psychometric
        motion_coh_vals = sorted(set(c for c, m in zip(motion_cohs, contexts) if m == ctx and c is not None))
        if motion_coh_vals:
            probs = []
            for coh in motion_coh_vals:
                trial_mask = ctx_mask & np.array([m == coh for m in motion_cohs])
                if trial_mask.sum() > 0:
                    probs.append(np.mean(choices[trial_mask] > 0))
                else:
                    probs.append(0.5)
            probs = np.array(probs)

            # Fit sigmoid
            try:
                popt, _ = curve_fit(_sigmoid, np.array(motion_coh_vals) * 100, probs,
                                    p0=[0, 25], maxfev=5000)
            except RuntimeError:
                popt = (0, 25)

            curves[(ctx, "motion")] = PsychometricCurve(
                coherences=np.array(motion_coh_vals),
                prob_right=probs,
                sigmoid_params=tuple(popt),
                label=f"{ctx}_motion",
            )

        # Color coherence psychometric
        color_coh_vals = sorted(set(c for c, m in zip(color_cohs, contexts) if m == ctx and c is not None))
        if color_coh_vals:
            probs = []
            for coh in color_coh_vals:
                trial_mask = ctx_mask & np.array([c == coh for c in color_cohs])
                if trial_mask.sum() > 0:
                    probs.append(np.mean(choices[trial_mask] > 0))
                else:
                    probs.append(0.5)
            probs = np.array(probs)

            try:
                popt, _ = curve_fit(_sigmoid, np.array(color_coh_vals) * 100, probs,
                                    p0=[0, 25], maxfev=5000)
            except RuntimeError:
                popt = (0, 25)

            curves[(ctx, "color")] = PsychometricCurve(
                coherences=np.array(color_coh_vals),
                prob_right=probs,
                sigmoid_params=tuple(popt),
                label=f"{ctx}_color",
            )

    return {
        "curves": curves,
        "choices": choices,
    }
