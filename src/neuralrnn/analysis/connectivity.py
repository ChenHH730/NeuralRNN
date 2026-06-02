"""Connectivity analysis for latent circuit models.

Validates that the inferred latent circuit connectivity w_rec agrees with
the high-dimensional RNN connectivity projected into the latent subspace:
    Q^T W_rec Q ≈ w_rec

Reference: Langdon & Engel (2025), Nature Neuroscience.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import numpy as np


@dataclass
class ConnectivityResult:
    """Result of connectivity analysis between latent circuit and RNN.

    Attributes:
        w_rec: Inferred latent connectivity (n x n).
        W_rec_projected: RNN connectivity projected to latent space Q^T W_rec Q (n x n).
        Q: Embedding matrix (n x N).
        correlation: Pearson correlation between w_rec and W_rec_projected.
    """
    w_rec: np.ndarray
    W_rec_projected: np.ndarray
    Q: np.ndarray
    correlation: float


def analyze_connectivity(latent_model, rnn_model) -> ConnectivityResult:
    """Analyze agreement between latent circuit and RNN connectivity.

    Computes the correlation between the inferred latent connectivity w_rec
    and the RNN connectivity conjugated by the embedding matrix Q:
        Q^T W_rec Q

    Args:
        latent_model: Trained LatentCircuitModel.
        rnn_model: Trained high-dimensional RNN model (e.g., EIRNN).

    Returns:
        ConnectivityResult with w_rec, projected W_rec, Q, and correlation.
    """
    with torch.no_grad():
        # Get latent connectivity
        w_rec = latent_model.w_rec.weight.data.detach().cpu().numpy()

        # Get RNN connectivity
        W_rec = rnn_model.h2h.weight.data.detach().cpu().numpy()
        # Apply Dale's law if applicable
        if hasattr(rnn_model, '_recurrent_weight'):
            W_rec_eff = rnn_model._recurrent_weight().detach().cpu().numpy()
        else:
            W_rec_eff = W_rec

        # Get embedding matrix Q
        Q = latent_model.embedding_matrix.detach().cpu().numpy()

        # Project: Q^T W_rec Q
        W_rec_projected = Q @ W_rec_eff @ Q.T

        # Compute Pearson correlation
        w_flat = w_rec.flatten()
        proj_flat = W_rec_projected.flatten()
        correlation = float(np.corrcoef(w_flat, proj_flat)[0, 1])

    return ConnectivityResult(
        w_rec=w_rec,
        W_rec_projected=W_rec_projected,
        Q=Q,
        correlation=correlation,
    )
