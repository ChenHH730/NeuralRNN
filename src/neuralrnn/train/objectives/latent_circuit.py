"""Latent circuit objective — fits a low-dimensional circuit to RNN responses.

The loss function combines:
1. MSE on behavioral output: ||readout(x) - z||^2
2. NMSE on embedded latent states: ||x @ Q - y||^2 / ||y_bar||^2

where x = latent states, Q = embedding matrix, y = RNN hidden states,
z = RNN behavioral output, y_bar = mean-centered y.

Reference: Langdon & Engel (2025), Nature Neuroscience.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .base import Objective


class LatentCircuitObjective(Objective):
    """Objective for fitting latent circuit models to RNN responses.

    Args:
        l_y: Weight for the NMSE term (default 1.0).
    """

    def __init__(self, l_y: float = 1.0) -> None:
        super().__init__()
        self.l_y = l_y

    def compute_loss(self, model, batch: dict) -> tuple:
        """Compute the latent circuit fitting loss.

        Args:
            model: LatentCircuitModel instance.
            batch: dict with keys:
                "inputs":     (B, T, K) — task inputs
                "targets":    (B, T, output_dim) — RNN behavioral output
                "rnn_states": (B, T, N) — RNN hidden states

        Returns:
            (loss, logs_dict) where logs_dict contains "mse_z" and "nmse_y".
        """
        inputs = batch["inputs"]
        z_target = batch["targets"]   # RNN behavioral output
        y_rnn = batch["rnn_states"]   # RNN hidden states

        # Forward pass through latent circuit
        out = model(inputs)
        x = out.states   # (B, T, n) — latent states
        # Behavioral output from latent circuit
        z_pred = out.outputs  # (B, T, output_dim)

        # MSE on behavioral output
        mse_z = torch.mean((z_pred - z_target) ** 2)

        # NMSE on embedded latent states
        # ||x @ Q - y||^2 / ||y_bar||^2
        Q = model.embedding_matrix  # (n, N)
        x_embedded = x @ Q  # (B, T, N) — embedded latent states

        y_bar = y_rnn - torch.mean(y_rnn, dim=[0, 1], keepdim=True)
        nmse_denom = torch.mean(y_bar ** 2).clamp_min(1e-8)
        nmse_y = torch.mean((x_embedded - y_rnn) ** 2) / nmse_denom

        loss = mse_z + self.l_y * nmse_y

        logs = {
            "loss": loss.item(),
            "mse_z": mse_z.item(),
            "nmse_y": nmse_y.item(),
        }
        return loss, logs
