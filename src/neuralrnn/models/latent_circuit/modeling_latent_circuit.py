"""Latent circuit model implementation.

Ported from Langdon & Engel (2025), Nature Neuroscience.
"Latent circuit inference from heterogeneous neural responses during cognitive tasks."

The latent circuit model is a low-dimensional recurrent circuit that is embedded
into a high-dimensional neural space via an orthonormal matrix Q (Cayley transform).

Dynamics (Euler discretization):
    x_t = (1 - alpha) * x_{t-1} + alpha * ReLU(w_rec @ x_{t-1} + w_in @ u_t + noise)

where alpha = dt / tau, noise ~ N(0, 2 * alpha * sigma_rec^2).

The embedding matrix Q maps latent states to the high-dimensional space:
    y = x @ Q   (embed: latent -> high-dim)
    x = y @ Q^T (project: high-dim -> latent)
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ...modeling_utils import NeuralDynamicsModel, DynamicsModelOutput
from ...auto.modeling_auto import register_model
from ...activations import get_activation
from .configuration_latent_circuit import LatentCircuitConfig


@register_model("latent_circuit")
class LatentCircuitModel(NeuralDynamicsModel):
    """Latent circuit model for inferring low-dimensional connectivity.

    This model implements a low-dimensional recurrent circuit (n nodes) that is
    embedded into an N-dimensional space via an orthonormal embedding matrix Q.
    The embedding is parameterized through the Cayley transform.

    Key components:
    - Recurrence: CTRNN-style Euler discretization with ReLU activation
    - Embedding: Orthonormal Q (n x N) via Cayley transform of learnable parameter a
    - Connectivity masks: diagonal input/output masks enforcing node identity
    - Readout: linear map from latent space to task output

    The model exposes embed() and project() methods for mapping between
    latent and high-dimensional spaces.
    """

    config_class = LatentCircuitConfig

    def __init__(self, config: LatentCircuitConfig) -> None:
        super().__init__(config)
        n = config.latent_dim       # number of latent nodes
        N = config.embedding_dim    # high-dimensional RNN size
        K = config.input_dim        # task input dimension
        O = config.output_dim       # task output dimension

        self.alpha = config.alpha  # Euler step size (resolved in config)
        self.sigma_rec = config.sigma_rec
        self.act = get_activation(config.activation)

        # Recurrent connectivity (n x n, no bias)
        self.w_rec = nn.Linear(n, n, bias=False)
        # Input connectivity (K -> n, no bias)
        self.w_in = nn.Linear(K, n, bias=False)
        # Readout connectivity (n -> O, no bias)
        self.w_out = nn.Linear(n, O, bias=False)

        # Learnable parameter for Cayley transform -> orthonormal Q
        self.a = nn.Parameter(torch.rand(N, N))

        # Connectivity masks (fixed, not learnable)
        # Input mask: each input connects to its designated node (diagonal)
        input_mask = torch.zeros(n, K)
        diag_size = min(n, K)
        input_mask[:diag_size, :diag_size] = torch.eye(diag_size)
        self.register_buffer("input_mask", input_mask)

        # Output mask: choice nodes connect to outputs (last output_dim nodes)
        output_mask = torch.zeros(O, n)
        output_mask[:, -O:] = torch.eye(O)
        self.register_buffer("output_mask", output_mask)

        # Initialize weights (matching reference initialization)
        self._init_weights()

        # Apply connectivity masks
        self.apply_constraints()
        self.apply_freeze_config()

    def _freeze_groups(self) -> dict[str, list[str]]:
        return {
            "input": [r"^w_in\."],
            "recurrent": [r"^w_rec\."],
            "output": [r"^w_out\."],
            "h0": [],  # initial state is fixed zero
        }

    def _init_weights(self) -> None:
        """Initialize weights to match reference implementation."""
        with torch.no_grad():
            self.w_rec.weight.normal_(mean=0.0, std=0.025)
            self.w_in.weight.normal_(mean=0.2, std=0.1)
            self.w_out.weight.normal_(mean=0.2, std=0.1)

    def cayley_transform(self, a: torch.Tensor) -> torch.Tensor:
        """Convert arbitrary matrix a into orthonormal Q via Cayley transform.

        Q_full = (I - skew) @ (I + skew)^{-1}
        where skew = (a - a^T) / 2

        Returns Q = Q_full[:n, :] of shape (n, N).
        """
        skew = (a - a.T) / 2
        N = a.shape[0]
        eye = torch.eye(N, device=a.device, dtype=a.dtype)
        Q_full = (eye - skew) @ torch.inverse(eye + skew)
        return Q_full[:self.config.latent_dim, :]

    @property
    def embedding_matrix(self) -> torch.Tensor:
        """Orthonormal embedding matrix Q of shape (n, N).

        Maps from latent space to high-dimensional space:
            y = x @ Q
        """
        return self.cayley_transform(self.a)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Map latent states to high-dimensional space: y = x @ Q.

        Args:
            x: Latent states (..., n).

        Returns:
            Embedded states (..., N).
        """
        return x @ self.embedding_matrix

    def project(self, y: torch.Tensor) -> torch.Tensor:
        """Project high-dimensional states to latent space: x = y @ Q^T.

        Args:
            y: High-dimensional states (..., N).

        Returns:
            Projected states (..., n).
        """
        return y @ self.embedding_matrix.T

    def apply_constraints(self) -> None:
        """Recompute Q and reapply connectivity masks.

        Must be called after each gradient step during training.
        Follows the reference implementation: mask * relu(weights).
        """
        with torch.no_grad():
            # Input mask: each input connects to its designated node (diagonal)
            self.w_in.weight.copy_(
                self.input_mask * torch.relu(self.w_in.weight)
            )
            # Output mask: choice nodes connect to outputs (last output_dim nodes)
            self.w_out.weight.copy_(
                self.output_mask * torch.relu(self.w_out.weight)
            )

    def init_state(self, batch_size: int, device: str = "cpu") -> torch.Tensor:
        """Initialize hidden states to zero (matching reference)."""
        return torch.zeros(batch_size, self.config.latent_dim, device=device)

    def recurrence(self, x_t: torch.Tensor, z_prev: torch.Tensor, *, inputs=None) -> torch.Tensor:
        """Single-step state transition in latent space.

        z_t = (1 - alpha) * z_{t-1} + alpha * ReLU(w_rec @ z_{t-1} + w_in @ x_t + noise)

        Args:
            x_t: Task input at current time step (batch, input_dim).
            z_prev: Previous latent state (batch, latent_dim).

        Returns:
            z_t: New latent state (batch, latent_dim).
        """
        pre = self.w_rec(z_prev) + self.w_in(x_t)

        # Add recurrent noise during training
        if self.sigma_rec > 0 and self.training:
            noise_std = (2 * self.alpha * self.sigma_rec ** 2) ** 0.5
            noise = noise_std * torch.randn_like(pre)
            pre = pre + noise

        # Euler discretization with the configured activation
        z_t = (1 - self.alpha) * z_prev + self.alpha * self.act(pre)
        return z_t

    def readout(self, z_t: torch.Tensor) -> torch.Tensor:
        """Readout from latent state to task output.

        Args:
            z_t: Latent state (batch, latent_dim).

        Returns:
            Task output (batch, output_dim).
        """
        return self.w_out(z_t)
