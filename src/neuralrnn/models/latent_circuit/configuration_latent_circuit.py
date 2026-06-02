"""Latent circuit configuration.

Ported from Langdon & Engel (2025), Nature Neuroscience.
The latent circuit model fits a low-dimensional recurrent circuit to neural
responses, simultaneously inferring connectivity and an embedding matrix.
"""
from __future__ import annotations

from ...configuration_utils import NeuralRNNConfig


class LatentCircuitConfig(NeuralRNNConfig):
    """Configuration for the latent circuit model.

    The latent circuit has n latent nodes (latent_dim) embedded into an
    N-dimensional space (embedding_dim) via an orthonormal matrix Q learned
    through the Cayley transform.

    Dynamics (Euler discretization):
        x_t = (1 - alpha) * x_{t-1} + alpha * ReLU(w_rec @ x_{t-1} + w_in @ u_t + noise)

    where alpha = dt / tau.

    Args:
        input_dim:      Task input dimension K (e.g. 6 for SiegelMiller task).
        latent_dim:     Number of latent nodes n (default 8).
        output_dim:     Task output dimension (e.g. 2 for binary choice).
        embedding_dim:  Dimension N of the high-dimensional RNN space.
        dt:             Discretization time step in ms (default 40).
        tau:            Time constant in ms (default 200). alpha = dt/tau = 0.2.
        sigma_rec:      Recurrent noise standard deviation (default 0.15).
        activation:     Nonlinearity (default "relu", only relu supported).
    """

    model_type = "latent_circuit"

    def __init__(
        self,
        input_dim: int = 6,
        latent_dim: int = 8,
        output_dim: int = 2,
        embedding_dim: int = 50,
        dt: float = 40.0,
        tau: float = 200.0,
        sigma_rec: float = 0.15,
        activation: str = "relu",
        **kwargs,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            latent_dim=latent_dim,
            output_dim=output_dim,
            dt=dt,
            activation=activation,
            **kwargs,
        )
        self.embedding_dim = embedding_dim
        self.tau = tau
        self.sigma_rec = sigma_rec
