"""Latent circuit configuration.

Ported from Langdon & Engel (2025), Nature Neuroscience.
The latent circuit model fits a low-dimensional recurrent circuit to neural
responses, simultaneously inferring connectivity and an embedding matrix.
"""
from __future__ import annotations

from ...configuration_utils import (
    NeuralRNNConfig, resolve_euler_alpha, validate_nonlinearity_mode,
)


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
        dt:             Discretization time step in ms (default None -> 40 ms,
                        giving alpha = dt/tau = 0.2).
        tau:            Time constant in ms (default 200).
        alpha:          Euler update fraction per step. When given explicitly it
                        takes precedence over dt/tau (priority: alpha > dt/tau);
                        see ``neuralrnn.configuration_utils.resolve_euler_alpha``.
        sigma_rec:      Recurrent noise standard deviation (default 0.15).
        activation:     Nonlinearity name for the recurrence. Supported: relu,
            tanh, sigmoid, softplus, leaky_relu/leakyrelu, elu, selu, gelu,
            silu/swish (default "relu"). Note that the connectivity masks in
            ``apply_constraints`` remain hard ReLU constraints regardless of
            this choice.
        nonlinearity_mode: Where the nonlinearity f sits in the Euler step
            (default "pre_activation", the family's native form):
            "pre_activation": z' = (1-α)z + α·f(pre);
            "post_blend":     z' = f((1-α)z + α·pre);
            "rate":           r = f(z); z' = (1-α)z + α·(w_rec@r + w_in@x).
            Recurrent noise is always sqrt(2α)·σ added on pre (family trait,
            all modes).
    """

    model_type = "latent_circuit"

    def __init__(
        self,
        input_dim: int = 6,
        latent_dim: int = 8,
        output_dim: int = 2,
        embedding_dim: int = 50,
        dt: float | None = None,
        tau: float = 200.0,
        alpha: float | None = None,
        sigma_rec: float = 0.15,
        activation: str = "relu",
        nonlinearity_mode: str = "pre_activation",
        **kwargs,
    ) -> None:
        alpha, dt = resolve_euler_alpha(dt, tau, alpha, default_dt=40.0, model_type=self.model_type)
        validate_nonlinearity_mode(nonlinearity_mode, model_type=self.model_type)
        super().__init__(
            input_dim=input_dim,
            latent_dim=latent_dim,
            output_dim=output_dim,
            dt=dt,
            activation=activation,
            **kwargs,
        )
        self.alpha = alpha
        self.embedding_dim = embedding_dim
        self.tau = tau
        self.sigma_rec = sigma_rec
        self.nonlinearity_mode = nonlinearity_mode
