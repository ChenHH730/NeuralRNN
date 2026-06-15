"""Low-rank RNN configuration.

Port of the LowRankRNN from Dubreuil et al. (2022) / Valente et al. (2022).
The recurrent weight matrix is parametrized as:
    W_rec = m @ n^T / N   (when scale_by_hidden_size=True)

where m, n are (N, rank) matrices. This constrains the recurrent dynamics
to a low-dimensional subspace spanned by the columns of m.

Note: Uses framework-standard naming — `activation` (not `non_linearity`)
and `output_activation` (not `output_non_linearity`).
"""
from __future__ import annotations

from ...configuration_utils import NeuralRNNConfig


class LowrankRNNConfig(NeuralRNNConfig):
    """Configuration for a low-rank RNN.

    Dynamics (Euler discretization with step alpha = dt/tau):
        r_t     = tanh(z_t + b)
        rec_t   = r_t @ n @ m^T / N                     (if scale_by_hidden_size)
        z_{t+1} = z_t + sigma * xi_t + alpha * (-z_t + rec_t + x_t @ Wi_full)
        y_t     = output_act(z_t) @ Wo_full / N          (if scale_by_hidden_size)

    where:
        Wi_full = (wi^T * si)^T   (input weights with per-channel scaling)
        Wo_full = wo * so         (output weights with per-channel scaling)
        xi_t ~ N(0, 1)            (Gaussian noise)

    The connectivity vectors m and n define the low-dimensional subspace
    in which the recurrent dynamics unfold. For rank-r networks, the
    columns of m span an r-dimensional subspace suitable for analysis.

    Args:
        input_dim: Number of input channels.
        latent_dim: Number of hidden units (N).
        output_dim: Number of output channels.
        rank: Rank of the recurrent connectivity matrix (default 1).
        alpha: dt/tau, the Euler discretization step (default 0.2).
        noise_std: Standard deviation of recurrent Gaussian noise (default 0.05).
        dt: Integration time step (optional; if set, alpha = dt/tau is computed).
        tau: Membrane time constant (default 100.0, used with dt).
        add_bias: Whether to add a bias term b to the pre-activation (default False).
        scale_by_hidden_size: If True, divide recurrent and output terms by N
            (default True, matches both original codebases).
        activation: Activation function for hidden state ('tanh' or 'relu').
        output_activation: Activation for readout ('tanh' or 'relu').
        train_wi: Whether to train input weights (default True).
        train_wo: Whether to train output weights (default True).
        train_wrec: Whether to train the low-rank m,n vectors (default True).
        train_h0: Whether to train initial state (default False).
    """

    model_type = "lowrank_rnn"

    def __init__(
        self,
        input_dim: int = 1,
        latent_dim: int = 500,
        output_dim: int = 1,
        rank: int = 1,
        alpha: float = 0.2,
        noise_std: float = 0.05,
        dt: float | None = None,
        tau: float = 100.0,
        add_bias: bool = False,
        scale_by_hidden_size: bool = True,
        activation: str = "tanh",
        output_activation: str = "tanh",
        train_wi: bool = True,
        train_wo: bool = True,
        train_wrec: bool = True,
        train_h0: bool = False,
        **kwargs,
    ) -> None:
        # Compute alpha from dt/tau if provided
        if dt is not None and alpha == 0.2:
            alpha = dt / tau
        super().__init__(
            input_dim=input_dim,
            latent_dim=latent_dim,
            output_dim=output_dim,
            dt=dt,
            activation=activation,
            **kwargs,
        )
        self.rank = rank
        self.alpha = alpha
        self.tau = tau
        self.noise_std = noise_std
        self.add_bias = add_bias
        self.scale_by_hidden_size = scale_by_hidden_size
        self.output_activation = output_activation
        self.train_wi = train_wi
        self.train_wo = train_wo
        self.train_wrec = train_wrec
        self.train_h0 = train_h0
