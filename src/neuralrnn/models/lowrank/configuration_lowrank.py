"""Low-rank RNN configuration.

Port of the LowRankRNN from Dubreuil et al. (2022) / Valente et al. (2022).
The recurrent weight matrix is parametrized as:
    W_rec = m @ n^T / N   (when scale_by_hidden_size=True)

where m, n are (N, rank) matrices. This constrains the recurrent dynamics
to a low-dimensional subspace spanned by the columns of m.

Note: Uses framework-standard naming — `activation` (not `non_linearity`),
`output_activation` (not `output_non_linearity`), and `sigma_rec` (not
`noise_std`).
"""
from __future__ import annotations

import warnings

from ...configuration_utils import NeuralRNNConfig, resolve_euler_alpha


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
        alpha: Euler update fraction per step. When given explicitly it takes
            precedence over dt/tau (priority: alpha > dt/tau); see
            ``neuralrnn.configuration_utils.resolve_euler_alpha``.
        dt: Integration time step (default None -> 20.0, the original code's
            value, giving alpha = dt/tau = 0.2).
        tau: Membrane time constant (default 100.0).
        sigma_rec: Standard deviation of recurrent Gaussian noise (default 0.05).
        add_bias: Whether to add a bias term b to the pre-activation (default False).
        scale_by_hidden_size: If True, divide recurrent and output terms by N
            (default True, matches both original codebases).
        activation: Activation function for hidden state. Supported: relu, tanh,
            sigmoid, softplus, leaky_relu/leakyrelu, elu, selu, gelu, silu/swish
            (default "tanh").
        output_activation: Activation for readout. Same supported names as
            ``activation`` (default "tanh").
        train_wi: Whether to train input weights wi (True, with si frozen) or
            the scaling si (False, with wi frozen). Selector, not a freeze flag:
            use ``freeze_input=True`` to freeze both (default True).
        train_wo: Whether to train output weights wo (True, with so frozen) or
            the scaling so (False, with wo frozen). Selector, not a freeze flag:
            use ``freeze_output=True`` to freeze both (default True).

    Freezing uses the framework-wide ``freeze_*`` flags (see NeuralRNNConfig):
    ``freeze_recurrent`` freezes the low-rank m,n factors, ``freeze_h0`` freezes
    the initial state. Note the family default ``freeze_h0=True`` (the original
    code never trained h0); pass ``freeze_h0=False`` to train it.
    Deprecated aliases ``train_wrec``/``train_h0`` are still accepted (with a
    DeprecationWarning) and mapped to ``freeze_recurrent``/``freeze_h0``.
    """

    model_type = "lowrank_rnn"

    def __init__(
        self,
        input_dim: int = 1,
        latent_dim: int = 500,
        output_dim: int = 1,
        rank: int = 1,
        alpha: float | None = None,
        sigma_rec: float = 0.05,
        dt: float | None = None,
        tau: float = 100.0,
        add_bias: bool = False,
        scale_by_hidden_size: bool = True,
        activation: str = "tanh",
        output_activation: str = "tanh",
        train_wi: bool = True,
        train_wo: bool = True,
        **kwargs,
    ) -> None:
        # Backward compatibility: old configs/code used `noise_std`.
        if "noise_std" in kwargs:
            noise_std = kwargs.pop("noise_std")
            warnings.warn(
                "LowrankRNNConfig: `noise_std` is deprecated, use `sigma_rec` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if sigma_rec == 0.05:
                sigma_rec = noise_std
        # Backward compatibility: `train_wrec`/`train_h0` were exact negations of
        # the framework-wide freeze flags and have been removed.
        if "train_wrec" in kwargs:
            train_wrec = kwargs.pop("train_wrec")
            warnings.warn(
                "LowrankRNNConfig: `train_wrec` is deprecated, use `freeze_recurrent` "
                "instead (freeze_recurrent = not train_wrec).",
                DeprecationWarning,
                stacklevel=2,
            )
            kwargs.setdefault("freeze_recurrent", not train_wrec)
        if "train_h0" in kwargs:
            train_h0 = kwargs.pop("train_h0")
            warnings.warn(
                "LowrankRNNConfig: `train_h0` is deprecated, use `freeze_h0` "
                "instead (freeze_h0 = not train_h0).",
                DeprecationWarning,
                stacklevel=2,
            )
            kwargs.setdefault("freeze_h0", not train_h0)
        else:
            # Family default: h0 is NOT trained (matches the original code's
            # train_h0=False). Explicit freeze_h0=False opts into training.
            kwargs.setdefault("freeze_h0", True)
        alpha, dt = resolve_euler_alpha(dt, tau, alpha, default_dt=20.0, model_type=self.model_type)
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
        self.sigma_rec = sigma_rec
        self.add_bias = add_bias
        self.scale_by_hidden_size = scale_by_hidden_size
        self.output_activation = output_activation
        self.train_wi = train_wi
        self.train_wo = train_wo
