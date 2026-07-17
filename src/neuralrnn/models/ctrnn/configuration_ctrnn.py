"""CTRNN family configurations (continuous-time RNN, including vanilla / EI variants).

Reference implementation: ported from nn-brain's RNN+DynamicalSystemAnalysis.ipynb / EI_RNN.ipynb.
Serves as the Contract-A copy template for "Paradigm A (task-optimized RNN)".
"""
from __future__ import annotations

from ...configuration_utils import NeuralRNNConfig, resolve_euler_alpha


class CTRNNConfig(NeuralRNNConfig):
    """Continuous-time RNN: τ dr/dt = -r + f(W_r r + W_x x + b), Euler-discretized with step dt.

    Args:
        input_dim:  Input dimension
        latent_dim: Number of hidden units M
        output_dim: Readout dimension (e.g. number of task classes)
        dt:         Discretization step; alpha = dt/tau. Default None -> 100.0
                    (family default, equivalent to alpha = 1.0).
        tau:        Time constant
        alpha:      Euler update fraction per step. When given explicitly it takes
                    precedence over dt/tau (priority: alpha > dt/tau > 1.0); see
                    ``neuralrnn.configuration_utils.resolve_euler_alpha``.
        activation: Nonlinearity name. Supported: relu, tanh, sigmoid, softplus,
            leaky_relu/leakyrelu, elu, selu, gelu, silu/swish (default "relu").
        dale:       Whether to enforce Dale constraints (excitatory/inhibitory separation); True for EI variant
        ei_ratio:   Fraction of excitatory units (effective when dale=True)
        trainable_h0: Whether the initial state is a trainable parameter.
            Structural switch: False -> h0 is a fixed buffer (not in
            named_parameters at all, stronger than freezing); True -> h0 is an
            nn.Parameter, which can still be frozen via freeze_h0=True.
        sigma_rec:  Standard deviation of recurrent noise (0 disables)
        relu_after_blend: True = f((1-α)z + α·pre) (original nn-brain formula);
                          False = (1-α)z + α·f(pre) (standard Euler discretization, default)
    """

    model_type = "ctrnn"

    def __init__(
        self,
        input_dim: int = 3,
        latent_dim: int = 64,
        output_dim: int = 3,
        dt: float | None = None,
        tau: float = 100.0,
        alpha: float | None = None,
        activation: str = "relu",
        dale: bool = False,
        ei_ratio: float = 0.8,
        trainable_h0: bool = False,
        sigma_rec: float = 0.0,
        relu_after_blend: bool = False,
        noise_alpha_scaling: bool = False,
        **kwargs,
    ) -> None:
        alpha, dt = resolve_euler_alpha(dt, tau, alpha, default_dt=100.0, model_type=self.model_type)
        super().__init__(input_dim=input_dim, latent_dim=latent_dim,
                         output_dim=output_dim, dt=dt, activation=activation, **kwargs)
        self.alpha = alpha
        self.tau = tau
        self.dale = dale
        self.ei_ratio = ei_ratio
        self.trainable_h0 = trainable_h0
        self.sigma_rec = sigma_rec
        self.relu_after_blend = relu_after_blend
        self.noise_alpha_scaling = noise_alpha_scaling


class VanillaRNNConfig(CTRNNConfig):
    """Discrete vanilla RNN (dt=None is equivalent to alpha=1)."""
    model_type = "vanilla_rnn"

    def __init__(self, **kwargs):
        kwargs.setdefault("dt", None)
        super().__init__(**kwargs)


class EIRNNConfig(CTRNNConfig):
    """Excitatory-Inhibitory RNN (Dale's principle enforced by default).

    Extended from CTRNNConfig with EI-specific parameters:
        readout_e_only: If True, readout only from excitatory units (first e_size units).
                        This matches the original E-I RNN paper (Song et al., 2016) where
                        long-range projections are exclusively excitatory.
        init_method:    Weight initialization method ('kaiming' or 'gamma').

    Reference:
        Song, H.F., Yang, G.R. and Wang, X.J., 2016.
        Training excitatory-inhibitory recurrent neural networks
        for cognitive tasks: a simple and flexible framework.
        PLoS computational biology, 12(2).
    """
    model_type = "ei_rnn"

    def __init__(self, readout_e_only: bool = True, init_method: str = "kaiming",
                 **kwargs):
        kwargs.setdefault("dale", True)
        super().__init__(**kwargs)
        self.readout_e_only = readout_e_only
        self.init_method = init_method
