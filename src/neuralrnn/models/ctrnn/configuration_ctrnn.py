"""CTRNN family configurations (continuous-time RNN, including vanilla / EI variants).

Reference implementation: ported from nn-brain's RNN+DynamicalSystemAnalysis.ipynb / EI_RNN.ipynb.
Serves as the Contract-A copy template for "Paradigm A (task-optimized RNN)".
"""
from __future__ import annotations

from ...configuration_utils import (
    NeuralRNNConfig, resolve_euler_alpha, validate_nonlinearity_mode,
)


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
        dale_signs: Optional per-unit sign vector (+1 excitatory / -1 inhibitory),
            length must equal latent_dim. When provided it implies dale=True and
            takes precedence over ei_ratio (which only supports a single global
            E/I split; dale_signs allows e.g. per-area 80/20 splits).
        trainable_h0: Whether the initial state is a trainable parameter.
            Structural switch: False -> h0 is a fixed buffer (not in
            named_parameters at all, stronger than freezing); True -> h0 is an
            nn.Parameter, which can still be frozen via freeze_h0=True.
        sigma_rec:  Standard deviation of recurrent noise (0 disables)
        noise_alpha_scaling: Legacy flag; noise std = sqrt(2 * alpha * sigma^2).
        noise_scaling: Explicit noise scaling mode, overrides noise_alpha_scaling when set:
            None               -> legacy behavior (noise_alpha_scaling flag);
            "sqrt_2alpha"      -> std = sqrt(2 * alpha * sigma^2) (same as legacy flag);
            "sqrt_2_over_alpha"-> std = sqrt(2 * sigma^2 / alpha) (recipe of
                                  Battista-2026: prefactor sqrt(2*sigma2rec/alpha)
                                  with sigma2rec the variance).
        nonlinearity_mode: Where the nonlinearity f sits in the Euler step
            (pre = W@state + B@x + b, noise added on pre):
            "pre_activation" (default): z' = (1-α)z + α·f(pre);
            "post_blend":               z' = f((1-α)z + α·pre) (nn-brain formula);
            "rate":                     r = f(z); z' = (1-α)z + α·(W@r + B@x + b)
                                        (classic firing-rate form; noise on pre stays
                                        inside the blend and is not rectified; readout
                                        stays from z).
            See ``neuralrnn.configuration_utils.SUPPORTED_NONLINEARITY_MODES``.
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
        dale_signs: list[float] | None = None,
        trainable_h0: bool = False,
        sigma_rec: float = 0.0,
        noise_alpha_scaling: bool = False,
        noise_scaling: str | None = None,
        nonlinearity_mode: str = "pre_activation",
        **kwargs,
    ) -> None:
        alpha, dt = resolve_euler_alpha(dt, tau, alpha, default_dt=100.0, model_type=self.model_type)
        validate_nonlinearity_mode(nonlinearity_mode, model_type=self.model_type)
        if noise_scaling not in (None, "sqrt_2alpha", "sqrt_2_over_alpha"):
            raise ValueError(
                f"noise_scaling must be None, 'sqrt_2alpha' or 'sqrt_2_over_alpha', got {noise_scaling!r}")
        super().__init__(input_dim=input_dim, latent_dim=latent_dim,
                         output_dim=output_dim, dt=dt, activation=activation, **kwargs)
        self.alpha = alpha
        self.tau = tau
        self.dale = dale or (dale_signs is not None)
        self.ei_ratio = ei_ratio
        if dale_signs is not None:
            if len(dale_signs) != latent_dim:
                raise ValueError(
                    f"dale_signs length {len(dale_signs)} != latent_dim {latent_dim}")
            if any(s not in (1, -1, 1.0, -1.0) for s in dale_signs):
                raise ValueError("dale_signs entries must be +1 (E) or -1 (I)")
        self.dale_signs = dale_signs
        self.trainable_h0 = trainable_h0
        self.sigma_rec = sigma_rec
        self.noise_alpha_scaling = noise_alpha_scaling
        self.noise_scaling = noise_scaling
        self.nonlinearity_mode = nonlinearity_mode


class EIRNNConfig(CTRNNConfig):
    """Excitatory-Inhibitory RNN (Dale's principle enforced by default).

    Extended from CTRNNConfig with EI-specific parameters:
        readout_e_only: If True, readout only from excitatory units (first e_size units).
                        This matches the original E-I RNN paper (Song et al., 2016) where
                        long-range projections are exclusively excitatory.
        init_method:    Weight initialization method ('kaiming' or 'gamma').
                        'gamma' samples |W_rec| ~ Gamma(4, 4) (recipe of Battista-2026).
        spectral_radius: If set, rescale the effective recurrent matrix |W| @ diag(sign)
                        to this spectral radius after init (e.g. 1.5 in Battista-2026).
        no_self_connections: If True, zero the W_rec diagonal after init.

    Reference:
        Song, H.F., Yang, G.R. and Wang, X.J., 2016.
        Training excitatory-inhibitory recurrent neural networks
        for cognitive tasks: a simple and flexible framework.
        PLoS computational biology, 12(2).
    """
    model_type = "ei_rnn"

    def __init__(self, readout_e_only: bool = True, init_method: str = "kaiming",
                 spectral_radius: float | None = None, no_self_connections: bool = False,
                 **kwargs):
        kwargs.setdefault("dale", True)
        if init_method not in ("kaiming", "gamma"):
            raise ValueError(f"init_method must be 'kaiming' or 'gamma', got {init_method!r}")
        super().__init__(**kwargs)
        self.readout_e_only = readout_e_only
        self.init_method = init_method
        self.spectral_radius = spectral_radius
        self.no_self_connections = no_self_connections
