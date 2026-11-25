"""CTRNN family configurations (continuous-time RNN, including vanilla / EI variants).

CTRNN reference: nn-brain's RNN+DynamicalSystemAnalysis.ipynb.
E-I reference: Song 2016 pycog, with configurable nn-brain tutorial alternatives.
Serves as the Contract-A copy template for "Paradigm A (task-optimized RNN)".
"""
from __future__ import annotations

import warnings

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
        noise_alpha_scaling: Multiply injection std by sqrt(2*alpha) when True.
        noise_position: "pre" (default), "post", or "state".
        evaluate_noise: Enable configured noise in eval mode (default False).
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
        nonlinearity_mode: str = "pre_activation",
        **kwargs,
    ) -> None:
        alpha, dt = resolve_euler_alpha(dt, tau, alpha, default_dt=100.0, model_type=self.model_type)
        validate_nonlinearity_mode(nonlinearity_mode, model_type=self.model_type)
        # Input-only migration keeps existing notebooks/checkpoints usable.
        # New configs store only sigma_rec and the original boolean flag.
        legacy_scaling = kwargs.pop("noise_scaling", None)
        if legacy_scaling not in (None, "sqrt_2alpha", "sqrt_2_over_alpha"):
            raise ValueError(f"Unknown legacy noise_scaling={legacy_scaling!r}")
        if legacy_scaling is not None:
            if legacy_scaling == "sqrt_2_over_alpha":
                sigma_rec = sigma_rec / alpha
            noise_alpha_scaling = True
            warnings.warn(
                "noise_scaling is obsolete; converted to equivalent sigma_rec and "
                "noise_alpha_scaling=True. The injected noise amplitude is preserved.",
                FutureWarning, stacklevel=2)
        super().__init__(input_dim=input_dim, latent_dim=latent_dim,
                         output_dim=output_dim, dt=dt, activation=activation,
                         sigma_rec=sigma_rec, noise_alpha_scaling=noise_alpha_scaling, **kwargs)
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
        self.nonlinearity_mode = nonlinearity_mode


class EIRNNConfig(CTRNNConfig):
    """Current-state E-I network with Dale sign constraints.

    ``nonlinearity_mode="rate"`` (default) uses activated currents in both
    recurrence and readout. Other modes store activity and read it directly.
    There is no separate observation-mode switch.

    Args:
        positive_input_weights: Constrain input weights to non-negative values.
        positive_output_weights: Constrain output magnitudes and apply the
            presynaptic E/I signs. With E-only readout, weights are non-negative.
        ei_positive_func: "rectify" (pycog default) or "abs"; used by all enabled
            sign constraints. Disabling ``dale`` removes recurrent column signs.
        readout_e_only: Select excitatory units (including custom dale_signs).
        no_self_connections: Enforce a zero recurrent diagonal on every call.
        in_mask: Binary (input_dim, latent_dim) connectivity, or None.
        rec_mask: Binary (latent_dim, latent_dim), target-by-source, or None.
        out_mask: Binary (latent_dim, output_dim), or None. E-only selection
            is applied after this full-population mask.
        init_method: "kaiming" (nn-brain recipe, default), "pycog" (Gamma(2,
            rate=20) recurrent magnitudes, mean-balanced default connectivity;
            uniform input/output magnitudes), or "gamma" (Battista Gamma(4,
            rate=4) recipe).
        spectral_radius: Initial effective recurrent radius, or None (default)
            to skip rescaling.
        h0_init: Initial current (default 0.0); trainable_h0 defaults to False.
        input_bias: Include a trainable input projection bias (default True).
        recurrent_bias: Include a trainable recurrent bias (default True).
        output_bias: Include a trainable output bias (default True).

    Defaults follow the nn-brain EI_RNN tutorial: dt=tau/5, rate dynamics, ReLU
    activation, E-only readout, unconstrained input/output weights with biases,
    no autapses, kaiming initialization without radius rescaling, and fixed
    h0=0. The pycog recipe remains available through explicit configuration.
    Noise stays opt-in (sigma_rec=0); its default position/scaling reproduce
    sqrt(2*alpha)*sigma_rec state increments. Task inputs and training objectives
    are configured separately. Masks describe plastic connectivity, not pycog's
    optional fixed non-trainable connections.
    """
    model_type = "ei_rnn"

    def __init__(self, readout_e_only: bool = True, init_method: str = "kaiming",
                 spectral_radius: float | None = None, no_self_connections: bool = True,
                 positive_input_weights: bool = False, positive_output_weights: bool = False,
                 ei_positive_func: str = "rectify", in_mask=None, rec_mask=None,
                 out_mask=None, h0_init: float = 0.0, input_bias: bool = True,
                 recurrent_bias: bool = True, output_bias: bool = True, **kwargs):
        if "readout_mode" in kwargs:
            raise TypeError("readout_mode was removed; E-I rate dynamics automatically read firing rates")
        kwargs.setdefault("dale", True)
        kwargs.setdefault("nonlinearity_mode", "rate")
        kwargs.setdefault("trainable_h0", False)
        kwargs.setdefault("noise_position", "state")
        kwargs.setdefault("noise_alpha_scaling", True)
        if kwargs.get("dt") is None and kwargs.get("alpha") is None:
            kwargs["dt"] = kwargs.get("tau", 100.0) / 5
        if ei_positive_func not in ("rectify", "abs"):
            raise ValueError("ei_positive_func must be 'rectify' or 'abs'")
        if init_method not in ("pycog", "kaiming", "gamma"):
            raise ValueError("init_method must be 'pycog', 'kaiming' or 'gamma'")
        for name, value in dict(readout_e_only=readout_e_only,
                no_self_connections=no_self_connections,
                positive_input_weights=positive_input_weights,
                positive_output_weights=positive_output_weights,
                input_bias=input_bias, recurrent_bias=recurrent_bias,
                output_bias=output_bias).items():
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be bool")
        import math
        if spectral_radius is not None and (not math.isfinite(spectral_radius) or spectral_radius <= 0):
            raise ValueError("spectral_radius must be finite and positive, or None")
        if not math.isfinite(h0_init):
            raise ValueError("h0_init must be finite")
        super().__init__(**kwargs)
        if not 0 <= self.ei_ratio <= 1:
            raise ValueError("ei_ratio must be between 0 and 1")
        self.readout_e_only = readout_e_only
        self.init_method = init_method
        self.spectral_radius = spectral_radius
        self.no_self_connections = no_self_connections
        self.positive_input_weights = positive_input_weights
        self.positive_output_weights = positive_output_weights
        self.ei_positive_func = ei_positive_func
        self.h0_init = float(h0_init)
        self.input_bias = input_bias
        self.recurrent_bias = recurrent_bias
        self.output_bias = output_bias
        # Normalize array/tensor masks once so AutoConfig serialization is JSON-safe.
        for name, value in dict(in_mask=in_mask, rec_mask=rec_mask, out_mask=out_mask).items():
            if hasattr(value, "tolist"):
                value = value.tolist()
            setattr(self, name, value)
