"""Gain RNN family configurations.

Unified "fixed weights + neuronal gain modulation" paradigm (see
``docs/papers/gain_rnn.md``): per-neuron gains/biases parameterize the firing-rate
function while connectivity can be hard-masked and/or frozen.

- ``GainRNNConfig``: base gain RNN. Rate map
      outside: r = g * phi(u + b)   (output gain == column scaling; Beiran & Litwin-Kumar 2025)
      inside:  r = phi(g * u + b)   (input gain == slope modulation; Stroud et al. 2018)
  The bias always sits inside phi.
- ``StpRNNConfig``: short-term plasticity as a *dynamic* gain parameterization
  (effective presynaptic gain = syn_x * syn_u, Tsodyks-Markram rate form).
  Family defaults reproduce the notebook-11 (Masse et al. 2019 style) model;
  ``stp_init="random"`` + per-trial ``stp_alpha`` reproduces Zhou & Buonomano 2024.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..constrained_rnn.configuration_constrained_rnn import ConstrainedRNNConfig

# Where the per-neuron gain sits relative to the nonlinearity (see gain_rnn.md §6):
#   "outside": r = g * phi(u + b)  — pure amplitude modulation, equivalent to
#              scaling column j of the recurrent matrix by g_j (presynaptic).
#   "inside":  r = phi(g * u + b)  — slope/at-origin sensitivity modulation;
#              saturation value unchanged.
# Note: for ReLU the two are exactly equivalent (scale degeneracy); negative
# gains flip E/I identity under Dale constraints. Use with care.
SUPPORTED_GAIN_POSITIONS = ("outside", "inside")

# Where recurrent noise enters the Euler step (orthogonal to noise_alpha_scaling):
#   "pre":  added on the pre-activation (scaled by alpha through the blend;
#           rectified in modes where f acts on pre). CTRNN lineage default.
#   "post": added on the leaked blend, before the final nonlinearity
#           (state-level noise; with noise_alpha_scaling=True the std is
#           sqrt(2*alpha)*sigma, matching both Masse-style (post_blend) and
#           Zhou & Buonomano 2024 (rate) discretizations).
SUPPORTED_NOISE_POSITIONS = ("pre", "post")

# How per-neuron STP constants (tau_x, tau_u, U) are initialized when they are
# given as scalars:
#   "constant":    every neuron gets (stp_tau_x, stp_tau_u, stp_U).
#   "alternating": even indices facilitating (tau_x_fac/tau_u_fac/U_fac),
#                  odd indices depressing (tau_x_dep/tau_u_dep/U_dep)
#                  (notebook 11 / Masse et al. 2019 "full" synapse config).
#   "random":      sampled from truncated normals (Zhou & Buonomano 2024):
#                  U ~ N(stp_U_mean, stp_U_std^2) truncated [stp_U_min, stp_U_max];
#                  tau_x, tau_u ~ N(stp_tau_mean, stp_tau_std^2) truncated
#                  [stp_tau_min, stp_tau_max], independently per neuron.
SUPPORTED_STP_INIT = ("constant", "alternating", "random")


def _validate_choice(value: str, supported: tuple[str, ...], field: str, model_type: str) -> str:
    if value not in supported:
        raise ValueError(
            f"{model_type}: unknown {field}={value!r}; supported: {list(supported)}"
        )
    return value


def validate_gain_position(position: str, *, model_type: str = "") -> str:
    """Validate a ``gain_position`` value (see ``SUPPORTED_GAIN_POSITIONS``)."""
    return _validate_choice(position, SUPPORTED_GAIN_POSITIONS, "gain_position", model_type)


def validate_noise_position(position: str, *, model_type: str = "") -> str:
    """Validate a ``noise_position`` value (see ``SUPPORTED_NOISE_POSITIONS``)."""
    return _validate_choice(position, SUPPORTED_NOISE_POSITIONS, "noise_position", model_type)


def validate_stp_init(init: str, *, model_type: str = "") -> str:
    """Validate a ``stp_init`` value (see ``SUPPORTED_STP_INIT``)."""
    return _validate_choice(init, SUPPORTED_STP_INIT, "stp_init", model_type)


def _param_to_list(x: Any) -> Any:
    """Convert a per-neuron init array/tensor/scalar to a JSON-serializable form."""
    if x is None or isinstance(x, (int, float, bool, str)):
        return x
    if isinstance(x, (list, tuple)):
        return list(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.generic,)):
        return x.item()
    if hasattr(x, "tolist"):  # torch.Tensor without a hard import
        return x.tolist()
    return x


class GainRNNConfig(ConstrainedRNNConfig):
    """Base gain RNN config: CTRNN lineage + masks + per-neuron gain/bias.

    Args:
        gain_position: "outside" (default; r = g*phi(u+b), output gain /
            presynaptic column scaling) or "inside" (r = phi(g*u+b), input
            gain / slope modulation). See ``SUPPORTED_GAIN_POSITIONS``.
        gain_init: Initial per-neuron gains; scalar (broadcast) or (M,) array.
        bias_init: Initial per-neuron biases; scalar (broadcast) or (M,) array.
        freeze_gain: Freeze the gain parameters (Layer-1 freeze vocabulary).
        freeze_bias: Freeze the bias parameters.
        noise_position: "pre" (default; noise on the pre-activation, CTRNN
            behavior) or "post" (noise on the leaked blend before the final
            nonlinearity). See ``SUPPORTED_NOISE_POSITIONS``.
        positive_input_weights: If True, ReLU is applied to the input weights
            in the forward pass (Dale-style soft constraint).
        positive_output_weights: If True, ReLU is applied to the readout
            weights in the forward pass.
        activation_params: Optional dict of kwargs forwarded to
            ``neuralrnn.activations.get_activation`` (e.g. {"beta": 1.0} for
            softplus, {"r0": 20.0, "rmax": 100.0} for piecewise_tanh).
        h0_init: Initial value for h0 at construction; scalar or (M,) array.
        nonlinearity_mode: Family default "rate" (state is a current; the rate
            map reads it and the readout consumes rates). All three CTRNN modes
            are supported; gain placement composes with each.
        activation: Family default "softplus".
        All ConstrainedRNNConfig / CTRNNConfig arguments are inherited
        (rec_mask/in_mask/out_mask, dale, freeze_input/freeze_recurrent/
        freeze_output/freeze_h0, sigma_rec, noise_alpha_scaling, ...).
    """

    model_type = "gain_rnn"

    def __init__(
        self,
        gain_position: str = "outside",
        gain_init: float | Any = 1.0,
        bias_init: float | Any = 0.0,
        freeze_gain: bool = False,
        freeze_bias: bool = False,
        noise_position: str = "pre",
        positive_input_weights: bool = False,
        positive_output_weights: bool = False,
        activation_params: dict | None = None,
        h0_init: float | Any = 0.0,
        **kwargs,
    ) -> None:
        kwargs.setdefault("nonlinearity_mode", "rate")
        kwargs.setdefault("activation", "softplus")
        validate_gain_position(gain_position, model_type=self.model_type)
        validate_noise_position(noise_position, model_type=self.model_type)
        if activation_params is not None and not isinstance(activation_params, dict):
            raise TypeError(
                f"{self.model_type}: activation_params must be a dict or None, "
                f"got {type(activation_params).__name__}"
            )
        super().__init__(**kwargs)
        self.gain_position = gain_position
        self.gain_init = gain_init
        self.bias_init = bias_init
        self.freeze_gain = freeze_gain
        self.freeze_bias = freeze_bias
        self.noise_position = noise_position
        self.positive_input_weights = positive_input_weights
        self.positive_output_weights = positive_output_weights
        self.activation_params = activation_params
        self.h0_init = h0_init

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["gain_init"] = _param_to_list(d.get("gain_init"))
        d["bias_init"] = _param_to_list(d.get("bias_init"))
        d["h0_init"] = _param_to_list(d.get("h0_init"))
        return d


class StpRNNConfig(GainRNNConfig):
    """STP RNN config: short-term plasticity as a dynamic gain parameterization.

    The static gain/bias are frozen at identity by default and the effective
    presynaptic gain is the dynamic factor syn_x * syn_u (Tsodyks-Markram rate
    form, cell-specific: shared by all synapses of the same presynaptic neuron):

        syn_x' = syn_x + (dt/tau_x)(1 - syn_x) - dt_sec * syn_u * syn_x * r
        syn_u' = syn_u + (dt/tau_u)(U_eff - syn_u) + dt_sec * U_eff(1 - syn_u) r
        U_eff  = clamp(stp_alpha * U, 0, 1);  rec_in = r * syn_x' * syn_u'

    with dt in ms and dt_sec = dt/1000 (the release terms assume rates in
    spikes/s, following both reference implementations). STP dynamics always
    use the physical dt; they are not affected by an explicit alpha override,
    so a physical dt is required (the model raises if config.dt is None).

    Family defaults reproduce the notebook-11 model (Masse et al. 2019 style):
    post_blend + relu + noise_position="post" + noise_alpha_scaling +
    positive input/output weights + h0_init=0.1 + frozen static gain/bias.
    For Zhou & Buonomano 2024 use nonlinearity_mode="rate",
    stp_init="random", positive_output_weights=False, and set the
    neuromodulator per trial via ``model.set_stp_alpha(...)``.

    Args:
        stp_tau_x: Depression time constant (ms); scalar or (M,) array.
            Arrays take precedence over ``stp_init``.
        stp_tau_u: Facilitation time constant (ms); scalar or (M,) array.
        stp_U: Baseline release probability; scalar or (M,) array.
        stp_init: "constant" | "alternating" | "random"; see
            ``SUPPORTED_STP_INIT``. Ignored when arrays are passed directly.
        tau_x_fac, tau_u_fac, U_fac: Facilitating triple for "alternating"
            (defaults match notebook 11: 1500 ms, 200 ms, 0.15).
        tau_x_dep, tau_u_dep, U_dep: Depressing triple for "alternating"
            (200 ms, 1500 ms, 0.45).
        stp_U_mean, stp_U_std, stp_U_min, stp_U_max: Truncated-normal
            parameters for U in "random" mode (Zhou & Buonomano 2024).
        stp_tau_mean, stp_tau_std, stp_tau_min, stp_tau_max: Truncated-normal
            parameters (ms) for tau_x/tau_u in "random" mode.
        stp_seed: Seed for the "random" STP sampler (torch.Generator).
        stp_alpha: Neuromodulator factor scaling U (trial-level cue, never
            trained); scalar or (M,) array. Stored as a runtime buffer;
            ``save/load`` restores the buffer value from the checkpoint.
        freeze_stp: Freeze the STP parameters (tau_x, tau_u, U). Default True,
            matching both reference papers (STP constants are never trained).
        init_method: "default" (standard Linear init) or "gamma" (notebook-11
            gamma-distribution init: h2h E columns ~ Gamma(gamma_shape_exc,
            gamma_scale), I columns ~ Gamma(gamma_shape_inh, gamma_scale),
            input2h ~ Gamma(0.2, gamma_scale), readout E columns ~
            Gamma(gamma_shape_exc, gamma_scale)).
        gamma_shape_exc, gamma_shape_inh, gamma_scale: Gamma init parameters.
        init_seed: Seed for the gamma init numpy RNG.
        All GainRNNConfig arguments are inherited.
    """

    model_type = "stp_rnn"

    def __init__(
        self,
        stp_tau_x: float | Any = 200.0,
        stp_tau_u: float | Any = 1500.0,
        stp_U: float | Any = 0.2,
        stp_init: str = "constant",
        tau_x_fac: float = 1500.0,
        tau_u_fac: float = 200.0,
        U_fac: float = 0.15,
        tau_x_dep: float = 200.0,
        tau_u_dep: float = 1500.0,
        U_dep: float = 0.45,
        stp_U_mean: float = 0.5,
        stp_U_std: float = 0.17,
        stp_U_min: float = 0.001,
        stp_U_max: float = 0.99,
        stp_tau_mean: float = 1000.0,
        stp_tau_std: float = 330.0,
        stp_tau_min: float = 100.0,
        stp_tau_max: float = 3000.0,
        stp_seed: int | None = None,
        stp_alpha: float | Any = 1.0,
        freeze_stp: bool = True,
        init_method: str = "default",
        gamma_shape_exc: float = 0.1,
        gamma_shape_inh: float = 0.2,
        gamma_scale: float = 1.0,
        init_seed: int | None = None,
        **kwargs,
    ) -> None:
        # Family defaults = notebook-11 native behavior.
        kwargs.setdefault("nonlinearity_mode", "post_blend")
        kwargs.setdefault("activation", "relu")
        kwargs.setdefault("noise_position", "post")
        kwargs.setdefault("noise_alpha_scaling", True)
        kwargs.setdefault("positive_input_weights", True)
        kwargs.setdefault("positive_output_weights", True)
        kwargs.setdefault("freeze_gain", True)
        kwargs.setdefault("freeze_bias", True)
        kwargs.setdefault("h0_init", 0.1)
        kwargs.setdefault("dt", 10.0)
        kwargs.setdefault("tau", 100.0)
        validate_stp_init(stp_init, model_type=self.model_type)
        if init_method not in ("default", "gamma"):
            raise ValueError(
                f"{self.model_type}: unknown init_method={init_method!r}; "
                "supported: ['default', 'gamma']"
            )
        super().__init__(**kwargs)
        self.stp_tau_x = stp_tau_x
        self.stp_tau_u = stp_tau_u
        self.stp_U = stp_U
        self.stp_init = stp_init
        self.tau_x_fac = tau_x_fac
        self.tau_u_fac = tau_u_fac
        self.U_fac = U_fac
        self.tau_x_dep = tau_x_dep
        self.tau_u_dep = tau_u_dep
        self.U_dep = U_dep
        self.stp_U_mean = stp_U_mean
        self.stp_U_std = stp_U_std
        self.stp_U_min = stp_U_min
        self.stp_U_max = stp_U_max
        self.stp_tau_mean = stp_tau_mean
        self.stp_tau_std = stp_tau_std
        self.stp_tau_min = stp_tau_min
        self.stp_tau_max = stp_tau_max
        self.stp_seed = stp_seed
        self.stp_alpha = stp_alpha
        self.freeze_stp = freeze_stp
        self.init_method = init_method
        self.gamma_shape_exc = gamma_shape_exc
        self.gamma_shape_inh = gamma_shape_inh
        self.gamma_scale = gamma_scale
        self.init_seed = init_seed

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["stp_tau_x"] = _param_to_list(d.get("stp_tau_x"))
        d["stp_tau_u"] = _param_to_list(d.get("stp_tau_u"))
        d["stp_U"] = _param_to_list(d.get("stp_U"))
        d["stp_alpha"] = _param_to_list(d.get("stp_alpha"))
        return d
