"""Configuration system base class (≈ transformers.PretrainedConfig).

All model configs inherit NeuralRNNConfig. Design goals:
- config is the *single source of truth* for model structure and hyperparameters, serializable to config.json.
- Dispatch via the model_type field in the AutoConfig registry.

Porters note: move all construction hyperparameters of the paper model into the corresponding
<Family>Config subclass; do not hard-code any structural parameters in the model's __init__.
"""
from __future__ import annotations

import json
import os
import warnings
from typing import Any

CONFIG_FILE_NAME = "config.json"

# Nonlinearity placement in the Euler step of continuous-time (leaky) RNNs.
# With pre = W@state + B@x + b (noise added on pre):
#   "pre_activation": z' = (1-alpha)*z + alpha*f(pre)   (standard Euler; default)
#   "post_blend":     z' = f((1-alpha)*z + alpha*pre)   (nn-brain / Masse-style formula)
#   "rate":           r = f(z); z' = (1-alpha)*z + alpha*(W@r + B@x + b)
#                     (classic firing-rate form; state is a current, f maps it to the rate)
SUPPORTED_NONLINEARITY_MODES = ("pre_activation", "post_blend", "rate")


def validate_nonlinearity_mode(mode: str, *, model_type: str = "") -> str:
    """Validate a ``nonlinearity_mode`` value for CTRNN-lineage configs.

    Args:
        mode: One of ``SUPPORTED_NONLINEARITY_MODES``.
        model_type: Prefix for the error message.

    Returns:
        The validated mode (unchanged).

    Raises:
        ValueError: If ``mode`` is not one of the supported values.
    """
    if mode not in SUPPORTED_NONLINEARITY_MODES:
        raise ValueError(
            f"{model_type}: unknown nonlinearity_mode={mode!r}; "
            f"supported: {list(SUPPORTED_NONLINEARITY_MODES)}"
        )
    return mode


def resolve_euler_alpha(
    dt: float | None,
    tau: float,
    alpha: float | None,
    *,
    default_dt: float | None = None,
    model_type: str = "",
) -> tuple[float, float | None]:
    """Unified Euler-step resolution for continuous-time configs.

    Every continuous-time model family exposes the same three knobs —
    ``alpha`` (update fraction per step), ``dt`` (physical time step) and
    ``tau`` (time constant) — resolved with a single deterministic priority:

    1. ``alpha`` explicitly given            -> use it directly (highest priority).
       If ``dt`` is also given and ``alpha != dt/tau``, a warning is emitted
       and ``alpha`` wins.
    2. ``alpha=None`` and ``dt`` given       -> ``alpha = dt / tau``.
    3. both None                             -> family default: ``dt = default_dt``
       and ``alpha = default_dt / tau``; with ``default_dt=None`` the model is
       fully discrete, ``alpha = 1.0``.

    Args:
        dt: Physical time step, or None.
        tau: Time constant (must be > 0).
        alpha: Explicit Euler update fraction, or None.
        default_dt: Family-specific fallback for ``dt`` when neither ``dt`` nor
            ``alpha`` is given; None means the family default is the discrete
            update (alpha = 1.0).
        model_type: Prefix for warning messages.

    Returns:
        (alpha, effective_dt): the resolved update fraction and the effective
        ``dt`` to store on the config (None when the model runs discrete).

    Raises:
        ValueError: If ``tau <= 0`` or the resolved/explicit ``alpha <= 0``.
    """
    if tau is None or tau <= 0:
        raise ValueError(f"{model_type}: tau must be a positive number, got {tau}")

    if alpha is not None and dt is not None:
        if abs(alpha - dt / tau) > 1e-6:
            warnings.warn(
                f"{model_type}: both alpha={alpha} and dt/tau={dt / tau:.6g} were given "
                f"and differ; alpha takes precedence (priority: alpha > dt/tau).",
                UserWarning,
                stacklevel=3,
            )
    elif alpha is None:
        if dt is None:
            dt = default_dt
        alpha = (dt / tau) if dt is not None else 1.0

    if alpha <= 0:
        raise ValueError(f"{model_type}: alpha must be > 0, got {alpha}")
    if alpha > 1:
        warnings.warn(
            f"{model_type}: alpha={alpha} > 1; explicit Euler integration may be unstable.",
            UserWarning,
            stacklevel=3,
        )
    return float(alpha), dt


class NeuralRNNConfig:
    """Base class for all dynamical-system model configs.

    Public fields correspond to the dimensions of the dynamical-system triple (F, G, initial value)
    in ARCHITECTURE §2.1:
        input_dim  : external input dimension K (0 if no input)
        latent_dim : latent state dimension M
        output_dim : readout dimension (usually == latent_dim in DSR)
        dt         : discretization step for continuous-time models (None for discrete models)
        activation : nonlinearity name. Supported names are defined in
            ``neuralrnn.activations.SUPPORTED_ACTIVATIONS`` (relu, tanh, sigmoid,
            softplus, leaky_relu/leakyrelu, elu, selu, gelu, silu/swish).
    Subclasses only need to add their own fields after super().__init__(...).
    """

    model_type: str = ""  # subclasses must set this; globally unique registry key, e.g. "shallow_plrnn"

    def __init__(
        self,
        input_dim: int = 0,
        latent_dim: int = 0,
        output_dim: int = 0,
        dt: float | None = None,
        activation: str = "relu",
        freeze_input: bool = False,
        freeze_recurrent: bool = False,
        freeze_output: bool = False,
        freeze_h0: bool = False,
        **kwargs: Any,
    ) -> None:
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.dt = dt
        self.activation = activation
        self.freeze_input = freeze_input
        self.freeze_recurrent = freeze_recurrent
        self.freeze_output = freeze_output
        self.freeze_h0 = freeze_h0
        # Forward unknown fields for forward compatibility (extra fields in old checkpoints do not error)
        for k, v in kwargs.items():
            setattr(self, k, v)

    # ---------- Serialization ----------
    def to_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        d["model_type"] = self.model_type
        return d

    def to_json_string(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False, sort_keys=True)

    def to_json_file(self, save_directory: str) -> str:
        os.makedirs(save_directory, exist_ok=True)
        path = os.path.join(save_directory, CONFIG_FILE_NAME)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json_string())
        return path

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "NeuralRNNConfig":
        d = dict(d)
        d.pop("model_type", None)  # carried by the concrete subclass itself
        return cls(**d)

    @classmethod
    def from_json_file(cls, json_file: str) -> "NeuralRNNConfig":
        with open(json_file, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def from_pretrained(cls, path: str) -> "NeuralRNNConfig":
        """Read from a directory or a config.json path. When calling on the base class, use AutoConfig so that
        model_type is dispatched to the correct subclass."""
        path = os.fspath(path)
        json_file = path if path.endswith(".json") else os.path.join(path, CONFIG_FILE_NAME)
        return cls.from_json_file(json_file)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.to_json_string()})"
