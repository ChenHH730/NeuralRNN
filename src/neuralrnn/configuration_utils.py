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
from dataclasses import dataclass, field, asdict, fields
from typing import Any

CONFIG_FILE_NAME = "config.json"


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
