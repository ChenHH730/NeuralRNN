"""AutoModel + model registry (≈ transformers.AutoModel).

To add a new model, just write in its modeling file:
    @register_model("my_family")
    class MyModel(NeuralDynamicsModel):
        config_class = MyConfig
        ...
After registration it can be instantiated globally via AutoModel.from_config / from_pretrained.
"""
from __future__ import annotations

import importlib
import os

from ..modeling_utils import NeuralDynamicsModel
from .configuration_auto import AutoConfig, CONFIG_REGISTRY

# model_type(str) -> model class
MODEL_REGISTRY: dict[str, type[NeuralDynamicsModel]] = {}

# Lazy module mapping: model_type -> "module.path". Avoids loading all heavy-dependency models at import time.
# Add one line here after porting a new model (or let the model module fill it directly via @register_model).
_LAZY_MODULES: dict[str, str] = {
    "ctrnn": "neuralrnn.models.ctrnn.modeling_ctrnn",
    "ei_rnn": "neuralrnn.models.ctrnn.modeling_ctrnn",
    "shallow_plrnn": "neuralrnn.models.plrnn.modeling_plrnn",
    "dend_plrnn": "neuralrnn.models.plrnn.modeling_plrnn",
    "alrnn": "neuralrnn.models.plrnn.modeling_plrnn",
    "latent_circuit": "neuralrnn.models.latent_circuit.modeling_latent_circuit",
    "tiny_rnn": "neuralrnn.models.tiny_rnn.modeling_tiny_rnn",
    "lowrank_rnn": "neuralrnn.models.lowrank.modeling_lowrank",
    # Add after porting:
    # "lfads": "neuralrnn.models.lfads.modeling_lfads",
}


def register_model(model_type: str):
    """Decorator: register a model class in MODEL_REGISTRY and its config class in CONFIG_REGISTRY."""
    def deco(cls: type[NeuralDynamicsModel]):
        MODEL_REGISTRY[model_type] = cls
        if getattr(cls, "config_class", None) is not None:
            CONFIG_REGISTRY[model_type] = cls.config_class
        return cls
    return deco


def _ensure_loaded(model_type: str) -> None:
    if model_type in MODEL_REGISTRY:
        return
    module_path = _LAZY_MODULES.get(model_type)
    if module_path is None:
        raise KeyError(
            f"Unknown model_type='{model_type}'. Registered: {sorted(set(MODEL_REGISTRY) | set(_LAZY_MODULES))}."
            f" If this is a newly ported model, register it in _LAZY_MODULES or ensure its module is imported.")
    importlib.import_module(module_path)  # trigger @register_model population
    if model_type not in MODEL_REGISTRY:
        raise KeyError(f"Module {module_path} did not register model_type='{model_type}'; check the @register_model decorator.")


class AutoModel:
    """Model factory dispatched by model_type."""

    @staticmethod
    def from_config(config) -> NeuralDynamicsModel:
        _ensure_loaded(config.model_type)
        return MODEL_REGISTRY[config.model_type](config)

    @staticmethod
    def from_pretrained(path: str, *, map_location: str = "cpu") -> NeuralDynamicsModel:
        path = os.fspath(path)
        config = AutoConfig.from_pretrained(path)
        _ensure_loaded(config.model_type)
        cls = MODEL_REGISTRY[config.model_type]
        return cls.from_pretrained(path, map_location=map_location)
