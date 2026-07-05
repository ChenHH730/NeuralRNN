"""AutoConfig + config registry (≈ transformers.AutoConfig).

Dispatch to the correct <Family>Config subclass based on the model_type field in config.json.
"""
from __future__ import annotations

import json
import os

from ..configuration_utils import NeuralRNNConfig, CONFIG_FILE_NAME

# model_type(str) -> Config subclass. Populated by register_model (see modeling_auto) or manually.
CONFIG_REGISTRY: dict[str, type[NeuralRNNConfig]] = {}


def register_config(model_type: str):
    def deco(cls: type[NeuralRNNConfig]):
        CONFIG_REGISTRY[model_type] = cls
        return cls
    return deco


def _read_model_type(path: str) -> str:
    path = os.fspath(path)
    json_file = path if path.endswith(".json") else os.path.join(path, CONFIG_FILE_NAME)
    with open(json_file, "r", encoding="utf-8") as f:
        return json.load(f)["model_type"]


def _ensure_config_loaded(model_type: str) -> None:
    if model_type in CONFIG_REGISTRY:
        return
    # Trigger import of the corresponding model module so its config gets registered
    # (shares the lazy mapping with modeling_auto)
    from .modeling_auto import _ensure_loaded
    _ensure_loaded(model_type)


class AutoConfig:
    """Config factory."""

    @staticmethod
    def for_model(model_type: str, **kwargs) -> NeuralRNNConfig:
        """Build a new config by model_type (with optional overrides)."""
        _ensure_config_loaded(model_type)
        return CONFIG_REGISTRY[model_type](**kwargs)

    @staticmethod
    def from_pretrained(path: str) -> NeuralRNNConfig:
        path = os.fspath(path)
        model_type = _read_model_type(path)
        _ensure_config_loaded(model_type)
        return CONFIG_REGISTRY[model_type].from_pretrained(path)
