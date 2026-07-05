"""Auto factory layer: dispatch configs and models by model_type (≈ transformers.models.auto)."""
from .configuration_auto import AutoConfig, CONFIG_REGISTRY, register_config
from .modeling_auto import AutoModel, MODEL_REGISTRY, register_model

__all__ = [
    "AutoConfig", "AutoModel",
    "register_config", "register_model",
    "CONFIG_REGISTRY", "MODEL_REGISTRY",
]
