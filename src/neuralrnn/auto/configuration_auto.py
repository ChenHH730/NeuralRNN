"""AutoConfig + 配置注册表（≈ transformers.AutoConfig）。

按 config.json 中的 model_type 字段，分发到正确的 <Family>Config 子类。
"""
from __future__ import annotations

import json
import os

from ..configuration_utils import NeuralRNNConfig, CONFIG_FILE_NAME

# model_type(str) -> Config 子类。由 register_model（见 modeling_auto）或手动填充。
CONFIG_REGISTRY: dict[str, type[NeuralRNNConfig]] = {}


def register_config(model_type: str):
    def deco(cls: type[NeuralRNNConfig]):
        CONFIG_REGISTRY[model_type] = cls
        return cls
    return deco


def _read_model_type(path: str) -> str:
    json_file = path if path.endswith(".json") else os.path.join(path, CONFIG_FILE_NAME)
    with open(json_file, "r", encoding="utf-8") as f:
        return json.load(f)["model_type"]


def _ensure_config_loaded(model_type: str) -> None:
    if model_type in CONFIG_REGISTRY:
        return
    # 触发对应模型模块导入，使其 config 被注册（与 modeling_auto 的懒加载共用映射）
    from .modeling_auto import _ensure_loaded
    _ensure_loaded(model_type)


class AutoConfig:
    """配置工厂。"""

    @staticmethod
    def for_model(model_type: str, **kwargs) -> NeuralRNNConfig:
        """按 model_type 构造一个新配置（带覆盖参数）。"""
        _ensure_config_loaded(model_type)
        return CONFIG_REGISTRY[model_type](**kwargs)

    @staticmethod
    def from_pretrained(path: str) -> NeuralRNNConfig:
        model_type = _read_model_type(path)
        _ensure_config_loaded(model_type)
        return CONFIG_REGISTRY[model_type].from_pretrained(path)
