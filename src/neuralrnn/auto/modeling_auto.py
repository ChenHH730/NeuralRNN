"""AutoModel + 模型注册表（≈ transformers.AutoModel）。

新模型接入只需在其 modeling 文件里：
    @register_model("my_family")
    class MyModel(NeuralDynamicsModel):
        config_class = MyConfig
        ...
注册后全局可通过 AutoModel.from_config / from_pretrained 实例化。
"""
from __future__ import annotations

import importlib
import os

from ..modeling_utils import NeuralDynamicsModel
from .configuration_auto import AutoConfig, CONFIG_REGISTRY

# model_type(str) -> 模型类
MODEL_REGISTRY: dict[str, type[NeuralDynamicsModel]] = {}

# Lazy modules mapping：model_type -> "模块路径"。避免导入时把所有重依赖模型一次性加载。
# 移植新模型后在此登记一行即可（也可在模型模块被导入时由 @register_model 直接填充）。
_LAZY_MODULES: dict[str, str] = {
    "ctrnn": "neuralrnn.models.ctrnn.modeling_ctrnn",
    "ei_rnn": "neuralrnn.models.ctrnn.modeling_ctrnn",
    "shallow_plrnn": "neuralrnn.models.plrnn.modeling_plrnn",
    "dend_plrnn": "neuralrnn.models.plrnn.modeling_plrnn",
    "alrnn": "neuralrnn.models.plrnn.modeling_plrnn",
    "latent_circuit": "neuralrnn.models.latent_circuit.modeling_latent_circuit",
    "tiny_rnn": "neuralrnn.models.tiny_rnn.modeling_tiny_rnn",
    # 移植后追加：
    # "lowrank_rnn": "neuralrnn.models.lowrank.modeling_lowrank",
    # "lfads": "neuralrnn.models.lfads.modeling_lfads",
}


def register_model(model_type: str):
    """装饰器：把模型类登记到 MODEL_REGISTRY，并把其 config 类登记到 CONFIG_REGISTRY。"""
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
            f"未知 model_type='{model_type}'。已注册: {sorted(set(MODEL_REGISTRY) | set(_LAZY_MODULES))}。"
            f" 若是新移植模型，请在 _LAZY_MODULES 登记或确保其模块已被 import。")
    importlib.import_module(module_path)  # 触发 @register_model 填充
    if model_type not in MODEL_REGISTRY:
        raise KeyError(f"模块 {module_path} 未注册 model_type='{model_type}'，检查 @register_model 装饰。")


class AutoModel:
    """按 model_type 分发的模型工厂。"""

    @staticmethod
    def from_config(config) -> NeuralDynamicsModel:
        _ensure_loaded(config.model_type)
        return MODEL_REGISTRY[config.model_type](config)

    @staticmethod
    def from_pretrained(path: str, *, map_location: str = "cpu") -> NeuralDynamicsModel:
        config = AutoConfig.from_pretrained(path)
        _ensure_loaded(config.model_type)
        cls = MODEL_REGISTRY[config.model_type]
        return cls.from_pretrained(path, map_location=map_location)
