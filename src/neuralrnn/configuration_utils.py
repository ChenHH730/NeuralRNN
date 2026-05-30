"""配置系统基类（≈ transformers.PretrainedConfig）。

所有模型的配置都继承 NeuralRNNConfig。设计目标：
- config 是模型结构与超参的*单一真相源*，可序列化为 config.json。
- 通过 model_type 字段在 AutoConfig 注册表中分发。

移植者注意：把论文模型的所有构造超参都搬进对应的 <Family>Config 子类，
不要在模型 __init__ 里硬编码任何结构参数。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict, fields
from typing import Any

CONFIG_FILE_NAME = "config.json"


class NeuralRNNConfig:
    """所有动力系统模型配置的基类。

    公共字段对应 ARCHITECTURE §2.1 的动力系统三元组 (F, G, 初值) 的维度：
        input_dim  : 外部输入维度 K（无输入则 0）
        latent_dim : 潜状态维度 M
        output_dim : 读出维度（DSR 中通常 == latent_dim）
        dt         : 连续时间模型的离散步长（离散模型为 None）
        activation : 非线性名称
    子类只需在 __init__ 中 super().__init__(...) 后追加自己的字段。
    """

    model_type: str = ""  # 子类必填，全局唯一注册键，如 "shallow_plrnn"

    def __init__(
        self,
        input_dim: int = 0,
        latent_dim: int = 0,
        output_dim: int = 0,
        dt: float | None = None,
        activation: str = "relu",
        **kwargs: Any,
    ) -> None:
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.dt = dt
        self.activation = activation
        # 透传未知字段，保证向前兼容（旧 checkpoint 多出的字段不报错）
        for k, v in kwargs.items():
            setattr(self, k, v)

    # ---------- 序列化 ----------
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
        d.pop("model_type", None)  # 由具体子类自带
        return cls(**d)

    @classmethod
    def from_json_file(cls, json_file: str) -> "NeuralRNNConfig":
        with open(json_file, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def from_pretrained(cls, path: str) -> "NeuralRNNConfig":
        """从目录或 config.json 路径读取。若在基类上调用，请用 AutoConfig 以按
        model_type 分发到正确子类。"""
        json_file = path if path.endswith(".json") else os.path.join(path, CONFIG_FILE_NAME)
        return cls.from_json_file(json_file)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.to_json_string()})"
