"""Neurogym 任务数据集（范式 A / 任务优化）。

移植自 RNN_DynamicalSystemAnalysis.ipynb 的取数方式：用 neurogym 构造认知任务
环境，包成 PyTorch dataloader，按 batch 产出 (inputs, targets)。这里在边界统一
转成 batch-first 的标准 batch dict（ARCHITECTURE §3.1 范式A）。

neurogym 为可选重依赖（见 pyproject [project.optional-dependencies] neurogym），
未安装时给出清晰提示。
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .base import BaseDataset


class NeurogymDataset(BaseDataset):
    kind = "neurogym"

    def __init__(self, env, dataset, input_dim: int, output_dim: int,
                 batch_size: int = 16, seq_len: int = 100):
        self.env = env                 # 保留底层 env，分析时取任务相关输入（如 0-coherence 均值输入）
        self._dataset = dataset        # neurogym Dataset（可调用，返回 (inputs, target) time-first）
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.seq_len = seq_len

    @classmethod
    def from_task(cls, task: str, *, batch_size: int = 16, seq_len: int = 100,
                  dt: int = 100, timing: dict | None = None, **env_kwargs: Any) -> "NeurogymDataset":
        """用任务名构造数据集，例如 task='PerceptualDecisionMaking-v0'。

        被 data/registry.py 的 load_dataset('perceptual_decision_making') 调用。
        额外 kwargs 透传给 neurogym 环境（dt / timing / 任务专属参数）。
        """
        try:
            import neurogym as ngym
        except ImportError as e:
            raise ImportError(
                "需要 neurogym：pip install 'neuralrnn[neurogym]' 或 pip install neurogym"
            ) from e

        kwargs = dict(env_kwargs)
        if timing is not None:
            kwargs["timing"] = timing
        dataset = ngym.Dataset(task, env_kwargs={"dt": dt, **kwargs},
                               batch_size=batch_size, seq_len=seq_len)
        env = dataset.env
        input_dim = env.observation_space.shape[0]
        # 分类任务用 n 类（CrossEntropy 目标）；回归任务用维度
        output_dim = int(getattr(env.action_space, "n", None)
                         or env.action_space.shape[0])
        return cls(env, dataset, input_dim, output_dim,
                   batch_size=batch_size, seq_len=seq_len)

    def sample_batch(self) -> dict[str, torch.Tensor]:
        # neurogym Dataset() 返回 time-first: inputs (T,B,obs), target (T,B)
        inputs, target = self._dataset()
        inputs = torch.as_tensor(inputs, dtype=torch.float32).permute(1, 0, 2)  # -> (B,T,obs)
        target = torch.as_tensor(np.asarray(target), dtype=torch.long).permute(1, 0)  # -> (B,T)
        return {"inputs": inputs, "targets": target, "mask": None}

    def task_input(self, kind: str = "stimulus") -> torch.Tensor:
        """返回用于不动点分析的"任务条件输入"（如决策任务的 0-coherence 均值输入）。
        默认返回零输入；具体任务在移植时按需覆盖（见 PORTING_GUIDE 配方1）。"""
        return torch.zeros(self.input_dim, dtype=torch.float32)
