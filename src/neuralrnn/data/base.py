"""数据层基类与张量约定。

统一 batch 格式（ARCHITECTURE §3.1），全部 batch-first：
  范式A(任务)   : {"inputs":(B,T,input_dim), "targets":(B,T)|(B,T,output_dim), "mask":(B,T)}
  范式B(重构)   : {"inputs":(B,T,N), "targets":(B,T,N), "external_inputs":(B,T,K)|None}
  行为          : {"action":..., "reward":..., "stage2":..., "mask":...}

移植者注意（契约 B）：多数数据只需写一个 loader 把原始文件读成上述 dict，
然后复用下面四个数据集类之一，并在 data/registry.py 登记 URL。
"""
from __future__ import annotations

from typing import Any, Iterator

import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """所有数据集的基类。子类实现 __len__/__getitem__ 返回标准 batch dict，
    或实现 sample_batch() 进行随机子序列采样（DSR 常用）。"""

    kind: str = ""            # "neurogym" / "timeseries" / "behavioral" / "trajectory"
    input_dim: int = 0
    output_dim: int = 0

    # 归一化器（z-score / min-max），分析时反变换用；无则 None
    normalizer: Any = None

    def sample_batch(self) -> dict[str, torch.Tensor]:
        """随机采一个 batch。默认走 __getitem__ + 简单堆叠；DSR 子类覆盖。"""
        raise NotImplementedError

    def __iter__(self) -> Iterator[dict]:
        # 让 next(iter(ds)) 可用：默认无限随机批（DSR 风格）
        while True:
            yield self.sample_batch()


class StandardScaler:
    """简单 z-score 归一化器，存均值/方差，支持反变换。"""
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, x: torch.Tensor) -> "StandardScaler":
        self.mean_ = x.mean(dim=0, keepdim=True)
        self.std_ = x.std(dim=0, keepdim=True).clamp_min(1e-8)
        return self

    def transform(self, x): return (x - self.mean_) / self.std_
    def inverse_transform(self, x): return x * self.std_ + self.mean_
    def fit_transform(self, x): return self.fit(x).transform(x)
