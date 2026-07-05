"""Data-layer base class and tensor conventions.

Unify batch format (ARCHITECTURE §3.1), all batch-first:
  Paradigm A (task)   : {"inputs":(B,T,input_dim), "targets":(B,T)|(B,T,output_dim), "mask":(B,T)}
  Paradigm B (reconstruction) : {"inputs":(B,T,N), "targets":(B,T,N), "external_inputs":(B,T,K)|None}
  Behavioral          : {"action":..., "reward":..., "stage2":..., "mask":...}

Porters note (contract B): most datasets only need a loader that reads raw files into the dict above,
then reuse one of the four dataset classes below and register the URL in data/registry.py.
"""
from __future__ import annotations

from typing import Any, Iterator

import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """Base class for all datasets. Subclasses implement __len__/__getitem__ returning a standard batch dict,
    or implement sample_batch() for random subsequence sampling (common in DSR)."""

    kind: str = ""            # "neurogym" / "timeseries" / "behavioral" / "trajectory"
    input_dim: int = 0
    output_dim: int = 0

    # Normalizer (z-score / min-max); used for inverse transform during analysis; None if absent
    normalizer: Any = None

    def sample_batch(self) -> dict[str, torch.Tensor]:
        """Randomly sample a batch. Default uses __getitem__ + simple stacking; DSR subclasses override."""
        raise NotImplementedError

    def __iter__(self) -> Iterator[dict]:
        # Make next(iter(ds)) available: default infinite random batches (DSR style)
        while True:
            yield self.sample_batch()


class StandardScaler:
    """Simple z-score normalizer storing mean/std and supporting inverse transform."""
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
