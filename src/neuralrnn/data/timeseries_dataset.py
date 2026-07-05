"""Time-series dataset (Paradigm B / DSR).

Ported from TimeSeriesDataset in CNS2023_tutorial.ipynb: slice (T×N) observations into subsequences
of length sequence_length; targets are inputs shifted right by one step; optional external inputs (T×K).

Note: the original sample_batch returned time-first; here we convert to batch-first at the boundary
(ARCHITECTURE §2.2) and pack into the standard batch dict.
"""
from __future__ import annotations

from random import randint

import numpy as np
import torch

from .base import BaseDataset, StandardScaler


class TimeSeriesDataset(BaseDataset):
    kind = "timeseries"

    def __init__(self, data, external_inputs=None, sequence_length: int = 200,
                 batch_size: int = 16, normalize: bool = False, test: np.ndarray | None = None,
                 dt: float | None = None):
        X = torch.as_tensor(np.asarray(data), dtype=torch.float32)
        self.normalizer = StandardScaler().fit(X) if normalize else None
        self.X = self.normalizer.transform(X) if self.normalizer else X
        self.T, self.N = self.X.shape
        self.input_dim = self.output_dim = self.N
        self.dim = self.N
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.dt = dt
        self.S = None if external_inputs is None else torch.as_tensor(
            np.asarray(external_inputs), dtype=torch.float32)
        if self.S is not None:
            assert self.S.shape[0] == self.T, "external_inputs must match the number of time steps in data"
        # Test set (for evaluation/generation); not sliced into batches
        self.test = torch.as_tensor(np.asarray(test), dtype=torch.float32) if test is not None else None

    def __len__(self):
        return max(self.T - self.sequence_length - 1, 0)

    def _slice(self, t: int):
        x = self.X[t:t + self.sequence_length]
        y = self.X[t + 1:t + self.sequence_length + 1]
        s = None if self.S is None else self.S[t:t + self.sequence_length]
        return x, y, s

    def sample_batch(self) -> dict:
        xs, ys, ss = [], [], []
        for _ in range(self.batch_size):
            x, y, s = self._slice(randint(0, len(self) - 1))
            xs.append(x); ys.append(y)
            if s is not None:
                ss.append(s)
        batch = {
            "inputs": torch.stack(xs),     # (B,T,N) batch-first
            "targets": torch.stack(ys),    # (B,T,N)
            "external_inputs": torch.stack(ss) if ss else None,
        }
        return batch

    @classmethod
    def from_npy(cls, train_path, test_path=None, dt=None, **kwargs) -> "TimeSeriesDataset":
        """Registry loader entry point: construct from .npy files (for the lorenz63 dataset)."""
        train = np.load(train_path).astype(np.float32)
        test = np.load(test_path).astype(np.float32) if test_path else None
        return cls(train, test=test, dt=dt, **kwargs)
