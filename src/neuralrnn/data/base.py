"""Data-layer base class and tensor conventions.

Unify batch format (ARCHITECTURE §3.1), all batch-first:
  Paradigm A (task)   : {"inputs":(B,T,input_dim), "targets":(B,T)|(B,T,output_dim), "mask":(B,T)}
  Paradigm B (reconstruction) : {"inputs":(B,T,N), "targets":(B,T,N), "external_inputs":(B,T,K)|None}
  Behavioral          : {"action":..., "reward":..., "stage2":..., "mask":...}

Porters note (contract B): most datasets only need a loader that reads raw files into the dict above,
then reuse one of the four dataset classes below and register the URL in data/registry.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

import torch
from torch.utils.data import Dataset


@dataclass
class Trials:
    """A small set of complete trials (returned by ``BaseDataset.sample_trials``).

    Same field layout as a trial-aligned dataset: ``inputs`` / ``targets`` /
    ``mask`` are zero-padded tensors and ``conditions`` is a list of per-trial
    dicts (with the unified ``epochs`` / ``n_steps`` / ``is_catch`` keys).
    """

    inputs: torch.Tensor
    targets: torch.Tensor
    mask: torch.Tensor | None
    conditions: list

    def __len__(self) -> int:
        return len(self.conditions)


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

    def sample_trials(self, n: int, seed: int | None = None) -> Trials:
        """Return n complete trials as a ``Trials`` object (for visualization/analysis
        without creating a second dataset). Implemented by trial-capable subclasses."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support sample_trials()."
        )

    def __iter__(self) -> Iterator[dict]:
        # Make next(iter(ds)) available: default infinite random batches (DSR style)
        while True:
            yield self.sample_batch()


def subset_trials(inputs: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor | None,
                  conditions: list, n: int, seed: int | None = None) -> Trials:
    """Select n trials from trial-aligned tensors -> Trials.

    seed=None takes the first n trials (deterministic); with a seed, draws a
    seeded random subset (without replacement when n <= N, else with).
    """
    n_total = inputs.shape[0]
    if seed is None:
        idx = torch.arange(min(n, n_total))
    else:
        g = torch.Generator().manual_seed(seed)
        if n <= n_total:
            idx = torch.randperm(n_total, generator=g)[:n]
        else:
            idx = torch.randint(0, n_total, (n,), generator=g)
    return Trials(
        inputs[idx], targets[idx],
        mask[idx] if mask is not None else None,
        [conditions[i] for i in idx.tolist()],
    )


class StandardScaler:
    """Simple z-score normalizer storing mean/std and supporting inverse transform."""
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, x: torch.Tensor) -> "StandardScaler":
        """Compute per-feature mean/std over dim 0. x: (N, ...) -> stats (1, ...)."""
        self.mean_ = x.mean(dim=0, keepdim=True)
        self.std_ = x.std(dim=0, keepdim=True).clamp_min(1e-8)
        return self

    def transform(self, x):
        """z-score normalize with the fitted statistics (broadcast over dim 0)."""
        return (x - self.mean_) / self.std_

    def inverse_transform(self, x):
        """Undo :meth:`transform`."""
        return x * self.std_ + self.mean_

    def fit_transform(self, x):
        """Fit on ``x`` then return its normalized version."""
        return self.fit(x).transform(x)
