"""Trial-aware time-series dataset for Paradigm B task-state reconstruction.

Unlike TimeSeriesDataset, which slices a single long (T, N) trajectory, this class
preserves the trial structure `(n_trials, trial_length, n_variable)`. This is
essential when each trial is an independent sequence (e.g. task-driven neural
activity) and sliding windows must not cross trial boundaries.
"""
from __future__ import annotations

from random import randint

import numpy as np
import torch

from .base import BaseDataset, StandardScaler


def _to_tensor(x) -> torch.Tensor | None:
    """Convert input to float32 torch.Tensor."""
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.float()
    return torch.as_tensor(np.asarray(x), dtype=torch.float32)


class TrialTimeseriesDataset(BaseDataset):
    """Dataset for trial-structured time series.

    Accepts 3D arrays of shape `(n_trials, trial_length, n_variable)`. Each trial
    is treated as an independent sequence. Targets default to inputs shifted right
    by one step within each trial. Train/test splits are performed trial-wise.

    Batch format (batch-first):
        {"inputs": (B, T, N), "targets": (B, T, N), "external_inputs": (B, T, K)|None}
    """

    kind = "trial_timeseries"

    def __init__(
        self,
        inputs: np.ndarray | torch.Tensor,
        targets: np.ndarray | torch.Tensor | None = None,
        external_inputs: np.ndarray | torch.Tensor | None = None,
        batch_size: int = 16,
        normalize: bool = False,
        normalize_externals: bool = False,
        test_fraction: float = 0.0,
        seed: int = 0,
    ) -> None:
        X = _to_tensor(inputs)
        Y = _to_tensor(targets)
        S = _to_tensor(external_inputs)

        if X.ndim != 3:
            raise ValueError(
                f"TrialTimeseriesDataset expects 3D inputs (n_trials, trial_length, n_variable), "
                f"got ndim={X.ndim}"
            )
        n_trials, trial_len, n_var = X.shape
        self.n_trials = n_trials
        self.trial_length = trial_len
        self.input_dim = n_var

        if Y is not None and Y.shape != X.shape:
            raise ValueError(
                f"targets shape {tuple(Y.shape)} must match inputs shape {tuple(X.shape)}"
            )
        if S is not None and S.shape[:2] != X.shape[:2]:
            raise ValueError(
                f"external_inputs shape {tuple(S.shape[:2])} must match inputs trial shape "
                f"{tuple(X.shape[:2])}"
            )

        # Train/test split (trial-wise)
        self._test_inputs = None
        self._test_targets = None
        self._test_external_inputs = None
        if test_fraction > 0.0:
            rng = torch.Generator().manual_seed(seed)
            n_test = max(1, int(n_trials * test_fraction))
            perm = torch.randperm(n_trials, generator=rng)
            test_idx = perm[:n_test].sort()[0]
            train_idx = perm[n_test:].sort()[0]

            self._test_inputs = X[test_idx]
            self._test_targets = Y[test_idx] if Y is not None else None
            self._test_external_inputs = S[test_idx] if S is not None else None

            X = X[train_idx]
            if Y is not None:
                Y = Y[train_idx]
            if S is not None:
                S = S[train_idx]

        # Normalize across all trials/time steps
        self.normalizer = StandardScaler().fit(X.reshape(-1, n_var)) if normalize else None
        if self.normalizer is not None:
            X = self.normalizer.transform(X.reshape(-1, n_var)).reshape(X.shape)
            if self._test_inputs is not None:
                self._test_inputs = self.normalizer.transform(
                    self._test_inputs.reshape(-1, n_var)
                ).reshape(self._test_inputs.shape)

        self.external_normalizer = None
        if normalize and normalize_externals and S is not None:
            ext_dim = S.shape[-1]
            self.external_normalizer = StandardScaler().fit(S.reshape(-1, ext_dim))
            S = self.external_normalizer.transform(S.reshape(-1, ext_dim)).reshape(S.shape)
            if self._test_external_inputs is not None:
                self._test_external_inputs = self.external_normalizer.transform(
                    self._test_external_inputs.reshape(-1, ext_dim)
                ).reshape(self._test_external_inputs.shape)

        self.X = X
        self.Y = Y if Y is not None else None
        self.S = S
        self.output_dim = self.input_dim
        self.batch_size = batch_size

        # Default targets: inputs shifted right by one step within each trial
        if self.Y is None:
            self.Y = torch.cat([self.X[:, 1:], self.X[:, -1:]], dim=1)

    def __len__(self) -> int:
        return self.X.shape[0]

    def _slice(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        x = self.X[idx]
        y = self.Y[idx]
        s = self.S[idx] if self.S is not None else None
        return x, y, s

    def sample_batch(self) -> dict[str, torch.Tensor]:
        """Sample a random batch of whole trials.

        Returns:
            {"inputs": (B, T, N), "targets": (B, T, N), "external_inputs": (B, T, K)|None}
        """
        xs, ys, extras = [], [], []
        n = len(self)
        for _ in range(self.batch_size):
            idx = randint(0, n - 1)
            x, y, s = self._slice(idx)
            xs.append(x)
            ys.append(y)
            if s is not None:
                extras.append(s)

        return {
            "inputs": torch.stack(xs),
            "targets": torch.stack(ys),
            "external_inputs": torch.stack(extras) if extras else None,
        }

    @property
    def test_set(self) -> TrialTimeseriesDataset | None:
        """The held-out test fraction as a separate TrialTimeseriesDataset."""
        if self._test_inputs is None:
            return None
        ds = TrialTimeseriesDataset.__new__(TrialTimeseriesDataset)
        ds.kind = self.kind
        ds.normalizer = self.normalizer
        ds.external_normalizer = self.external_normalizer
        ds.X = self._test_inputs
        ds.Y = self._test_targets if self._test_targets is not None else (
            torch.cat([ds.X[:, 1:], ds.X[:, -1:]], dim=1)
        )
        ds.S = self._test_external_inputs
        ds.n_trials = ds.X.shape[0]
        ds.trial_length = ds.X.shape[1]
        ds.input_dim = ds.X.shape[2]
        ds.output_dim = ds.input_dim
        ds.batch_size = self.batch_size
        ds._test_inputs = None
        ds._test_targets = None
        ds._test_external_inputs = None
        return ds

    @property
    def test(self) -> torch.Tensor | None:
        """Full test trajectory flattened across trials (convenience alias)."""
        if self._test_inputs is not None:
            return self._test_inputs.reshape(-1, self.input_dim)
        return None

    @classmethod
    def from_arrays(
        cls,
        inputs: np.ndarray | torch.Tensor,
        targets: np.ndarray | torch.Tensor | None = None,
        external_inputs: np.ndarray | torch.Tensor | None = None,
        normalize_externals: bool = False,
        **kwargs,
    ) -> TrialTimeseriesDataset:
        """Convenience constructor from numpy arrays or torch tensors.

        Args:
            inputs: (n_trials, trial_length, n_variable) array of observations.
            targets: optional (n_trials, trial_length, n_variable) array.
            external_inputs: optional (n_trials, trial_length, n_external) array.
            normalize_externals: If True and normalize=True, fit a separate StandardScaler
                on ``external_inputs`` and transform them independently.
            **kwargs: passed to TrialTimeseriesDataset.__init__ (batch_size, normalize,
                      test_fraction, seed).
        """
        return cls(
            inputs,
            targets=targets,
            external_inputs=external_inputs,
            normalize_externals=normalize_externals,
            **kwargs,
        )
