"""Custom dataset for importing user-generated data into the NeuralRNN framework.

Supports three use cases:
  1. Supervised (Paradigm A): input-output pairs for task optimization
  2. Time-series reconstruction (Paradigm B): observed trajectories for DSR,
     with optional internal states for teacher forcing
  3. Free-running generation: input-only data for model rollout evaluation

Input formats:
  - NumPy arrays: (T, D) or (T,) or (B, T, D)
  - Torch tensors: same shapes
  - .npz files with keys "inputs", "targets", "internal_states", "external_inputs"
  - MATLAB .mat files via scipy.io.loadmat

Batch format (batch-first):
  Paradigm A (supervised):
    {"inputs": (B,T,input_dim), "targets": (B,T,output_dim), "mask": (B,T)|None}
  Paradigm B (timeseries):
    {"inputs": (B,T,N), "targets": (B,T,N), "external_inputs": (B,T,K)|None}

Usage:
    from neuralrnn.data.custom_dataset import CustomDataset

    # From arrays
    ds = CustomDataset.from_arrays(trajectory, mode="timeseries", sequence_length=200)

    # From .npz
    ds = CustomDataset.from_npz("my_data.npz", sequence_length=150)

    # From .mat
    ds = CustomDataset.from_mat("neural_data.mat", variable_map={"inputs": "stim"})
"""
from __future__ import annotations

from pathlib import Path
from random import randint

import numpy as np
import torch

from .base import BaseDataset, StandardScaler


def _to_tensor(x) -> torch.Tensor:
    """Convert input to float32 torch.Tensor."""
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.float()
    return torch.as_tensor(np.asarray(x), dtype=torch.float32)


def _ensure_2d(x: torch.Tensor, keep_3d: bool = False) -> torch.Tensor:
    """Ensure tensor is at least 2D. If 1D (T,), add trailing dim.
    If keep_3d=True and input is 3D, return as-is (for supervised trial data)."""
    if x.ndim == 1:
        return x.unsqueeze(-1)
    if x.ndim == 3 and not keep_3d:
        # (B, T, D) -> (B*T, D)
        B, T, D = x.shape
        return x.reshape(B * T, D)
    return x


class CustomDataset(BaseDataset):
    """User-generated dataset for custom inputs, outputs, and optional internal states.

    Two modes:
      - "supervised": inputs + targets for task optimization (Paradigm A)
      - "timeseries": observed time series for DSR (Paradigm B),
        with optional internal_states for teacher forcing

    The mode is auto-detected from the provided data if set to "auto":
      - If targets are provided and inputs.shape != targets.shape -> supervised
      - If targets are provided and inputs.shape == targets.shape -> timeseries
      - If only inputs are provided -> timeseries (targets = inputs shifted right)
    """

    kind = "custom"

    def __init__(
        self,
        inputs: np.ndarray | torch.Tensor,
        targets: np.ndarray | torch.Tensor | None = None,
        internal_states: np.ndarray | torch.Tensor | None = None,
        external_inputs: np.ndarray | torch.Tensor | None = None,
        sequence_length: int = 200,
        batch_size: int = 16,
        mode: str = "auto",
        normalize: bool = False,
        normalize_externals: bool = False,
        test_fraction: float = 0.0,
        seed: int = 0,
    ) -> None:
        # Convert to tensors
        X = _to_tensor(inputs)
        Y = _to_tensor(targets)
        IS = _to_tensor(internal_states)
        S = _to_tensor(external_inputs)

        # Auto-detect mode before reshaping (3D input with different-dim target -> supervised)
        if mode == "auto":
            if Y is not None and X.ndim >= 2 and X.shape[-1] != Y.shape[-1]:
                mode = "supervised"
            else:
                mode = "timeseries"
        assert mode in ("supervised", "timeseries"), f"Unknown mode: {mode}"
        self.mode = mode

        # Reshape: for supervised mode, preserve 3D (B,T,D) trial structure
        keep_3d = (self.mode == "supervised" and X.ndim == 3)
        X = _ensure_2d(X, keep_3d=keep_3d)
        if Y is not None:
            Y = _ensure_2d(Y, keep_3d=keep_3d)
        if IS is not None:
            IS = _ensure_2d(IS)
        if S is not None:
            S = _ensure_2d(S)

        # Train/test split
        T_total = X.shape[0]
        if test_fraction > 0.0:
            rng = torch.Generator().manual_seed(seed)
            n_test = max(1, int(T_total * test_fraction))
            perm = torch.randperm(T_total, generator=rng)
            test_idx = perm[:n_test]
            train_idx = perm[n_test:]
            train_idx, _ = train_idx.sort()
            test_idx, _ = test_idx.sort()

            self._test_inputs = X[test_idx]
            self._test_targets = Y[test_idx] if Y is not None else None
            self._test_internal_states = IS[test_idx] if IS is not None else None
            self._test_external_inputs = S[test_idx] if S is not None else None

            X = X[train_idx]
            if Y is not None:
                Y = Y[train_idx]
            if IS is not None:
                IS = IS[train_idx]
            if S is not None:
                S = S[train_idx]
        else:
            self._test_inputs = None
            self._test_targets = None
            self._test_internal_states = None
            self._test_external_inputs = None

        # Normalize
        self.normalizer = StandardScaler().fit(X) if normalize else None
        if self.normalizer:
            X = self.normalizer.transform(X)
            if self._test_inputs is not None:
                self._test_inputs = self.normalizer.transform(self._test_inputs)

        # Optional normalization of external inputs (independent from observations)
        self.external_normalizer = None
        if normalize and normalize_externals and S is not None:
            self.external_normalizer = StandardScaler().fit(S)
            S = self.external_normalizer.transform(S)
            if self._test_external_inputs is not None:
                self._test_external_inputs = self.external_normalizer.transform(
                    self._test_external_inputs
                )

        # Store training data
        self.X = X  # (T, N)
        self.Y = Y  # (T, output_dim) or None
        self.IS = IS  # (T, latent_dim) or None — internal states
        self.S = S  # (T, K) or None — external inputs

        if self.X.ndim == 3:
            # 3D supervised: (B, T, D)
            self.T, self.N = self.X.shape[1], self.X.shape[2]
        else:
            self.T, self.N = self.X.shape
        self.input_dim = self.N
        self.output_dim = self.Y.shape[-1] if self.Y is not None else self.N
        self.sequence_length = sequence_length
        self.batch_size = batch_size

        # For Paradigm B: if targets not provided, use shifted inputs
        if self.mode == "timeseries" and self.Y is None:
            self.Y = self.X.clone()  # Targets = inputs (shifted by 1 in sample_batch)

    def __len__(self) -> int:
        # For 3D supervised data: number of trials
        if self.mode == "supervised" and self.X.ndim == 3:
            return self.X.shape[0]
        return max(self.T - self.sequence_length - 1, 0)

    def _slice_supervised(self, t: int) -> tuple:
        """Slice for supervised mode: (input_seq, target_seq, mask_seq).
        For 3D data: t indexes trials, returns the full trial.
        For 2D data: t indexes time, returns a subsequence."""
        if self.X.ndim == 3:
            # 3D: (B, T, D) — t is trial index
            x = self.X[t]   # (T, D)
            y = self.Y[t]   # (T, output_dim)
            mask = torch.ones(x.shape[0], dtype=torch.float32)
        else:
            # 2D: (T_total, D) — t is time index
            x = self.X[t:t + self.sequence_length]
            y = self.Y[t:t + self.sequence_length]
            mask = torch.ones(self.sequence_length, dtype=torch.float32)
        return x, y, mask

    def _slice_timeseries(self, t: int) -> tuple:
        """Slice for timeseries mode: (input_seq, target_seq, ext_input_seq).
        targets = inputs shifted right by 1."""
        x = self.X[t:t + self.sequence_length]
        y = self.X[t + 1:t + self.sequence_length + 1]
        s = self.S[t:t + self.sequence_length] if self.S is not None else None
        return x, y, s

    def sample_batch(self) -> dict[str, torch.Tensor]:
        """Sample a random batch of subsequences.

        Returns:
            For supervised mode:
                {"inputs": (B,T,input_dim), "targets": (B,T,output_dim), "mask": (B,T)}
            For timeseries mode:
                {"inputs": (B,T,N), "targets": (B,T,N), "external_inputs": (B,T,K)|None}
        """
        xs, ys, extras = [], [], []

        for _ in range(self.batch_size):
            t = randint(0, len(self) - 1)

            if self.mode == "supervised":
                x, y, mask = self._slice_supervised(t)
                xs.append(x)
                ys.append(y)
                extras.append(mask)
            else:  # timeseries
                x, y, s = self._slice_timeseries(t)
                xs.append(x)
                ys.append(y)
                if s is not None:
                    extras.append(s)

        if self.mode == "supervised":
            return {
                "inputs": torch.stack(xs),      # (B,T,input_dim)
                "targets": torch.stack(ys),     # (B,T,output_dim)
                "mask": torch.stack(extras),    # (B,T)
            }
        else:
            return {
                "inputs": torch.stack(xs),      # (B,T,N)
                "targets": torch.stack(ys),     # (B,T,N)
                "external_inputs": torch.stack(extras) if extras else None,
            }

    @property
    def test_set(self) -> CustomDataset | None:
        """The held-out test fraction as a separate CustomDataset (if test_fraction > 0)."""
        if self._test_inputs is None:
            return None
        ds = CustomDataset.__new__(CustomDataset)
        ds.kind = self.kind
        ds.mode = self.mode
        ds.normalizer = self.normalizer
        ds.external_normalizer = self.external_normalizer
        ds.X = self._test_inputs
        ds.Y = self._test_targets if self._test_targets is not None else (
            self._test_inputs.clone() if self.mode == "timeseries" else None
        )
        ds.IS = self._test_internal_states
        ds.S = self._test_external_inputs
        if ds.X.ndim == 3:
            ds.T, ds.N = ds.X.shape[1], ds.X.shape[2]
        else:
            ds.T, ds.N = ds.X.shape
        ds.input_dim = self.input_dim
        ds.output_dim = self.output_dim
        ds.sequence_length = self.sequence_length
        ds.batch_size = self.batch_size
        # Test set has no further split
        ds._test_inputs = None
        ds._test_targets = None
        ds._test_internal_states = None
        ds._test_external_inputs = None
        return ds

    @property
    def test(self) -> torch.Tensor | None:
        """Full test trajectory for DSR evaluation (convenience alias)."""
        if self._test_inputs is not None:
            return self._test_inputs
        return None

    # ====================== Class method constructors ======================

    @classmethod
    def from_arrays(
        cls,
        inputs: np.ndarray | torch.Tensor,
        targets: np.ndarray | torch.Tensor | None = None,
        internal_states: np.ndarray | torch.Tensor | None = None,
        external_inputs: np.ndarray | torch.Tensor | None = None,
        normalize_externals: bool = False,
        **kwargs,
    ) -> CustomDataset:
        """Convenience constructor from numpy arrays or torch tensors.

        Args:
            inputs: (T, D) or (T,) or (B, T, D) array of inputs/observations.
            targets: (T, D') or (T,) array. For supervised: class labels or regression targets.
                     For timeseries: same as inputs (auto-generated if None).
            internal_states: (T, M) optional internal latent states (e.g. for teacher forcing).
            external_inputs: (T, K) optional external inputs / covariates.
            normalize_externals: If True and normalize=True, fit a separate StandardScaler
                on ``external_inputs`` and transform them independently of ``inputs``.
            **kwargs: passed to CustomDataset.__init__ (sequence_length, batch_size, mode,
                      normalize, test_fraction, seed).

        Returns:
            CustomDataset instance.

        Examples:
            # Paradigm A: supervised
            ds = CustomDataset.from_arrays(X, targets=Y, mode="supervised")

            # Paradigm B: DSR
            ds = CustomDataset.from_arrays(trajectory, mode="timeseries")

            # Paradigm B with internal states
            ds = CustomDataset.from_arrays(traj, internal_states=states)
        """
        return cls(inputs, targets=targets, internal_states=internal_states,
                   external_inputs=external_inputs, normalize_externals=normalize_externals,
                   **kwargs)

    @classmethod
    def from_dict(cls, data: dict, normalize_externals: bool = False, **kwargs) -> CustomDataset:
        """Construct from a dict with keys "inputs", "targets" (optional),
        "internal_states" (optional), "external_inputs" (optional).

        Useful for loading from preprocessed data structures or .npz files.

        Args:
            data: dict with array-valued keys.
            normalize_externals: If True and normalize=True, normalize ``external_inputs``
                with a separate StandardScaler.
            **kwargs: passed to CustomDataset.__init__.

        Returns:
            CustomDataset instance.
        """
        return cls(
            inputs=data["inputs"],
            targets=data.get("targets"),
            internal_states=data.get("internal_states"),
            external_inputs=data.get("external_inputs"),
            normalize_externals=normalize_externals,
            **kwargs,
        )

    @classmethod
    def from_npz(cls, path: str, normalize_externals: bool = False, **kwargs) -> CustomDataset:
        """Load from a .npz file.

        Expected keys: "inputs" (required), "targets" (optional),
        "internal_states" (optional), "external_inputs" (optional).

        Args:
            path: path to .npz file.
            normalize_externals: If True and normalize=True, normalize ``external_inputs``
                with a separate StandardScaler.
            **kwargs: passed to CustomDataset.__init__.

        Returns:
            CustomDataset instance.
        """
        data = np.load(path, allow_pickle=False)
        return cls.from_dict({k: data[k] for k in data.files},
                             normalize_externals=normalize_externals, **kwargs)

    @classmethod
    def from_mat(
        cls,
        path: str,
        variable_map: dict[str, str] | None = None,
        normalize_externals: bool = False,
        **kwargs,
    ) -> CustomDataset:
        """Load from a MATLAB .mat file (requires scipy).

        Args:
            path: path to .mat file.
            variable_map: dict mapping expected keys ("inputs", "targets",
                "internal_states", "external_inputs") to .mat variable names.
                If None, uses the default names directly.
            normalize_externals: If True and normalize=True, normalize ``external_inputs``
                with a separate StandardScaler.
            **kwargs: passed to CustomDataset.__init__.

        Returns:
            CustomDataset instance.
        """
        try:
            from scipy.io import loadmat
        except ImportError as e:
            raise ImportError(
                "Loading .mat files requires scipy: pip install scipy"
            ) from e

        mat = loadmat(path, squeeze_me=True)
        default_map = {
            "inputs": "inputs",
            "targets": "targets",
            "internal_states": "internal_states",
            "external_inputs": "external_inputs",
        }
        vmap = {**default_map, **(variable_map or {})}

        def _get(key: str):
            mat_name = vmap.get(key, key)
            if mat_name in mat:
                arr = np.asarray(mat[mat_name], dtype=np.float32)
                # squeeze singleton dims from loadmat
                if arr.ndim == 0:
                    arr = arr.reshape(1)
                return arr
            return None

        return cls(
            inputs=_get("inputs"),
            targets=_get("targets"),
            internal_states=_get("internal_states"),
            external_inputs=_get("external_inputs"),
            normalize_externals=normalize_externals,
            **kwargs,
        )
