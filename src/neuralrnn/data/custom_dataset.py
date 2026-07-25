"""Custom dataset for importing user-generated data into the NeuralRNN framework.

Supports three use cases:
  1. Supervised (Paradigm A): input-output pairs for task optimization
  2. Reconstruction (Paradigm B): observed trajectories for DSR / activity
     reconstruction, with optional behavioral targets and loss mask
  3. Free-running generation: input-only data for model rollout evaluation

Input formats:
  - NumPy arrays: (T, D) or (T,) or (B, T, D)
  - Torch tensors: same shapes
  - .npz files with keys "inputs", "targets", "activity", "external_inputs", "mask"
  - MATLAB .mat files via scipy.io.loadmat

Batch format (batch-first):
  Paradigm A (supervised):
    {"inputs": (B,T,input_dim), "targets": (B,T,output_dim), "mask": (B,T)|None}
  Paradigm B (reconstruction):
    {"activity": (B,T,N), "inputs": (B,T,K)|None, "targets": (B,T,O)|None, "mask": (B,T)|None}
    where "inputs" are external/task inputs and "activity" is the observed trajectory.

Usage:
    from neuralrnn.data.custom_dataset import CustomDataset

    # From arrays (reconstruction mode)
    ds = CustomDataset.from_arrays(trajectory, mode="reconstruction", sequence_length=200)

    # From .npz
    ds = CustomDataset.from_npz("my_data.npz", sequence_length=150)

    # From .mat
    ds = CustomDataset.from_mat("neural_data.mat", variable_map={"inputs": "stim"})
"""
from __future__ import annotations

import warnings
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
    """User-generated dataset for custom inputs, outputs, and optional observations.

    Two modes:
      - "supervised": inputs + targets for task optimization (Paradigm A)
      - "reconstruction": observed activity for DSR / activity reconstruction
        (Paradigm B), with optional external inputs, behavioral targets, and mask.

    The mode is auto-detected from the provided data if set to "auto":
      - If targets are provided and inputs.shape != targets.shape -> supervised
      - Otherwise -> reconstruction
    """

    kind = "custom"

    def __init__(
        self,
        inputs: np.ndarray | torch.Tensor,
        targets: np.ndarray | torch.Tensor | None = None,
        internal_states: np.ndarray | torch.Tensor | None = None,
        external_inputs: np.ndarray | torch.Tensor | None = None,
        mask: np.ndarray | torch.Tensor | None = None,
        sequence_length: int = 200,
        batch_size: int = 16,
        mode: str = "auto",
        normalize: bool = False,
        normalize_externals: bool = False,
        test_fraction: float = 0.0,
        seed: int = 0,
    ) -> None:
        # Deprecated alias handling
        if mode == "timeseries":
            warnings.warn(
                'CustomDataset mode="timeseries" is deprecated; use mode="reconstruction" instead.',
                DeprecationWarning,
                stacklevel=2,
            )
            mode = "reconstruction"

        # Convert to tensors
        X = _to_tensor(inputs)
        Y = _to_tensor(targets)
        IS = _to_tensor(internal_states)
        S = _to_tensor(external_inputs)
        M = _to_tensor(mask)

        # Auto-detect mode before reshaping (3D input with different-dim target -> supervised)
        if mode == "auto":
            if Y is not None and X.ndim >= 2 and X.shape[-1] != Y.shape[-1]:
                mode = "supervised"
            else:
                mode = "reconstruction"
        if mode not in ("supervised", "reconstruction"):
            raise ValueError(f"Unknown mode: {mode!r}. Expected 'supervised' or 'reconstruction'.")
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
        if M is not None:
            M = _ensure_2d(M)

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
            self._test_mask = M[test_idx] if M is not None else None

            X = X[train_idx]
            if Y is not None:
                Y = Y[train_idx]
            if IS is not None:
                IS = IS[train_idx]
            if S is not None:
                S = S[train_idx]
            if M is not None:
                M = M[train_idx]
        else:
            self._test_inputs = None
            self._test_targets = None
            self._test_internal_states = None
            self._test_external_inputs = None
            self._test_mask = None

        # Normalize
        if self.mode == "supervised":
            self.normalizer = StandardScaler().fit(X) if normalize else None
            if self.normalizer:
                X = self.normalizer.transform(X)
                if self._test_inputs is not None:
                    self._test_inputs = self.normalizer.transform(self._test_inputs)
        else:
            # reconstruction mode: normalize the observation trajectory (activity)
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
        self.X = X  # supervised: (T, input_dim) or (B, T, input_dim); reconstruction: (T, N) activity
        self.Y = Y  # supervised: targets; reconstruction: optional behavior targets
        self.IS = IS  # deprecated internal states (stored but not used in sample_batch)
        self.S = S  # external inputs (reconstruction mode) or None
        self.M = M  # optional mask

        if self.X.ndim == 3:
            # 3D supervised: (B, T, D)
            self.T, self.N = self.X.shape[1], self.X.shape[2]
        else:
            self.T, self.N = self.X.shape
        self.input_dim = self.N
        self.output_dim = self.Y.shape[-1] if self.Y is not None else self.N
        self.sequence_length = sequence_length
        self.batch_size = batch_size

    def __len__(self) -> int:
        # For 3D supervised data: number of trials
        if self.mode == "supervised" and self.X.ndim == 3:
            return self.X.shape[0]
        if self.mode == "reconstruction":
            return max(self.T - self.sequence_length, 0)
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

    def _slice_reconstruction(self, t: int) -> tuple:
        """Slice for reconstruction mode: (activity, inputs, targets, mask)."""
        act = self.X[t:t + self.sequence_length]
        inp = self.S[t:t + self.sequence_length] if self.S is not None else None
        tgt = self.Y[t:t + self.sequence_length] if self.Y is not None else None
        msk = self.M[t:t + self.sequence_length] if self.M is not None else None
        return act, inp, tgt, msk

    def sample_batch(self) -> dict[str, torch.Tensor]:
        """Sample a random batch of subsequences.

        Returns:
            For supervised mode:
                {"inputs": (B,T,input_dim), "targets": (B,T,output_dim), "mask": (B,T)}
            For reconstruction mode:
                {"activity": (B,T,N), "inputs": (B,T,K)|None, "targets": (B,T,O)|None,
                 "mask": (B,T)|None}
                Only non-None optional keys are included.
        """
        xs, ys, extras = [], [], []
        # reconstruction mode may also produce targets and mask
        ss, ts, ms = [], [], []

        for _ in range(self.batch_size):
            t = randint(0, max(len(self) - 1, 0))

            if self.mode == "supervised":
                x, y, mask = self._slice_supervised(t)
                xs.append(x)
                ys.append(y)
                extras.append(mask)
            else:  # reconstruction
                act, inp, tgt, msk = self._slice_reconstruction(t)
                xs.append(act)
                if inp is not None:
                    ss.append(inp)
                if tgt is not None:
                    ts.append(tgt)
                if msk is not None:
                    ms.append(msk)

        if self.mode == "supervised":
            return {
                "inputs": torch.stack(xs),      # (B,T,input_dim)
                "targets": torch.stack(ys),     # (B,T,output_dim)
                "mask": torch.stack(extras),    # (B,T)
            }
        else:
            batch = {"activity": torch.stack(xs)}  # (B,T,N)
            if ss:
                batch["inputs"] = torch.stack(ss)  # (B,T,K)
            if ts:
                batch["targets"] = torch.stack(ts)  # (B,T,O)
            if ms:
                batch["mask"] = torch.stack(ms)     # (B,T)
            return batch

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
        if self.mode == "supervised":
            ds.Y = self._test_targets
        else:
            ds.Y = self._test_targets  # behavior targets (may be None)
        ds.IS = self._test_internal_states
        ds.S = self._test_external_inputs
        ds.M = self._test_mask
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
        ds._test_mask = None
        return ds

    @property
    def test(self) -> torch.Tensor | None:
        """Full test trajectory for DSR evaluation (convenience alias)."""
        if self._test_inputs is not None:
            return self._test_inputs
        return None

    @property
    def activity(self) -> torch.Tensor | None:
        """Observation trajectory (reconstruction mode). Alias for self.X."""
        return self.X

    # ====================== Class method constructors ======================

    @classmethod
    def from_arrays(
        cls,
        inputs: np.ndarray | torch.Tensor,
        targets: np.ndarray | torch.Tensor | None = None,
        internal_states: np.ndarray | torch.Tensor | None = None,
        external_inputs: np.ndarray | torch.Tensor | None = None,
        mask: np.ndarray | torch.Tensor | None = None,
        normalize_externals: bool = False,
        **kwargs,
    ) -> CustomDataset:
        """Convenience constructor from numpy arrays or torch tensors.

        Args:
            inputs: (T, D) or (T,) or (B, T, D) array.
                For supervised: task inputs.
                For reconstruction: observed activity / trajectory.
            targets: (T, D') or (T,) array. For supervised: class labels or regression targets.
                For reconstruction: optional behavioral targets.
            internal_states: (T, M) optional internal latent states (deprecated; unused).
            external_inputs: (T, K) optional external inputs / covariates (reconstruction mode).
            mask: (T,) or (T, 1) optional per-timestep mask (reconstruction mode).
            normalize_externals: If True and normalize=True, fit a separate StandardScaler
                on ``external_inputs`` and transform them independently of ``inputs``.
            **kwargs: passed to CustomDataset.__init__ (sequence_length, batch_size, mode,
                      normalize, test_fraction, seed).

        Returns:
            CustomDataset instance.

        Examples:
            # Paradigm A: supervised
            ds = CustomDataset.from_arrays(X, targets=Y, mode="supervised")

            # Paradigm B: DSR (teacher forcing)
            ds = CustomDataset.from_arrays(trajectory, mode="reconstruction", sequence_length=200)

            # Paradigm B: activity reconstruction with behavior targets
            ds = CustomDataset.from_arrays(
                traj, targets=behavior, external_inputs=ext_inp, mask=msk,
                mode="reconstruction"
            )
        """
        return cls(inputs, targets=targets, internal_states=internal_states,
                   external_inputs=external_inputs, mask=mask,
                   normalize_externals=normalize_externals, **kwargs)

    @classmethod
    def from_dict(cls, data: dict, normalize_externals: bool = False, **kwargs) -> CustomDataset:
        """Construct from a dict with keys "inputs", "targets" (optional),
        "external_inputs" (optional), "mask" (optional).

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
            mask=data.get("mask"),
            normalize_externals=normalize_externals,
            **kwargs,
        )

    @classmethod
    def from_npz(cls, path: str, normalize_externals: bool = False, **kwargs) -> CustomDataset:
        """Load from a .npz file.

        Expected keys: "inputs" (required), "targets" (optional),
        "external_inputs" (optional), "mask" (optional).

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
                "external_inputs", "mask") to .mat variable names.
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
            "mask": "mask",
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
            mask=_get("mask"),
            normalize_externals=normalize_externals,
            **kwargs,
        )
