"""Reconstruction dataset — pre-computed teacher data for activity reconstruction.

Holds task inputs, recorded teacher activity (hidden states or firing rates),
and optionally teacher behavioral outputs and a loss mask. Used to fit a
student model to a trained teacher RNN (or to recorded neural data).

Also supports single-trajectory DSR time series (e.g. Lorenz-63) via
``from_timeseries``: windows are sliced lazily from the stored series at
``sample_batch()`` time and exposed under the same unified batch keys
("activity" = observations, "inputs" = optional external inputs).
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from .base import BaseDataset, StandardScaler


class ReconstructionDataset(BaseDataset):
    """Dataset for reconstructing teacher neural activity (and behavior).

    Stores pre-computed data, typically from running a trained teacher RNN
    on task inputs:
    - inputs: task inputs
    - activity: recorded teacher activity (hidden states or firing rates)
    - targets: (optional) teacher behavioral output (readout)
    - mask: (optional) per-timestep loss mask

    Attributes:
        kind: "reconstruction"
        inputs: (N, T, K) tensor — task inputs
        activity: (N, T, A) tensor — recorded teacher activity
        targets: (N, T, O) tensor or None — teacher behavioral output
        mask: (N, T) or (N, T, 1) tensor or None — loss mask
        batch_size: batch size for sample_batch()
    """

    kind = "reconstruction"

    def __init__(
        self,
        inputs: torch.Tensor,
        activity: torch.Tensor,
        targets: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        batch_size: int = 128,
    ) -> None:
        super().__init__()
        self.inputs = inputs
        self.activity = activity
        self.targets = targets
        self.mask = mask
        self.batch_size = batch_size
        self.input_dim = inputs.shape[-1]
        self.output_dim = targets.shape[-1] if targets is not None else 0
        self._timeseries = False

    @classmethod
    def from_timeseries(
        cls,
        data,
        external_inputs=None,
        sequence_length: int = 200,
        batch_size: int = 16,
        normalize: bool = False,
        test: np.ndarray | None = None,
        dt: float | None = None,
    ) -> "ReconstructionDataset":
        """Create a windowed DSR dataset from a single (T, N) observation trajectory.

        Windows are sliced lazily at sample_batch() time (the series is stored
        once, windows are not materialized). Batches use the unified keys:
        "activity" = (B, T, N) observation windows, "inputs" = (B, T, K)
        external-input windows (only when external_inputs is given).

        Attributes exposed for evaluation: .X (normalized series), .T, .N, .dim,
        .test (held-out series, not sliced), .normalizer, .sequence_length, .dt.
        """
        obj = cls.__new__(cls)
        BaseDataset.__init__(obj)
        X = torch.as_tensor(np.asarray(data), dtype=torch.float32)
        obj.normalizer = StandardScaler().fit(X) if normalize else None
        obj.X = obj.normalizer.transform(X) if obj.normalizer else X
        obj.T, obj.N = obj.X.shape
        obj.dim = obj.N
        obj.sequence_length = sequence_length
        obj.batch_size = batch_size
        obj.dt = dt
        obj.S = None if external_inputs is None else torch.as_tensor(
            np.asarray(external_inputs), dtype=torch.float32)
        if obj.S is not None:
            assert obj.S.shape[0] == obj.T, \
                "external_inputs must match the number of time steps in data"
        # Test set (for evaluation/generation); not sliced into batches
        obj.test = torch.as_tensor(np.asarray(test), dtype=torch.float32) \
            if test is not None else None
        obj._timeseries = True
        obj.input_dim = 0 if obj.S is None else obj.S.shape[-1]
        obj.output_dim = obj.N
        return obj

    @classmethod
    def from_npy(cls, train_path, test_path=None, dt=None, **kwargs) -> "ReconstructionDataset":
        """Registry loader entry point: construct from .npy files (for the lorenz63 dataset)."""
        train = np.load(train_path).astype(np.float32)
        test = np.load(test_path).astype(np.float32) if test_path else None
        return cls.from_timeseries(train, test=test, dt=dt, **kwargs)

    @classmethod
    def from_rnn_and_task(
        cls,
        rnn_model,
        task_dataset,
        batch_size: int = 128,
        device: str = "cpu",
        activity_fn: str | Callable | None = None,
        record_targets: bool = True,
    ) -> "ReconstructionDataset":
        """Create dataset by running a trained teacher RNN on task data.

        Args:
            rnn_model: Trained teacher RNN model (NeuralDynamicsModel).
            task_dataset: Task dataset providing trial inputs. Must expose
                either ``get_all_trials()`` (returning a dict with "inputs"
                and optional "targets"/"mask") or ``.inputs`` (with optional
                ``.targets`` / ``.mask``) attributes.
            batch_size: Batch size for sample_batch().
            device: Device to run the teacher RNN on.
            activity_fn: What to record as teacher activity:
                None (default) — hidden states (``out.states``);
                "firing_rates" — ``rnn_model.get_firing_rates(out.states)``;
                callable — ``activity_fn(rnn_model, out) -> Tensor``.
            record_targets: If True, store the teacher behavioral output
                (``out.outputs``) as reconstruction targets.

        Returns:
            ReconstructionDataset instance.
        """
        rnn_model.eval()
        with torch.no_grad():
            if hasattr(task_dataset, "get_all_trials"):
                data = task_dataset.get_all_trials()
                inputs = data["inputs"].to(device)
                mask = data.get("mask")
            else:
                inputs = task_dataset.inputs.to(device)
                mask = getattr(task_dataset, "mask", None)
            out = rnn_model(inputs)
            if activity_fn is None:
                activity = out.states  # (N, T, latent_dim)
            elif activity_fn == "firing_rates":
                activity = rnn_model.get_firing_rates(out.states)
            else:
                activity = activity_fn(rnn_model, out)
            targets = out.outputs if record_targets else None

        return cls(
            inputs=inputs.cpu(),
            activity=activity.cpu(),
            targets=targets.cpu() if targets is not None else None,
            mask=mask.cpu() if mask is not None else None,
            batch_size=batch_size,
        )

    def sample_batch(self) -> dict[str, torch.Tensor]:
        """Sample a random batch.

        Returns:
            dict with keys:
                "inputs":   (B, T, K) — task inputs
                "activity": (B, T, A) — recorded teacher activity
                "targets":  (B, T, O) — teacher behavioral output
                                (only if provided at construction)
                "mask":     (B, T) | (B, T, 1) — loss mask
                                (only if provided at construction)

            Timeseries mode (from_timeseries): "activity" holds random
            observation windows of length sequence_length, "inputs" holds the
            matching external-input windows (only when external inputs exist).
        """
        if self._timeseries:
            starts = torch.randint(0, len(self), (self.batch_size,))
            batch = {"activity": torch.stack(
                [self.X[t:t + self.sequence_length] for t in starts])}
            if self.S is not None:
                batch["inputs"] = torch.stack(
                    [self.S[t:t + self.sequence_length] for t in starts])
            return batch
        n = self.inputs.shape[0]
        idx = torch.randint(0, n, (self.batch_size,))
        batch = {
            "inputs": self.inputs[idx],
            "activity": self.activity[idx],
        }
        if self.targets is not None:
            batch["targets"] = self.targets[idx]
        if self.mask is not None:
            batch["mask"] = self.mask[idx]
        return batch

    def __len__(self) -> int:
        if self._timeseries:
            return max(self.T - self.sequence_length, 0)
        return self.inputs.shape[0]
