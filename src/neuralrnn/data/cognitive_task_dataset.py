"""Cognitive task dataset — wraps task generators for Paradigm A.

Provides a unified interface for cognitive task data (inputs, targets, masks)
compatible with the NeuralRNN Trainer and Objective classes.

Two modes (``mode=`` in ``from_task``; see docs/DATA_REFACTOR.md §2.2):

* ``"aligned"`` (default): pre-generated trial-aligned tensors — the original
  behavior. ``inputs`` / ``targets`` / ``mask`` / ``conditions`` attributes,
  ``get_all_trials()``, ``__len__``, whole-trial ``sample_batch()``.
* ``"streaming"``: the generated trials form a pool; ``sample_batch()``
  concatenates randomly drawn trials per batch row into ``seq_len`` windows
  (windows cross trial boundaries, like neurogym streaming). ``mask`` is None
  and the trial-aligned interface raises RuntimeError.
"""
from __future__ import annotations

from typing import Any

import torch
import numpy as np

from .base import BaseDataset, Trials, subset_trials
from .tasks import TASK_CLASSES, resolve_task_name


class CognitiveTaskDataset(BaseDataset):
    """Dataset wrapping cognitive task generators (see module docstring).

    Mask format: boolean/float tensor mask (N, T, output_dim), 1=valid,
    0=ignore. A legacy index-array mask path (np.ndarray) is retained for
    legacy custom tasks: targets are pre-sliced and padded back to full length.
    """

    kind = "cognitive_task"

    def __init__(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
        conditions: list,
        task_name: str = "",
        batch_size: int = 128,
        mode: str = "aligned",
        seq_len: int = 100,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if mode not in ("aligned", "streaming"):
            raise ValueError(f"mode must be 'aligned' or 'streaming', got {mode!r}")
        self.mode = mode
        self.seq_len = seq_len
        self.task_name = task_name
        self.batch_size = batch_size
        self.input_dim = inputs.shape[-1]
        self.output_dim = targets.shape[-1]
        self._gen = torch.Generator().manual_seed(seed) if seed is not None else None
        if mode == "aligned":
            self._inputs = inputs
            self._targets = targets
            self._mask = mask
            self._conditions = conditions
        else:
            # Trial pool for streaming; each trial is trimmed to its true
            # length (conditions[i]["n_steps"]) when concatenated.
            self._pool = (inputs, targets, mask, conditions)

    # ------------------------------------------------------------------ aligned interface

    def _require_aligned(self, what: str) -> None:
        if self.mode != "aligned":
            raise RuntimeError(
                f"This CognitiveTaskDataset is in streaming mode (seq_len windows over "
                f"concatenated trials); '{what}' requires mode='aligned'."
            )

    @property
    def inputs(self) -> torch.Tensor:
        self._require_aligned("inputs")
        return self._inputs

    @property
    def targets(self) -> torch.Tensor:
        self._require_aligned("targets")
        return self._targets

    @property
    def mask(self) -> torch.Tensor:
        self._require_aligned("mask")
        return self._mask

    @property
    def conditions(self) -> list:
        self._require_aligned("conditions")
        return self._conditions

    @classmethod
    def from_task(cls, task_name: str, batch_size: int = 128,
                  mode: str = "aligned", seq_len: int = 100, **kwargs) -> "CognitiveTaskDataset":
        """Create dataset from a named task.

        Args:
            task_name: Task name (canonical or deprecated alias; warns on alias).
            batch_size: Batch size for sample_batch().
            mode: "aligned" (default) or "streaming" (see module docstring).
            seq_len: Window length for streaming mode.
            **kwargs: Arguments passed to the task constructor (e.g. n_trials,
                sigma_in, catch_fraction, seed; deprecated names warn).

        Returns:
            CognitiveTaskDataset instance.
        """
        canonical = resolve_task_name(task_name)  # raises ValueError for unknown
        seed = kwargs.get("seed")
        task = TASK_CLASSES[canonical].from_kwargs(**kwargs)
        result = task.generate_trials()

        # Handle different return formats
        if len(result) == 4:
            inputs, targets, mask_or_indices, conditions = result
        else:
            raise ValueError(f"Expected 4 return values, got {len(result)}")

        # Normalize mask format to float tensor
        if isinstance(mask_or_indices, np.ndarray):
            # Index array mask (legacy custom tasks)
            # targets are already pre-sliced, need to pad back to full length
            training_mask = mask_or_indices
            full_targets = torch.zeros(inputs.shape[0], inputs.shape[1], targets.shape[-1])
            full_targets[:, training_mask, :] = targets
            targets = full_targets
            mask = torch.zeros_like(targets)
            mask[:, training_mask, :] = 1.0
        elif isinstance(mask_or_indices, torch.Tensor):
            mask = mask_or_indices.float()
        else:
            raise ValueError(f"Unexpected mask type: {type(mask_or_indices)}")

        return cls(
            inputs=inputs,
            targets=targets,
            mask=mask,
            conditions=conditions,
            task_name=canonical,
            batch_size=batch_size,
            mode=mode,
            seq_len=seq_len,
            seed=seed,
        )

    # ------------------------------------------------------------------ sampling

    def sample_batch(self) -> dict[str, torch.Tensor]:
        """Sample a batch.

        Aligned mode: random whole trials with replacement —
            {"inputs": (B,T,input_dim), "targets": (B,T,output_dim), "mask": (B,T,output_dim)}
        Streaming mode: concatenated-trial windows that may cross trial
        boundaries — {"inputs": (B,seq_len,input_dim), "targets": (B,seq_len,output_dim),
        "mask": None}.
        """
        if self.mode == "streaming":
            return self._sample_stream()
        n = self._inputs.shape[0]
        idx = torch.randint(0, n, (self.batch_size,))
        return {
            "inputs": self._inputs[idx],
            "targets": self._targets[idx],
            "mask": self._mask[idx],
        }

    def _sample_stream(self) -> dict[str, torch.Tensor]:
        pool_inputs, pool_targets, _, pool_conditions = self._pool
        n = pool_inputs.shape[0]
        rows_in, rows_tg = [], []
        for _ in range(self.batch_size):
            parts_in, parts_tg, total = [], [], 0
            while total < self.seq_len:
                idx = int(torch.randint(0, n, (1,), generator=self._gen))
                tlen = int(pool_conditions[idx].get("n_steps", pool_inputs.shape[1]))
                parts_in.append(pool_inputs[idx, :tlen])
                parts_tg.append(pool_targets[idx, :tlen])
                total += tlen
            rows_in.append(torch.cat(parts_in, dim=0)[: self.seq_len])
            rows_tg.append(torch.cat(parts_tg, dim=0)[: self.seq_len])
        return {
            "inputs": torch.stack(rows_in),
            "targets": torch.stack(rows_tg),
            "mask": None,
        }

    def sample_trials(self, n: int, seed: int | None = None) -> Trials:
        """Return n complete trials as a ``Trials`` object (no new dataset needed).

        Aligned mode: subset of the pre-generated trials. Streaming mode:
        trials drawn from the pool (with replacement when n exceeds the pool),
        trimmed to their true lengths and zero-padded to the longest.
        """
        if self.mode == "aligned":
            return subset_trials(self._inputs, self._targets, self._mask,
                                 self._conditions, n, seed)
        pool_inputs, pool_targets, _, pool_conditions = self._pool
        n_total = pool_inputs.shape[0]
        if seed is None:
            idx = torch.arange(min(n, n_total))
        else:
            g = torch.Generator().manual_seed(seed)
            idx = (torch.randperm(n_total, generator=g)[:n] if n <= n_total
                   else torch.randint(0, n_total, (n,), generator=g))
        lengths = [int(pool_conditions[i].get("n_steps", pool_inputs.shape[1]))
                   for i in idx.tolist()]
        t_max = max(lengths)
        inputs = torch.zeros(len(idx), t_max, self.input_dim)
        targets = torch.zeros(len(idx), t_max, *pool_targets.shape[2:],
                              dtype=pool_targets.dtype)
        mask = torch.zeros(len(idx), t_max, *pool_targets.shape[2:])
        conditions = []
        for row, (i, tlen) in enumerate(zip(idx.tolist(), lengths)):
            inputs[row, :tlen] = pool_inputs[i, :tlen]
            targets[row, :tlen] = pool_targets[i, :tlen]
            mask[row, :tlen] = 1.0
            conditions.append(pool_conditions[i])
        return Trials(inputs, targets, mask, conditions)

    def get_all_trials(self) -> dict[str, torch.Tensor]:
        """Return all trials (for analysis; aligned mode only).

        Returns:
            dict with keys: "inputs", "targets", "mask"
        """
        self._require_aligned("get_all_trials")
        return {
            "inputs": self._inputs,
            "targets": self._targets,
            "mask": self._mask,
        }

    def __len__(self) -> int:
        self._require_aligned("__len__")
        return self._inputs.shape[0]
