"""Cognitive task dataset — wraps task generators for Paradigm A.

Provides a unified interface for cognitive task data (inputs, targets, masks)
compatible with the NeuralRNN Trainer and Objective classes.
"""
from __future__ import annotations

from typing import Any

import torch
import numpy as np

from .base import BaseDataset
from .tasks import TASK_REGISTRY


class CognitiveTaskDataset(BaseDataset):
    """Dataset wrapping cognitive task generators.

    Supports two mask formats:
    - Boolean tensor mask: (N, T, output_dim), 1=valid, 0=ignore
    - Index array mask (training_mask): targets are pre-sliced

    For index-array tasks, the dataset stores the full-length inputs and
    pre-sliced targets, and generates a boolean mask from the indices.

    Attributes:
        kind: "cognitive_task"
        inputs: (N, T, input_dim) tensor
        targets: (N, T, output_dim) tensor (full-length, padded if needed)
        mask: (N, T, output_dim) tensor (boolean)
        conditions: list of dicts with trial metadata
        task_name: name of the task
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
    ) -> None:
        super().__init__()
        self.inputs = inputs
        self.targets = targets
        self.mask = mask
        self.conditions = conditions
        self.task_name = task_name
        self.batch_size = batch_size
        self.input_dim = inputs.shape[-1]
        self.output_dim = targets.shape[-1]

    @classmethod
    def from_task(cls, task_name: str, batch_size: int = 128, **kwargs) -> "CognitiveTaskDataset":
        """Create dataset from a named task generator.

        Args:
            task_name: Name of the task (must be in TASK_REGISTRY).
            batch_size: Batch size for sample_batch().
            **kwargs: Arguments passed to the task's generate_trials().

        Returns:
            CognitiveTaskDataset instance.
        """
        if task_name not in TASK_REGISTRY:
            raise ValueError(
                f"Unknown task '{task_name}'. Available: {list(TASK_REGISTRY.keys())}"
            )

        gen_fn = TASK_REGISTRY[task_name]
        result = gen_fn(**kwargs)

        # Handle different return formats
        if len(result) == 4:
            inputs, targets, mask_or_indices, conditions = result
        else:
            raise ValueError(f"Expected 4 return values, got {len(result)}")

        # Normalize mask format to boolean tensor
        if isinstance(mask_or_indices, np.ndarray):
            # Index array mask (ManteTask, DelayMatchToSample)
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
            task_name=task_name,
            batch_size=batch_size,
        )

    def sample_batch(self) -> dict[str, torch.Tensor]:
        """Sample a random batch of trials.

        Returns:
            dict with keys:
                "inputs":  (batch_size, T, input_dim)
                "targets": (batch_size, T, output_dim)
                "mask":    (batch_size, T, output_dim)
        """
        n = self.inputs.shape[0]
        idx = torch.randint(0, n, (self.batch_size,))
        return {
            "inputs": self.inputs[idx],
            "targets": self.targets[idx],
            "mask": self.mask[idx],
        }

    def get_all_trials(self) -> dict[str, torch.Tensor]:
        """Return all trials (for analysis).

        Returns:
            dict with keys: "inputs", "targets", "mask"
        """
        return {
            "inputs": self.inputs,
            "targets": self.targets,
            "mask": self.mask,
        }

    def __len__(self) -> int:
        return self.inputs.shape[0]
