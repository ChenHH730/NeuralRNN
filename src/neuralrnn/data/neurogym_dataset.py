"""Neurogym task dataset (Paradigm A / task optimization).

Ported from the data-extraction style of RNN_DynamicalSystemAnalysis.ipynb: build a cognitive-task
environment with neurogym, wrap it as a PyTorch dataloader, and yield (inputs, targets) per batch.
Here we convert to the batch-first standard batch dict at the boundary (ARCHITECTURE §3.1 Paradigm A).

neurogym is an optional heavy dependency (see pyproject [project.optional-dependencies] neurogym);
a clear hint is shown when not installed.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .base import BaseDataset


class NeurogymDataset(BaseDataset):
    kind = "neurogym"

    def __init__(self, env, dataset, input_dim: int, output_dim: int,
                 batch_size: int = 16, seq_len: int = 100):
        self.env = env                 # Keep underlying env for task-related input during analysis (e.g., mean input at 0-coherence)
        self._dataset = dataset        # neurogym Dataset (callable, returns (inputs, target) time-first)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.seq_len = seq_len

    @classmethod
    def from_task(cls, task: str, *, batch_size: int = 16, seq_len: int = 100,
                  dt: int = 100, timing: dict | None = None, **env_kwargs: Any) -> "NeurogymDataset":
        """Construct dataset from a task name, e.g. task='PerceptualDecisionMaking-v0'.

        Called by load_dataset('perceptual_decision_making') in data/registry.py.
        Extra kwargs are forwarded to the neurogym environment (dt / timing / task-specific parameters).
        """
        try:
            import neurogym as ngym
        except ImportError as e:
            raise ImportError(
                "neurogym is required: pip install 'neuralrnn[neurogym]' or pip install neurogym"
            ) from e

        kwargs = dict(env_kwargs)
        if timing is not None:
            kwargs["timing"] = timing
        dataset = ngym.Dataset(task, env_kwargs={"dt": dt, **kwargs},
                               batch_size=batch_size, seq_len=seq_len)
        env = dataset.env
        input_dim = env.observation_space.shape[0]
        # Classification tasks use n classes (CrossEntropy target); regression tasks use dimension
        output_dim = int(getattr(env.action_space, "n", None)
                         or env.action_space.shape[0])
        return cls(env, dataset, input_dim, output_dim,
                   batch_size=batch_size, seq_len=seq_len)

    def sample_batch(self) -> dict[str, torch.Tensor]:
        # neurogym Dataset() returns time-first: inputs (T,B,obs), target (T,B)
        inputs, target = self._dataset()
        inputs = torch.as_tensor(inputs, dtype=torch.float32).permute(1, 0, 2)  # -> (B,T,obs)
        target = torch.as_tensor(np.asarray(target), dtype=torch.long).permute(1, 0)  # -> (B,T)
        return {"inputs": inputs, "targets": target, "mask": None}

    def task_input(self, kind: str = "stimulus") -> torch.Tensor:
        """Return the 'task-condition input' for fixed-point analysis (e.g., mean input at 0-coherence for decision tasks).
        Defaults to zero input; override per task during porting as needed (see PORTING_GUIDE recipe 1)."""
        return torch.zeros(self.input_dim, dtype=torch.float32)
