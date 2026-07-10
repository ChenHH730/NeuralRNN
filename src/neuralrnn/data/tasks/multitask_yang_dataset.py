"""Multitask Yang et al. (2019) dataset wrapper for NeuralRNN training.

Provides a `MultitaskYangDataset` that samples tasks according to rule
probabilities and yields batch-first dicts compatible with the generic
`Trainer` and `SupervisedObjective`.
"""
from __future__ import annotations

import numpy as np
import torch

from ..base import BaseDataset
from .multitask_yang_task import generate_trials, RULES_ALL


class MultitaskYangDataset(BaseDataset):
    """Dataset that interleaves the 20 Yang et al. (2019) tasks.

    Args:
        rules: list of rule names to train on. If None, uses all 20 rules.
        rule_prob_map: dict mapping rule name to relative sampling probability.
            Rules not in the map default to 1.0. The original paper oversamples
            contextdm1 and contextdm2 by 5x.
        batch_size: number of trials per sampled task.
        mode: 'random' or 'test'.
        sigma_x: input noise scale.
        seed: random seed.
    """

    kind = "neurogym"  # Paradigm A supervised task data

    def __init__(
        self,
        rules: list[str] | None = None,
        rule_prob_map: dict[str, float] | None = None,
        batch_size: int = 64,
        mode: str = "random",
        sigma_x: float = 0.01,
        seed: int | None = None,
    ) -> None:
        self.rules = list(rules) if rules is not None else list(RULES_ALL)
        self.rule_prob_map = rule_prob_map or {}
        self.batch_size = batch_size
        self.mode = mode
        self.sigma_x = sigma_x

        self.input_dim = 85
        self.output_dim = 33

        self.rng = np.random.default_rng(seed)
        self._torch_rng = torch.Generator()
        if seed is not None:
            self._torch_rng.manual_seed(seed)

        # Normalize rule probabilities
        probs = np.array([self.rule_prob_map.get(r, 1.0) for r in self.rules], dtype=float)
        self.rule_probs = probs / probs.sum()

    def sample_batch(self) -> dict[str, torch.Tensor]:
        """Sample one task according to rule_probs and generate a batch."""
        rule = self.rng.choice(self.rules, p=self.rule_probs)
        inputs, targets, mask, conditions = generate_trials(
            rule,
            n_trials=self.batch_size,
            mode=self.mode,
            sigma_x=self.sigma_x,
            seed=int(self.rng.integers(0, 2**31)),
        )
        return {
            "inputs": torch.from_numpy(inputs),
            "targets": torch.from_numpy(targets),
            "mask": torch.from_numpy(mask),
            "rule": rule,
            "conditions": conditions,
        }

    def sample_task(self, rule: str, n_trials: int | None = None) -> dict[str, torch.Tensor]:
        """Generate a batch for a specific task."""
        if n_trials is None:
            n_trials = self.batch_size
        inputs, targets, mask, conditions = generate_trials(
            rule,
            n_trials=n_trials,
            mode=self.mode,
            sigma_x=self.sigma_x,
            seed=int(self.rng.integers(0, 2**31)),
        )
        return {
            "inputs": torch.from_numpy(inputs),
            "targets": torch.from_numpy(targets),
            "mask": torch.from_numpy(mask),
            "rule": rule,
            "conditions": conditions,
        }
