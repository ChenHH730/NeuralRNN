"""Multitask Driscoll et al. (2024) dataset wrapper for NeuralRNN training.

Provides a `MultitaskFlexibleDataset` that samples tasks according to rule
probabilities and yields batch-first dicts compatible with the generic
`Trainer` and `SupervisedObjective`.

Following the original paper, context-dependent delayed DM tasks
(`contextdelaydm1` and `contextdelaydm2`) are oversampled 5x by default to
prevent the network from learning a suboptimal strategy (integrating both
modalities regardless of context, which yields at most ~75% accuracy).
"""
from __future__ import annotations

import numpy as np
import torch

from ..base import BaseDataset
from .multitask_flexible_task import generate_trials, RULES_ALL


__all__ = ["MultitaskFlexibleDataset", "get_flexible_rule_prob_map"]


def get_flexible_rule_prob_map(oversample_context: bool | float = True) -> dict[str, float]:
    """Build the rule probability map used by Driscoll et al. (2024).

    The paper explicitly oversamples ContextIntModality1 and ContextIntModality2
    (here `contextdelaydm1` and `contextdelaydm2`) during training to prevent the
    network from adopting a suboptimal strategy (integrating both modalities
    equally, which yields at most ~75% accuracy).

    Args:
        oversample_context: If True, use the paper's default 5x oversampling.
            If False, use equal probabilities for all tasks. If a float, use
            that value as the relative weight for the two context tasks.
    """
    if oversample_context is False:
        return {}
    weight = 5.0 if oversample_context is True else float(oversample_context)
    return {
        'contextdelaydm1': weight,
        'contextdelaydm2': weight,
    }


class MultitaskFlexibleDataset(BaseDataset):
    """Dataset that interleaves the 15 Driscoll et al. (2024) tasks.

    Args:
        rules: list of rule names to train on. If None, uses all 15 rules.
        rule_prob_map: dict mapping rule name to relative sampling probability.
            Rules not in the map default to 1.0. If None, the paper's default
            5x oversampling of context-dependent delayed DM tasks is used.
            Pass an empty dict (or set `oversample_context=False`) for uniform
            sampling.
        oversample_context: convenience flag to enable/disable the paper's 5x
            oversampling. Only used when `rule_prob_map` is None.
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
        oversample_context: bool | float = True,
        batch_size: int = 64,
        mode: str = "random",
        sigma_x: float = 0.01,
        seed: int | None = None,
    ) -> None:
        self.rules = list(rules) if rules is not None else list(RULES_ALL)
        if rule_prob_map is not None:
            self.rule_prob_map = rule_prob_map
        else:
            self.rule_prob_map = get_flexible_rule_prob_map(oversample_context)
        self.oversample_context = oversample_context
        self.batch_size = batch_size
        self.mode = mode
        self.sigma_x = sigma_x

        self.input_dim = 20
        self.output_dim = 3

        self.rng = np.random.default_rng(seed)

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
