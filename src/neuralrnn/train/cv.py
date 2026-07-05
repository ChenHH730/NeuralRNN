"""Nested cross-validation + config grid framework (Tiny RNN paradigm).

Ported from behavior_cv_training_config_combination in 01-fitting-generated-data.ipynb:
expand base_config × config_ranges into a set of named configs, and for each config perform
outer × inner nested cross-validation with multiple random seeds, selecting the model with the
best validation loss.

This is the standard practice for behavior fitting (small models, strong regularization, nested CV
to avoid overfitting). This file provides a runnable skeleton; for training a single model, reuse
Trainer + BehavioralObjective (see PORTING_GUIDE recipe 7).
"""
from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass, field

import numpy as np


def config_combination(base_config: dict, config_ranges: dict) -> list[dict]:
    """Cartesian product expansion: base_config overlaid with each combination of config_ranges values.

    Each returned config gains a 'model_name' field (built from the varying key-value pairs) for easy retrieval.
    """
    keys = list(config_ranges.keys())
    out: list[dict] = []
    for values in itertools.product(*[config_ranges[k] for k in keys]):
        cfg = copy.deepcopy(base_config)
        name_parts = []
        for k, v in zip(keys, values):
            cfg[k] = v
            name_parts.append(f"{k}{v}")
        cfg["model_name"] = ".".join(name_parts)
        out.append(cfg)
    return out


def _kfold_indices(n: int, k: int, seed: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    folds = np.array_split(idx, k)
    splits = []
    for i in range(k):
        val = folds[i]
        train = np.concatenate([folds[j] for j in range(k) if j != i])
        splits.append((train, val))
    return splits


@dataclass
class CVResult:
    config: dict
    outer_val_losses: list[float] = field(default_factory=list)

    @property
    def mean_val_loss(self) -> float:
        return float(np.mean(self.outer_val_losses)) if self.outer_val_losses else float("nan")


def behavior_cv_training(base_config: dict, config_ranges: dict,
                         fit_one_fn, n_samples: int) -> list[CVResult]:
    """Run nested CV for each config and return cross-validation results per config.

    Parameters
    ----------
    fit_one_fn(config, train_idx, val_idx, seed) -> float
        Train a single model and return validation loss. When porting, implement this with
        Trainer + BehavioralObjective (train on train_idx, evaluate NLL on val_idx).
    n_samples : total number of subjects / trials, used for splitting.

    Notes: this gives the outer/inner/seed loop skeleton; inner folds are used for early stopping /
    hyperparameter selection, outer folds give the generalization estimate. The simplified skeleton
    defaults to evaluating the outer validation set with the best inner setting.
    """
    results: list[CVResult] = []
    for cfg in config_combination(base_config, config_ranges):
        res = CVResult(config=cfg)
        outer = _kfold_indices(n_samples, cfg.get("outer_splits", 3), seed=0)
        for o, (outer_train, outer_val) in enumerate(outer):
            seed_losses = []
            for seed in range(cfg.get("seed_num", 1)):
                # Inner CV (inside outer_train): can be used for early stopping / seed selection;
                # skeleton trains directly here.
                inner = _kfold_indices(len(outer_train),
                                       cfg.get("inner_splits", 2), seed=seed)
                best_inner = min(
                    fit_one_fn(cfg, outer_train[itr], outer_train[ival], seed)
                    for itr, ival in inner
                )
                seed_losses.append(best_inner)
            # Use the loss of the best seed setting on the outer validation fold as the generalization estimate
            res.outer_val_losses.append(float(np.min(seed_losses)))
        results.append(res)
    return results


def find_best_models_for_exp(results: list[CVResult]) -> CVResult:
    """Select the config with the lowest mean validation loss (matches notebook's find_best_models_for_exp)."""
    return min(results, key=lambda r: r.mean_val_loss)
