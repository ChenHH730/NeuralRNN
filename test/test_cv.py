"""Tests for cross-validation utilities in train/cv.py.

Covers:
  - config_combination Cartesian product and model_name generation
  - _kfold_indices split invariants
  - behavior_cv_training smoke test with a mock fit_one_fn
  - find_best_models_for_exp selection logic
"""
from __future__ import annotations

import numpy as np
import pytest

from neuralrnn.train.cv import (
    config_combination,
    _kfold_indices,
    behavior_cv_training,
    find_best_models_for_exp,
    CVResult,
)


def test_config_combination_cartesian_product():
    base = {"model_type": "ctrnn", "epochs": 10}
    ranges = {"lr": [0.01, 0.1], "dropout": [0.0, 0.5]}
    configs = config_combination(base, ranges)

    assert len(configs) == 4
    assert all("model_name" in c for c in configs)
    assert all(c["model_type"] == "ctrnn" and c["epochs"] == 10 for c in configs)
    # Every combination appears exactly once.
    pairs = [(c["lr"], c["dropout"]) for c in configs]
    assert sorted(pairs) == [(0.01, 0.0), (0.01, 0.5), (0.1, 0.0), (0.1, 0.5)]
    names = {c["model_name"] for c in configs}
    assert names == {"lr0.01.dropout0.0", "lr0.01.dropout0.5", "lr0.1.dropout0.0", "lr0.1.dropout0.5"}


def test_kfold_indices_disjoint_and_cover_all_indices():
    n = 20
    k = 4
    splits = _kfold_indices(n, k, seed=42)
    assert len(splits) == k

    all_indices = set(range(n))
    for train, val in splits:
        assert len(train) + len(val) == n
        assert len(set(train) & set(val)) == 0
        assert set(train) | set(val) == all_indices


def test_behavior_cv_training_smoke():
    def mock_fit(config, train_idx, val_idx, seed):
        # Lower learning rate => lower validation loss, for selection testing.
        return float(config["lr"]) * len(val_idx)

    base = {"model_type": "ctrnn", "lr": 0.01, "outer_splits": 2, "inner_splits": 2, "seed_num": 1}
    ranges = {"lr": [0.01, 0.1]}
    results = behavior_cv_training(base, ranges, mock_fit, n_samples=16)

    assert len(results) == 2
    assert all(isinstance(r, CVResult) for r in results)
    assert all(len(r.outer_val_losses) == 2 for r in results)
    assert all(np.isfinite(r.mean_val_loss) for r in results)


def test_find_best_models_selects_lowest_mean_val_loss():
    a = CVResult(config={"lr": 0.1}, outer_val_losses=[0.5, 0.6])
    b = CVResult(config={"lr": 0.01}, outer_val_losses=[0.2, 0.3])
    c = CVResult(config={"lr": 1.0}, outer_val_losses=[0.9, 1.0])

    best = find_best_models_for_exp([a, b, c])
    assert best.config["lr"] == 0.01
    assert np.isclose(best.mean_val_loss, 0.25)
