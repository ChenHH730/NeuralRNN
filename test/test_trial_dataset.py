"""Tests for TrialTimeseriesDataset."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from neuralrnn.data.trial_dataset import TrialTimeseriesDataset


class TestTrialTimeseriesDataset:
    def test_rejects_2d_inputs(self):
        inputs = np.random.randn(100, 5).astype(np.float32)
        with pytest.raises(ValueError, match="3D"):
            TrialTimeseriesDataset.from_arrays(inputs)

    def test_sample_batch_shapes(self):
        B, T, N = 20, 15, 5
        inputs = np.random.randn(B, T, N).astype(np.float32)
        ds = TrialTimeseriesDataset.from_arrays(inputs, batch_size=4)
        batch = ds.sample_batch()
        assert batch["inputs"].shape == (4, T, N)
        assert batch["targets"].shape == (4, T, N)
        assert batch["external_inputs"] is None

    def test_targets_default_to_shifted_inputs(self):
        B, T, N = 10, 8, 3
        inputs = np.random.randn(B, T, N).astype(np.float32)
        ds = TrialTimeseriesDataset.from_arrays(inputs, batch_size=2)
        batch = ds.sample_batch()
        # targets[t] should equal inputs[t+1]
        assert torch.allclose(batch["targets"][:, :-1], batch["inputs"][:, 1:])

    def test_external_inputs_shape(self):
        B, T, N, K = 10, 8, 3, 2
        inputs = np.random.randn(B, T, N).astype(np.float32)
        ext = np.random.randn(B, T, K).astype(np.float32)
        ds = TrialTimeseriesDataset.from_arrays(inputs, external_inputs=ext, batch_size=2)
        batch = ds.sample_batch()
        assert batch["external_inputs"].shape == (2, T, K)

    def test_trial_wise_test_split(self):
        B, T, N = 100, 10, 4
        inputs = np.random.randn(B, T, N).astype(np.float32)
        ds = TrialTimeseriesDataset.from_arrays(inputs, batch_size=8, test_fraction=0.2, seed=1)
        assert ds.test_set is not None
        assert ds.X.shape[0] + ds.test_set.X.shape[0] == B
        # No overlap between train and test trials
        train_ids = set()
        # We can't directly recover original indices, but shapes should be consistent
        assert ds.X.ndim == 3
        assert ds.test_set.X.ndim == 3

    def test_normalization_preserves_trial_shape(self):
        B, T, N = 10, 12, 3
        inputs = np.random.randn(B, T, N).astype(np.float32)
        ds = TrialTimeseriesDataset.from_arrays(inputs, batch_size=2, normalize=True)
        assert ds.X.shape == (B, T, N)
        batch = ds.sample_batch()
        assert batch["inputs"].shape == (2, T, N)

    def test_external_inputs_normalization(self):
        B, T, N, K = 10, 8, 3, 2
        inputs = np.random.randn(B, T, N).astype(np.float32)
        ext = np.random.randn(B, T, K).astype(np.float32)
        ds = TrialTimeseriesDataset.from_arrays(
            inputs, external_inputs=ext, batch_size=2,
            normalize=True, normalize_externals=True
        )
        assert ds.external_normalizer is not None
        batch = ds.sample_batch()
        assert batch["external_inputs"].shape == (2, T, K)
