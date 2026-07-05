"""Tests for the NeuralRNN data layer.

Covers:
  - BaseDataset subclassing contract
  - TimeSeriesDataset construction and batch shapes
  - CustomDataset with inputs/targets/states
  - Registered dataset names and load_dataset entry point
  - download.py cache helpers (without network access)
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from neuralrnn.data import BaseDataset, TimeSeriesDataset, CustomDataset, TrialTimeseriesDataset
from neuralrnn.data.registry import DATASET_REGISTRY, load_dataset
from neuralrnn.data import download as download_module


class _MinimalToyDataset(BaseDataset):
    kind = "timeseries"

    def __init__(self, input_dim: int = 2, output_dim: int = 2, B: int = 4, T: int = 10):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.B = B
        self.T = T

    def sample_batch(self):
        x = torch.randn(self.B, self.T, self.input_dim)
        y = torch.randn(self.B, self.T, self.output_dim)
        return {"inputs": x, "targets": y}


class TestBaseDatasetContract:
    def test_subclass_exposes_kind_and_batch(self):
        ds = _MinimalToyDataset()
        batch = ds.sample_batch()
        assert ds.kind == "timeseries"
        assert "inputs" in batch
        assert "targets" in batch
        assert batch["inputs"].shape == (4, 10, 2)


class TestTimeSeriesDataset:
    def test_sample_batch_shapes(self):
        data = np.random.randn(500, 3).astype(np.float32)
        ds = TimeSeriesDataset(data, sequence_length=50, batch_size=8)
        batch = ds.sample_batch()

        assert ds.input_dim == 3
        assert ds.output_dim == 3
        assert batch["inputs"].shape == (8, 50, 3)
        assert batch["targets"].shape == (8, 50, 3)
        assert batch["external_inputs"] is None


class TestCustomDataset:
    def test_timeseries_mode_shapes(self):
        T, N = 300, 4
        inputs = np.random.randn(T, N).astype(np.float32)
        targets = np.random.randn(T, N).astype(np.float32)
        ds = CustomDataset.from_arrays(
            inputs, targets=targets, mode="timeseries", sequence_length=40, batch_size=6
        )
        batch = ds.sample_batch()

        assert ds.mode == "timeseries"
        assert batch["inputs"].shape == (6, 40, 4)
        assert batch["targets"].shape == (6, 40, 4)
        assert batch["external_inputs"] is None

    def test_supervised_mode_shapes(self):
        B, T, D, O = 5, 20, 3, 2
        inputs = np.random.randn(B, T, D).astype(np.float32)
        targets = np.random.randn(B, T, O).astype(np.float32)
        ds = CustomDataset.from_arrays(
            inputs, targets=targets, mode="supervised", batch_size=2
        )
        batch = ds.sample_batch()

        assert ds.mode == "supervised"
        assert batch["inputs"].shape == (2, T, D)
        assert batch["targets"].shape == (2, T, O)
        assert batch["mask"].shape == (2, T)

    def test_optional_internal_states_stored(self):
        T, N, M = 200, 3, 8
        inputs = np.random.randn(T, N).astype(np.float32)
        states = np.random.randn(T, M).astype(np.float32)
        ds = CustomDataset.from_arrays(
            inputs, internal_states=states, mode="timeseries", sequence_length=30, batch_size=4
        )
        assert ds.IS is not None
        assert ds.IS.shape == (T, M)


class TestTrialTimeseriesDataset:
    def test_trial_dataset_registered_and_runs(self):
        B, T, N = 16, 10, 4
        inputs = np.random.randn(B, T, N).astype(np.float32)
        ds = TrialTimeseriesDataset.from_arrays(inputs, batch_size=4)
        batch = ds.sample_batch()
        assert batch["inputs"].shape == (4, T, N)
        assert batch["targets"].shape == (4, T, N)

    def test_trial_dataset_preserves_trials_in_test_split(self):
        B, T, N = 50, 8, 3
        inputs = np.random.randn(B, T, N).astype(np.float32)
        ds = TrialTimeseriesDataset.from_arrays(inputs, batch_size=4, test_fraction=0.2, seed=0)
        assert ds.test_set is not None
        assert ds.X.shape[0] + ds.test_set.X.shape[0] == B


class TestDatasetRegistry:
    def test_known_datasets_are_registered(self):
        for name in [
            "perceptual_decision_making",
            "delay_comparison",
            "lorenz63",
            "mante",
            "bartolo_monkey",
        ]:
            assert name in DATASET_REGISTRY

    def test_load_dataset_raises_for_unknown_name(self):
        with pytest.raises(KeyError):
            load_dataset("definitely_not_a_registered_dataset")

    def test_load_perceptual_decision_making_if_neurogym_installed(self):
        pytest.importorskip("neurogym")
        ds = load_dataset("perceptual_decision_making", batch_size=4, seq_len=50)
        batch = ds.sample_batch()
        assert batch["inputs"].ndim == 3
        assert batch["targets"].ndim == 2
        assert batch["inputs"].shape[0] == batch["targets"].shape[0]


class TestDownloadHelpers:
    def test_cache_root_returns_existing_path(self, tmp_path, monkeypatch):
        # Temporarily redirect cache so we do not touch the user's home cache.
        monkeypatch.setenv("NEURALRNN_CACHE", str(tmp_path))
        root = download_module.cache_root()
        assert root.exists()
        assert root == tmp_path

    def test_sha256_on_temporary_file(self, tmp_path):
        path = tmp_path / "hello.txt"
        path.write_bytes(b"hello")
        digest = download_module._sha256(path)
        assert isinstance(digest, str)
        assert len(digest) == 64
