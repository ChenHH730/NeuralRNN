"""Tests for the NeuralRNN data layer.

Covers:
  - BaseDataset subclassing contract
  - ReconstructionDataset.from_timeseries construction and batch shapes
  - CustomDataset with inputs/targets/states
  - Registered dataset names and load_dataset entry point
  - download.py cache helpers (without network access)
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from neuralrnn.data import BaseDataset, CustomDataset, ReconstructionDataset
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


class TestFromTimeseries:
    def test_sample_batch_shapes(self):
        data = np.random.randn(500, 3).astype(np.float32)
        ds = ReconstructionDataset.from_timeseries(data, sequence_length=50, batch_size=8)
        batch = ds.sample_batch()

        assert ds.output_dim == 3
        assert ds.input_dim == 0
        assert batch["activity"].shape == (8, 50, 3)
        assert "inputs" not in batch
        assert len(ds) == 500 - 50

    def test_external_inputs_and_normalization(self):
        data = np.random.randn(300, 2).astype(np.float32) * 5 + 10
        ext = np.random.randn(300, 1).astype(np.float32)
        ds = ReconstructionDataset.from_timeseries(
            data, external_inputs=ext, sequence_length=20, batch_size=4, normalize=True)
        batch = ds.sample_batch()
        assert batch["activity"].shape == (4, 20, 2)
        assert batch["inputs"].shape == (4, 20, 1)
        assert ds.input_dim == 1
        assert ds.normalizer is not None
        assert abs(float(ds.X.mean())) < 0.2  # roughly standardized


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


class TestDatasetRegistry:
    def test_known_datasets_are_registered(self):
        for name in [
            "perceptual_decision_making",
            "delay_comparison",
            "lorenz63",
            "mante",
            "siegel_miller",
            "dms_continuous",
            "wm_angle",
            "wm_frequency",
            "bartolo_monkey",
        ]:
            assert name in DATASET_REGISTRY

    def test_load_cognitive_tasks(self):
        ds = load_dataset("mante", n_trials=2, n_coh=2, batch_size=4)
        assert ds.inputs.shape == (16, 75, 6)
        assert ds.targets.shape == (16, 75, 2)
        assert ds.mask.shape == (16, 75, 2)

        ds_alias = load_dataset("siegel_miller", n_trials=2, n_coh=2, batch_size=4)
        assert ds_alias.inputs.shape == ds.inputs.shape

        ds_cont = load_dataset("dms_continuous", n_trials=2, n_coh=2, batch_size=4)
        assert ds_cont.inputs.shape[-1] == 4
        assert ds_cont.targets.shape[-1] == 2
        assert ds_cont.mask.shape == ds_cont.targets.shape

        ds_wm = load_dataset("wm_angle", n_trials=4, batch_size=4)
        assert ds_wm.inputs.shape[-1] == 2
        assert ds_wm.targets.shape[-1] == 2

        ds_romo = load_dataset("wm_frequency", num_trials=4, batch_size=4)
        assert ds_romo.inputs.shape[-1] == 1
        assert ds_romo.targets.shape[-1] == 1

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
