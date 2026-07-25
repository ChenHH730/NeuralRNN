"""Tests for CustomDataset reconstruction mode and objective compatibility."""
from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch

from neuralrnn.data.custom_dataset import CustomDataset
from neuralrnn.train.objectives.reconstruction import ReconstructionObjective
from neuralrnn.train.objectives.teacher_forcing import TeacherForcingObjective


def test_reconstruction_mode_batch_keys():
    """Reconstruction mode sample_batch returns unified keys."""
    traj = np.random.randn(500, 3).astype(np.float32)
    ds = CustomDataset.from_arrays(traj, mode="reconstruction", sequence_length=50, batch_size=4)
    batch = ds.sample_batch()

    assert "activity" in batch
    assert "inputs" not in batch  # no external inputs provided
    assert "targets" not in batch
    assert "mask" not in batch
    assert batch["activity"].shape == (4, 50, 3)


def test_reconstruction_mode_with_optional_fields():
    """Reconstruction mode includes inputs, targets, mask when provided."""
    T, N, K, O = 500, 3, 2, 1
    traj = np.random.randn(T, N).astype(np.float32)
    ext = np.random.randn(T, K).astype(np.float32)
    tgt = np.random.randn(T, O).astype(np.float32)
    msk = np.ones((T, 1), dtype=np.float32)

    ds = CustomDataset.from_arrays(
        traj, targets=tgt, external_inputs=ext, mask=msk,
        mode="reconstruction", sequence_length=50, batch_size=4,
    )
    batch = ds.sample_batch()

    assert batch["activity"].shape == (4, 50, N)
    assert batch["inputs"].shape == (4, 50, K)
    assert batch["targets"].shape == (4, 50, O)
    assert batch["mask"].shape == (4, 50, 1)


def test_reconstruction_mode_auto_detect():
    """Auto-detect falls to reconstruction when inputs and targets have same shape."""
    traj = np.random.randn(500, 3).astype(np.float32)
    ds = CustomDataset.from_arrays(traj, mode="auto", sequence_length=50, batch_size=4)
    assert ds.mode == "reconstruction"


def test_reconstruction_mode_external_inputs_only():
    """Only external_inputs provided: batch has inputs but no targets/mask."""
    T, N, K = 500, 3, 2
    traj = np.random.randn(T, N).astype(np.float32)
    ext = np.random.randn(T, K).astype(np.float32)

    ds = CustomDataset.from_arrays(
        traj, external_inputs=ext,
        mode="reconstruction", sequence_length=50, batch_size=4,
    )
    batch = ds.sample_batch()
    assert "activity" in batch
    assert "inputs" in batch
    assert "targets" not in batch
    assert "mask" not in batch


def test_reconstruction_with_teacher_forcing_objective():
    """CustomDataset reconstruction mode feeds TeacherForcingObjective correctly."""
    from neuralrnn import AutoConfig, AutoModel

    torch.manual_seed(0)
    np.random.seed(0)

    n_steps = 1000
    dt = 0.05
    t = np.arange(n_steps) * dt
    trajectory = np.zeros((n_steps, 2), dtype=np.float32)
    trajectory[0] = [1.0, 0.0]
    for i in range(1, n_steps):
        x, y = trajectory[i - 1]
        trajectory[i, 0] = x + dt * y
        trajectory[i, 1] = y + dt * (-0.5 * y - 4.0 * x)

    ds = CustomDataset.from_arrays(
        trajectory, mode="reconstruction", sequence_length=50, batch_size=4,
    )
    batch = ds.sample_batch()
    assert "activity" in batch

    cfg = AutoConfig.for_model(
        "shallow_plrnn", latent_dim=2, hidden_dim=16, output_dim=2, autonomous=True,
    )
    model = AutoModel.from_config(cfg)
    obj = TeacherForcingObjective(alpha=0.1)
    loss, logs = obj.compute_loss(model, batch)
    assert isinstance(loss, torch.Tensor)
    assert "loss" in logs
    assert loss.item() >= 0


def test_reconstruction_with_reconstruction_objective():
    """CustomDataset reconstruction mode feeds ReconstructionObjective correctly."""
    from neuralrnn import AutoConfig, AutoModel

    torch.manual_seed(1)
    B, T, K, N, O = 32, 20, 2, 4, 1
    inputs = torch.randn(B, T, K)
    activity = torch.randn(B, T, N)
    targets = torch.randn(B, T, O)
    mask = torch.ones(B, T)

    # Use CustomDataset in reconstruction mode with pre-shaped tensors
    # Note: CustomDataset expects time-first 2D arrays, so we flatten batch dim
    ds = CustomDataset.from_arrays(
        inputs=activity.reshape(-1, N).numpy(),
        targets=targets.reshape(-1, O).numpy(),
        external_inputs=inputs.reshape(-1, K).numpy(),
        mask=mask.reshape(-1, 1).numpy(),
        mode="reconstruction",
        sequence_length=T,
        batch_size=8,
    )
    batch = ds.sample_batch()

    cfg = AutoConfig.for_model("ctrnn", input_dim=K, latent_dim=N, output_dim=O, dt=20, tau=100)
    model = AutoModel.from_config(cfg)
    obj = ReconstructionObjective(behavior_weight=1.0, activity_weight=1.0, state_map="identity")
    loss, logs = obj.compute_loss(model, batch)
    assert isinstance(loss, torch.Tensor)
    assert "loss" in logs
    assert "mse_z" in logs
    assert "nmse_y" in logs


def test_timeseries_deprecated_alias():
    """mode='timeseries' is accepted but raises a DeprecationWarning."""
    traj = np.random.randn(100, 2).astype(np.float32)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ds = CustomDataset.from_arrays(traj, mode="timeseries", sequence_length=20, batch_size=2)
        assert ds.mode == "reconstruction"
        assert any(issubclass(warning.category, DeprecationWarning) for warning in w)


def test_reconstruction_test_set():
    """Test set property works in reconstruction mode."""
    traj = np.random.randn(200, 3).astype(np.float32)
    ext = np.random.randn(200, 2).astype(np.float32)
    tgt = np.random.randn(200, 1).astype(np.float32)
    msk = np.ones((200, 1), dtype=np.float32)

    ds = CustomDataset.from_arrays(
        traj, targets=tgt, external_inputs=ext, mask=msk,
        mode="reconstruction", sequence_length=30, batch_size=4, test_fraction=0.2, seed=0,
    )
    assert ds.test_set is not None
    test_batch = ds.test_set.sample_batch()
    assert "activity" in test_batch
    assert "inputs" in test_batch
    assert "targets" in test_batch
    assert "mask" in test_batch


def test_reconstruction_len():
    """__len__ for reconstruction mode is T - sequence_length."""
    traj = np.random.randn(100, 2).astype(np.float32)
    ds = CustomDataset.from_arrays(traj, mode="reconstruction", sequence_length=30, batch_size=2)
    assert len(ds) == 100 - 30


def test_reconstruction_activity_alias():
    """activity property returns X in reconstruction mode."""
    traj = np.random.randn(100, 2).astype(np.float32)
    ds = CustomDataset.from_arrays(traj, mode="reconstruction", sequence_length=20, batch_size=2)
    assert torch.allclose(ds.activity, ds.X)
