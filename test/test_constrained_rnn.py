"""Tests for constrained RNN models."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from neuralrnn import AutoConfig, AutoModel
from neuralrnn.models.constrained_rnn import (
    ConstrainedRNNConfig,
    ConstrainedRNNModel,
    ModularRNNConfig,
    SERNNConfig,
    SparseRNNConfig,
)
from neuralrnn.train.objectives.constrained import ConstrainedSupervisedObjective


@pytest.mark.parametrize("model_type", ["constrained_rnn", "se_rnn", "sparse_rnn", "modular_rnn"])
def test_autoconfig_automodel(model_type):
    cfg = AutoConfig.for_model(
        model_type,
        input_dim=3,
        latent_dim=64,
        output_dim=3,
        dt=100.0,
        tau=100.0,
    )
    model = AutoModel.from_config(cfg)
    assert model.config.model_type == model_type
    x = torch.randn(2, 10, 3)
    out = model(x)
    assert out.states.shape == (2, 10, 64)
    assert out.outputs.shape == (2, 10, 3)


def test_custom_masks_remain_zero():
    M = 8
    rec_mask = np.eye(M, dtype=np.float32)
    in_mask = np.ones((3, M), dtype=np.float32)
    in_mask[:, 0] = 0.0
    out_mask = np.ones((M, 2), dtype=np.float32)
    out_mask[0, :] = 0.0

    cfg = ConstrainedRNNConfig(
        input_dim=3,
        latent_dim=M,
        output_dim=2,
        dt=10.0,
        tau=10.0,
        rec_mask=rec_mask,
        in_mask=in_mask,
        out_mask=out_mask,
    )
    model = ConstrainedRNNModel(cfg)

    # Verify initial projection
    assert (model.h2h.weight * (1 - model.rec_mask)).abs().max().item() == 0.0
    assert (model.input2h.weight * (1 - model.in_mask.t())).abs().max().item() == 0.0
    assert (model.readout_layer.weight * (1 - model.out_mask.t())).abs().max().item() == 0.0

    # Take an optimizer step and verify masks still hold
    opt = torch.optim.Adam(model.parameters(), lr=0.1)
    x = torch.randn(2, 5, 3)
    out = model(x)
    loss = out.outputs.mean()
    loss.backward()
    opt.step()
    model._apply_masks_to_weights()

    assert (model.h2h.weight * (1 - model.rec_mask)).abs().max().item() == 0.0
    assert (model.input2h.weight * (1 - model.in_mask.t())).abs().max().item() == 0.0
    assert (model.readout_layer.weight * (1 - model.out_mask.t())).abs().max().item() == 0.0


def test_sparse_rnn_mask_density():
    M = 100
    sparsity = 0.05
    cfg = SparseRNNConfig(
        input_dim=3,
        latent_dim=M,
        output_dim=3,
        sparsity=sparsity,
        allow_self_connections=False,
        seed=0,
    )
    model = AutoModel.from_config(cfg)
    rec_mask = model.rec_mask.cpu().numpy()
    np.fill_diagonal(rec_mask, 0.0)  # autapses excluded
    density = rec_mask.sum() / (M * M)
    assert abs(density - sparsity) < 0.02
    assert np.diag(model.rec_mask.cpu().numpy()).sum() == 0.0


def test_modular_rnn_block_structure():
    M = 60
    n_modules = 6
    cfg = ModularRNNConfig(
        input_dim=3,
        latent_dim=M,
        output_dim=3,
        n_modules=n_modules,
        p_inter=0.0,
        intra_density=1.0,
        allow_self_connections=False,
        seed=0,
    )
    model = AutoModel.from_config(cfg)
    mask = model.rec_mask.cpu().numpy()
    module_size = M // n_modules
    for i in range(n_modules):
        for j in range(n_modules):
            block = mask[i * module_size : (i + 1) * module_size,
                         j * module_size : (j + 1) * module_size]
            if i == j:
                # diagonal removed
                assert block.sum() == module_size * (module_size - 1)
            else:
                assert block.sum() == 0.0


def test_se_rnn_regularizer_finite():
    cfg = SERNNConfig(
        input_dim=3,
        latent_dim=100,
        output_dim=3,
        grid_shape=(5, 5, 4),
        se1_weight=0.5,
        comms_factor=1.0,
    )
    model = AutoModel.from_config(cfg)
    reg = model.constraint_loss()
    assert torch.isfinite(reg)
    assert reg.item() >= 0.0


def test_constrained_objective_adds_regularizer():
    cfg = SERNNConfig(
        input_dim=2,
        latent_dim=16,
        output_dim=2,
        grid_shape=(4, 4),
        se1_weight=1.0,
    )
    model = AutoModel.from_config(cfg)
    obj = ConstrainedSupervisedObjective(task_type="classification", constraint_weight=1.0)
    batch = {
        "inputs": torch.randn(4, 5, 2),
        "targets": torch.randint(0, 2, (4, 5)),
    }
    loss, logs = obj.compute_loss(model, batch)
    assert "constraint_loss" in logs
    assert "task_loss" in logs
    assert torch.isfinite(loss)


def test_save_load_roundtrip(tmp_path):
    cfg = SparseRNNConfig(
        input_dim=3,
        latent_dim=32,
        output_dim=3,
        sparsity=0.1,
        seed=7,
    )
    model = AutoModel.from_config(cfg)
    save_dir = tmp_path / "sparse_rnn_test"
    model.save_pretrained(str(save_dir))

    model2 = AutoModel.from_pretrained(str(save_dir))
    assert isinstance(model2.config, SparseRNNConfig)
    assert torch.allclose(model.rec_mask, model2.rec_mask)
    assert torch.allclose(model.h2h.weight, model2.h2h.weight)

    x = torch.randn(2, 5, 3)
    with torch.no_grad():
        out1 = model(x)
        out2 = model2(x)
    assert torch.allclose(out1.outputs, out2.outputs, atol=1e-6)
