"""Tests for DendPLRNN and ALRNN model variants."""
from __future__ import annotations

import pytest
import torch

from neuralrnn import AutoConfig, AutoModel
from neuralrnn.analysis.fixed_points import find_fixed_points
from neuralrnn.train.objectives.teacher_forcing import TeacherForcingObjective


def _random_batch(B: int, T: int, N: int, K: int = 0):
    batch = {"activity": torch.randn(B, T, N)}
    if K > 0:
        batch["inputs"] = torch.randn(B, T, K)
    return batch


class TestDendPLRNN:
    def test_construct_from_config(self):
        cfg = AutoConfig.for_model("dend_plrnn", latent_dim=4, n_bases=3, input_dim=0)
        model = AutoModel.from_config(cfg)
        assert model.config.model_type == "dend_plrnn"
        assert model.config.n_bases == 3

    def test_forward_shape_autonomous(self):
        cfg = AutoConfig.for_model("dend_plrnn", latent_dim=4, n_bases=3, input_dim=0)
        model = AutoModel.from_config(cfg)
        z0 = torch.randn(2, 4)
        out = model.forward(None, initial_state=z0, n_steps=10)
        assert out.states.shape == (2, 10, 4)
        assert out.outputs.shape == (2, 10, 4)

    def test_forward_shape_with_inputs(self):
        cfg = AutoConfig.for_model("dend_plrnn", latent_dim=4, n_bases=3, input_dim=2)
        model = AutoModel.from_config(cfg)
        batch = _random_batch(2, 10, 4, K=2)
        out = model.forward(batch["inputs"])
        assert out.states.shape == (2, 10, 4)

    def test_jacobian_matches_autodiff(self):
        cfg = AutoConfig.for_model("dend_plrnn", latent_dim=3, n_bases=2, input_dim=0)
        model = AutoModel.from_config(cfg)
        z = torch.randn(3, requires_grad=True)
        J_analytic = model.jacobian(z)
        J_auto = torch.autograd.functional.jacobian(
            lambda zz: model.recurrence(None, zz.unsqueeze(0)).squeeze(0), z
        )
        assert torch.allclose(J_analytic, J_auto, atol=1e-5)

    def test_analytic_fixed_points_runs(self):
        cfg = AutoConfig.for_model("dend_plrnn", latent_dim=3, n_bases=2, input_dim=0)
        model = AutoModel.from_config(cfg)
        assert model.supports_analytic_fixed_points
        fps = find_fixed_points(model, backend="analytic", max_order=1)
        assert isinstance(fps.points, list)

    def test_analytic_parameters_task_input(self):
        cfg = AutoConfig.for_model("dend_plrnn", latent_dim=3, n_bases=2,
                                   input_dim=2, autonomous=False)
        model = AutoModel.from_config(cfg)
        task_input = torch.tensor([0.5, -0.3], dtype=torch.float32)
        p = model.analytic_parameters(task_input=task_input)
        assert set(p.keys()) == {"A", "W1", "W2", "h1", "h2"}
        p_auto = model.analytic_parameters()
        assert not torch.allclose(p["h1"], p_auto["h1"])

    def test_teacher_forcing_loss(self):
        cfg = AutoConfig.for_model("dend_plrnn", latent_dim=4, n_bases=2, input_dim=0)
        model = AutoModel.from_config(cfg)
        batch = _random_batch(2, 10, 4)
        obj = TeacherForcingObjective(alpha=0.1)
        loss, info = obj.compute_loss(model, batch)
        assert loss.ndim == 0
        assert "loss" in info


class TestALRNN:
    def test_construct_from_config(self):
        cfg = AutoConfig.for_model("alrnn", latent_dim=5, n_linear=3, input_dim=0)
        model = AutoModel.from_config(cfg)
        assert model.config.model_type == "alrnn"
        assert model.n_linear == 3
        assert model.n_relu == 2

    def test_n_linear_equal_latent_dim_raises(self):
        cfg = AutoConfig.for_model("alrnn", latent_dim=3, n_linear=3, input_dim=0)
        with pytest.raises(ValueError):
            AutoModel.from_config(cfg)

    def test_forward_shape_autonomous(self):
        cfg = AutoConfig.for_model("alrnn", latent_dim=5, n_linear=3, input_dim=0)
        model = AutoModel.from_config(cfg)
        z0 = torch.randn(2, 5)
        out = model.forward(None, initial_state=z0, n_steps=10)
        assert out.states.shape == (2, 10, 5)

    def test_forward_shape_with_inputs(self):
        cfg = AutoConfig.for_model("alrnn", latent_dim=5, n_linear=3, input_dim=2)
        model = AutoModel.from_config(cfg)
        batch = _random_batch(2, 10, 5, K=2)
        out = model.forward(batch["inputs"])
        assert out.states.shape == (2, 10, 5)

    def test_jacobian_matches_autodiff(self):
        cfg = AutoConfig.for_model("alrnn", latent_dim=4, n_linear=2, input_dim=0)
        model = AutoModel.from_config(cfg)
        z = torch.randn(4, requires_grad=True)
        J_analytic = model.jacobian(z)
        J_auto = torch.autograd.functional.jacobian(
            lambda zz: model.recurrence(None, zz.unsqueeze(0)).squeeze(0), z
        )
        assert torch.allclose(J_analytic, J_auto, atol=1e-5)

    def test_analytic_fixed_points_runs(self):
        cfg = AutoConfig.for_model("alrnn", latent_dim=4, n_linear=2, input_dim=0)
        model = AutoModel.from_config(cfg)
        assert model.supports_analytic_fixed_points
        fps = find_fixed_points(model, backend="analytic", max_order=1)
        assert isinstance(fps.points, list)

    def test_analytic_parameters_task_input(self):
        cfg = AutoConfig.for_model("alrnn", latent_dim=4, n_linear=2,
                                   input_dim=2, autonomous=False)
        model = AutoModel.from_config(cfg)
        task_input = torch.tensor([0.5, -0.3], dtype=torch.float32)
        p = model.analytic_parameters(task_input=task_input)
        assert set(p.keys()) == {"A", "W1", "W2", "h1", "h2"}
        p_auto = model.analytic_parameters()
        assert not torch.allclose(p["h1"], p_auto["h1"])

    def test_sparse_teacher_forcing_loss(self):
        cfg = AutoConfig.for_model("alrnn", latent_dim=4, n_linear=2, input_dim=0)
        model = AutoModel.from_config(cfg)
        batch = _random_batch(2, 20, 4)
        obj = TeacherForcingObjective(alpha=0.5, forcing_interval=5)
        loss, info = obj.compute_loss(model, batch)
        assert loss.ndim == 0


class TestShallowPLRNNBackwardCompat:
    """Ensure existing shallowPLRNN behavior is unchanged."""

    def test_construct_and_forward(self):
        cfg = AutoConfig.for_model("shallow_plrnn", latent_dim=3, hidden_dim=10, input_dim=0)
        model = AutoModel.from_config(cfg)
        z0 = torch.randn(1, 3)
        out = model.forward(None, initial_state=z0, n_steps=5)
        assert out.states.shape == (1, 5, 3)

    def test_analytic_parameters_keys(self):
        cfg = AutoConfig.for_model("shallow_plrnn", latent_dim=3, hidden_dim=10, input_dim=0)
        model = AutoModel.from_config(cfg)
        p = model.analytic_parameters()
        assert set(p.keys()) == {"A", "W1", "W2", "h1", "h2"}

    def test_analytic_parameters_task_input(self):
        cfg = AutoConfig.for_model("shallow_plrnn", latent_dim=3, hidden_dim=10,
                                   input_dim=2, autonomous=False)
        model = AutoModel.from_config(cfg)
        task_input = torch.tensor([0.5, -0.3], dtype=torch.float32)
        p = model.analytic_parameters(task_input=task_input)
        assert set(p.keys()) == {"A", "W1", "W2", "h1", "h2"}
        # h1 should differ from the autonomous case
        p_auto = model.analytic_parameters()
        assert not torch.allclose(p["h1"], p_auto["h1"])

    def test_jacobian_matches_autodiff(self):
        cfg = AutoConfig.for_model("shallow_plrnn", latent_dim=3, hidden_dim=10, input_dim=0)
        model = AutoModel.from_config(cfg)
        z = torch.randn(3, requires_grad=True)
        J_analytic = model.jacobian(z)
        J_auto = torch.autograd.functional.jacobian(
            lambda zz: model.recurrence(None, zz.unsqueeze(0)).squeeze(0), z
        )
        assert torch.allclose(J_analytic, J_auto, atol=1e-5)


class TestSaveLoad:
    def test_dend_roundtrip(self, tmp_path):
        cfg = AutoConfig.for_model("dend_plrnn", latent_dim=3, n_bases=2, input_dim=0)
        model = AutoModel.from_config(cfg)
        model.save_pretrained(tmp_path / "dend")
        restored = AutoModel.from_pretrained(tmp_path / "dend")
        assert restored.config.model_type == "dend_plrnn"
        z0 = torch.randn(1, 3)
        out1 = model.forward(None, initial_state=z0, n_steps=5).states
        out2 = restored.forward(None, initial_state=z0, n_steps=5).states
        assert torch.allclose(out1, out2)

    def test_alrnn_roundtrip(self, tmp_path):
        cfg = AutoConfig.for_model("alrnn", latent_dim=4, n_linear=2, input_dim=0)
        model = AutoModel.from_config(cfg)
        model.save_pretrained(tmp_path / "alrnn")
        restored = AutoModel.from_pretrained(tmp_path / "alrnn")
        assert restored.config.model_type == "alrnn"
        z0 = torch.randn(1, 4)
        out1 = model.forward(None, initial_state=z0, n_steps=5).states
        out2 = restored.forward(None, initial_state=z0, n_steps=5).states
        assert torch.allclose(out1, out2)
