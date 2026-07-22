"""Tests for the PLRNN DSR refactor (task 2): paper inits, clipped basis,
M>N readout, unified teacher forcing, from_timeseries, radam/exponential lr."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from neuralrnn import (AutoConfig, AutoModel, Trainer, TrainingArguments,
                       TeacherForcingObjective, ReconstructionDataset)


class TestPaperInit:
    def test_alrnn_paper_init(self):
        torch.manual_seed(0)
        cfg = AutoConfig.for_model("alrnn", latent_dim=20, output_dim=3,
                                   n_linear=18, input_dim=0, init_scheme="paper")
        m = AutoModel.from_config(cfg)
        assert 0 < m.A.min() <= m.A.max() <= 1.0
        assert abs(m.W.std().item() - 0.01) < 0.005
        assert torch.all(m.h == 0)

    def test_dend_paper_init(self):
        torch.manual_seed(0)
        cfg = AutoConfig.for_model("dend_plrnn", latent_dim=22, n_bases=20,
                                   input_dim=0, init_scheme="paper",
                                   threshold_range=(-3.0, 3.0))
        m = AutoModel.from_config(cfg)
        # AW = (I + R^T R/M)/lambda_max has spectral radius 1; A = diag(AW) in (0, 1]
        assert 0 < m.A.min() <= m.A.max() <= 1.0 + 1e-6
        rb = 1.0 / np.sqrt(20)
        assert m.alphas.abs().max() <= rb + 1e-6
        assert m.H.min() >= -3.0 and m.H.max() <= 3.0
        assert torch.all(torch.diagonal(m.W) == 0)  # W = AW - diag(AW)

    def test_invalid_init_scheme_raises(self):
        with pytest.raises(ValueError):
            AutoConfig.for_model("alrnn", latent_dim=5, n_linear=3,
                                 init_scheme="bogus")
        with pytest.raises(ValueError):
            AutoConfig.for_model("dend_plrnn", latent_dim=5, init_scheme="bogus")


class TestClippedBasis:
    def test_clipped_basis_formula(self):
        cfg = AutoConfig.for_model("dend_plrnn", latent_dim=4, n_bases=3,
                                   input_dim=0, use_clipping=True)
        m = AutoModel.from_config(cfg)
        z = torch.randn(2, 4)
        manual = ((m.alphas.view(1, 1, -1)
                   * (torch.relu(z.unsqueeze(-1) - m.H.unsqueeze(0))
                      - torch.relu(z.unsqueeze(-1)))).sum(dim=-1))
        assert torch.allclose(m._basis(z), manual)

    def test_clipped_jacobian_matches_autodiff(self):
        cfg = AutoConfig.for_model("dend_plrnn", latent_dim=4, n_bases=3,
                                   input_dim=0, use_clipping=True)
        m = AutoModel.from_config(cfg)
        z = torch.randn(4, requires_grad=True)
        J_ana = m.jacobian(z)
        J_auto = torch.autograd.functional.jacobian(
            lambda zz: m.recurrence(None, zz.unsqueeze(0)).squeeze(0), z)
        assert torch.allclose(J_ana, J_auto, atol=1e-5)

    def test_clipped_analytic_parameters_reproduce_recurrence(self):
        cfg = AutoConfig.for_model("dend_plrnn", latent_dim=4, n_bases=3,
                                   input_dim=0, use_clipping=True)
        m = AutoModel.from_config(cfg)
        p = m.analytic_parameters()
        assert p["W1"].shape == (4, 4 * 4)      # M x M*(B+1)

        def shallow_step(zz):
            return p["A"] @ zz + p["W1"] @ torch.relu(p["W2"] @ zz + p["h2"]) + p["h1"]

        for _ in range(5):
            z = torch.randn(4)
            assert torch.allclose(shallow_step(z), m.recurrence(None, z), atol=1e-6)

    def test_clipping_disables_hard_clamp(self):
        cfg = AutoConfig.for_model("dend_plrnn", latent_dim=3, n_bases=2,
                                   input_dim=0, use_clipping=True, clip_range=0.1)
        m = AutoModel.from_config(cfg)
        # With use_clipping, clip_range must NOT hard-clamp the state
        z = torch.full((1, 3), 100.0)
        out = m.recurrence(None, z)
        assert out.abs().max() > 0.1


class TestReadoutSlicing:
    @pytest.mark.parametrize("model_type,kwargs", [
        ("shallow_plrnn", {"hidden_dim": 10}),
        ("dend_plrnn", {"n_bases": 3}),
        ("alrnn", {"n_linear": 5}),
    ])
    def test_m_greater_n_readout(self, model_type, kwargs):
        cfg = AutoConfig.for_model(model_type, latent_dim=8, output_dim=3,
                                   input_dim=0, **kwargs)
        m = AutoModel.from_config(cfg)
        out = m(n_steps=5, initial_state=m.init_state(2, "cpu"))
        assert out.states.shape == (2, 5, 8)
        assert out.outputs.shape == (2, 5, 3)
        assert torch.allclose(out.outputs, out.states[..., :3])

    def test_default_output_dim_unchanged(self):
        cfg = AutoConfig.for_model("shallow_plrnn", latent_dim=4, hidden_dim=6,
                                   input_dim=0)
        m = AutoModel.from_config(cfg)
        out = m(n_steps=5, initial_state=m.init_state(2, "cpu"))
        assert out.outputs.shape == (2, 5, 4)


class TestUnifiedTeacherForcing:
    def test_loss_uses_activity_and_optional_inputs(self):
        cfg = AutoConfig.for_model("alrnn", latent_dim=8, output_dim=3,
                                   n_linear=6, input_dim=2)
        m = AutoModel.from_config(cfg)
        batch = {"activity": torch.randn(2, 20, 3),
                 "inputs": torch.randn(2, 20, 2)}
        obj = TeacherForcingObjective(alpha=1.0, forcing_interval=5)
        loss, info = obj.compute_loss(m, batch)
        assert loss.ndim == 0 and "alpha" in info

    def test_autonomous_batch_without_inputs(self):
        cfg = AutoConfig.for_model("shallow_plrnn", latent_dim=3, hidden_dim=8,
                                   input_dim=0)
        m = AutoModel.from_config(cfg)
        obj = TeacherForcingObjective(alpha=0.1)
        loss, _ = obj.compute_loss(m, {"activity": torch.randn(2, 10, 3)})
        assert loss.ndim == 0

    def test_initial_forcing_is_hard(self):
        # Reference behavior: the initial latent always starts from the first
        # observation (alpha=1), regardless of the forcing alpha.
        cfg = AutoConfig.for_model("alrnn", latent_dim=8, output_dim=3,
                                   n_linear=6, input_dim=0)
        m = AutoModel.from_config(cfg)
        X = torch.randn(1, 5, 3)
        obj_soft = TeacherForcingObjective(alpha=0.0)  # free-running after init
        # With alpha=0 the rollout is free from a hard-forced init:
        # predictions equal generate() from init with first 3 dims = X[0].
        loss, _ = obj_soft.compute_loss(m, {"activity": X})
        z0 = m.init_state(1, "cpu")
        z0[:, :3] = X[:, 0]
        traj = m.generate(z0, 4)                       # (1,5,M): init + 4 steps
        preds = traj[:, 1:, :3]                        # predictions of x_1..x_4
        ref = torch.nn.functional.mse_loss(preds, X[:, 1:])
        assert abs(loss.item() - ref.item()) < 1e-6

    def test_tf_noise_changes_loss(self):
        cfg = AutoConfig.for_model("shallow_plrnn", latent_dim=3, hidden_dim=8,
                                   input_dim=0)
        m = AutoModel.from_config(cfg)
        batch = {"activity": torch.randn(2, 10, 3)}
        clean, _ = TeacherForcingObjective(alpha=0.5).compute_loss(m, batch)
        noisy, _ = TeacherForcingObjective(alpha=0.5, tf_noise=0.05).compute_loss(m, batch)
        assert clean.item() != noisy.item()


class TestFromTimeseries:
    def test_window_batches_and_len(self):
        data = np.random.randn(500, 3).astype(np.float32)
        ds = ReconstructionDataset.from_timeseries(data, sequence_length=50,
                                                   batch_size=8, normalize=True)
        batch = ds.sample_batch()
        assert batch["activity"].shape == (8, 50, 3)
        assert "inputs" not in batch
        assert len(ds) == 450
        assert ds.N == 3 and ds.output_dim == 3 and ds.input_dim == 0

    def test_external_inputs(self):
        data = np.random.randn(200, 2).astype(np.float32)
        ext = np.random.randn(200, 4).astype(np.float32)
        ds = ReconstructionDataset.from_timeseries(data, external_inputs=ext,
                                                   sequence_length=10, batch_size=4)
        batch = ds.sample_batch()
        assert batch["inputs"].shape == (4, 10, 4)
        assert ds.input_dim == 4

    def test_test_series_stored_unsliced(self):
        data = np.random.randn(300, 2).astype(np.float32)
        test = np.random.randn(100, 2).astype(np.float32)
        ds = ReconstructionDataset.from_timeseries(data, test=test, dt=0.01)
        assert ds.test.shape == (100, 2)
        assert ds.dt == 0.01

    def test_from_npy(self, tmp_path):
        train = np.random.randn(400, 3).astype(np.float32)
        test = np.random.randn(100, 3).astype(np.float32)
        np.save(tmp_path / "train.npy", train)
        np.save(tmp_path / "test.npy", test)
        ds = ReconstructionDataset.from_npy(str(tmp_path / "train.npy"),
                                            test_path=str(tmp_path / "test.npy"),
                                            sequence_length=20, batch_size=4,
                                            normalize=True, dt=0.01)
        assert ds.test.shape == (100, 3)
        assert ds.sample_batch()["activity"].shape == (4, 20, 3)


class TestLearnZ0:
    @pytest.mark.parametrize("model_type,kwargs", [
        ("dend_plrnn", {"n_bases": 3}),
        ("alrnn", {"n_linear": 5}),
    ])
    def test_lift_shape_and_hard_set(self, model_type, kwargs):
        cfg = AutoConfig.for_model(model_type, latent_dim=8, output_dim=3,
                                   input_dim=0, learn_z0=True, **kwargs)
        m = AutoModel.from_config(cfg)
        assert m.B is not None and m.B.shape == (3, 8)
        x0 = torch.randn(2, 3)
        z = m.init_state_from_obs(x0)
        assert z.shape == (2, 8)
        assert torch.allclose(z[:, :3], x0)       # observed dims hard-set
        hidden_expected = (x0 @ m.B)[:, 3:]
        assert torch.allclose(z[:, 3:], hidden_expected)

    @pytest.mark.parametrize("model_type,kwargs", [
        ("dend_plrnn", {"n_bases": 3}),
        ("alrnn", {"n_linear": 5}),
    ])
    def test_default_zero_hidden(self, model_type, kwargs):
        cfg = AutoConfig.for_model(model_type, latent_dim=8, output_dim=3,
                                   input_dim=0, **kwargs)
        m = AutoModel.from_config(cfg)
        assert m.B is None
        x0 = torch.randn(2, 3)
        z = m.init_state_from_obs(x0)
        assert torch.allclose(z[:, :3], x0)
        assert torch.all(z[:, 3:] == 0)

    def test_lift_is_trainable_and_saved(self, tmp_path):
        cfg = AutoConfig.for_model("alrnn", latent_dim=8, output_dim=3,
                                   n_linear=6, input_dim=0, learn_z0=True)
        m = AutoModel.from_config(cfg)
        assert m.B.requires_grad
        m.save_pretrained(tmp_path / "m")
        m2 = AutoModel.from_pretrained(tmp_path / "m")
        assert m2.B is not None and torch.allclose(m2.B, m.B)

    def test_teacher_forcing_uses_lift(self):
        cfg = AutoConfig.for_model("alrnn", latent_dim=8, output_dim=3,
                                   n_linear=6, input_dim=0, learn_z0=True)
        m = AutoModel.from_config(cfg)
        obj = TeacherForcingObjective(alpha=1.0, forcing_interval=4)
        loss, _ = obj.compute_loss(m, {"activity": torch.randn(2, 12, 3)})
        assert loss.ndim == 0
        # B should receive gradients
        loss.backward()
        assert m.B.grad is not None


class TestOptimizers:
    def _toy(self):
        class _Toy:
            kind = "reconstruction"

            def sample_batch(self):
                return {"activity": torch.randn(4, 10, 3)}

        cfg = AutoConfig.for_model("shallow_plrnn", latent_dim=3, hidden_dim=8,
                                   input_dim=0)
        return AutoModel.from_config(cfg), _Toy()

    def test_radam_builds(self):
        from neuralrnn.train.trainer import _build_optimizer
        model, _ = self._toy()
        args = TrainingArguments(optimizer="radam")
        opt = _build_optimizer(model.parameters(), args)
        assert isinstance(opt, torch.optim.RAdam)

    def test_exponential_requires_lr_end(self):
        from neuralrnn.train.trainer import _build_optimizer, _build_scheduler
        model, _ = self._toy()
        args = TrainingArguments(lr_scheduler="exponential")
        opt = _build_optimizer(model.parameters(), args)
        with pytest.raises(ValueError):
            _build_scheduler(opt, args)

    def test_exponential_decays_to_lr_end(self):
        model, ds = self._toy()
        args = TrainingArguments(max_steps=20, learning_rate=1e-3,
                                 lr_scheduler="exponential", lr_end=1e-5,
                                 optimizer="radam", log_every=0,
                                 disable_progress_bar=True)
        trainer = Trainer(model, ds, TeacherForcingObjective(alpha=0.1), args)
        trainer.train()
        lr_final = trainer.optimizer.param_groups[0]["lr"]
        assert abs(lr_final - 1e-5) / 1e-5 < 0.2  # close to target after 20 steps

    def test_eval_metric_keep_best_restores_eval_best(self):
        model, ds = self._toy()
        calls = {"n": 0}

        def eval_fn(m):
            calls["n"] += 1
            # decreasing metric so the last eval is best
            return {"d_stsp": 1.0 / calls["n"]}

        args = TrainingArguments(max_steps=30, log_every=0, eval_every=10,
                                 eval_metric="d_stsp", keep_best=True,
                                 disable_progress_bar=True)
        trainer = Trainer(model, ds, TeacherForcingObjective(alpha=0.1), args,
                          eval_fn=eval_fn)
        trainer.train()
        assert trainer._best_state_dict_eval is not None
        assert trainer._best_metric == pytest.approx(1.0 / calls["n"])
