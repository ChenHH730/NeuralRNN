"""Tests for training objectives.

Covers the three main training paradigms used in NeuralRNN:
  - SupervisedObjective (classification / regression, paradigm A)
  - TeacherForcingObjective (dynamics reconstruction, paradigm B)
  - BehavioralObjective (behavior fitting, Tiny RNN paradigm)
  - VariationalObjective (LFADS-style ELBO skeleton)

Style follows the existing test suite: torch is lazily imported, and toy
inline dataset classes provide minimal data contracts.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from neuralrnn import AutoConfig, AutoModel, NeuralDynamicsModel, NeuralRNNConfig
from neuralrnn import (
    SupervisedObjective, TeacherForcingObjective,
    BehavioralObjective, VariationalObjective,
    LatentCircuitObjective, ConstrainedSupervisedObjective,
    RegularizedSupervisedObjective,
    build_objective, OBJECTIVE_REGISTRY, AutoObjective,
)
from neuralrnn.data import BaseDataset
from neuralrnn.modeling_utils import DynamicsModelOutput


class _ToyTask(BaseDataset):
    kind = "neurogym"

    def __init__(self, input_dim: int = 3, n_actions: int = 2, T: int = 20, B: int = 8):
        self.input_dim, self.output_dim, self.T, self.B = input_dim, n_actions, T, B

    def sample_batch(self):
        x = torch.randn(self.B, self.T, self.input_dim)
        y = torch.randint(0, self.output_dim, (self.B, self.T))
        return {"inputs": x, "targets": y, "mask": None}


class _ToyRegression(BaseDataset):
    kind = "neurogym"

    def __init__(self, input_dim: int = 3, output_dim: int = 2, T: int = 20, B: int = 8):
        self.input_dim, self.output_dim, self.T, self.B = input_dim, output_dim, T, B

    def sample_batch(self):
        x = torch.randn(self.B, self.T, self.input_dim)
        y = torch.randn(self.B, self.T, self.output_dim)
        return {"inputs": x, "targets": y, "mask": None}


class _ToyTimeSeries(BaseDataset):
    kind = "timeseries"

    def __init__(self, N: int = 3, T: int = 400, B: int = 8, L: int = 50):
        self.N, self.B, self.L = N, B, L
        t = np.linspace(0, 20 * np.pi, T)
        self.X = torch.tensor(
            np.stack([np.sin(t), np.cos(t), np.sin(2 * t)][:N], axis=1),
            dtype=torch.float32,
        )
        self.input_dim = self.output_dim = N

    def sample_batch(self):
        xs, ys = [], []
        for _ in range(self.B):
            s = np.random.randint(0, self.X.shape[0] - self.L - 2)
            xs.append(self.X[s : s + self.L])
            ys.append(self.X[s + 1 : s + self.L + 1])
        return {
            "inputs": torch.stack(xs),
            "targets": torch.stack(ys),
            "external_inputs": None,
        }


class _TinyVariationalModel(NeuralDynamicsModel):
    """Minimal model that exposes rates + kl in output.extras."""

    def __init__(self, input_dim: int, latent_dim: int, output_dim: int):
        config = NeuralRNNConfig(
            input_dim=input_dim, latent_dim=latent_dim, output_dim=output_dim
        )
        super().__init__(config)
        self.encoder = torch.nn.Linear(input_dim, latent_dim)
        self.readout_layer = torch.nn.Linear(latent_dim, output_dim)

    def recurrence(self, x_t, z_prev, *, inputs=None):
        return torch.tanh(self.encoder(x_t) + z_prev)

    def readout(self, z_t):
        return self.readout_layer(z_t)

    def forward(self, inputs, **kwargs):
        # Simple feed-forward-through-time for smoke testing.
        B, T, _ = inputs.shape
        z = self.init_state(B, device=inputs.device)
        states, outputs = [], []
        for t in range(T):
            z = self.recurrence(inputs[:, t], z)
            states.append(z)
            outputs.append(self.readout(z))
        rates = torch.relu(torch.stack(outputs, dim=1)) + 1e-3
        kl = torch.tensor(0.05, device=inputs.device)
        return DynamicsModelOutput(
            outputs=rates,
            states=torch.stack(states, dim=1),
            extras={"rates": rates, "kl": kl},
        )


class TestBuildObjective:
    def test_build_from_string(self):
        obj = build_objective("supervised", task_type="classification")
        assert isinstance(obj, SupervisedObjective)
        assert obj.task_type == "classification"

    def test_build_from_instance_passes_through(self):
        obj = SupervisedObjective("regression")
        assert build_objective(obj) is obj

    def test_build_from_dict(self):
        obj = build_objective({"name": "teacher_forcing", "alpha": 0.2})
        assert isinstance(obj, TeacherForcingObjective)
        assert obj.alpha == pytest.approx(0.2)

    def test_registry_contains_all_objectives(self):
        expected = {
            "supervised", "regularized_supervised", "teacher_forcing",
            "behavioral", "variational", "latent_circuit", "constrained_supervised",
        }
        assert expected.issubset(set(OBJECTIVE_REGISTRY))

    def test_unknown_objective_raises(self):
        with pytest.raises(ValueError, match="Unknown objective"):
            build_objective("not_an_objective")

    def test_auto_objective_from_name(self):
        obj = AutoObjective.from_name("supervised", task_type="regression")
        assert isinstance(obj, SupervisedObjective)
        assert obj.task_type == "regression"


class TestSupervisedObjective:
    def test_classification_returns_scalar_loss_and_logs(self):
        ds = _ToyTask()
        cfg = AutoConfig.for_model("ctrnn", input_dim=3, latent_dim=16, output_dim=2)
        model = AutoModel.from_config(cfg)

        obj = SupervisedObjective("classification")
        batch = ds.sample_batch()
        loss, logs = obj.compute_loss(model, batch)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert "loss" in logs
        assert "acc" in logs
        assert 0.0 <= logs["acc"] <= 1.0

    def test_regression_returns_scalar_loss_and_logs(self):
        ds = _ToyRegression()
        cfg = AutoConfig.for_model("ctrnn", input_dim=3, latent_dim=16, output_dim=2)
        model = AutoModel.from_config(cfg)

        obj = SupervisedObjective("regression")
        batch = ds.sample_batch()
        loss, logs = obj.compute_loss(model, batch)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert "loss" in logs


class TestRegularizedSupervisedObjective:
    def test_activity_regularizer(self):
        ds = _ToyRegression()
        cfg = AutoConfig.for_model("ctrnn", input_dim=3, latent_dim=16, output_dim=2)
        model = AutoModel.from_config(cfg)

        obj = RegularizedSupervisedObjective(
            task_type="regression",
            activity_weight=0.1,
        )
        batch = ds.sample_batch()
        loss, logs = obj.compute_loss(model, batch)
        assert "activity_loss" in logs
        assert "task_loss" in logs
        assert loss.item() > logs["task_loss"]

    def test_weight_regularizer(self):
        ds = _ToyRegression()
        cfg = AutoConfig.for_model("ctrnn", input_dim=3, latent_dim=16, output_dim=2)
        model = AutoModel.from_config(cfg)

        obj = RegularizedSupervisedObjective(
            task_type="regression",
            weight_weight=0.1,
        )
        batch = ds.sample_batch()
        loss, logs = obj.compute_loss(model, batch)
        assert "weight_loss" in logs
        assert loss.item() > logs["task_loss"]

    def test_ortho_regularizer_safe_on_missing_attributes(self):
        ds = _ToyRegression()
        cfg = AutoConfig.for_model("ctrnn", input_dim=3, latent_dim=16, output_dim=2)
        model = AutoModel.from_config(cfg)

        obj = RegularizedSupervisedObjective(
            task_type="regression",
            ortho_weight=1.0,
            ortho_input_name="missing_attr",
            ortho_output_name="missing_attr",
        )
        batch = ds.sample_batch()
        loss, logs = obj.compute_loss(model, batch)
        assert "ortho_loss" in logs
        assert logs["ortho_loss"] == 0.0
        assert loss.item() == pytest.approx(logs["task_loss"])

    def test_classification_path(self):
        ds = _ToyTask()
        cfg = AutoConfig.for_model("ctrnn", input_dim=3, latent_dim=16, output_dim=2)
        model = AutoModel.from_config(cfg)

        obj = RegularizedSupervisedObjective(
            task_type="classification",
            activity_weight=0.01,
            weight_weight=0.01,
        )
        batch = ds.sample_batch()
        loss, logs = obj.compute_loss(model, batch)
        assert "acc" in logs
        assert "activity_loss" in logs
        assert "weight_loss" in logs

    def test_global_reductions_match_notebook_objectives(self):
        """Verify equivalence with OrthogonalityObjective / MultitaskObjective conventions.

        The reference notebooks compute the task MSE and activity penalty with a
        global (batch-level) reduction, and the weight penalty as a raw sum of
        squares.  This test checks that RegularizedSupervisedObjective reproduces
        those exact values.
        """
        ds = _ToyRegression(B=4, T=5, input_dim=3, output_dim=2)
        cfg = AutoConfig.for_model("ctrnn", input_dim=3, latent_dim=16, output_dim=2)
        model = AutoModel.from_config(cfg)

        obj = RegularizedSupervisedObjective(
            task_type="regression",
            activity_weight=1e-6,
            weight_weight=1e-6,
            weight_reduce="sum",
            mse_reduce="global",
            activity_reduce="global",
        )
        batch = ds.sample_batch()
        loss, logs = obj.compute_loss(model, batch)

        # Hand-compute expected terms for verification.
        out = model(batch["inputs"])
        y, target = out.outputs, batch["targets"]
        mask = batch.get("mask")
        assert mask is None  # _ToyRegression does not provide a mask
        expected_mse = ((y - target) ** 2).mean()
        expected_activity = (out.states ** 2).mean()
        expected_weight = sum((p ** 2).sum() for p in model.parameters() if p.requires_grad)

        assert logs["task_loss"] == pytest.approx(expected_mse.item())
        assert logs["activity_loss"] == pytest.approx(expected_activity.item())
        assert logs["weight_loss"] == pytest.approx(expected_weight.item())
        expected_total = (
            expected_mse
            + 1e-6 * expected_activity
            + 1e-6 * expected_weight
        )
        assert loss.item() == pytest.approx(expected_total.item())

    def test_global_mse_with_mask(self):
        """When mse_reduce='global', the masked MSE matches (err*m).sum()/m.sum()."""
        ds = _ToyRegression(B=3, T=4, input_dim=3, output_dim=2)
        cfg = AutoConfig.for_model("ctrnn", input_dim=3, latent_dim=8, output_dim=2)
        model = AutoModel.from_config(cfg)

        batch = ds.sample_batch()
        # Inject a non-trivial mask to make global vs per-trial reduction differ.
        mask = torch.ones(batch["targets"].shape[:2])
        mask[:, 0] = 0.0
        batch["mask"] = mask

        obj = RegularizedSupervisedObjective(
            task_type="regression",
            mse_reduce="global",
        )
        loss, logs = obj.compute_loss(model, batch)

        out = model(batch["inputs"])
        err = (out.outputs - batch["targets"]) ** 2
        m = mask.unsqueeze(-1)
        expected = (err * m).sum() / m.sum()
        assert logs["task_loss"] == pytest.approx(expected.item())


class TestTeacherForcingObjective:
    def test_compute_loss_on_toy_timeseries(self):
        ds = _ToyTimeSeries()
        cfg = AutoConfig.for_model(
            "shallow_plrnn",
            input_dim=0,
            latent_dim=3,
            output_dim=3,
            hidden_dim=20,
            autonomous=True,
        )
        model = AutoModel.from_config(cfg)

        obj = TeacherForcingObjective(alpha=0.2)
        batch = ds.sample_batch()
        loss, logs = obj.compute_loss(model, batch)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert "loss" in logs
        assert "alpha" in logs


class TestBehavioralObjective:
    def test_compute_loss_with_toy_task_and_rnn(self):
        ds = _ToyTask(input_dim=3, n_actions=2)
        cfg = AutoConfig.for_model("ctrnn", input_dim=3, latent_dim=16, output_dim=2)
        model = AutoModel.from_config(cfg)

        obj = BehavioralObjective()
        batch = ds.sample_batch()
        loss, logs = obj.compute_loss(model, batch)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert "loss" in logs
        assert "nll" in logs


class TestVariationalObjective:
    def test_compute_loss_returns_scalar(self):
        B, T, N = 4, 10, 2
        model = _TinyVariationalModel(input_dim=N, latent_dim=4, output_dim=N)
        obj = VariationalObjective(kl_weight=1.0, likelihood="poisson")

        batch = {
            "inputs": torch.randn(B, T, N),
            "targets": torch.poisson(torch.rand(B, T, N) * 2.0),
        }
        loss, logs = obj.compute_loss(model, batch)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert "loss" in logs
        assert "recon" in logs
        assert "kl" in logs
