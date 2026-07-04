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
from neuralrnn import SupervisedObjective, TeacherForcingObjective
from neuralrnn import BehavioralObjective, VariationalObjective
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
