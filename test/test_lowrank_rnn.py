"""Smoke tests for the LowrankRNN model.

Verifies construction, forward pass in both return_modes, SVD reparametrization,
a single supervised training step, save/load roundtrip, and freeze-flag sanity.
"""
import pytest

torch = pytest.importorskip("torch")

from neuralrnn import AutoConfig, AutoModel, NeuralDynamicsModel
from neuralrnn import Trainer, TrainingArguments, SupervisedObjective
from neuralrnn.data import BaseDataset


class _ToyRegression(BaseDataset):
    kind = "timeseries"

    def __init__(self, input_dim=3, output_dim=2, T=20, B=8):
        self.input_dim, self.output_dim, self.T, self.B = input_dim, output_dim, T, B

    def sample_batch(self):
        x = torch.randn(self.B, self.T, self.input_dim)
        y = torch.randn(self.B, self.T, self.output_dim)
        return {"inputs": x, "targets": y, "mask": None}


class TestLowrankRNNConstruction:

    def test_autoconfig_automodel(self):
        cfg = AutoConfig.for_model(
            "lowrank_rnn", input_dim=3, latent_dim=16, output_dim=2, rank=2,
            noise_std=0.0
        )
        model = AutoModel.from_config(cfg)
        assert isinstance(model, NeuralDynamicsModel)
        assert model.config.model_type == "lowrank_rnn"
        assert model.config.rank == 2
        assert model.hidden_size == 16


class TestLowrankRNNForward:

    def _make(self):
        cfg = AutoConfig.for_model(
            "lowrank_rnn", input_dim=3, latent_dim=16, output_dim=2, rank=2,
            noise_std=0.0
        )
        return AutoModel.from_config(cfg)

    def test_forward_return_dynamics_tuple(self):
        model = self._make()
        B, T = 4, 10
        x = torch.randn(B, T, 3)
        out = model(x, return_dynamics=True)
        # LowrankRNN returns a (outputs, trajectories) tuple when return_dynamics=True.
        assert isinstance(out, tuple)
        outputs, trajectories = out
        assert outputs.shape == (B, T, 2)
        assert trajectories.shape == (B, T + 1, 16)

    def test_forward_return_states_container(self):
        model = self._make()
        B, T = 4, 10
        x = torch.randn(B, T, 3)
        out = model(x, return_states=True)
        assert hasattr(out, "outputs") and hasattr(out, "states")
        assert out.outputs.shape == (B, T, 2)
        # Unlike return_dynamics, return_states does not prepend the initial state.
        assert out.states.shape == (B, T, 16)


class TestLowrankRNNUtilities:

    def test_svd_reparametrization_runs(self):
        cfg = AutoConfig.for_model(
            "lowrank_rnn", input_dim=3, latent_dim=16, output_dim=2, rank=2,
            noise_std=0.0
        )
        model = AutoModel.from_config(cfg)
        model.svd_reparametrization()
        assert model.m.shape == (16, 2)
        assert model.n.shape == (16, 2)


class TestLowrankRNNTraining:

    def test_train_one_step_regression(self):
        ds = _ToyRegression(input_dim=3, output_dim=2)
        cfg = AutoConfig.for_model(
            "lowrank_rnn", input_dim=3, latent_dim=16, output_dim=2, rank=2,
            noise_std=0.0
        )
        model = AutoModel.from_config(cfg)
        hist = Trainer(model, ds, SupervisedObjective("regression"),
                       TrainingArguments(max_steps=1, log_every=0)).train()
        assert len(hist) == 1
        assert "loss" in hist[0]


class TestLowrankRNNSaveLoad:

    def test_save_load_roundtrip(self, tmp_path):
        cfg = AutoConfig.for_model(
            "lowrank_rnn", input_dim=3, latent_dim=16, output_dim=2, rank=2,
            noise_std=0.0
        )
        model = AutoModel.from_config(cfg)
        model.eval()
        x = torch.randn(3, 7, 3)
        out_before = model(x, return_states=True)

        model.save_pretrained(str(tmp_path))
        reloaded = AutoModel.from_pretrained(str(tmp_path))
        reloaded.eval()
        out_after = reloaded(x, return_states=True)

        assert torch.allclose(out_before.outputs, out_after.outputs, atol=1e-6)
        assert torch.allclose(out_before.states, out_after.states, atol=1e-6)


class TestLowrankRNNFreezeSanity:

    def test_freeze_flags_override_train_flags(self):
        cfg = AutoConfig.for_model(
            "lowrank_rnn", input_dim=3, latent_dim=16, output_dim=2, rank=2,
            train_wi=True, train_wo=True, train_wrec=True, train_h0=True,
            freeze_input=True, freeze_recurrent=True
        )
        model = AutoModel.from_config(cfg)
        for n, p in model.named_parameters():
            if n in ("wi", "si", "m", "n"):
                assert not p.requires_grad, f"{n} should be frozen"
