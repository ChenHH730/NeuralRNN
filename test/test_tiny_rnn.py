"""Smoke tests for the TinyRNN model (GRU-based behavioral RNN).

Verifies construction via AutoConfig/AutoModel, forward shapes, both readout
modes, a single training step, and save/load roundtrip.
"""
import pytest

torch = pytest.importorskip("torch")

from neuralrnn import AutoConfig, AutoModel, NeuralDynamicsModel
from neuralrnn import Trainer, TrainingArguments, BehavioralObjective
from neuralrnn.data import BaseDataset


class _ToyBehavioral(BaseDataset):
    kind = "behavioral"

    def __init__(self, input_dim=3, n_actions=2, T=20, B=8):
        self.input_dim, self.output_dim, self.T, self.B = input_dim, n_actions, T, B

    def sample_batch(self):
        x = torch.randn(self.B, self.T, self.input_dim)
        y = torch.randint(0, self.output_dim, (self.B, self.T))
        return {"inputs": x, "targets": y, "mask": None}


class TestTinyRNNConstruction:

    def test_autoconfig_automodel(self):
        cfg = AutoConfig.for_model("tiny_rnn", input_dim=3, latent_dim=4, output_dim=2)
        model = AutoModel.from_config(cfg)
        assert isinstance(model, NeuralDynamicsModel)
        assert model.config.model_type == "tiny_rnn"
        assert model.config.latent_dim == 4

    def test_forward_shape_return_states(self):
        cfg = AutoConfig.for_model("tiny_rnn", input_dim=3, latent_dim=4, output_dim=2)
        model = AutoModel.from_config(cfg)
        B, T = 4, 10
        out = model(torch.randn(B, T, 3), return_states=True)
        assert out.outputs.shape == (B, T, 2)
        assert out.states.shape == (B, T, 4)

    def test_readout_fc_true(self):
        cfg = AutoConfig.for_model(
            "tiny_rnn", input_dim=3, latent_dim=4, output_dim=2, readout_FC=True
        )
        model = AutoModel.from_config(cfg)
        assert hasattr(model, "readout_layer")
        out = model(torch.randn(2, 5, 3), return_states=True)
        assert out.outputs.shape == (2, 5, 2)

    def test_readout_fc_false(self):
        # Diagonal readout requires latent_dim == output_dim.
        cfg = AutoConfig.for_model(
            "tiny_rnn", input_dim=3, latent_dim=4, output_dim=4, readout_FC=False
        )
        model = AutoModel.from_config(cfg)
        assert hasattr(model, "readout_coef")
        out = model(torch.randn(2, 5, 3), return_states=True)
        assert out.outputs.shape == (2, 5, 4)


class TestTinyRNNTraining:

    def test_train_one_step_behavioral_classification(self):
        ds = _ToyBehavioral(input_dim=3, n_actions=2)
        cfg = AutoConfig.for_model("tiny_rnn", input_dim=3, latent_dim=4, output_dim=2)
        model = AutoModel.from_config(cfg)
        hist = Trainer(model, ds, BehavioralObjective(),
                       TrainingArguments(max_steps=1, log_every=0)).train()
        assert len(hist) == 1
        assert "loss" in hist[0]


class TestTinyRNNSaveLoad:

    def test_save_load_roundtrip(self, tmp_path):
        cfg = AutoConfig.for_model("tiny_rnn", input_dim=3, latent_dim=4, output_dim=2)
        model = AutoModel.from_config(cfg)
        x = torch.randn(3, 7, 3)
        out_before = model(x, return_states=True)

        model.save_pretrained(str(tmp_path))
        reloaded = AutoModel.from_pretrained(str(tmp_path))
        out_after = reloaded(x, return_states=True)

        assert torch.allclose(out_before.outputs, out_after.outputs, atol=1e-6)
        assert torch.allclose(out_before.states, out_after.states, atol=1e-6)

    def test_save_load_diagonal_readout(self, tmp_path):
        cfg = AutoConfig.for_model(
            "tiny_rnn", input_dim=3, latent_dim=3, output_dim=3, readout_FC=False
        )
        model = AutoModel.from_config(cfg)
        x = torch.randn(2, 5, 3)
        out_before = model(x, return_states=True)

        model.save_pretrained(str(tmp_path))
        reloaded = AutoModel.from_pretrained(str(tmp_path))
        out_after = reloaded(x, return_states=True)

        assert torch.allclose(out_before.outputs, out_after.outputs, atol=1e-6)
        assert torch.allclose(out_before.states, out_after.states, atol=1e-6)
