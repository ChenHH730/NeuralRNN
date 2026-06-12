"""Tests for dropout support in Trainer.

Verifies that:
- Dropout mask shapes are correct
- dropout_rate=0 gives identical results to standard forward
- Training with dropout runs without errors
- Participation-based sampling works
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from neuralrnn import AutoConfig, AutoModel, Trainer, TrainingArguments
from neuralrnn import SupervisedObjective
from neuralrnn.data import BaseDataset
from neuralrnn.modeling_utils import DynamicsModelOutput


class _ToyTask(BaseDataset):
    kind = "neurogym"

    def __init__(self, input_dim=3, n_actions=2, T=20, B=8):
        self.input_dim, self.output_dim, self.T, self.B = input_dim, n_actions, T, B

    def sample_batch(self):
        x = torch.randn(self.B, self.T, self.input_dim)
        y = torch.randint(0, self.output_dim, (self.B, self.T))
        return {"inputs": x, "targets": y, "mask": None}


class TestDropoutMask:
    def test_uniform_mask_shape(self):
        cfg = AutoConfig.for_model("ctrnn", input_dim=3, latent_dim=16, output_dim=2)
        model = AutoModel.from_config(cfg)
        mask = model._sample_dropout_mask(16, 0.2, "uniform", 1.0, None, torch.device("cpu"))
        assert mask.shape == (16,)

    def test_uniform_mask_values_are_0_or_1(self):
        cfg = AutoConfig.for_model("ctrnn", input_dim=3, latent_dim=16, output_dim=2)
        model = AutoModel.from_config(cfg)
        mask = model._sample_dropout_mask(16, 0.2, "uniform", 1.0, None, torch.device("cpu"))
        assert ((mask == 0) | (mask == 1)).all()

    def test_participation_mask_requires_tensor(self):
        cfg = AutoConfig.for_model("ctrnn", input_dim=3, latent_dim=16, output_dim=2)
        model = AutoModel.from_config(cfg)
        with pytest.raises(ValueError, match="participation"):
            model._sample_dropout_mask(16, 0.2, "participation", 1.0, None, torch.device("cpu"))

    def test_participation_mask_with_tensor(self):
        cfg = AutoConfig.for_model("ctrnn", input_dim=3, latent_dim=16, output_dim=2)
        model = AutoModel.from_config(cfg)
        part = torch.rand(16)
        mask = model._sample_dropout_mask(16, 0.2, "participation", 1.0, part, torch.device("cpu"))
        assert mask.shape == (16,)
        assert ((mask == 0) | (mask == 1)).all()

    def test_mask_at_least_one_neuron(self):
        """Even with very high dropout rate, at least one neuron should survive."""
        cfg = AutoConfig.for_model("ctrnn", input_dim=3, latent_dim=16, output_dim=2)
        model = AutoModel.from_config(cfg)
        # With rate=0.99 and M=16, it's very likely all get dropped
        # The safeguard should ensure at least 1 survives
        mask = model._sample_dropout_mask(16, 0.99, "uniform", 1.0, None, torch.device("cpu"))
        assert mask.sum() >= 1


class TestForwardWithDropout:
    def test_rate_zero_matches_standard_forward(self):
        """dropout_rate=0 should return identical results to model.forward()."""
        cfg = AutoConfig.for_model("ctrnn", input_dim=3, latent_dim=16, output_dim=2)
        model = AutoModel.from_config(cfg)
        model.eval()

        x = torch.randn(4, 10, 3)

        out_std = model(x, return_states=True)
        sc, oc, sd, od = model.forward_with_dropout(x, dropout_rate=0.0)

        assert torch.allclose(out_std.states, sc, atol=1e-6)
        assert torch.allclose(out_std.outputs, oc, atol=1e-6)
        assert torch.allclose(sc, sd, atol=1e-6)  # clean == dropped when rate=0
        assert torch.allclose(oc, od, atol=1e-6)

    def test_output_shapes(self):
        cfg = AutoConfig.for_model("ctrnn", input_dim=3, latent_dim=16, output_dim=2)
        model = AutoModel.from_config(cfg)
        model.eval()

        x = torch.randn(4, 10, 3)
        sc, oc, sd, od = model.forward_with_dropout(x, dropout_rate=0.2)

        assert sc.shape == (4, 10, 16)
        assert oc.shape == (4, 10, 2)
        assert sd.shape == (4, 10, 16)
        assert od.shape == (4, 10, 2)

    def test_dropout_changes_states(self):
        """With dropout enabled, dropped states should differ from clean."""
        cfg = AutoConfig.for_model("ctrnn", input_dim=3, latent_dim=16, output_dim=2)
        model = AutoModel.from_config(cfg)
        model.eval()

        torch.manual_seed(42)
        x = torch.randn(4, 10, 3)
        sc, oc, sd, od = model.forward_with_dropout(x, dropout_rate=0.5)

        # At least some states should differ
        assert not torch.allclose(sc, sd, atol=1e-6)


class TestDropoutTraining:
    def test_training_runs_with_dropout(self):
        """Trainer should run without errors when dropout is enabled."""
        ds = _ToyTask()
        cfg = AutoConfig.for_model("ctrnn", input_dim=3, latent_dim=16, output_dim=2)
        model = AutoModel.from_config(cfg)

        hist = Trainer(model, ds, SupervisedObjective("classification"),
                       TrainingArguments(max_steps=3, log_every=0,
                                         dropout_rate=0.2)).train()
        assert len(hist) == 3
        assert all("loss" in h for h in hist)

    def test_training_runs_without_dropout(self):
        """Trainer should still work with dropout_rate=0."""
        ds = _ToyTask()
        cfg = AutoConfig.for_model("ctrnn", input_dim=3, latent_dim=16, output_dim=2)
        model = AutoModel.from_config(cfg)

        hist = Trainer(model, ds, SupervisedObjective("classification"),
                       TrainingArguments(max_steps=3, log_every=0,
                                         dropout_rate=0.0)).train()
        assert len(hist) == 3

    def test_participation_sampling_training(self):
        """Training with participation-based dropout should work."""
        ds = _ToyTask()
        cfg = AutoConfig.for_model("ctrnn", input_dim=3, latent_dim=16, output_dim=2)
        model = AutoModel.from_config(cfg)

        hist = Trainer(model, ds, SupervisedObjective("classification"),
                       TrainingArguments(max_steps=3, log_every=0,
                                         dropout_rate=0.1,
                                         dropout_sampling="participation")).train()
        assert len(hist) == 3
