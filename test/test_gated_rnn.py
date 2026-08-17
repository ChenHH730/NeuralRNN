"""Tests for the gated_rnn family (GatedRNNModel: GRU/LSTM; TinyRNNModel specialization).

Covers registration and family wiring, manual recurrence vs nn.GRU/nn.LSTM
full-sequence parity, the LSTM packed-state convention, freeze groups,
dtype casting, the L1 hook, save/load roundtrips, and the unified
behavioral/supervised objective semantics.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F

from neuralrnn import AutoConfig, AutoModel
from neuralrnn import Trainer, TrainingArguments, BehavioralObjective, SupervisedObjective
from neuralrnn.data import BaseDataset
from neuralrnn.models.gated_rnn import (
    GatedRNNConfig,
    GatedRNNModel,
    TinyRNNConfig,
    TinyRNNModel,
)


class _ToyBehavioral(BaseDataset):
    kind = "behavioral"

    def __init__(self, input_dim=3, n_actions=2, T=20, B=8):
        self.input_dim, self.output_dim, self.T, self.B = input_dim, n_actions, T, B

    def sample_batch(self):
        x = torch.randn(self.B, self.T, self.input_dim)
        y = torch.randint(0, self.output_dim, (self.B, self.T))
        return {"inputs": x, "targets": y, "mask": None}


def _make_gated(rnn_type="GRU", **overrides):
    kw = dict(input_dim=3, latent_dim=4, output_dim=2, rnn_type=rnn_type)
    kw.update(overrides)
    return AutoModel.from_config(AutoConfig.for_model("gated_rnn", **kw))


def _make_tiny(**overrides):
    kw = dict(input_dim=3, latent_dim=2, output_dim=2)
    kw.update(overrides)
    return AutoModel.from_config(AutoConfig.for_model("tiny_rnn", **kw))


# ============================ registration / defaults ============================

class TestRegistration:

    def test_gated_gru_autoconfig_automodel(self):
        model = _make_gated("GRU")
        assert isinstance(model, GatedRNNModel)
        assert model.config.model_type == "gated_rnn"
        assert hasattr(model, "gru") and model.rnn is model.gru
        out = model(torch.randn(4, 10, 3))
        assert out.outputs.shape == (4, 10, 2)
        assert out.states.shape == (4, 10, 4)

    def test_gated_lstm_autoconfig_automodel(self):
        model = _make_gated("LSTM")
        assert isinstance(model, GatedRNNModel)
        assert hasattr(model, "lstm") and model.rnn is model.lstm
        out = model(torch.randn(4, 10, 3))
        assert out.outputs.shape == (4, 10, 2)
        assert out.states.shape == (4, 10, 4)

    def test_tiny_is_gated_subclass(self):
        assert issubclass(TinyRNNModel, GatedRNNModel)
        assert issubclass(TinyRNNConfig, GatedRNNConfig)
        model = _make_tiny()
        assert isinstance(model, GatedRNNModel)
        assert model.config.model_type == "tiny_rnn"
        # Legacy attribute name preserved (checkpoint compatibility)
        assert hasattr(model, "gru")

    def test_rnn_type_case_normalized(self):
        cfg = AutoConfig.for_model("gated_rnn", input_dim=1, latent_dim=2,
                                   output_dim=1, rnn_type="gru")
        assert cfg.rnn_type == "GRU"

    def test_invalid_rnn_type_raises(self):
        with pytest.raises(ValueError, match="rnn_type"):
            GatedRNNConfig(input_dim=1, latent_dim=2, output_dim=1, rnn_type="RNN")

    def test_trainable_c0_requires_lstm(self):
        with pytest.raises(ValueError, match="trainable_c0"):
            GatedRNNConfig(input_dim=1, latent_dim=2, output_dim=1,
                           rnn_type="GRU", trainable_c0=True)

    def test_tiny_rejects_lstm(self):
        cfg = AutoConfig.for_model("tiny_rnn", input_dim=3, latent_dim=2,
                                   output_dim=2, rnn_type="LSTM")
        with pytest.raises(ValueError, match="Unsupported rnn_type"):
            AutoModel.from_config(cfg)


# ============================ recurrence <-> forward parity ============================

class TestRecurrenceForwardParity:

    def test_gru_step_loop_matches_forward(self):
        model = _make_gated("GRU", latent_dim=5)
        model.eval()
        x = torch.randn(3, 12, 3)
        with torch.no_grad():
            out = model(x)
            z = model.init_state(3)
            zs = []
            for t in range(x.shape[1]):
                z = model.recurrence(x[:, t], z)
                zs.append(z)
            zt = torch.stack(zs, dim=1)
        assert torch.allclose(zt, out.states, atol=1e-6)

    def test_lstm_packed_step_loop_matches_forward(self):
        model = _make_gated("LSTM", latent_dim=5)
        model.eval()
        x = torch.randn(3, 12, 3)
        with torch.no_grad():
            out = model(x)
            z = model.init_state(3)  # packed [h, c], (B, 2M)
            assert z.shape == (3, 10)
            hs = []
            for t in range(x.shape[1]):
                z = model.recurrence(x[:, t], z)
                hs.append(z[..., :5])
            ht = torch.stack(hs, dim=1)
        assert torch.allclose(ht, out.states, atol=1e-6)

    def test_lstm_unpacked_fallback_equals_zero_cell(self):
        model = _make_gated("LSTM", latent_dim=4)
        x_t = torch.randn(6, 3)
        h = torch.randn(6, 4)
        packed = torch.cat([h, torch.zeros_like(h)], dim=-1)
        assert torch.allclose(model.recurrence(x_t, h),
                              model.recurrence(x_t, packed), atol=1e-6)

    def test_lstm_readout_uses_h_part(self):
        model = _make_gated("LSTM", latent_dim=4)
        h = torch.randn(6, 4)
        c = torch.randn(6, 4)
        packed = torch.cat([h, c], dim=-1)
        assert torch.allclose(model.readout(packed), model.readout(h), atol=1e-6)

    def test_lstm_forward_with_unpacked_initial_state(self):
        model = _make_gated("LSTM", latent_dim=4)
        x = torch.randn(2, 6, 3)
        h0 = torch.zeros(2, 4)
        out1 = model(x, initial_state=h0)
        out2 = model(x, initial_state=torch.cat([h0, torch.zeros_like(h0)], dim=-1))
        assert torch.allclose(out1.outputs, out2.outputs, atol=1e-6)

    def test_forward_requires_inputs(self):
        model = _make_gated("GRU")
        with pytest.raises(ValueError, match="input"):
            model(None, n_steps=5)


# ============================ h0 / c0 ============================

class TestInitialState:

    def test_trainable_h0(self):
        model = _make_gated("GRU", trainable_h0=True)
        assert isinstance(model.h0, torch.nn.Parameter) and model.h0.requires_grad

    def test_trainable_c0_lstm(self):
        model = _make_gated("LSTM", trainable_c0=True)
        assert isinstance(model.c0, torch.nn.Parameter) and model.c0.requires_grad

    def test_buffer_h0_default(self):
        model = _make_gated("LSTM")
        assert not isinstance(model.h0, torch.nn.Parameter)
        assert not isinstance(model.c0, torch.nn.Parameter)
        names = dict(model.named_parameters())
        assert "h0" not in names and "c0" not in names


# ============================ freeze groups ============================

class TestFreeze:

    def test_freeze_groups_gru(self):
        model = _make_gated("GRU")
        frozen = model.freeze_parameters(groups="recurrent")
        assert sorted(frozen) == ["gru.bias_hh_l0", "gru.weight_hh_l0"]
        assert not model.gru.weight_hh_l0.requires_grad
        assert model.gru.weight_ih_l0.requires_grad

    def test_freeze_groups_lstm(self):
        model = _make_gated("LSTM")
        frozen = model.freeze_parameters(groups=["input", "output"])
        assert sorted(frozen) == [
            "lstm.bias_ih_l0", "lstm.weight_ih_l0",
            "readout_layer.bias", "readout_layer.weight",
        ]

    def test_freeze_h0_covers_c0(self):
        model = _make_gated("LSTM", trainable_h0=True, trainable_c0=True,
                            freeze_h0=True)
        assert not model.h0.requires_grad
        assert not model.c0.requires_grad

    def test_config_freeze_flags(self):
        model = _make_gated("GRU", freeze_recurrent=True)
        assert not model.gru.weight_hh_l0.requires_grad

    def test_tiny_freeze_diagonal_readout(self):
        model = _make_tiny(latent_dim=2, output_dim=2, readout_FC=False,
                           freeze_output=True)
        assert not model.readout_coef.requires_grad


# ============================ dtype / L1 ============================

class TestDtypeAndL1:

    def test_float64_cast(self):
        model = _make_gated("GRU", dtype="float64")
        assert model.gru.weight_hh_l0.dtype == torch.float64
        assert model.h0.dtype == torch.float64
        out = model(torch.randn(2, 5, 3, dtype=torch.float64))
        assert out.outputs.dtype == torch.float64

    def test_invalid_dtype_raises(self):
        with pytest.raises(ValueError, match="dtype"):
            GatedRNNConfig(input_dim=1, latent_dim=2, output_dim=1, dtype="float16")

    def test_get_l1_loss(self):
        model = _make_gated("GRU")
        expected = model.gru.weight_hh_l0.abs().sum()
        assert torch.allclose(model.get_l1_loss(), expected)

    def test_get_l1_loss_lstm(self):
        model = _make_gated("LSTM")
        expected = model.lstm.weight_hh_l0.abs().sum()
        assert torch.allclose(model.get_l1_loss(), expected)


# ============================ save / load ============================

class TestSaveLoad:

    @pytest.mark.parametrize("rnn_type", ["GRU", "LSTM"])
    def test_gated_roundtrip(self, tmp_path, rnn_type):
        model = _make_gated(rnn_type, trainable_h0=True)
        x = torch.randn(3, 7, 3)
        out_before = model(x)
        model.save_pretrained(str(tmp_path))
        reloaded = AutoModel.from_pretrained(str(tmp_path))
        out_after = reloaded(x)
        assert reloaded.config.rnn_type == rnn_type
        assert torch.allclose(out_before.outputs, out_after.outputs, atol=1e-6)
        assert torch.allclose(out_before.states, out_after.states, atol=1e-6)

    def test_tiny_state_dict_keys_unchanged(self):
        """Legacy checkpoints use gru.* / readout_layer.* / h0 key names."""
        model = _make_tiny()
        assert set(model.state_dict()) == {
            "gru.weight_ih_l0", "gru.weight_hh_l0",
            "gru.bias_ih_l0", "gru.bias_hh_l0",
            "readout_layer.weight", "readout_layer.bias", "h0",
        }
        model_diag = _make_tiny(latent_dim=2, output_dim=2, readout_FC=False)
        assert set(model_diag.state_dict()) == {
            "gru.weight_ih_l0", "gru.weight_hh_l0",
            "gru.bias_ih_l0", "gru.bias_hh_l0",
            "readout_coef", "h0",
        }

    def test_tiny_config_roundtrip_fields(self, tmp_path):
        cfg = AutoConfig.for_model("tiny_rnn", input_dim=3, latent_dim=2,
                                   output_dim=2, output_h0=True, l1_weight=1e-4,
                                   dtype="float64")
        model = AutoModel.from_config(cfg)
        model.save_pretrained(str(tmp_path))
        reloaded = AutoModel.from_pretrained(str(tmp_path))
        rc = reloaded.config
        assert rc.model_type == "tiny_rnn"
        assert rc.output_h0 is True and rc.readout_FC is True
        assert rc.l1_weight == pytest.approx(1e-4)
        assert rc.dtype == "float64"
        assert reloaded.gru.weight_hh_l0.dtype == torch.float64


# ============================ objective unification ============================

class TestObjectiveUnification:

    def test_behavioral_is_supervised_subclass(self):
        assert issubclass(BehavioralObjective, SupervisedObjective)

    def test_supervised_output_h0_alignment(self):
        """SupervisedObjective drops the leading output step for output_h0 models."""
        torch.manual_seed(0)
        model = _make_tiny(output_h0=True, l1_weight=0.0)
        ds = _ToyBehavioral()
        batch = ds.sample_batch()
        loss, logs = SupervisedObjective("classification").compute_loss(model, batch)
        with torch.no_grad():
            logits = model(batch["inputs"]).outputs[:, :-1]
            target = batch["targets"].long()
            expected = F.cross_entropy(logits.reshape(-1, 2), target.reshape(-1)).mean()
        assert loss.item() == pytest.approx(expected.item(), rel=1e-6)
        assert "acc" in logs

    def test_behavioral_loss_equals_ce_plus_l1(self):
        torch.manual_seed(0)
        model = _make_tiny(output_h0=True, l1_weight=1e-3)
        ds = _ToyBehavioral()
        batch = ds.sample_batch()
        loss, logs = BehavioralObjective().compute_loss(model, batch)
        with torch.no_grad():
            logits = model(batch["inputs"]).outputs[:, :-1]
            target = batch["targets"].long()
            ce = F.cross_entropy(logits.reshape(-1, 2), target.reshape(-1)).mean()
            expected = ce + 1e-3 * model.gru.weight_hh_l0.abs().sum()
        assert loss.item() == pytest.approx(expected.item(), rel=1e-6)
        assert logs["nll"] == pytest.approx(ce.item(), rel=1e-6)
        assert "l1" in logs and "acc" in logs

    def test_behavioral_explicit_l1_weight_overrides_config(self):
        torch.manual_seed(0)
        model = _make_tiny(l1_weight=1e-3)
        batch = _ToyBehavioral().sample_batch()
        loss_off, logs_off = BehavioralObjective(l1_weight=0.0).compute_loss(model, batch)
        assert "l1" not in logs_off
        loss_on, logs_on = BehavioralObjective(l1_weight=0.5).compute_loss(model, batch)
        assert "l1" in logs_on
        assert loss_on.item() > loss_off.item()

    def test_behavioral_without_l1_hook(self):
        """Models without get_l1_loss / l1_weight config fall back to plain CE."""
        torch.manual_seed(0)
        cfg = AutoConfig.for_model("ctrnn", input_dim=3, latent_dim=6, output_dim=2)
        model = AutoModel.from_config(cfg)
        batch = _ToyBehavioral().sample_batch()
        loss, logs = BehavioralObjective().compute_loss(model, batch)
        assert "nll" in logs and "l1" not in logs

    def test_behavioral_train_one_step(self):
        ds = _ToyBehavioral(input_dim=3, n_actions=2)
        model = _make_tiny(output_h0=True)
        hist = Trainer(model, ds, BehavioralObjective(),
                       TrainingArguments(max_steps=1, log_every=0)).train()
        assert len(hist) == 1 and "loss" in hist[0]
