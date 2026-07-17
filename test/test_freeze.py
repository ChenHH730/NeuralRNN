"""Tests for the parameter-freezing API (ESN / reservoir-computing support).

Verifies that config-level freeze flags and programmatic freeze_parameters
correctly set requires_grad without breaking construction, training, or
save/load roundtrips.
"""
import copy

import pytest

torch = pytest.importorskip("torch")

from neuralrnn import AutoConfig, AutoModel, NeuralDynamicsModel
from neuralrnn import Trainer, TrainingArguments, SupervisedObjective
from neuralrnn.data import BaseDataset


class _ToyTask(BaseDataset):
    kind = "neurogym"

    def __init__(self, input_dim=3, n_actions=2, T=20, B=8):
        self.input_dim, self.output_dim, self.T, self.B = input_dim, n_actions, T, B

    def sample_batch(self):
        x = torch.randn(self.B, self.T, self.input_dim)
        y = torch.randint(0, self.output_dim, (self.B, self.T))
        return {"inputs": x, "targets": y, "mask": None}

    def task_input(self):
        return torch.zeros(self.input_dim)


def _param_requires_grad(model):
    return {n: p.requires_grad for n, p in model.named_parameters()}


def _group_param_names(model, group):
    return sorted(model._match_parameters([group], None))


# ============================ CTRNN ============================
class TestCTRNNFreeze:

    def _make(self, **freeze):
        cfg = AutoConfig.for_model(
            "ctrnn", input_dim=3, latent_dim=16, output_dim=2,
            trainable_h0=True, **freeze
        )
        return AutoModel.from_config(cfg)

    def test_freeze_input(self):
        model = self._make(freeze_input=True)
        grads = _param_requires_grad(model)
        for n in _group_param_names(model, "input"):
            assert not grads[n], f"{n} should be frozen"
        for g in ("recurrent", "output", "h0"):
            for n in _group_param_names(model, g):
                assert grads[n], f"{n} should be trainable"

    def test_freeze_recurrent(self):
        model = self._make(freeze_recurrent=True)
        grads = _param_requires_grad(model)
        for n in _group_param_names(model, "recurrent"):
            assert not grads[n], f"{n} should be frozen"
        for g in ("input", "output", "h0"):
            for n in _group_param_names(model, g):
                assert grads[n], f"{n} should be trainable"

    def test_freeze_output(self):
        model = self._make(freeze_output=True)
        grads = _param_requires_grad(model)
        for n in _group_param_names(model, "output"):
            assert not grads[n], f"{n} should be frozen"
        for g in ("input", "recurrent", "h0"):
            for n in _group_param_names(model, g):
                assert grads[n], f"{n} should be trainable"

    def test_freeze_h0(self):
        model = self._make(freeze_h0=True)
        grads = _param_requires_grad(model)
        for n in _group_param_names(model, "h0"):
            assert not grads[n], f"{n} should be frozen"
        for g in ("input", "recurrent", "output"):
            for n in _group_param_names(model, g):
                assert grads[n], f"{n} should be trainable"

    def test_freeze_all(self):
        model = self._make(
            freeze_input=True, freeze_recurrent=True,
            freeze_output=True, freeze_h0=True
        )
        grads = _param_requires_grad(model)
        for n, rg in grads.items():
            assert not rg, f"{n} should be frozen"

    def test_only_output_changes_during_training(self):
        model = self._make(
            freeze_input=True, freeze_recurrent=True, freeze_h0=True
        )
        state_before = copy.deepcopy(model.state_dict())
        ds = _ToyTask(input_dim=3, n_actions=2)
        Trainer(model, ds, SupervisedObjective("classification"),
                TrainingArguments(max_steps=1, log_every=0)).train()
        state_after = model.state_dict()

        for n, p_before in state_before.items():
            p_after = state_after[n]
            if n.startswith("readout_layer"):
                assert not torch.allclose(p_before, p_after, atol=1e-10), \
                    f"{n} should have changed during training"
            else:
                assert torch.allclose(p_before, p_after, atol=1e-10), \
                    f"{n} should remain frozen but changed"

    def test_config_roundtrip_preserves_freeze(self, tmp_path):
        cfg = AutoConfig.for_model(
            "ctrnn", input_dim=3, latent_dim=16, output_dim=2,
            trainable_h0=True,
            freeze_input=True, freeze_recurrent=True,
            freeze_output=True, freeze_h0=True
        )
        model = AutoModel.from_config(cfg)
        before = _param_requires_grad(model)
        model.save_pretrained(str(tmp_path))
        reloaded = AutoModel.from_pretrained(str(tmp_path))
        after = _param_requires_grad(reloaded)

        assert reloaded.config.freeze_input
        assert reloaded.config.freeze_recurrent
        assert reloaded.config.freeze_output
        assert reloaded.config.freeze_h0
        assert before == after


# ============================ LowrankRNN ============================
class TestLowrankRNNFreeze:

    def _make(self, train_wi=None, train_wo=None, **freeze):
        kw = dict(input_dim=3, latent_dim=16, output_dim=2, rank=2)
        if train_wi is not None:
            kw["train_wi"] = train_wi
        if train_wo is not None:
            kw["train_wo"] = train_wo
        cfg = AutoConfig.for_model("lowrank_rnn", **kw, **freeze)
        return AutoModel.from_config(cfg)

    def test_freeze_flags_override_train_flags(self):
        # Even though train_wi/train_wo are True (and freeze_h0 defaulted),
        # freeze_* flags should keep the parameters frozen.
        model = self._make(
            freeze_input=True, freeze_recurrent=True,
            freeze_output=True, freeze_h0=True
        )
        grads = _param_requires_grad(model)
        for n, rg in grads.items():
            assert not rg, f"{n} should be frozen"

    def test_partial_freeze_input_overrides_train_wi(self):
        # train_wi=True leaves wi trainable but si frozen by design.
        # freeze_input=True should freeze both wi and si.
        model = self._make(train_wi=True, freeze_input=True)
        grads = _param_requires_grad(model)
        for n in _group_param_names(model, "input"):
            assert not grads[n], f"{n} should be frozen despite train_wi=True"
        # Recurrent/output main weights should remain trainable.
        for n in _group_param_names(model, "recurrent"):
            assert grads[n], f"{n} should be trainable"
        for n in _group_param_names(model, "output"):
            if n == "so":
                # so is frozen whenever train_wo=True (default) by design.
                continue
            assert grads[n], f"{n} should be trainable"

    def test_train_flags_work_without_freeze(self):
        model = self._make(
            train_wi=False, train_wo=False, freeze_recurrent=True
        )
        grads = _param_requires_grad(model)
        # Main weights frozen; scaling factors remain trainable by design.
        # (h0 is frozen by default: LowrankRNNConfig sets freeze_h0=True.)
        assert not grads["wi"]
        assert not grads["wo"]
        assert not grads["m"]
        assert not grads["n"]
        assert not grads["h0"]
        assert grads["si"]
        assert grads["so"]

    def test_train_wi_true_unfrozen_without_freeze(self):
        model = self._make(train_wi=True, train_wo=False, freeze_recurrent=True)
        grads = _param_requires_grad(model)
        # wi trainable, si frozen by design when train_wi=True.
        assert grads["wi"]
        assert not grads["si"]
        # Recurrent frozen.
        assert not grads["m"]
        assert not grads["n"]

    def test_h0_frozen_by_default_unless_opted_in(self):
        # Family default freeze_h0=True (original code never trained h0).
        model = self._make()
        assert not _param_requires_grad(model)["h0"]
        model = self._make(freeze_h0=False)
        assert _param_requires_grad(model)["h0"]

    def test_deprecated_train_flags_map_to_freeze(self):
        # Old-style flags still work via deprecation shim.
        with pytest.warns(DeprecationWarning, match="freeze_recurrent"):
            cfg = AutoConfig.for_model(
                "lowrank_rnn", input_dim=3, latent_dim=16, output_dim=2,
                rank=2, train_wrec=False
            )
        assert cfg.freeze_recurrent is True
        model = AutoModel.from_config(cfg)
        grads = _param_requires_grad(model)
        assert not grads["m"] and not grads["n"]

        with pytest.warns(DeprecationWarning, match="freeze_h0"):
            cfg = AutoConfig.for_model(
                "lowrank_rnn", input_dim=3, latent_dim=16, output_dim=2,
                rank=2, train_h0=True
            )
        assert cfg.freeze_h0 is False  # train_h0=True -> explicitly trainable
        model = AutoModel.from_config(cfg)
        assert _param_requires_grad(model)["h0"]


# ============================ TinyRNN ============================
class TestTinyRNNFreeze:

    def _make(self, **freeze):
        cfg = AutoConfig.for_model(
            "tiny_rnn", input_dim=3, latent_dim=4, output_dim=2,
            trainable_h0=True, **freeze
        )
        return AutoModel.from_config(cfg)

    def test_freeze_input(self):
        model = self._make(freeze_input=True)
        grads = _param_requires_grad(model)
        # GRU couples input-to-hidden weights/biases in a single gate matrix,
        # which the model exposes as the "input" group.
        for n in _group_param_names(model, "input"):
            assert not grads[n], f"{n} should be frozen"
        for n in _group_param_names(model, "recurrent"):
            assert grads[n], f"{n} should be trainable"
        for n in _group_param_names(model, "output"):
            assert grads[n], f"{n} should be trainable"

    def test_freeze_recurrent(self):
        model = self._make(freeze_recurrent=True)
        grads = _param_requires_grad(model)
        for n in _group_param_names(model, "recurrent"):
            assert not grads[n], f"{n} should be frozen"
        for n in _group_param_names(model, "input"):
            assert grads[n], f"{n} should be trainable"
        for n in _group_param_names(model, "output"):
            assert grads[n], f"{n} should be trainable"

    def test_freeze_output(self):
        model = self._make(freeze_output=True)
        grads = _param_requires_grad(model)
        for n in _group_param_names(model, "output"):
            assert not grads[n], f"{n} should be frozen"
        for n in _group_param_names(model, "input"):
            assert grads[n], f"{n} should be trainable"
        for n in _group_param_names(model, "recurrent"):
            assert grads[n], f"{n} should be trainable"

    def test_freeze_h0(self):
        model = self._make(freeze_h0=True)
        grads = _param_requires_grad(model)
        for n in _group_param_names(model, "h0"):
            assert not grads[n], f"{n} should be frozen"
        for g in ("input", "recurrent", "output"):
            for n in _group_param_names(model, g):
                assert grads[n], f"{n} should be trainable"


# ============================ ShallowPLRNN & LatentCircuit ============================
class TestOtherModelsFreeze:

    def test_shallow_plrnn_freeze_groups(self):
        cfg = AutoConfig.for_model(
            "shallow_plrnn", input_dim=3, latent_dim=4, output_dim=4,
            hidden_dim=8, freeze_input=True, freeze_recurrent=True
        )
        model = AutoModel.from_config(cfg)
        grads = _param_requires_grad(model)
        for n in _group_param_names(model, "input"):
            assert not grads[n], f"{n} should be frozen"
        for n in _group_param_names(model, "recurrent"):
            assert not grads[n], f"{n} should be frozen"

    def test_latent_circuit_freeze_groups(self):
        cfg = AutoConfig.for_model(
            "latent_circuit", input_dim=3, latent_dim=4, output_dim=2,
            embedding_dim=16, freeze_input=True, freeze_recurrent=True,
            freeze_output=True
        )
        model = AutoModel.from_config(cfg)
        grads = _param_requires_grad(model)
        for g in ("input", "recurrent", "output"):
            for n in _group_param_names(model, g):
                assert not grads[n], f"{n} should be frozen"


# ============================ Base pattern API ============================
class TestBasePatternAPI:

    def test_freeze_parameters_by_pattern(self):
        cfg = AutoConfig.for_model("ctrnn", input_dim=3, latent_dim=16, output_dim=2)
        model = AutoModel.from_config(cfg)
        frozen = model.freeze_parameters(patterns=[r"^h2h\.weight$"])
        assert "h2h.weight" in frozen
        assert model.get_parameter("h2h.weight").requires_grad is False
        assert model.get_parameter("h2h.bias").requires_grad is True

    def test_unfreeze_parameters_by_pattern(self):
        cfg = AutoConfig.for_model("ctrnn", input_dim=3, latent_dim=16, output_dim=2)
        model = AutoModel.from_config(cfg)
        model.freeze_parameters(patterns=[r"^h2h\."])
        model.unfreeze_parameters(patterns=[r"^h2h\.bias$"])
        assert model.get_parameter("h2h.bias").requires_grad is True
        assert model.get_parameter("h2h.weight").requires_grad is False

    def test_freeze_unknown_group_raises(self):
        cfg = AutoConfig.for_model("ctrnn", input_dim=3, latent_dim=16, output_dim=2)
        model = AutoModel.from_config(cfg)
        with pytest.raises(ValueError, match="Unknown freeze group"):
            model.freeze_parameters(groups="nonexistent")
