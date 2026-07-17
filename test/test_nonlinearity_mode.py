"""Tests for the unified nonlinearity placement (`nonlinearity_mode`).

Three modes (pre = W@state + B@x + b, noise on pre):
    "pre_activation": z' = (1-α)z + α·f(pre)            (default, standard Euler)
    "post_blend":     z' = f((1-α)z + α·pre)            (nn-brain / Masse-style)
    "rate":           r = f(z); z' = (1-α)z + α·(W@r + B@x + b)

See ``neuralrnn.configuration_utils.SUPPORTED_NONLINEARITY_MODES``.
"""
import json
import warnings

import pytest
import torch

from neuralrnn import AutoConfig, AutoModel, SUPPORTED_NONLINEARITY_MODES
from neuralrnn.configuration_utils import validate_nonlinearity_mode

torch = pytest.importorskip("torch")


# ---------------- validator unit tests ----------------

def test_supported_modes_constant():
    assert SUPPORTED_NONLINEARITY_MODES == ("pre_activation", "post_blend", "rate")


def test_validate_ok_and_passthrough():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for mode in SUPPORTED_NONLINEARITY_MODES:
            assert validate_nonlinearity_mode(mode) == mode


def test_validate_invalid_raises():
    with pytest.raises(ValueError, match="nonlinearity_mode"):
        validate_nonlinearity_mode("banana", model_type="t")


# ---------------- per-family defaults ----------------

@pytest.mark.parametrize("model_type", [
    "ctrnn", "ei_rnn", "constrained_rnn", "se_rnn", "sparse_rnn", "modular_rnn",
    "latent_circuit",
])
def test_default_pre_activation(model_type):
    assert AutoConfig.for_model(model_type).nonlinearity_mode == "pre_activation"


def test_default_lowrank_is_rate():
    assert AutoConfig.for_model("lowrank_rnn").nonlinearity_mode == "rate"


@pytest.mark.parametrize("model_type", ["ctrnn", "ei_rnn", "constrained_rnn",
                                        "latent_circuit", "lowrank_rnn"])
def test_invalid_mode_raises_per_family(model_type):
    with pytest.raises(ValueError, match="nonlinearity_mode"):
        AutoConfig.for_model(model_type, nonlinearity_mode="banana")


# ---------------- CTRNN: hand-computed single steps ----------------

def _tiny_ctrnn(mode):
    cfg = AutoConfig.for_model(
        "ctrnn", input_dim=1, latent_dim=2, output_dim=1,
        alpha=0.5, activation="tanh", sigma_rec=0.0, nonlinearity_mode=mode)
    model = AutoModel.from_config(cfg).eval()
    with torch.no_grad():
        model.input2h.weight.copy_(torch.tensor([[0.5], [-0.25]]))
        model.input2h.bias.copy_(torch.tensor([0.1, -0.2]))
        model.h2h.weight.copy_(torch.tensor([[0.3, -0.1], [0.2, 0.4]]))
        model.h2h.bias.copy_(torch.tensor([0.05, -0.15]))
        model.readout_layer.weight.copy_(torch.tensor([[0.7, -0.6]]))
        model.readout_layer.bias.copy_(torch.tensor([0.02]))
    return model


def _ctrnn_pre(model, rec_in, x):
    return (x @ model.input2h.weight.t() + model.input2h.bias
            + rec_in @ model.h2h.weight.t() + model.h2h.bias)


@pytest.mark.parametrize("mode", list(SUPPORTED_NONLINEARITY_MODES))
def test_ctrnn_single_step_hand_computed(mode):
    model = _tiny_ctrnn(mode)
    x = torch.tensor([[2.0]])
    z = torch.tensor([[1.0, -0.5]])
    with torch.no_grad():
        if mode == "pre_activation":
            expected = 0.5 * z + 0.5 * torch.tanh(_ctrnn_pre(model, z, x))
        elif mode == "post_blend":
            expected = torch.tanh(0.5 * z + 0.5 * _ctrnn_pre(model, z, x))
        else:  # rate
            expected = 0.5 * z + 0.5 * _ctrnn_pre(model, torch.tanh(z), x)
        got = model.recurrence(x, z)
    assert torch.allclose(got, expected, atol=1e-6), mode


def test_ctrnn_rate_mode_readout_stays_from_state():
    """rate mode does not reroute the readout through f: readout(z) = W_out z + b."""
    model = _tiny_ctrnn("rate")
    z = torch.tensor([[0.3, -0.8]])
    with torch.no_grad():
        expected = z @ model.readout_layer.weight.t() + model.readout_layer.bias
        got = model.readout(z)
    assert torch.allclose(got, expected, atol=1e-6)


def test_rate_keeps_subthreshold_memory_post_blend_rectifies():
    """With relu and a strongly negative drive, 'rate' lets z go negative
    (subthreshold memory) while 'post_blend' rectifies the whole update."""
    models = {mode: _tiny_ctrnn(mode) for mode in ("rate", "post_blend")}
    for model in models.values():
        with torch.no_grad():
            model.h2h.bias.copy_(torch.tensor([-10.0, -10.0]))
            model.input2h.bias.zero_()
            model.config.activation = "relu"
            model.act = torch.relu
    x = torch.zeros(1, 1)
    z0 = torch.zeros(1, 2)
    with torch.no_grad():
        z_rate = models["rate"].recurrence(x, z0)
        z_post = models["post_blend"].recurrence(x, z0)
    assert (z_rate < 0).all()
    assert (z_post >= 0).all()


def test_ctrnn_modes_agree_at_alpha_1():
    """At alpha=1 (no leak) pre_activation and post_blend are the same map."""
    pa = _tiny_ctrnn("pre_activation")
    pb = _tiny_ctrnn("post_blend")
    pa.alpha = pb.alpha = 1.0
    x = torch.tensor([[2.0]])
    z = torch.tensor([[1.0, -0.5]])
    with torch.no_grad():
        assert torch.allclose(pa.recurrence(x, z), pb.recurrence(x, z), atol=1e-6)


# ---------------- constrained family: masks compose with every mode ----------------

def test_constrained_rate_mode_with_masks_hand_computed():
    rec_mask = [[1.0, 0.0], [0.0, 1.0]]  # off-diagonal removed
    cfg = AutoConfig.for_model(
        "constrained_rnn", input_dim=1, latent_dim=2, output_dim=1,
        alpha=0.5, activation="tanh", sigma_rec=0.0,
        rec_mask=rec_mask, nonlinearity_mode="rate")
    model = AutoModel.from_config(cfg).eval()
    with torch.no_grad():
        model.input2h.weight.copy_(torch.tensor([[0.5], [-0.25]]))
        model.input2h.bias.copy_(torch.tensor([0.1, -0.2]))
        model.h2h.weight.copy_(torch.tensor([[0.3, -0.1], [0.2, 0.4]]))
        model.h2h.bias.copy_(torch.tensor([0.05, -0.15]))
    x = torch.tensor([[2.0]])
    z = torch.tensor([[1.0, -0.5]])
    w_eff = model.h2h.weight * torch.tensor(rec_mask)
    with torch.no_grad():
        pre = (x @ model.input2h.weight.t() + model.input2h.bias
               + torch.tanh(z) @ w_eff.t() + model.h2h.bias)
        expected = 0.5 * z + 0.5 * pre
        got = model.recurrence(x, z)
    assert torch.allclose(got, expected, atol=1e-6)


@pytest.mark.parametrize("model_type,extra", [
    ("se_rnn", {"grid_shape": (2, 2), "latent_dim": 4, "embedding_dim": 2}),
    ("sparse_rnn", {"latent_dim": 4, "sparsity": 0.5}),
    ("modular_rnn", {"latent_dim": 4, "n_modules": 2}),
])
def test_variant_config_rebuild_preserves_mode(model_type, extra):
    """SE/Sparse/Modular rebuild a ConstrainedRNNConfig from to_dict(); the mode
    must survive that round-trip."""
    cfg = AutoConfig.for_model(model_type, input_dim=2, output_dim=1,
                               nonlinearity_mode="post_blend", **extra)
    model = AutoModel.from_config(cfg)
    assert model.config.nonlinearity_mode == "post_blend"


# ---------------- latent_circuit: hand-computed single steps ----------------

def _tiny_latent_circuit(mode):
    cfg = AutoConfig.for_model(
        "latent_circuit", input_dim=1, latent_dim=2, output_dim=1,
        embedding_dim=4, alpha=0.5, activation="relu", sigma_rec=0.0,
        nonlinearity_mode=mode)
    model = AutoModel.from_config(cfg).eval()
    with torch.no_grad():
        model.w_rec.weight.copy_(torch.tensor([[0.3, -0.1], [0.2, 0.4]]))
        model.w_in.weight.copy_(torch.tensor([[0.5], [-0.25]]))
    return model


@pytest.mark.parametrize("mode", list(SUPPORTED_NONLINEARITY_MODES))
def test_latent_circuit_single_step_hand_computed(mode):
    model = _tiny_latent_circuit(mode)
    x = torch.tensor([[2.0]])
    z = torch.tensor([[1.0, -0.5]])
    with torch.no_grad():
        if mode == "rate":
            pre = torch.relu(z) @ model.w_rec.weight.t() + x @ model.w_in.weight.t()
            expected = 0.5 * z + 0.5 * pre
        elif mode == "post_blend":
            pre = z @ model.w_rec.weight.t() + x @ model.w_in.weight.t()
            expected = torch.relu(0.5 * z + 0.5 * pre)
        else:
            pre = z @ model.w_rec.weight.t() + x @ model.w_in.weight.t()
            expected = 0.5 * z + 0.5 * torch.relu(pre)
        got = model.recurrence(x, z)
    assert torch.allclose(got, expected, atol=1e-6), mode


# ---------------- lowrank: native rate mode + other modes ----------------

def _tiny_lowrank(mode):
    cfg = AutoConfig.for_model(
        "lowrank_rnn", input_dim=1, latent_dim=3, output_dim=1, rank=1,
        alpha=0.5, activation="tanh", sigma_rec=0.0, add_bias=True,
        nonlinearity_mode=mode)
    model = AutoModel.from_config(cfg).eval()
    with torch.no_grad():
        model.m.copy_(torch.tensor([[0.4], [-0.2], [0.1]]))
        model.n.copy_(torch.tensor([[0.3], [0.5], [-0.6]]))
        model.wi.copy_(torch.tensor([[0.7, -0.4, 0.2]]))
        model.si.copy_(torch.tensor([1.5]))
        model.b.copy_(torch.tensor([0.05, -0.1, 0.15]))
        model.wo.copy_(torch.tensor([[0.6], [-0.3], [0.9]]))
        model.so.copy_(torch.tensor([0.8]))
        model.h0.copy_(torch.tensor([0.2, -0.3, 0.4]))
    model._define_proxy_parameters()
    return model


def _lowrank_rec(model, rec_in):
    return rec_in @ model.n @ model.m.t() / model.config.latent_dim


@pytest.mark.parametrize("mode", list(SUPPORTED_NONLINEARITY_MODES))
def test_lowrank_single_step_hand_computed(mode):
    model = _tiny_lowrank(mode)
    x = torch.tensor([[1.5]])
    z = torch.tensor([[0.6, -0.2, 0.1]])
    with torch.no_grad():
        inp = x @ model.wi_full
        if mode == "rate":
            # bias stays inside f (reference form)
            expected = z + 0.5 * (-z + _lowrank_rec(model, torch.tanh(z + model.b)) + inp)
        elif mode == "post_blend":
            pre = _lowrank_rec(model, z) + inp + model.b
            expected = torch.tanh(0.5 * z + 0.5 * pre)
        else:
            pre = _lowrank_rec(model, z) + inp + model.b
            expected = 0.5 * z + 0.5 * torch.tanh(pre)
        got = model.recurrence(x, z)
    assert torch.allclose(got, expected, atol=1e-6), mode


def test_lowrank_forward_rate_mode_step0_no_bias_asymmetry():
    """rate mode keeps the reference forward: step-0 rate is f(h0) WITHOUT bias,
    later steps f(h + b). recurrence() (with bias) must match forward from t>=1."""
    model = _tiny_lowrank("rate")
    x = torch.tensor([[[1.5], [-0.5], [0.25]]])  # (B=1, T=3, K=1)
    with torch.no_grad():
        out = model(x)
        states = out.states[0]  # (T, N) post-step states
        # Hand-built reference rollout replicating the asymmetry
        h = model.h0.unsqueeze(0).clone()
        r = torch.tanh(h)  # step 0: NO bias
        manual = []
        for t in range(3):
            xt = x[:, t, :]
            h = h + 0.5 * (-h + _lowrank_rec(model, r) + xt @ model.wi_full)
            r = torch.tanh(h + model.b)  # later steps: with bias
            manual.append(h)
        manual = torch.cat(manual, dim=0)
        assert torch.allclose(states, manual, atol=1e-6)
        # recurrence() agrees with the forward step for t >= 1
        h1 = states[0:1]
        h2_direct = states[1:2]
        h2_via_recurrence = model.recurrence(x[:, 1, :], h1)
        assert torch.allclose(h2_via_recurrence, h2_direct, atol=1e-6)


@pytest.mark.parametrize("mode", ["pre_activation", "post_blend"])
def test_lowrank_forward_matches_recurrence_loop(mode):
    """Outside rate mode there is no step-0 asymmetry: forward must equal a
    manual recurrence loop exactly."""
    model = _tiny_lowrank(mode)
    x = torch.randn(2, 6, 1)
    with torch.no_grad():
        states = model(x).states
        z = model.init_state(2)
        manual = []
        for t in range(6):
            z = model.recurrence(x[:, t, :], z)
            manual.append(z)
        manual = torch.stack(manual, dim=1)
    assert torch.allclose(states, manual, atol=1e-6), mode


def test_lowrank_readout_unchanged_by_mode():
    """output_activation on the state is a family trait independent of the mode."""
    model = _tiny_lowrank("rate")
    z = torch.tensor([[0.3, -0.8, 0.5]])
    with torch.no_grad():
        expected = torch.tanh(z) @ model.wo_full / model.config.latent_dim
        got = model.readout(z)
    assert torch.allclose(got, expected, atol=1e-6)


# ---------------- serialization ----------------

@pytest.mark.parametrize("model_type,mode", [
    ("ctrnn", "post_blend"),
    ("ei_rnn", "rate"),
    ("constrained_rnn", "rate"),
    ("latent_circuit", "post_blend"),
    ("lowrank_rnn", "pre_activation"),
])
def test_config_roundtrip_preserves_mode(tmp_path, model_type, mode):
    cfg = AutoConfig.for_model(model_type, nonlinearity_mode=mode)
    cfg.to_json_file(str(tmp_path))
    raw = (tmp_path / "config.json").read_text(encoding="utf-8")
    assert "nonlinearity_mode" in raw
    assert "relu_after_blend" not in raw
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cfg2 = AutoConfig.from_pretrained(str(tmp_path))
    assert cfg2.nonlinearity_mode == mode


def test_legacy_checkpoint_without_mode_field_loads_default(tmp_path):
    """Old configs predate the field: they must load with the family default."""
    old = {"model_type": "ctrnn", "input_dim": 2, "latent_dim": 8, "output_dim": 1,
           "dt": 10.0, "tau": 50.0, "activation": "relu"}
    (tmp_path / "config.json").write_text(json.dumps(old))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cfg = AutoConfig.from_pretrained(str(tmp_path))
    assert cfg.nonlinearity_mode == "pre_activation"


def test_forward_all_modes_all_families_smoke():
    for model_type in ["ctrnn", "ei_rnn", "constrained_rnn", "latent_circuit", "lowrank_rnn"]:
        for mode in SUPPORTED_NONLINEARITY_MODES:
            cfg = AutoConfig.for_model(model_type, input_dim=3, latent_dim=8,
                                       output_dim=2, sigma_rec=0.0,
                                       nonlinearity_mode=mode)
            model = AutoModel.from_config(cfg).eval()
            out = model(torch.randn(2, 5, 3))
            assert out.outputs.shape == (2, 5, 2)
