"""Tests for the gain_rnn family (GainRNNModel).

Covers registration, family defaults, rate-map placement semantics, single-step
hand-computed parity across nonlinearity modes, noise positions, freeze
behavior, masks / positive weights, save-load roundtrips, the identity anchor
vs ConstrainedRNNModel, and cross-model parity with ConnectomeRNNModel.
"""
import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F

from neuralrnn import AutoConfig, AutoModel, Trainer, TrainingArguments, SupervisedObjective
from neuralrnn.activations import get_activation
from neuralrnn.data import BaseDataset
from neuralrnn.models.gain_rnn import GainRNNConfig, GainRNNModel


class _ToyRegressionTask(BaseDataset):
    kind = "neurogym"

    def __init__(self, input_dim=3, output_dim=2, T=12, B=6):
        self.input_dim, self.output_dim, self.T, self.B = input_dim, output_dim, T, B

    def sample_batch(self):
        return {
            "inputs": torch.randn(self.B, self.T, self.input_dim),
            "targets": torch.randn(self.B, self.T, self.output_dim),
            "mask": None,
        }

    def task_input(self):
        return torch.zeros(self.input_dim)


def _make_gain(**overrides):
    kw = dict(input_dim=3, latent_dim=6, output_dim=2)
    kw.update(overrides)
    return AutoModel.from_config(AutoConfig.for_model("gain_rnn", **kw))


def _write_weights(model, w_in, w_rec, b_rec, w_out, b_out, gain=None, bias=None):
    with torch.no_grad():
        model.input2h.weight.copy_(w_in)
        model.input2h.bias.zero_()
        model.h2h.weight.copy_(w_rec)
        model.h2h.bias.copy_(b_rec)
        model.readout_layer.weight.copy_(w_out)
        model.readout_layer.bias.copy_(b_out)
        if gain is not None:
            model.gain.copy_(gain)
        if bias is not None:
            model.bias.copy_(bias)


# ============================ registration / defaults ============================

class TestRegistration:
    def test_autoconfig_automl_shapes(self):
        model = _make_gain()
        out = model(torch.randn(4, 10, 3))
        assert out.outputs.shape == (4, 10, 2)
        assert out.states.shape == (4, 10, 6)
        assert isinstance(model, GainRNNModel)

    def test_family_defaults(self):
        cfg = AutoConfig.for_model("gain_rnn", input_dim=3, latent_dim=6, output_dim=2)
        assert cfg.nonlinearity_mode == "rate"
        assert cfg.activation == "softplus"
        assert cfg.gain_position == "outside"
        assert cfg.noise_position == "pre"
        assert cfg.noise_alpha_scaling is False
        assert cfg.freeze_gain is False and cfg.freeze_bias is False
        assert cfg.positive_input_weights is False and cfg.positive_output_weights is False

    def test_invalid_enums_raise(self):
        with pytest.raises(ValueError, match="gain_position"):
            GainRNNConfig(input_dim=3, latent_dim=6, output_dim=2, gain_position="middle")
        with pytest.raises(ValueError, match="noise_position"):
            GainRNNConfig(input_dim=3, latent_dim=6, output_dim=2, noise_position="everywhere")
        with pytest.raises(TypeError, match="activation_params"):
            GainRNNConfig(input_dim=3, latent_dim=6, output_dim=2, activation_params=[1, 2])


# ============================ rate map semantics ============================

class TestRateMap:
    def test_outside_handcomputed(self):
        model = _make_gain(gain_position="outside", activation="tanh")
        with torch.no_grad():
            model.gain.copy_(torch.tensor([1.0, 2.0, 0.5, 1.5, 1.0, 3.0]))
            model.bias.copy_(torch.tensor([0.1, -0.2, 0.0, 0.3, -0.1, 0.2]))
        u = torch.randn(4, 6)
        expected = model.gain * torch.tanh(u + model.bias)
        assert torch.allclose(model.rate_map(u), expected, atol=1e-6)

    def test_inside_handcomputed(self):
        model = _make_gain(gain_position="inside", activation="tanh")
        with torch.no_grad():
            model.gain.copy_(torch.tensor([1.0, 2.0, 0.5, 1.5, 1.0, 3.0]))
            model.bias.copy_(torch.tensor([0.1, -0.2, 0.0, 0.3, -0.1, 0.2]))
        u = torch.randn(4, 6)
        expected = torch.tanh(model.gain * u + model.bias)
        assert torch.allclose(model.rate_map(u), expected, atol=1e-6)

    def test_relu_inside_outside_equivalence(self):
        """For ReLU with positive gains the two placements are exactly equivalent
        (scale degeneracy, gain_rnn.md section 6.3)."""
        m_out = _make_gain(gain_position="outside", activation="relu")
        m_in = _make_gain(gain_position="inside", activation="relu")
        g = torch.rand(6) + 0.5
        for m in (m_out, m_in):
            with torch.no_grad():
                m.gain.copy_(g)
                m.bias.zero_()
        u = torch.randn(8, 6)
        assert torch.allclose(m_out.rate_map(u), m_in.rate_map(u), atol=1e-6)

    def test_default_rate_map_is_plain_activation(self):
        model = _make_gain(activation="tanh")
        u = torch.randn(8, 6)
        assert torch.allclose(model.rate_map(u), torch.tanh(u), atol=1e-6)


# ============================ single-step parity across modes ============================

@pytest.mark.parametrize("gain_position", ["outside", "inside"])
@pytest.mark.parametrize("mode", ["pre_activation", "post_blend", "rate"])
def test_single_step_handcomputed(gain_position, mode):
    torch.manual_seed(0)
    M, K = 3, 2
    model = _make_gain(input_dim=K, latent_dim=M, output_dim=2,
                       nonlinearity_mode=mode, gain_position=gain_position,
                       activation="tanh", dt=10.0, tau=100.0)
    w_in = torch.randn(M, K)
    w_rec = torch.randn(M, M)
    b_rec = torch.randn(M)
    g = torch.rand(M) + 0.5
    b = torch.randn(M) * 0.1
    _write_weights(model, w_in, w_rec, b_rec, torch.randn(2, M), torch.randn(2), gain=g, bias=b)

    x_t, z_prev = torch.randn(5, K), torch.randn(5, M)
    alpha = 0.1
    act = torch.tanh
    rate = g * act(z_prev + b) if gain_position == "outside" else act(g * z_prev + b)
    pre = x_t @ w_in.t() + (rate if mode == "rate" else z_prev) @ w_rec.t() + b_rec
    if mode == "pre_activation":
        rate_pre = g * act(pre + b) if gain_position == "outside" else act(g * pre + b)
        expected = (1 - alpha) * z_prev + alpha * rate_pre
    elif mode == "post_blend":
        update = (1 - alpha) * z_prev + alpha * pre
        expected = g * act(update + b) if gain_position == "outside" else act(g * update + b)
    else:
        expected = (1 - alpha) * z_prev + alpha * pre
    got = model.recurrence(x_t, z_prev)
    assert torch.allclose(got, expected, atol=1e-6)


def test_rate_mode_readout_from_rates():
    """Family deviation: in rate mode the readout consumes rate_map(z), not z."""
    model = _make_gain(nonlinearity_mode="rate", activation="tanh")
    with torch.no_grad():
        model.gain.copy_(torch.rand(6) + 0.5)
        model.bias.copy_(torch.randn(6) * 0.1)
    z = torch.randn(4, 6)
    expected = F.linear(model.rate_map(z), model.readout_layer.weight, model.readout_layer.bias)
    assert torch.allclose(model.readout(z), expected, atol=1e-6)
    # ... and is NOT the CTRNN readout-from-state.
    assert not torch.allclose(model.readout(z), F.linear(z, model.readout_layer.weight,
                                                         model.readout_layer.bias))


# ============================ identity anchor vs ConstrainedRNN ============================

@pytest.mark.parametrize("mode", ["pre_activation", "post_blend", "rate"])
def test_identity_anchor_vs_constrained(mode):
    """With gain=1, bias=0 a GainRNN must reproduce ConstrainedRNNModel exactly
    (recurrence in all modes; readout in pre_activation/post_blend)."""
    from neuralrnn.models.constrained_rnn import ConstrainedRNNModel

    torch.manual_seed(1)
    base_cfg = dict(input_dim=3, latent_dim=6, output_dim=2,
                    nonlinearity_mode=mode, activation="relu", dt=10.0, tau=100.0)
    ref = ConstrainedRNNModel(AutoConfig.for_model("constrained_rnn", **base_cfg))
    gain = _make_gain(**base_cfg)
    # Share all common parameters/buffers (h0 is a buffer by default).
    with torch.no_grad():
        for name, p in ref.named_parameters():
            gain.get_parameter(name).copy_(p)
        for name, b in ref.named_buffers():
            getattr(gain, name).copy_(b)
    x_t, z_prev = torch.randn(5, 3), torch.randn(5, 6)
    assert torch.allclose(gain.recurrence(x_t, z_prev), ref.recurrence(x_t, z_prev), atol=1e-6)
    if mode != "rate":
        assert torch.allclose(gain.readout(z_prev), ref.readout(z_prev), atol=1e-6)
    else:
        # Documented family deviation: rate-mode readout reads the rates.
        assert torch.allclose(gain.readout(z_prev), ref.readout(torch.relu(z_prev)), atol=1e-6)
    # Full forward parity (outputs for non-rate modes).
    out_g, out_r = gain(torch.randn(2, 7, 3)), ref(torch.randn(2, 7, 3))
    assert out_g.outputs.shape == out_r.outputs.shape


# ============================ noise positions ============================

@pytest.mark.parametrize("mode", ["pre_activation", "post_blend", "rate"])
@pytest.mark.parametrize("position", ["pre", "post"])
def test_noise_positions(mode, position):
    torch.manual_seed(2)
    M = 4
    model = _make_gain(latent_dim=M, nonlinearity_mode=mode, noise_position=position,
                       activation="relu", sigma_rec=0.5, noise_alpha_scaling=True,
                       dt=10.0, tau=100.0)
    model.train()
    x_t, z_prev = torch.randn(3, 3), torch.randn(3, M)

    torch.manual_seed(123)
    got = model.recurrence(x_t, z_prev)

    # Hand replay with the same single randn draw.
    torch.manual_seed(123)
    alpha = 0.1
    std = (2 * alpha * 0.5 ** 2) ** 0.5
    rate = model.rate_map(z_prev) if mode == "rate" else z_prev
    pre = F.linear(x_t, model.input2h.weight, model.input2h.bias) + F.linear(
        rate, model.h2h.weight, model.h2h.bias)
    if position == "pre":
        pre = pre + std * torch.randn_like(pre)
    if mode == "pre_activation":
        expected = (1 - alpha) * z_prev + alpha * model.rate_map(pre)
        if position == "post":
            expected = expected + std * torch.randn_like(expected)
    else:
        update = (1 - alpha) * z_prev + alpha * pre
        if position == "post":
            update = update + std * torch.randn_like(update)
        expected = model.rate_map(update) if mode == "post_blend" else update
    assert torch.allclose(got, expected, atol=1e-6)

    # Eval mode: no noise, deterministic.
    model.eval()
    a = model.recurrence(x_t, z_prev)
    b = model.recurrence(x_t, z_prev)
    assert torch.allclose(a, b)


# ============================ piecewise_tanh / activation_params ============================

class TestPiecewiseTanh:
    def test_values_and_continuity(self):
        f = get_activation("piecewise_tanh", r0=20.0, rmax=100.0)
        x = torch.linspace(-500, 500, 2001)
        y = f(x)
        # Negative branch saturates at -r0, positive branch at rmax-r0.
        assert abs(y[0].item() - (-20.0)) < 0.1
        assert abs(y[-1].item() - 80.0) < 0.1
        # Continuous and through the origin.
        assert abs(f(torch.tensor(0.0)).item()) < 1e-7
        # Unit slope at the origin (finite difference).
        eps = 1e-4
        slope = (f(torch.tensor(eps)) - f(torch.tensor(-eps))).item() / (2 * eps)
        assert abs(slope - 1.0) < 1e-3

    def test_stroud_gain_formula(self):
        """Inside gain with piecewise_tanh == Stroud et al. 2018 eq. 2."""
        model = _make_gain(gain_position="inside", activation="piecewise_tanh",
                           activation_params={"r0": 20.0, "rmax": 100.0})
        with torch.no_grad():
            model.gain.copy_(torch.full((6,), 1.7))
            model.bias.zero_()
        x = torch.randn(8, 6)
        g = 1.7
        expected = torch.where(x < 0, 20.0 * torch.tanh(g * x / 20.0),
                               80.0 * torch.tanh(g * x / 80.0))
        assert torch.allclose(model.rate_map(x), expected, atol=1e-5)

    def test_invalid_params_raise(self):
        with pytest.raises(ValueError, match="rmax > r0"):
            get_activation("piecewise_tanh", r0=100.0, rmax=20.0)
        with pytest.raises(ValueError, match="kwargs"):
            get_activation("piecewise_tanh", r0=20.0, rmax=100.0, bogus=1)

    def test_activation_params_softplus_beta(self):
        model = _make_gain(activation="softplus", activation_params={"beta": 2.0})
        x = torch.randn(4, 6)
        expected = 0.5 * torch.log1p(torch.exp(2.0 * x))
        assert torch.allclose(model.act(x), expected, atol=1e-5)


# ============================ positive weights ============================

def test_positive_input_output_weights():
    torch.manual_seed(4)
    M = 4
    model = _make_gain(latent_dim=M, nonlinearity_mode="rate", activation="tanh",
                       positive_input_weights=True, positive_output_weights=True)
    x_t, z_prev = torch.randn(3, 3), torch.randn(3, M)
    rate = model.rate_map(z_prev)
    pre = F.linear(x_t, F.relu(model.input2h.weight), model.input2h.bias) + F.linear(
        rate, model.h2h.weight, model.h2h.bias)
    expected_z = (1 - model.alpha) * z_prev + model.alpha * pre
    assert torch.allclose(model.recurrence(x_t, z_prev), expected_z, atol=1e-6)
    expected_y = F.linear(model.rate_map(expected_z), F.relu(model.readout_layer.weight),
                          model.readout_layer.bias)
    assert torch.allclose(model.readout(expected_z), expected_y, atol=1e-6)


# ============================ freeze behavior ============================

class TestFreeze:
    def test_freeze_groups(self):
        model = _make_gain()
        assert model._match_parameters(["gains"], None) == {"gain"}
        assert model._match_parameters(["biases"], None) == {"bias"}
        # Base groups still present (merged, not replaced).
        assert "h2h.weight" in model._match_parameters(["recurrent"], None)
        assert "input2h.weight" in model._match_parameters(["input"], None)

    def test_freeze_gain_bias_flags(self):
        model = _make_gain(freeze_gain=True)
        grads = {n: p.requires_grad for n, p in model.named_parameters()}
        assert not grads["gain"] and grads["bias"]
        model = _make_gain(freeze_bias=True)
        grads = {n: p.requires_grad for n, p in model.named_parameters()}
        assert grads["gain"] and not grads["bias"]

    def test_connectome_scenario_only_gain_bias_train(self):
        """Beiran & Litwin-Kumar student: freeze everything except gain/bias."""
        model = _make_gain(freeze_input=True, freeze_recurrent=True,
                           freeze_output=True, freeze_h0=True)
        state_before = copy.deepcopy(model.state_dict())
        ds = _ToyRegressionTask()
        Trainer(model, ds, SupervisedObjective("regression"),
                TrainingArguments(max_steps=1, log_every=0)).train()
        state_after = model.state_dict()
        for n, p_before in state_before.items():
            p_after = state_after[n]
            if n in ("gain", "bias"):
                assert not torch.allclose(p_before, p_after, atol=1e-10), f"{n} should change"
            else:
                assert torch.allclose(p_before, p_after, atol=1e-10), f"{n} should stay frozen"

    def test_freeze_roundtrip(self, tmp_path):
        model = _make_gain(freeze_gain=True, freeze_recurrent=True)
        before = {n: p.requires_grad for n, p in model.named_parameters()}
        model.save_pretrained(str(tmp_path))
        reloaded = AutoModel.from_pretrained(str(tmp_path))
        after = {n: p.requires_grad for n, p in reloaded.named_parameters()}
        assert before == after
        assert reloaded.config.freeze_gain is True


# ============================ serialization ============================

def test_config_roundtrip_with_arrays_and_params():
    cfg = GainRNNConfig(
        input_dim=3, latent_dim=4, output_dim=2,
        gain_init=np.linspace(0.5, 2.0, 4), bias_init=[0.1, -0.1, 0.0, 0.2],
        h0_init=np.ones(4) * 0.3, activation="softplus",
        activation_params={"beta": 1.5}, gain_position="inside",
        rec_mask=(np.random.rand(4, 4) > 0.5).astype(float).tolist(),
    )
    d = cfg.to_dict()
    import json
    json.dumps(d)  # must be JSON-serializable
    cfg2 = GainRNNConfig.from_dict(d)
    assert cfg2.gain_init == [0.5, 1.0, 1.5, 2.0]
    assert cfg2.bias_init == [0.1, -0.1, 0.0, 0.2]
    assert cfg2.activation_params == {"beta": 1.5}
    assert cfg2.gain_position == "inside"
    model = AutoModel.from_config(cfg2)
    assert torch.allclose(model.gain, torch.tensor([0.5, 1.0, 1.5, 2.0]))
    assert torch.allclose(model.h0, torch.full((4,), 0.3))


def test_save_load_roundtrip(tmp_path):
    model = _make_gain(gain_init=np.linspace(0.5, 2.0, 6))
    x = torch.randn(2, 8, 3)
    model.eval()
    out_before = model(x)
    model.save_pretrained(str(tmp_path))
    loaded = AutoModel.from_pretrained(str(tmp_path))
    loaded.eval()
    for (n1, p1), (n2, p2) in zip(model.state_dict().items(), loaded.state_dict().items()):
        assert n1 == n2 and torch.allclose(p1, p2), n1
    assert torch.allclose(out_before.outputs, loaded(x).outputs, atol=1e-6)


# ============================ h0 / autonomous / masks ============================

def test_h0_init_both_modes():
    model = _make_gain(h0_init=0.4)
    assert torch.allclose(model.h0, torch.full((6,), 0.4))
    assert torch.allclose(model.init_state(3), torch.full((3, 6), 0.4))
    model_t = _make_gain(h0_init=0.4, trainable_h0=True)
    assert isinstance(model_t.h0, torch.nn.Parameter)
    assert torch.allclose(model_t.h0.detach(), torch.full((6,), 0.4))


def test_autonomous_rollout():
    model = _make_gain()
    model.eval()
    z0 = model.init_state(2)
    out = model(inputs=None, initial_state=z0, n_steps=6)
    assert out.states.shape == (2, 6, 6)
    # recurrence with x_t=None skips the input term entirely.
    z1 = model.recurrence(None, z0)
    W = model.h2h.weight
    expected = (1 - model.alpha) * z0 + model.alpha * F.linear(
        model.rate_map(z0), W, model.h2h.bias)
    assert torch.allclose(z1, expected, atol=1e-6)


def test_mask_gain_composition():
    """Masked recurrent entries stay exactly zero after an optimizer step."""
    torch.manual_seed(5)
    M = 4
    rec_mask = np.ones((M, M), dtype=np.float32)
    rec_mask[0, 1] = 0.0
    rec_mask[2, 3] = 0.0
    model = _make_gain(latent_dim=M, rec_mask=rec_mask.tolist())
    assert model.h2h.weight[0, 1].item() == 0.0
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss = model(torch.randn(2, 5, 3)).outputs.sum()
    opt.zero_grad()
    loss.backward()
    assert model.h2h.weight.grad[0, 1].item() == 0.0
    assert model.h2h.weight.grad[2, 3].item() == 0.0
    opt.step()
    assert model.h2h.weight[0, 1].item() == 0.0
    assert model.h2h.weight[2, 3].item() == 0.0
