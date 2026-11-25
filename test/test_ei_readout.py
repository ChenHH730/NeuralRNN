"""E-I native current/rate semantics, constraints, serialization and RL integration."""
import pytest
import torch
from neuralrnn import AutoConfig, AutoModel


def make(**kwargs):
    return AutoModel.from_config(AutoConfig.for_model(
        "ei_rnn", input_dim=2, latent_dim=5, output_dim=2, alpha=.2,
        sigma_rec=0, **kwargs)).eval()


def test_factory_defaults_and_removed_option():
    m = make()
    assert m.config.nonlinearity_mode == "rate"
    assert "readout_mode" not in m.config.to_dict()
    assert not m.config.positive_input_weights
    assert not m.config.positive_output_weights
    assert m.config.ei_positive_func == "rectify"
    assert m.config.init_method == "kaiming"
    assert m.config.spectral_radius is None
    assert not m.h0.requires_grad
    torch.testing.assert_close(m.h0, torch.zeros_like(m.h0))
    assert m.input2h.bias is not None
    assert m.h2h.bias is not None
    assert m.readout_layer.bias is not None
    with pytest.raises(TypeError, match="removed"):
        make(readout_mode="rate")


@pytest.mark.parametrize("mode", ["pre_activation", "post_blend", "rate"])
@pytest.mark.parametrize("e_only", [False, True])
def test_native_readout_roundtrip(tmp_path, mode, e_only):
    m = make(nonlinearity_mode=mode, readout_e_only=e_only)
    state = torch.tensor([[-1., .4, -.2, .8, -.7]])
    features = state.relu() if mode == "rate" else state
    if e_only:
        features = features[:, :m.e_size]
    expected = torch.nn.functional.linear(features, m._output_weight(), m.readout_layer.bias)
    torch.testing.assert_close(m.readout(state), expected)
    inputs = torch.randn(1, 6, 2)
    before = m(inputs, initial_state=state)
    m.save_pretrained(tmp_path)
    loaded = AutoModel.from_pretrained(tmp_path).eval()
    after = loaded(inputs, initial_state=state)
    torch.testing.assert_close(after.states, before.states, rtol=0, atol=0)
    torch.testing.assert_close(after.outputs, before.outputs, rtol=0, atol=0)


@pytest.mark.parametrize("positive_in", [False, True])
@pytest.mark.parametrize("positive_out", [False, True])
@pytest.mark.parametrize("func", ["rectify", "abs"])
@pytest.mark.parametrize("e_only", [False, True])
def test_sign_switches_and_negative_parameter_gradients(positive_in, positive_out, func, e_only):
    m = make(positive_input_weights=positive_in, positive_output_weights=positive_out,
             ei_positive_func=func, readout_e_only=e_only)
    with torch.no_grad():
        m.input2h.weight.fill_(-.3)
        m.readout_layer.weight.fill_(-.3)
        m.h2h.weight.fill_(-.3)
    m._input_weight().sum().backward()
    expected = (-1 if func == "abs" else 0) if positive_in else 1
    torch.testing.assert_close(m.input2h.weight.grad, torch.full_like(m.input2h.weight, expected))
    if positive_in:
        assert (m._input_weight() >= 0).all()
    else:
        assert (m._input_weight() < 0).all()
    output = m._output_weight()
    if not positive_out:
        assert (output < 0).all()
    else:
        assert (output[:, :m.e_size] >= 0).all()
        if not e_only:
            assert (output[:, m.e_size:] <= 0).all()
    # Constraints remain effective even after arbitrary raw parameter changes.
    assert torch.count_nonzero(m._recurrent_weight().diagonal()) == 0
    m._recurrent_weight().sum().backward()
    assert torch.count_nonzero(m.h2h.weight.grad.diagonal()) == 0


def test_interleaved_ei_masks_freeze_and_roundtrip(tmp_path):
    im = torch.ones(2, 5); im[0, 1] = 0
    rm = torch.ones(5, 5); rm[2, 0] = 0
    om = torch.ones(5, 2); om[3, 1] = 0
    m = make(dale_signs=[-1, 1, -1, 1, 1], in_mask=im, rec_mask=rm, out_mask=om,
             freeze_output=True, input_bias=True, recurrent_bias=True, output_bias=True)
    assert m.e_size == 3 and m.excitatory_indices.tolist() == [1, 3, 4]
    assert all(not p.requires_grad for p in m.readout_layer.parameters())
    assert m._input_weight()[1, 0] == 0 and m._recurrent_weight()[2, 0] == 0
    assert m._output_weight()[1, 1] == 0
    with torch.no_grad(): m.h2h.weight.fill_(.2)
    m._recurrent_weight().sum().backward()
    assert m.h2h.weight.grad[2, 0] == 0
    m.save_pretrained(tmp_path)
    other = AutoModel.from_pretrained(tmp_path).eval()
    z = torch.randn(2, 5)
    torch.testing.assert_close(m.readout(z), other.readout(z))


@pytest.mark.parametrize("mode", ["rate", "pre_activation", "post_blend"])
@pytest.mark.parametrize("e_only", [False, True])
def test_actor_critic_heads_use_native_ei_activity(mode, e_only):
    core = make(nonlinearity_mode=mode, dale_signs=[-1, 1, -1, 1, 1]).config.to_dict()
    agent = AutoModel.from_config(AutoConfig.for_model(
        "actor_critic", core_config=core, action_dim=2, readout_e_only=e_only,
        aux_head_out_dim=3, aux_input="state", policy_logit_scale=.7)).eval()
    state = torch.tensor([[-1., .4, -.2, .8, -.7]])
    features = state.relu() if mode == "rate" else state
    if e_only:
        features = features[:, [1, 3, 4]]
    torch.testing.assert_close(agent.readout(state), agent.actor(features) * .7)
    torch.testing.assert_close(agent.value(state), agent.critic(features).squeeze(-1))
    torch.testing.assert_close(agent.aux_logits(state), agent.aux(features))


@pytest.mark.parametrize("kwargs", [dict(ei_positive_func="bad"), dict(ei_ratio=1.2),
    dict(in_mask=[[1]]), dict(spectral_radius=0), dict(positive_input_weights="yes"),
    dict(dale_signs=[-1]*5), dict(h0_init=float("nan"))])
def test_invalid_options(kwargs):
    with pytest.raises(ValueError): make(**kwargs)
