"""Public noise controls, equation checks and old-config compatibility."""
import warnings

import pytest
import torch

from neuralrnn import AutoConfig, AutoModel
from neuralrnn.analysis import linearize


def make(kind='ctrnn', **kwargs):
    return AutoModel.from_config(AutoConfig.for_model(
        kind, input_dim=2, latent_dim=4, output_dim=2, **kwargs))


FAMILIES = ['ctrnn', 'ei_rnn', 'constrained_rnn', 'gain_rnn', 'stp_rnn',
            'lowrank_rnn', 'latent_circuit', 'shallow_plrnn', 'dend_plrnn', 'alrnn']


@pytest.mark.parametrize('kind', FAMILIES)
@pytest.mark.parametrize('position', ['pre', 'post', 'state'])
def test_public_train_eval_switches(kind, position):
    torch.manual_seed(53)
    model = make(kind, sigma_rec=.3, noise_position=position)
    x = torch.randn(8, 5, 2)
    def run():
        return model(x).states.detach()
    model.eval()
    assert torch.equal(run(), run())
    model.evaluate_noise = True
    assert not torch.equal(run(), run())
    model.evaluate_noise = False
    model.train()
    assert not torch.equal(run(), run())
    model.sigma_rec = 0
    assert torch.equal(run(), run())
    assert model.config.sigma_rec == 0


@pytest.mark.parametrize('mode', ['rate', 'pre_activation', 'post_blend'])
@pytest.mark.parametrize('position', ['pre', 'post', 'state'])
@pytest.mark.parametrize('scaling', [False, True])
def test_ctrnn_noise_equation(mode, position, scaling):
    model = make(alpha=.2, activation='relu', sigma_rec=.3,
                 nonlinearity_mode=mode, noise_position=position,
                 noise_alpha_scaling=scaling)
    model._randn_like = torch.ones_like
    x = torch.tensor([[.1, -.2]])
    z = torch.tensor([[-.7, .3, -.1, .5]])
    recurrent = torch.relu(z) if mode == 'rate' else z
    pre = model.h2h(recurrent) + model.input2h(x)
    s = .3 * ((2 * .2) ** .5 if scaling else 1)
    if position == 'pre':
        pre = pre + s
    expected = .8 * z + .2 * (torch.relu(pre) if mode == 'pre_activation' else pre)
    if position == 'post':
        expected = expected + s
    if mode == 'post_blend':
        expected = torch.relu(expected)
    if position == 'state':
        expected = expected + s
    torch.testing.assert_close(model.recurrence(x, z), expected)


@pytest.mark.parametrize('kind', ['shallow_plrnn', 'dend_plrnn', 'alrnn'])
@pytest.mark.parametrize('position', ['pre', 'post', 'state'])
def test_plrnn_equations_and_gradients(kind, position):
    torch.manual_seed(31)
    model = make(kind, sigma_rec=.2, noise_position=position)
    model._randn_like = torch.ones_like
    z = torch.tensor([[-.4, .3, -.2, .1]], requires_grad=True)
    x = torch.tensor([[.1, -.3]])
    s = .2 if position == 'pre' else 0
    if kind == 'shallow_plrnn':
        expected = model.A*z + torch.relu(z @ model.W2.T + model.h2+s) @ model.W1.T + model.h1
    elif kind == 'dend_plrnn':
        basis = (model.alphas * torch.relu((z+s).unsqueeze(-1)-model.H)).sum(-1)
        expected = model.A*z + basis @ model.W.T + model.h
    else:
        phi = torch.cat([(z+s)[:, :model.n_linear], torch.relu((z+s)[:, model.n_linear:])], -1)
        expected = model.A*z + phi @ model.W.T + model.h
    expected = expected + x @ model.C.T
    if position != 'pre':
        expected = expected + .2
    actual = model.recurrence(x, z)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert torch.isfinite(z.grad).all()
    model.evaluate_noise = True
    noisy_jac = linearize(model, z.detach()[0]).jacobian
    model.sigma_rec = 0
    torch.testing.assert_close(noisy_jac, linearize(model, z.detach()[0]).jacobian)


@pytest.mark.parametrize('kind', ['dend_plrnn', 'alrnn'])
def test_state_noise_is_after_clamp(kind):
    model = make(kind, sigma_rec=2, noise_position='post', clip_range=.1)
    model._randn_like = torch.ones_like
    z = torch.zeros(2, 4)
    x = torch.zeros(2, 2)
    assert model.recurrence(x, z).abs().max() <= .1
    model.config.noise_position = 'state'
    assert model.recurrence(x, z).min() >= 1.9


@pytest.mark.parametrize('kind,position,scaling', [
    ('ctrnn', 'pre', False), ('stp_rnn', 'post', True),
    ('lowrank_rnn', 'state', False), ('latent_circuit', 'pre', True),
    ('shallow_plrnn', 'pre', False),
])
def test_family_defaults(kind, position, scaling):
    config = AutoConfig.for_model(kind)
    assert config.noise_position == position
    assert config.noise_alpha_scaling == scaling
    assert config.evaluate_noise is False


@pytest.mark.parametrize('mode', ['rate', 'pre_activation', 'post_blend'])
def test_legacy_battista_config_migrates_once(mode, tmp_path):
    with pytest.warns(FutureWarning, match='amplitude is preserved'):
        model = make(alpha=.2, sigma_rec=.15, noise_scaling='sqrt_2_over_alpha',
                     nonlinearity_mode=mode, evaluate_noise=True)
    assert model.config.sigma_rec == pytest.approx(.75)
    assert model._noise_std() == pytest.approx(.15 * (2/.2)**.5)
    assert 'noise_scaling' not in model.config.to_dict()
    model.save_pretrained(tmp_path)
    with warnings.catch_warnings(record=True) as seen:
        loaded = AutoModel.from_pretrained(tmp_path)
    assert not any('noise_scaling' in str(w.message) for w in seen)
    assert loaded.config.sigma_rec == model.config.sigma_rec
    assert loaded.evaluate_noise is True
    x = torch.randn(2, 3, 2)
    model.eval()
    loaded.eval()
    torch.manual_seed(7)
    expected = model(x).states
    torch.manual_seed(7)
    torch.testing.assert_close(loaded(x).states, expected)


@pytest.mark.parametrize('kind,opts', [('gated_rnn', {'rnn_type':'GRU'}),
                                    ('gated_rnn', {'rnn_type':'LSTM'}), ('tiny_rnn', {})])
def test_gated_noise_warning_and_deterministic_output(kind, opts):
    with pytest.warns(UserWarning, match='not supported'):
        model = make(kind, sigma_rec=.1, evaluate_noise=True, **opts)
    x = torch.randn(2, 3, 2)
    with warnings.catch_warnings(record=True) as seen:
        assert torch.equal(model(x).states, model(x).states)
    assert not seen  # Do not flood every training step with warnings.
    with warnings.catch_warnings(record=True) as seen:
        model = make(kind, **opts)
    assert not seen
    model.sigma_rec = .2
    with pytest.warns(UserWarning, match='not supported'):
        model(x)


def test_actor_critic_eval_flag_roundtrip(tmp_path):
    core = AutoConfig.for_model('ctrnn', input_dim=2, latent_dim=4, sigma_rec=.2)
    model = AutoModel.from_config(AutoConfig.for_model(
        'actor_critic', core_config=core.to_dict(), action_dim=2))
    model.evaluate_noise = True
    assert model.core.evaluate_noise
    model.save_pretrained(tmp_path)
    loaded = AutoModel.from_pretrained(tmp_path)
    assert loaded.evaluate_noise and loaded.core.evaluate_noise
    loaded.evaluate_noise = False
    assert not loaded.core.evaluate_noise


@pytest.mark.parametrize('enabled', [False, True])
def test_evaluation_helpers_inherit_flag_and_restore_mode(enabled):
    from types import SimpleNamespace
    from neuralrnn.analysis.perturbation import compute_choice
    from neuralrnn.analysis.psychometric import compute_psychometric
    from neuralrnn.data import ReconstructionDataset

    model = make(sigma_rec=.2, evaluate_noise=enabled)
    x = torch.randn(3, 4, 2)
    seen = []
    def observe(module, args):
        seen.append((module.training, module.recurrent_noise_enabled))
    handle = model.register_forward_pre_hook(observe)
    try:
        compute_choice(model, x)
        compute_psychometric(model, x, [{}]*3)
        ReconstructionDataset.from_rnn_and_task(model, SimpleNamespace(inputs=x))
    finally:
        handle.remove()
    assert len(seen) == 3
    assert all(mode is False and noise is enabled for mode, noise in seen)
    assert model.training
    assert model.evaluate_noise is enabled


@pytest.mark.parametrize('kind', ['shallow_plrnn', 'dend_plrnn', 'alrnn'])
def test_native_discrete_scaling(kind):
    model = make(kind, sigma_rec=.3, noise_alpha_scaling=True, noise_position='state')
    model._randn_like = torch.ones_like
    x, z = torch.zeros(2, 2), torch.zeros(2, 4)
    model.eval()
    base = model.recurrence(x, z)
    model.evaluate_noise = True
    torch.testing.assert_close(model.recurrence(x, z)-base, torch.full_like(z, .3 * 2**.5))


@pytest.mark.parametrize('kwargs', [dict(sigma_rec=-1), dict(sigma_rec=float('nan')),
    dict(sigma_rec=float('inf')), dict(evaluate_noise='yes'),
    dict(noise_alpha_scaling='yes'), dict(noise_position='unknown')])
def test_invalid_noise_configuration(kwargs):
    with pytest.raises(ValueError):
        AutoConfig.for_model('ctrnn', **kwargs)
