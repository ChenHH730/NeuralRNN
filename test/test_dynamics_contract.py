"""Revision contracts checked against independent maps and explicit equations."""
import numpy as np
import pytest
import torch

from neuralrnn import AutoConfig, AutoModel, NeuralDynamicsModel, NeuralRNNConfig
from neuralrnn.analysis import linearize, max_lyapunov_exponent
from neuralrnn.analysis.fixed_points import (NumericFixedPointFinder,
    GolubFixedPointFinder, ScipyFixedPointFinder)
from neuralrnn.train.objectives.teacher_forcing import TeacherForcingObjective


def make(kind, **kwargs):
    return AutoModel.from_config(AutoConfig.for_model(
        kind, input_dim=2, latent_dim=4, output_dim=2, **kwargs))


class Affine(NeuralDynamicsModel):
    def __init__(self, diagonal=(0.5, 0.8), bias=(0.1, -0.2)):
        super().__init__(NeuralRNNConfig(input_dim=0, latent_dim=2, output_dim=2))
        self.A = torch.nn.Parameter(torch.diag(torch.tensor(diagonal, dtype=torch.float64)))
        self.b = torch.nn.Parameter(torch.tensor(bias, dtype=torch.float64), requires_grad=False)

    def recurrence(self, x_t, z_prev, *, inputs=None):
        return z_prev @ self.A.T + self.b

    def readout(self, z):
        return z


@pytest.mark.parametrize('cell', ['GRU', 'LSTM'])
def test_complete_trajectory_matches_fused_cell_and_gradients(cell):
    torch.manual_seed(8)
    model = make('gated_rnn', rnn_type=cell, dtype='float64')
    x = torch.randn(3, 7, 2, dtype=torch.float64)
    z0 = torch.randn(3, model.state_dim, dtype=torch.float64)
    packed = model.full_state_trajectory(x, initial_state=z0)
    fused = model(x, initial_state=z0)
    torch.testing.assert_close(packed[..., :4], fused.states, atol=1e-12, rtol=1e-12)
    if cell == 'LSTM':
        # Independently run fused one-step cells to obtain every cell state.
        state = (z0[:, :4].unsqueeze(0), z0[:, 4:].unsqueeze(0))
        cells = []
        for t in range(x.shape[1]):
            _, state = model.rnn(x[:, t:t+1], state)
            cells.append(state[1][0])
        torch.testing.assert_close(packed[..., 4:], torch.stack(cells, 1))
    params = tuple(model.rnn.parameters())
    g1 = torch.autograd.grad(packed[..., :4].square().sum(), params)
    g2 = torch.autograd.grad(fused.states.square().sum(), params)
    for a, b in zip(g1, g2):
        torch.testing.assert_close(a, b)


@pytest.mark.parametrize('kind,opts', [('gated_rnn', {'rnn_type':'LSTM'}), ('stp_rnn', {})])
def test_full_jacobian_matches_finite_difference_and_trajectory(kind, opts):
    torch.manual_seed(4)
    model = make(kind, **opts).double()
    x = torch.ones(2, 5, 2, dtype=torch.float64) * 0.2
    with model.evaluation_context(noise=False):
        z0 = model.init_state(2).clone()
        z0[:, :4] = 0.4
        if kind == 'stp_rnn':
            z0[:, 4:8] = 0.7
            z0[:, 8:] = 0.3
        full = model.full_state_trajectory(x, initial_state=z0)
        out = model(x, initial_state=z0)
        torch.testing.assert_close(full[..., :4], out.states)
        if kind == 'stp_rnn':
            torch.testing.assert_close(full, out.extras['full_states'])
        z = z0[0]
        lin = linearize(model, z, task_input=x[0, 0])
        eps = 1e-6
        cols = []
        for delta in torch.eye(model.state_dim, dtype=z.dtype) * eps:
            cols.append((model.recurrence(x[:1, 0], (z+delta)[None])[0] -
                         model.recurrence(x[:1, 0], (z-delta)[None])[0]) / (2*eps))
        np.testing.assert_allclose(lin.jacobian, torch.stack(cols, 1).detach(), atol=1e-7)
        with pytest.raises(ValueError, match='complete state'):
            linearize(model, z[:4], task_input=x[0, 0])


@pytest.mark.parametrize('finder', [NumericFixedPointFinder(n_iters=1200, lr=.03, speed_tol=1e-7),
    GolubFixedPointFinder(n_iters=1200, n_candidates=4, tol_q=1e-14, noise_scale=0),
    ScipyFixedPointFinder(n_candidates=4)])
def test_affine_roots_and_flags(finder):
    model = Affine()
    model.train()
    initial_flags = [p.requires_grad for p in model.parameters()]
    fps = finder.find(model, init_states=torch.zeros(4, 2, dtype=torch.float64))
    assert len(fps) == 1
    expected = torch.linalg.solve(torch.eye(2, dtype=torch.float64)-model.A, model.b)
    np.testing.assert_allclose(fps.points[0].z, expected.detach(), atol=1e-5)
    assert fps.points[0].is_stable
    assert model.training and [p.requires_grad for p in model.parameters()] == initial_flags


@pytest.mark.parametrize('finder', [NumericFixedPointFinder(n_iters=3),
    GolubFixedPointFinder(n_iters=3, n_candidates=3), ScipyFixedPointFinder(n_candidates=3)])
def test_no_root_and_exception_restore(finder):
    model = Affine((1., 1.), (1., 1.))
    assert len(finder.find(model)) == 0
    flags = [p.requires_grad for p in model.parameters()]
    original = model.recurrence
    def broken(*args, **kwargs):
        raise RuntimeError('deliberate failure')
    model.recurrence = broken
    # SciPy intentionally skips failed candidates, others propagate.
    try:
        finder.find(model)
    except RuntimeError as exc:
        assert 'deliberate failure' in str(exc)
    assert model.training and [p.requires_grad for p in model.parameters()] == flags
    model.recurrence = original


def test_lyapunov_maximum_tail_and_units():
    model = Affine((0.4, 1.2), (0., 0.))
    # Dominant direction is NOT the first coordinate, T is not divisible by ons.
    for ons in (1, 3, 7):
        value = max_lyapunov_exponent(model, torch.zeros(2, dtype=torch.float64),
                                     T=23, T_trans=0, ons=ons, dt=1)
        assert value == pytest.approx(np.log(1.2), abs=1e-12)
    scaled = max_lyapunov_exponent(model, torch.zeros(2, dtype=torch.float64),
                                  T=23, T_trans=0, ons=3, dt=.1)
    assert scaled == pytest.approx(np.log(1.2)/.1, abs=1e-12)
    assert model.training


@pytest.mark.parametrize('kind', ['ctrnn','ei_rnn','constrained_rnn','gain_rnn',
                                  'stp_rnn','lowrank_rnn','latent_circuit'])
@pytest.mark.parametrize('training', [False, True])
@pytest.mark.parametrize('enabled', [False, True])
def test_noise_policy_and_generator(kind, training, enabled):
    model = make(kind, sigma_rec=.3)
    model.train(training)
    x = torch.ones(6, 4, 2)
    global_rng = torch.random.get_rng_state().clone()
    def run(seed):
        with model.noise_context(enabled, generator=torch.Generator().manual_seed(seed)):
            return model(x).states
    a, b, c = run(71), run(71), run(72)
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert torch.equal(a, c) != enabled
    assert model.training == training
    assert torch.equal(global_rng, torch.random.get_rng_state())


@pytest.mark.parametrize('scaling,variance', [(None,.04*.09),
    ('sqrt_2alpha', .04*2*.2*.09), ('sqrt_2_over_alpha',2*.2*.09)])
def test_noise_increment_variance(scaling, variance):
    model = make('ctrnn', alpha=.2, sigma_rec=.3, noise_scaling=scaling,
                 nonlinearity_mode='rate')
    z = torch.zeros(40000, 4)
    x = torch.zeros(40000, 2)
    with model.noise_context(False):
        base = model.recurrence(x, z)
    with model.evaluation_context(noise=True, generator=torch.Generator().manual_seed(5)):
        noise = model.recurrence(x, z)-base
    assert float(noise.var()) == pytest.approx(variance, rel=.02)


def test_eval_helpers_restore_mode_and_use_explicit_rng():
    from neuralrnn.analysis.psychometric import compute_psychometric
    from neuralrnn.data import ReconstructionDataset
    from types import SimpleNamespace
    model = make('ctrnn', sigma_rec=.5)
    x = torch.ones(20, 5, 2)
    dataset = SimpleNamespace(inputs=x)
    def collect():
        return ReconstructionDataset.from_rnn_and_task(model, dataset, recurrent_noise=True,
                    noise_generator=torch.Generator().manual_seed(9))
    a, b = collect(), collect()
    torch.testing.assert_close(a.activity, b.activity)
    assert model.training
    def curve():
        return compute_psychometric(model, x, [{}]*20, recurrent_noise=True,
                    noise_generator=torch.Generator().manual_seed(9))['choices']
    np.testing.assert_array_equal(curve(), curve())
    assert model.training


def test_teacher_forcing_explicit_mapping_and_validation():
    model = make('gated_rnn', rnn_type='LSTM')
    batch = {'activity':torch.randn(2, 5, 2), 'inputs':torch.randn(2, 5, 2)}
    with pytest.raises(ValueError, match='identity observation'):
        TeacherForcingObjective().compute_loss(model, batch)
    def lift(model, obs, pred):
        return torch.cat([obs, pred[..., 2:]], -1)
    loss, _ = TeacherForcingObjective(observation_to_state=lift).compute_loss(model, batch)
    loss.backward()
    assert torch.isfinite(loss)
    for kwargs in ({'alpha':-1}, {'alpha':1.1}, {'forcing_interval':1.5}, {'tf_noise':-1}):
        with pytest.raises(ValueError):
            TeacherForcingObjective(**kwargs)


def test_actor_critic_lstm_uses_hidden_features_and_complete_memory():
    model = AutoModel.from_config(AutoConfig.for_model('actor_critic',
        core_config={'model_type':'gated_rnn','input_dim':2,'latent_dim':4,
                     'output_dim':2,'rnn_type':'LSTM'}, action_dim=2))
    z = model.init_state(3)
    assert model.state_dim == 8
    result = model.step(torch.ones(3, 2), z)
    assert result[-1].shape == (3, 8)
    assert result[3].shape == (3,)


@pytest.mark.parametrize('kind,opts', [('gated_rnn', {'rnn_type':'LSTM'}), ('stp_rnn', {})])
@pytest.mark.parametrize('backend', ['numeric','golub','scipy'])
def test_finders_use_complete_state_at_known_auxiliary_equilibrium(kind, opts, backend):
    model = make(kind, **opts).double()
    with torch.no_grad():
        if kind == 'gated_rnn':
            for p in model.rnn.parameters():
                p.zero_()
        else:
            model.h0.zero_()
            model.h2h.weight.zero_()
            model.h2h.bias.zero_()
            model.input2h.weight.zero_()
            model.input2h.bias.zero_()
    z = model.init_state(3).detach().clone()
    factories = {'numeric':lambda:NumericFixedPointFinder(n_iters=0,speed_tol=1e-10),
        'golub':lambda:GolubFixedPointFinder(n_iters=0,n_candidates=3,noise_scale=0,tol_q=1e-20),
        'scipy':lambda:ScipyFixedPointFinder(n_candidates=3)}
    finder = factories[backend]()
    fps = finder.find(model,init_states=z,task_input=torch.zeros(2,dtype=torch.float64))
    assert len(fps) == 1
    assert fps.points[0].z.shape == (model.state_dim,)
    assert fps.points[0].eigenvalues.shape == (model.state_dim,)
    with pytest.raises(ValueError,match='complete state'):
        finder.find(model,init_states=z[:,:4])


@pytest.mark.parametrize('kind,opts,multiplier', [
    ('constrained_rnn',{'noise_alpha_scaling':True}, .2**2*2*.2),
    ('latent_circuit',{}, .2**2*2*.2),
    ('gain_rnn',{'noise_alpha_scaling':True,'noise_position':'pre'},.2**2*2*.2),
    ('gain_rnn',{'noise_alpha_scaling':True,'noise_position':'post'},2*.2),
    ('stp_rnn',{'noise_alpha_scaling':True,'noise_position':'post'},2*.2),
    ('lowrank_rnn',{},1.),
])
def test_family_noise_variances(kind,opts,multiplier):
    model = make(kind,alpha=.2,sigma_rec=.3,nonlinearity_mode='rate',**opts)
    z = model.init_state(30000)
    x = torch.zeros(30000,2)
    with model.noise_context(False):
        base = model.recurrence(x,z)
    with model.evaluation_context(noise=True,generator=torch.Generator().manual_seed(12)):
        delta = (model.recurrence(x,z)-base)[...,:4]
    assert float(delta.var()) == pytest.approx(.3**2*multiplier,rel=.02)


@pytest.mark.parametrize('alpha,interval',[(0.,None),(.3,None),(1.,3),(.4,2)])
def test_gtf_stf_loss_matches_independent_step_schedule(alpha,interval):
    model = Affine()
    model.identity_observation = True
    x = torch.arange(16,dtype=torch.float64).reshape(1,8,2)/10
    z = x[:,0]
    predictions=[]
    for t in range(7):
        if t and (interval is None or t % interval == 0):
            z = alpha*x[:,t] + (1-alpha)*z
        z = z*torch.tensor([.5,.8],dtype=z.dtype)+torch.tensor([.1,-.2],dtype=z.dtype)
        predictions.append(z)
    expected=(torch.stack(predictions,1)-x[:,1:]).square().mean()
    loss,_=TeacherForcingObjective(alpha=alpha,forcing_interval=interval).compute_loss(model,{'activity':x})
    torch.testing.assert_close(loss,expected)


def test_mixed_module_modes_restore_after_failure():
    model=make('gated_rnn',rnn_type='LSTM')
    model.train()
    model.readout_layer.eval()
    flags=[p.requires_grad for p in model.parameters()]
    modes=[m.training for m in model.modules()]
    with model.noise_context(True):
        with pytest.raises(RuntimeError):
            with model.evaluation_context(noise=False,strict_state=True):
                next(model.parameters()).requires_grad_(False)
                raise RuntimeError('failure')
        assert model.recurrent_noise_enabled
    assert modes == [m.training for m in model.modules()]
    assert flags == [p.requires_grad for p in model.parameters()]


def test_lyapunov_evaluates_jacobian_before_transition():
    class Nonlinear(Affine):
        def recurrence(self,x_t,z_prev,*,inputs=None):
            return .8*torch.tanh(z_prev)
    model=Nonlinear()
    z=torch.tensor([.7,.4],dtype=torch.float64)
    logs=torch.zeros_like(z)
    for _ in range(5):
        logs += torch.log(.8*(1-torch.tanh(z)**2))
        z=.8*torch.tanh(z)
    value=max_lyapunov_exponent(model,torch.tensor([.7,.4],dtype=torch.float64),
                                T=5,T_trans=0,ons=2,dt=1)
    assert value == pytest.approx(float(logs.max()/5),abs=1e-12)


def test_complete_state_pca_and_vector_field():
    from neuralrnn.analysis.dimensionality import collect_states, fit_pca
    from neuralrnn.analysis.vector_field import compute_vector_field
    class Data:
        def sample_batch(self):
            return {'inputs':torch.ones(3,6,2)}
    model=make('gated_rnn',rnn_type='LSTM')
    states=collect_states(model,Data(),full_state=True)
    assert states.shape == (18,8)
    pca=fit_pca(states)
    field=compute_vector_field(model,pca.components,pca.mean,task_input=torch.zeros(2),n_grid=3)
    assert field.speed.shape == (3,3)
    assert model.training
    with pytest.raises(ValueError,match='complete state'):
        compute_vector_field(model,pca.components[:,:4],pca.mean[:4],n_grid=3)


def test_lstm_dropout_keeps_cell_memory_shape():
    model=make('gated_rnn',rnn_type='LSTM')
    sc,oc,sd,od=model.forward_with_dropout(torch.ones(3,5,2),dropout_rate=.2)
    assert sc.shape == sd.shape == (3,5,4)
    assert oc.shape == od.shape == (3,5,2)
    od.sum().backward()


def test_metric_input_guards_and_identity_spectrum():
    from neuralrnn.analysis.stsp_metrics import (power_spectrum_error,
        state_space_divergence_binning, state_space_divergence_gmm)
    x=np.random.default_rng(2).normal(size=(100,2))
    assert power_spectrum_error(x,x,smoothing=0) == pytest.approx(0,abs=1e-7)
    with pytest.raises(ValueError):
        power_spectrum_error(x,x[:-1])
    with pytest.raises(ValueError):
        power_spectrum_error(np.ones_like(x),x)
    with pytest.raises(ValueError):
        state_space_divergence_binning(x,np.ones_like(x))
    assert np.isfinite(state_space_divergence_gmm(x[:50],x,mc_n=50))


def test_perturbation_restores_weights_on_failure(monkeypatch):
    from neuralrnn.analysis.perturbation import analyze_perturbation, PerturbationSpec
    from neuralrnn.analysis import psychometric
    from types import SimpleNamespace
    model=make('ctrnn')
    latent=SimpleNamespace(embedding_matrix=torch.eye(4))
    dataset=SimpleNamespace(conditions=[{}]*3,get_all_trials=lambda:{'inputs':torch.ones(3,5,2)})
    original=model.h2h.weight.detach().clone()
    calls=[]
    def evaluate(*args,**kwargs):
        calls.append(kwargs)
        if len(calls)==2:
            raise RuntimeError('evaluation failed')
        return {}
    monkeypatch.setattr(psychometric,'compute_psychometric',evaluate)
    with pytest.raises(RuntimeError,match='evaluation failed'):
        analyze_perturbation(model,latent,dataset,PerturbationSpec(0,1,.3),
                             recurrent_noise=True,noise_seed=7)
    torch.testing.assert_close(original,model.h2h.weight)
    assert all(call['recurrent_noise'] for call in calls)
    assert all(call['noise_generator'].initial_seed()==7 for call in calls)
