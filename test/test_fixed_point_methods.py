"""Known roots, periodic orbits and failure diagnostics for revision W1.2."""
import numpy as np
import pytest
import torch
from scipy.optimize import brentq

from neuralrnn import AutoConfig, AutoModel, NeuralDynamicsModel, NeuralRNNConfig
from neuralrnn.analysis import find_fixed_points, linearize, classify_fixed_point
from neuralrnn.analysis.fixed_points import (
    NumericFixedPointFinder, GolubFixedPointFinder, ScipyFixedPointFinder,
    AnalyticPLRNNFixedPointFinder, _orbits_equal,
)


class AffineMap(NeuralDynamicsModel):
    def __init__(self, A=((.5, .1), (0., .8)), b=(.1, -.2)):
        super().__init__(NeuralRNNConfig(input_dim=1, latent_dim=2, output_dim=2))
        self.A = torch.nn.Parameter(torch.tensor(A, dtype=torch.float64))
        self.b = torch.nn.Parameter(torch.tensor(b, dtype=torch.float64), requires_grad=False)

    def recurrence(self, x_t, z_prev, *, inputs=None):
        return z_prev @ self.A.T + self.b + (0 if x_t is None else .2 * x_t)

    def readout(self, z):
        return z


def finder(backend, **kwargs):
    settings = dict(
        numeric=dict(n_iters=1800, lr=.04, speed_tol=1e-7),
        golub=dict(n_iters=1800, tol_q=5e-15, noise_scale=0, n_candidates=3, seed=3),
        scipy=dict(speed_tol=1e-8, xtol=1e-10, maxfev=1000),
    )[backend]
    settings.update(kwargs)
    return settings


@pytest.mark.parametrize('backend', ['numeric', 'golub', 'scipy'])
def test_affine_nonzero_input_and_candidate_accounting(backend):
    model = AffineMap()
    initial = torch.tensor([[-1., 1.], [0., 0.], [1., -1.]], dtype=torch.float64)
    fps = find_fixed_points(model, backend=backend, init_states=initial,
                            task_input=torch.tensor([.7]), **finder(backend))
    expected = np.linalg.solve(np.eye(2) - model.A.detach().numpy(), model.b.numpy() + .14)
    assert len(fps) == 1
    np.testing.assert_allclose(fps.points[0].z, expected, atol=2e-6)
    assert len(fps.diagnostics) == 3
    assert all(d.converged for d in fps.diagnostics)
    assert sum(d.accepted for d in fps.diagnostics) == 1
    assert sum(d.status == 'duplicate' for d in fps.diagnostics) == 2
    assert fps.metadata['dtype'] == 'torch.float64'
    assert fps.metadata['n_candidates_actual'] == 3
    assert fps.points[0].stability == 'stable'


@pytest.mark.parametrize('backend', ['numeric', 'golub', 'scipy'])
def test_no_root_has_diagnostics_without_false_solution(backend):
    model = AffineMap(A=np.eye(2), b=(1., 1.))
    fps = find_fixed_points(model, backend=backend, init_states=torch.zeros(3, 2),
                            **finder(backend))
    assert len(fps) == 0
    assert len(fps.diagnostics) == 3
    assert not any(d.converged for d in fps.diagnostics)
    np.testing.assert_allclose([d.speed for d in fps.diagnostics], np.sqrt(2))


@pytest.mark.parametrize('radius, label, stable', [
    (.8, 'stable', True), (-1.2, 'unstable', False),
    (1., 'near_neutral', None), (1.0000005, 'near_neutral', None),
    (.9999995, 'near_neutral', None),
])
def test_discrete_spectrum_and_neutral_band(radius, label, stable):
    model = AffineMap(A=np.diag([radius, .5]), b=(0., 0.))
    fps = find_fixed_points(model, backend='numeric', n_iters=0,
                            init_states=torch.zeros(1, 2), speed_tol=1e-12)
    lin = linearize(model, torch.zeros(2))
    assert fps.points[0].stability == lin.stability == label
    assert fps.points[0].is_stable is stable and lin.is_stable is stable
    assert classify_fixed_point(lin) == ('saddle(1)' if label == 'unstable' else label)


def tent_map():
    # F(z)=1-2*relu(z). Independent truths: unstable root 1/3;
    # stable period-two orbit {-1,1}, derivative product 0*(-2)=0.
    model = AutoModel.from_config(AutoConfig.for_model('shallow_plrnn', input_dim=0,
        latent_dim=1, hidden_dim=1, output_dim=1, autonomous=True)).double()
    with torch.no_grad():
        model.A.zero_()
        model.W1.fill_(-2)
        model.W2.fill_(1)
        model.h1.fill_(1)
        model.h2.zero_()
    return model


def test_piecewise_roots_cycles_and_primitive_period():
    fps = find_fixed_points(tent_map(), backend='analytic', max_order=4,
        outer_it=120, inner_it=10, speed_tol=1e-12, dedup_tol=1e-8, seed=4)
    assert sorted(p.order for p in fps) == [1, 2]
    root = next(p for p in fps if p.order == 1)
    cycle = next(p for p in fps if p.order == 2)
    np.testing.assert_allclose(root.z, [1 / 3], atol=1e-12)
    np.testing.assert_allclose(np.sort(cycle.cycle[:, 0]), [-1, 1], atol=1e-12)
    assert root.stability == 'unstable' and cycle.stability == 'stable'
    np.testing.assert_allclose(cycle.eigenvalues, [0.], atol=1e-12)
    assert all(p.speed < 1e-12 for p in fps)
    assert fps.metadata['orders'][3]['lower_period'] > 0
    assert all(s['attempted'] <= 1200 for s in fps.metadata['orders'])


def test_cycle_equivalence_uses_all_coordinates_and_order():
    orbit = np.array([[0., 0.], [1., 2.], [3., 4.]])
    assert _orbits_equal(orbit, np.roll(orbit, 2, axis=0), 1e-8, np.ones(2))
    assert not _orbits_equal(orbit, orbit[::-1], 1e-8, np.ones(2))
    different = orbit.copy()
    different[1, 1] += .1
    assert not _orbits_equal(orbit, different, 1e-8, np.ones(2))


def test_relu_kink_is_separate_from_smooth_linear_stability():
    model = tent_map()
    with torch.no_grad():
        model.h1.zero_()
        model.W1.fill_(.5)
    fps = find_fixed_points(model, backend='analytic', seed=2,
                            outer_it=20, inner_it=5, speed_tol=1e-12)
    assert len(fps) == 1
    assert fps.points[0].stability == 'nondifferentiable'
    assert fps.points[0].is_stable is None


def test_analytic_no_root_and_real_map_residual(monkeypatch):
    model = tent_map()
    with torch.no_grad():
        model.A.fill_(1)
        model.W1.zero_()
    fps = find_fixed_points(model, backend='analytic', outer_it=10, inner_it=4, seed=1)
    assert len(fps) == 0
    assert fps.metadata['orders'][0]['singular'] == 10
    model = tent_map()
    original = model.recurrence
    monkeypatch.setattr(model, 'recurrence', lambda x, z: original(x, z) + .01)
    fps = find_fixed_points(model, backend='analytic', outer_it=20, inner_it=5,
                            seed=3, speed_tol=1e-8)
    assert len(fps) == 0 and fps.diagnostics
    assert all(not d.converged for d in fps.diagnostics)
    assert fps.diagnostics[0].speed == pytest.approx(.01)


def test_small_ctrnn_against_independent_scalar_root_and_jacobian():
    model = AutoModel.from_config(AutoConfig.for_model('ctrnn', input_dim=1,
        latent_dim=1, output_dim=1, activation='tanh', alpha=.2,
        nonlinearity_mode='pre_activation')).double()
    with torch.no_grad():
        model.h2h.weight.fill_(.4)
        model.h2h.bias.fill_(.1)
        model.input2h.weight.fill_(.3)
        model.input2h.bias.fill_(-.05)
    root = brentq(lambda z: np.tanh(.4 * z + .3 * .7 + .05) - z, -2, 2, xtol=1e-14)
    expected_j = .8 + .2 * .4 * (1 - np.tanh(.4 * root + .26)**2)
    fps = find_fixed_points(model, backend='scipy', task_input=torch.tensor([.7], dtype=torch.float64),
                            init_states=torch.tensor([[-1.], [0.], [1.]]), speed_tol=1e-10)
    assert len(fps) == 1
    np.testing.assert_allclose(fps.points[0].z, [root], atol=1e-10)
    np.testing.assert_allclose(fps.points[0].eigenvalues, [expected_j], atol=1e-10)


def test_dedup_scales_all_complete_coordinates_and_threshold_units():
    model = AffineMap(A=np.eye(2), b=(0., 0.))
    initial = torch.tensor([[0., 0.], [0., 2.]])
    raw = find_fixed_points(model, backend='numeric', n_iters=0, init_states=initial, dedup_tol=.01)
    scaled = find_fixed_points(model, backend='numeric', n_iters=0, init_states=initial,
                               dedup_tol=.01, state_scale=[1., 1000.])
    assert len(raw) == 2 and len(scaled) == 1
    translation = AffineMap(A=np.eye(2), b=(.03, .04))
    fps = find_fixed_points(translation, backend='numeric', n_iters=0,
                            init_states=initial, speed_tol=.01)
    assert len(fps) == 0
    assert fps.diagnostics[0].speed == pytest.approx(.05)
    assert fps.diagnostics[0].q == pytest.approx(.00125)


def test_scipy_exception_is_visible_and_flags_restore(monkeypatch):
    model = AffineMap()
    flags = [p.requires_grad for p in model.parameters()]
    def broken(*args):
        raise RuntimeError('known failure')
    monkeypatch.setattr(model, 'recurrence', broken)
    fps = ScipyFixedPointFinder(n_candidates=2).find(model)
    assert len(fps) == 0 and len(fps.diagnostics) == 2
    assert all(d.status == 'solver_error' and 'known failure' in d.message for d in fps.diagnostics)
    assert model.training and [p.requires_grad for p in model.parameters()] == flags


@pytest.mark.parametrize('backend', ['numeric', 'golub', 'scipy'])
def test_nan_map_never_yields_a_fixed_point(backend, monkeypatch):
    model = AffineMap()
    monkeypatch.setattr(model, 'recurrence', lambda x, z: z * float('nan'))
    fps = find_fixed_points(model, backend=backend, init_states=torch.zeros(3, 2), **finder(backend))
    assert len(fps) == 0
    assert all(not d.converged for d in fps.diagnostics)


@pytest.mark.parametrize('factory,kwargs', [
    (NumericFixedPointFinder, dict(speed_tol=-1)), (GolubFixedPointFinder, dict(tol_q=0)),
    (ScipyFixedPointFinder, dict(mode='typo')), (AnalyticPLRNNFixedPointFinder, dict(max_order=0)),
    (NumericFixedPointFinder, dict(stability_tol=-1)), (NumericFixedPointFinder, dict(n_iters=1.5)),
])
def test_invalid_protocol_is_rejected(factory, kwargs):
    with pytest.raises(ValueError):
        factory(**kwargs)


def test_dispatch_guards_and_local_seed():
    model = AffineMap()
    with pytest.raises(ValueError, match='Unknown'):
        find_fixed_points(model, backend='typo')
    with pytest.raises(ValueError, match='max_order'):
        find_fixed_points(model, backend='numeric', max_order=2)
    with pytest.raises(ValueError, match='nonempty'):
        find_fixed_points(model, backend='numeric', init_states=torch.empty(0, 2))
    with pytest.raises(ValueError, match='state_scale'):
        find_fixed_points(model, backend='numeric', state_scale=[1.])
    torch_state = torch.random.get_rng_state().clone()
    a = NumericFixedPointFinder(n_iters=0, n_candidates=3, seed=9).find(model)
    b = NumericFixedPointFinder(n_iters=0, n_candidates=3, seed=9).find(model)
    np.testing.assert_array_equal(a.diagnostics[0].z, b.diagnostics[0].z)
    assert torch.equal(torch_state, torch.random.get_rng_state())
