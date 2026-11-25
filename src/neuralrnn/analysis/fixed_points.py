"""Fixed points and periodic orbits of the complete deterministic discrete map.

Accepted points and rejected candidate diagnostics are separate. Residuals are
Euclidean norms in the original state units. See docs/FIXED_POINT_METHODS.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import torch

from ..modeling_utils import NeuralDynamicsModel
from ._context import deterministic_analysis
from ._stability import spectral_stability


@dataclass
class FixedPoint:
    """Accepted root or orbit; speed is its measured maximum step residual."""
    z: np.ndarray
    speed: float
    eigenvalues: np.ndarray | None = None
    is_stable: bool | None = None
    order: int = 1
    cycle: np.ndarray | None = None
    stability: str = 'unknown'
    spectral_radius: float | None = None


@dataclass
class CandidateDiagnostic:
    """One solver outcome. Converged means the residual criterion passed.

    Solver termination success is recorded separately; duplicates can be
    converged but not accepted. Failed candidates never enter ``points``.
    """
    z: np.ndarray
    speed: float
    q: float
    converged: bool
    accepted: bool = False
    status: str = 'unconverged'
    initial_state: np.ndarray | None = None
    iterations: int | None = None
    nfev: int | None = None
    solver_success: bool | None = None
    message: str = ''
    duplicate_of: int | None = None
    order: int = 1


@dataclass
class FixedPointSet:
    """Accepted results, candidate diagnostics and a reproducible protocol."""
    points: list[FixedPoint] = field(default_factory=list)
    diagnostics: list[CandidateDiagnostic] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def coords(self) -> np.ndarray:
        return np.stack([p.z for p in self.points]) if self.points else np.empty((0,))

    def __len__(self):
        return len(self.points)

    def __iter__(self):
        return iter(self.points)


def _positive(name, value, zero=False):
    if not np.isfinite(value) or (value < 0 if zero else value <= 0):
        raise ValueError(f'{name} must be finite and {"nonnegative" if zero else "positive"}')


def _integer(name, value, zero=False):
    _positive(name, value, zero)
    if isinstance(value, bool) or int(value) != value:
        raise ValueError(f'{name} must be an integer')


def _scale(state_scale, width):
    scale = np.ones(width) if state_scale is None else np.asarray(state_scale, dtype=float)
    if scale.shape != (width,) or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError('state_scale must contain one finite positive value per complete state coordinate')
    return scale


def _input(model, task_input):
    if task_input is None:
        return None
    p = next(model.parameters())
    x = torch.as_tensor(task_input, device=p.device, dtype=p.dtype)
    if x.ndim != 1 or x.shape[0] != model.config.input_dim or not torch.isfinite(x).all():
        raise ValueError('task_input must be a finite (input_dim,) constant input')
    return x[None]


def _states(model, init_states):
    p = next(model.parameters())
    z = torch.as_tensor(init_states, device=p.device, dtype=p.dtype).detach()
    model.validate_state(z)
    if z.ndim < 1 or z.numel() == 0 or not torch.isfinite(z).all():
        raise ValueError('init_states must be nonempty and finite')
    return z.reshape(-1, model.state_dim).clone()


def _metadata(finder, model, xin, source, initial=None):
    p = next(model.parameters())
    params = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in vars(finder).items()}
    return dict(backend=type(finder).__name__, parameters=params,
                state_dim=model.state_dim, dtype=str(p.dtype), device=str(p.device),
                task_input=None if xin is None else xin[0].detach().cpu().tolist(),
                candidate_source=source, n_candidates_actual=None if initial is None else len(initial),
                initial_min=None if initial is None else initial.min(0).values.detach().cpu().tolist(),
                initial_max=None if initial is None else initial.max(0).values.detach().cpu().tolist(),
                distance='Euclidean norm of (z1-z2)/state_scale',
                state_scale=_scale(finder.state_scale, model.state_dim).tolist(),
                residual='norm(F(z)-z), unscaled state units',
                recurrent_noise=False, search_is_exhaustive=False)


def _point(z, speed, eig, stability_tol, **kwargs):
    label, stable, _, _, radius = spectral_stability(eig, stability_tol)
    return FixedPoint(z=z, speed=speed, eigenvalues=eig, is_stable=stable,
                      stability=label, spectral_radius=radius, **kwargs)


def _finish(model, z, speeds, initial, finder, xin, metadata, iterations, records=None):
    coords = z.detach().cpu().numpy()
    speeds = np.asarray(speeds, dtype=float)
    scale = _scale(finder.state_scale, model.state_dim)
    threshold = np.sqrt(2 * finder.tol_q) if hasattr(finder, 'tol_q') else finder.speed_tol
    metadata.update(residual_threshold=float(threshold), acceptance='speed < residual_threshold')
    fps = FixedPointSet(metadata=metadata)
    for i, (c, sp) in enumerate(zip(coords, speeds)):
        finite = bool(np.isfinite(c).all() and np.isfinite(sp))
        record = {} if records is None else records[i]
        converged = finite and sp < threshold and not record.get('error', False)
        fps.diagnostics.append(CandidateDiagnostic(
            z=c.copy(), speed=float(sp), q=float(0.5 * sp**2), converged=converged,
            status='converged' if converged else ('nonfinite' if not finite else 'unconverged'),
            initial_state=initial[i].detach().cpu().numpy().copy(),
            iterations=record.get('iterations', iterations), nfev=record.get('nfev'),
            solver_success=record.get('success'), message=record.get('message', '')))
        if record.get('error', False):
            fps.diagnostics[-1].status = 'solver_error'
    for i in np.argsort(speeds):
        d = fps.diagnostics[i]
        if not d.converged:
            continue
        duplicate = next((j for j, p in enumerate(fps.points)
                          if np.linalg.norm((d.z - p.z) / scale) < finder.dedup_tol), None)
        if duplicate is not None:
            d.status, d.duplicate_of = 'duplicate', duplicate
            continue
        state = torch.as_tensor(d.z, dtype=z.dtype, device=z.device)
        J = model.jacobian(state, inputs=xin).detach().cpu().numpy()
        eig = np.linalg.eigvals(J) if np.isfinite(J).all() else None
        fps.points.append(_point(d.z, d.speed, eig, finder.stability_tol))
        d.accepted, d.status = True, 'accepted'
    metadata.update(n_converged=sum(d.converged for d in fps.diagnostics), n_accepted=len(fps))
    return fps


class NumericFixedPointFinder:
    """Adam minimizes mean((F(z)-z)**2) over candidates and coordinates.

    ``speed_tol`` applies to the per-candidate L2 norm, not the mean loss.
    The legacy default 0.1 is retained; scientific runs must calibrate it.
    Supplied complete states are used directly, without resampling.

    ``init_positive`` selects the default candidate distribution: True samples
    uniform [0, init_scale) (non-negative activity), False samples symmetric
    uniform [-init_scale/2, init_scale/2). None (default) picks automatically:
    signed currents (``nonlinearity_mode="rate"``) get the symmetric range,
    everything else the non-negative one.
    """
    def __init__(self, n_candidates=64, n_iters=10000, lr=1e-3,
                 speed_tol=1e-1, dedup_tol=1e-2, init_scale=3.0,
                 init_positive=None, stability_tol=1e-6, state_scale=None, seed=None):
        for name, value in [('n_candidates', n_candidates), ('n_iters', n_iters)]:
            _integer(name, value, zero=name == 'n_iters')
        for name, value in [('lr', lr), ('speed_tol', speed_tol), ('dedup_tol', dedup_tol)]:
            _positive(name, value)
        _positive('init_scale', init_scale, zero=True)
        if init_positive not in (None, True, False):
            raise ValueError('init_positive must be True, False or None (auto)')
        spectral_stability(None, stability_tol)
        self.n_candidates, self.n_iters, self.lr = n_candidates, n_iters, lr
        self.speed_tol, self.dedup_tol = speed_tol, dedup_tol
        self.init_scale, self.init_positive = init_scale, init_positive
        self.stability_tol, self.state_scale, self.seed = stability_tol, state_scale, seed

    @torch.no_grad()
    def _speed(self, model, z, task_input):
        return (model.recurrence(task_input, z) - z).norm(dim=-1)

    @deterministic_analysis
    def find(self, model, *, task_input=None, init_states=None):
        p = next(model.parameters())
        xin = _input(model, task_input)
        _scale(self.state_scale, model.state_dim)
        if init_states is None:
            generator = None if self.seed is None else torch.Generator(device=p.device).manual_seed(self.seed)
            init_positive = self.init_positive
            if init_positive is None:
                # Rate-mode models store signed currents; others store activity.
                init_positive = getattr(model.config, 'nonlinearity_mode', None) != 'rate'
            z = model.init_state(self.n_candidates, p.device).detach().clone().to(p.dtype)
            activity = torch.rand(self.n_candidates, model.config.latent_dim,
                                  dtype=p.dtype, device=p.device, generator=generator) * self.init_scale
            z[:, :model.config.latent_dim] = activity if init_positive else activity - self.init_scale / 2
        else:
            init_positive = None
            z = _states(model, init_states)
        initial = z.clone()
        metadata = _metadata(self, model, xin,
                             'uniform activity; auxiliary init_state' if init_positive else
                             'uniform signed; auxiliary init_state' if init_positive is False
                             else 'all supplied complete states', initial)
        for p in model.parameters():
            p.requires_grad_(False)
        z = z.detach().requires_grad_(True)
        opt = torch.optim.Adam([z], lr=self.lr)
        steps = 0
        for _ in range(self.n_iters):
            opt.zero_grad()
            loss = (model.recurrence(xin, z) - z).square().mean()
            if not torch.isfinite(loss):
                metadata['termination'] = 'nonfinite objective'
                break
            loss.backward()
            opt.step()
            steps += 1
        metadata.setdefault('termination', 'iteration budget exhausted')
        return _finish(model, z, self._speed(model, z, xin).cpu().numpy(), initial,
                       self, xin, metadata, steps)


class GolubFixedPointFinder:
    """Golub-inspired gradient search with per-candidate decreasing rates.

    This is not a full port of the reference adaptive optimizer. Minimize
    q=0.5*||F(z)-z||**2, reject uphill updates and multiply their rates by
    decrease_factor. Supplied trajectories are resampled and perturbed in
    every complete-state coordinate. Preserve the model's numeric precision.
    """
    def __init__(self, n_candidates=1000, n_iters=5000, initial_rate=1.0,
                 decrease_factor=.95, tol_q=1e-9, noise_scale=.05,
                 dedup_tol=1e-2, seed=None, stability_tol=1e-6, state_scale=None):
        _integer('n_candidates', n_candidates)
        _integer('n_iters', n_iters, zero=True)
        for name, value in [('initial_rate', initial_rate), ('tol_q', tol_q), ('dedup_tol', dedup_tol)]:
            _positive(name, value)
        _positive('noise_scale', noise_scale, zero=True)
        if not 0 < decrease_factor < 1:
            raise ValueError('decrease_factor must be in (0, 1)')
        spectral_stability(None, stability_tol)
        self.n_candidates, self.n_iters = n_candidates, n_iters
        self.initial_rate, self.decrease_factor = initial_rate, decrease_factor
        self.tol_q, self.noise_scale, self.dedup_tol = tol_q, noise_scale, dedup_tol
        self.seed, self.stability_tol, self.state_scale = seed, stability_tol, state_scale

    def _prepare_init_states(self, init_states, M):
        rng = np.random.RandomState(self.seed)
        if init_states is None:
            return torch.from_numpy(rng.rand(self.n_candidates, M) * self.noise_scale)
        states = torch.as_tensor(init_states).detach().cpu().numpy().reshape(-1, M)
        idx = rng.choice(len(states), self.n_candidates, replace=self.n_candidates > len(states))
        return torch.from_numpy(states[idx] + rng.randn(self.n_candidates, M) * self.noise_scale)

    @deterministic_analysis
    def find(self, model, *, task_input=None, init_states=None):
        p = next(model.parameters())
        xin = _input(model, task_input)
        _scale(self.state_scale, model.state_dim)
        if init_states is not None:
            init_states = _states(model, init_states)
        z = self._prepare_init_states(init_states, model.state_dim).to(device=p.device, dtype=p.dtype)
        if init_states is None:
            width = model.config.latent_dim
            z[:, width:] = model.init_state(self.n_candidates, p.device).detach()[:, width:]
        initial = z.clone()
        metadata = _metadata(self, model, xin, 'uniform activity; auxiliary init_state' if init_states is None
                             else 'resampled supplied states plus Gaussian noise', initial)
        for p in model.parameters():
            p.requires_grad_(False)
        z.requires_grad_(True)
        rates = torch.full((len(z),), self.initial_rate, device=z.device, dtype=z.dtype)
        steps = 0
        for _ in range(self.n_iters):
            q = .5 * (model.recurrence(xin, z) - z).square().sum(-1)
            if not torch.isfinite(q).all():
                metadata['termination'] = 'nonfinite objective'
                break
            if (q < self.tol_q).all():
                metadata['termination'] = 'all candidates passed residual criterion'
                break
            grad = torch.autograd.grad(q.sum(), z)[0]
            with torch.no_grad():
                new = z - rates[:, None] * grad
                q_new = .5 * (model.recurrence(xin, new) - new).square().sum(-1)
                rejected = (q_new > q) | ~torch.isfinite(q_new)
                rates[rejected] *= self.decrease_factor
                new[rejected] = z[rejected]
                z.copy_(new)
            steps += 1
        metadata.setdefault('termination', 'iteration budget exhausted')
        metadata['final_rate_range'] = [float(rates.min()), float(rates.max())]
        with torch.no_grad():
            speeds = (model.recurrence(xin, z) - z).norm(dim=-1).cpu().numpy()
        return _finish(model, z, speeds, initial, self, xin, metadata, steps)


def _construct_relu_matrix(number_quadrant: int, dim: int) -> np.ndarray:
    bits = format(number_quadrant, f"0{dim}b")[::-1]
    return np.diag(np.array([bool(int(b)) for b in bits]))


def _relu_matrix_list(dim: int, order: int, rng=np.random) -> np.ndarray:
    out = np.empty((dim, dim, order))
    for i in range(order):
        n = int(np.floor(rng.rand() * (2 ** dim)))
        out[:, :, i] = _construct_relu_matrix(n, dim)
    return out


def _get_factors(A, W1, W2, D_list, order):
    factor_z = np.eye(A.shape[0])
    factor_h1 = np.eye(A.shape[0])
    factor_h2 = W1.dot(D_list[:, :, 0])
    for i in range(order - 1):
        factor_z = (A + W1.dot(D_list[:, :, i]).dot(W2)).dot(factor_z)
        factor_h1 = (A + W1.dot(D_list[:, :, i + 1]).dot(W2)).dot(factor_h1) + np.eye(A.shape[0])
        factor_h2 = (A + W1.dot(D_list[:, :, i + 1]).dot(W2)).dot(factor_h2) + W1.dot(D_list[:, :, i + 1])
    factor_z = (A + W1.dot(D_list[:, :, order - 1]).dot(W2)).dot(factor_z)
    return factor_z, factor_h1, factor_h2


def _cycle_point_candidate(A, W1, W2, h1, h2, D_list, order):
    z_f, h1_f, h2_f = _get_factors(A, W1, W2, D_list, order)
    try:
        return np.linalg.solve(np.eye(A.shape[0]) - z_f, h1_f.dot(h1) + h2_f.dot(h2))
    except np.linalg.LinAlgError:
        return None


def _latent_step(z, A, W1, W2, h1, h2):
    return A.dot(z) + W1.dot(np.maximum(W2.dot(z) + h2, 0)) + h1


def _latent_series(steps, A, W1, W2, h1, h2, dz, z0):
    z = z0 if z0 is not None else np.random.randn(dz)
    traj = [z]
    for _ in range(1, steps):
        z = _latent_step(z, A, W1, W2, h1, h2)
        traj.append(z)
    return traj


def _get_eigvals(A, W1, W2, D_list, order):
    # A may be an (M,) diagonal vector (shallowPLRNN/dendPLRNN) or a full (M,M) matrix (effective ALRNN form).
    A_mat = np.diag(A) if A.ndim == 1 else A
    e = np.eye(A.shape[0])
    for i in range(order):
        e = (A_mat + W1.dot(D_list[:, :, i]).dot(W2)).dot(e)
    return np.linalg.eigvals(e)


def _orbits_equal(first, second, tolerance, scale):
    """Compare whole orbits modulo cyclic rotation, never time reversal."""
    a, b = np.asarray(first), np.asarray(second)
    return a.shape == b.shape and any(
        np.max(np.linalg.norm((a - np.roll(b, shift, axis=0)) / scale, axis=1)) < tolerance
        for shift in range(len(a)))


def _minimal_period(orbit, tolerance, scale):
    for period in range(1, len(orbit)):
        if len(orbit) % period == 0 and np.max(np.linalg.norm(
                (orbit - np.roll(orbit, period, axis=0)) / scale, axis=1)) < tolerance:
            return period
    return len(orbit)


def _scy_fi(A, W1, W2, h1, h2, order, found_lower, outer_it=300, inner_it=100,
            dedup_tol=1e-2, *, state_scale=None, rng=None, stats=None):
    """Bounded SCYFI region search, not exhaustive region enumeration.

    Each restart refines at most inner_it region sequences. Repeated lower
    periods are rejected even when the lower-order search missed them.
    """
    rng = np.random if rng is None else rng
    stats = {} if stats is None else stats
    stats.update(attempted=0, singular=0, inconsistent=0, lower_period=0, duplicates=0)
    scale = _scale(state_scale, len(h1))
    cycles, eigvals = [], []
    for _ in range(outer_it):
        D = _relu_matrix_list(len(h2), order, rng)
        for _ in range(inner_it):
            stats['attempted'] += 1
            zc = _cycle_point_candidate(A, W1, W2, h1, h2, D, order)
            if zc is None or not np.isfinite(zc).all():
                stats['singular'] += 1
                break
            orbit = np.asarray(_latent_series(order, A, W1, W2, h1, h2, len(h1), zc))
            actual_D = np.stack([np.diag(W2 @ z + h2 > 0) for z in orbit], axis=-1)
            if not np.array_equal(D, actual_D):
                stats['inconsistent'] += 1
                D = actual_D
                continue
            if _minimal_period(orbit, dedup_tol, scale) < order:
                stats['lower_period'] += 1
            elif any(_orbits_equal(orbit, old, dedup_tol, scale) for old in cycles):
                stats['duplicates'] += 1
            else:
                cycles.append(orbit)
                eigvals.append(_get_eigvals(A, W1, W2, D, order))
            break
    return cycles, eigvals


class AnalyticPLRNNFixedPointFinder:
    """Solve consistent piecewise-affine regions; verify in the actual map.

    ``speed`` is max_j ||F(z_j)-z_(j+1 mod k)||. The returned eigenvalues
    belong to the k-step monodromy, not a continuous-time generator.
    """
    def __init__(self, max_order=1, outer_it=300, inner_it=100, dedup_tol=1e-2,
                 speed_tol=1e-5, stability_tol=1e-6, state_scale=None,
                 kink_tol=1e-8, seed=None):
        for name, value in [('max_order', max_order), ('outer_it', outer_it), ('inner_it', inner_it)]:
            _integer(name, value)
        _positive('dedup_tol', dedup_tol)
        _positive('speed_tol', speed_tol)
        _positive('kink_tol', kink_tol, zero=True)
        spectral_stability(None, stability_tol)
        self.max_order, self.outer_it, self.inner_it = max_order, outer_it, inner_it
        self.dedup_tol, self.speed_tol = dedup_tol, speed_tol
        self.stability_tol, self.state_scale = stability_tol, state_scale
        self.kink_tol, self.seed = kink_tol, seed

    @deterministic_analysis
    def find(self, model, *, task_input=None):
        if not model.supports_analytic_fixed_points:
            raise RuntimeError(f'{type(model).__name__} does not support analytic fixed points')
        # Autonomous models historically ignore task_input.
        effective_input = None if getattr(model.config, 'autonomous', False) else task_input
        xin = _input(model, effective_input)
        params = model.analytic_parameters(task_input=None if xin is None else xin[0])
        required = {'A', 'W1', 'W2', 'h1', 'h2'}
        if not required.issubset(params):
            raise RuntimeError(f'analytic_parameters must contain {required}')
        A, W1, W2, h1, h2 = [params[k].detach().cpu().numpy().astype(np.float64)
                              for k in ('A', 'W1', 'W2', 'h1', 'h2')]
        if A.ndim == 1:
            A = np.diag(A)
        p = next(model.parameters())
        metadata = _metadata(self, model, xin, 'random activation-region sequences')
        metadata.update(solver_dtype='float64', residual_threshold=self.speed_tol,
                        residual='max_j norm(F(z_j)-z_(j+1 mod k)), model dtype', orders=[])
        fps = FixedPointSet(metadata=metadata)
        rng = np.random if self.seed is None else np.random.RandomState(self.seed)
        found_lower = []
        for order in range(1, self.max_order + 1):
            stats = dict(order=order)
            cycles, _ = _scy_fi(A, W1, W2, h1, h2, order, found_lower,
                                self.outer_it, self.inner_it, self.dedup_tol,
                                state_scale=self.state_scale, rng=rng, stats=stats)
            found_lower.append(cycles)
            metadata['orders'].append(stats)
            for orbit in cycles:
                z = torch.as_tensor(orbit, dtype=p.dtype, device=p.device)
                with torch.no_grad():
                    residuals = (model.recurrence(xin, z) - z.roll(-1, 0)).norm(dim=-1)
                speed = float(residuals.max())
                converged = np.isfinite(speed) and speed < self.speed_tol
                diagnostic = CandidateDiagnostic(z=z[0].cpu().numpy(), speed=speed,
                    q=.5 * speed**2, converged=bool(converged), order=order,
                    status='unconverged', message='region-consistent candidate; residual checked in model dtype')
                fps.diagnostics.append(diagnostic)
                if not converged:
                    continue
                product = np.eye(model.state_dim)
                for state in z:
                    product = model.jacobian(state, inputs=xin).detach().cpu().numpy() @ product
                eig = np.linalg.eigvals(product)
                point = _point(z[0].cpu().numpy(), speed, eig, self.stability_tol,
                               order=order, cycle=z.cpu().numpy() if order > 1 else None)
                if np.any(np.abs(orbit @ W2.T + h2) <= self.kink_tol):
                    point.stability, point.is_stable = 'nondifferentiable', None
                fps.points.append(point)
                diagnostic.accepted, diagnostic.status = True, 'accepted'
        metadata.update(n_converged=sum(d.converged for d in fps.diagnostics), n_accepted=len(fps))
        return fps


class ScipyFixedPointFinder:
    """MINPACK hybrid root search or Powell minimization of q=0.5*||F-z||².

    ``speed_tol`` is independent of solver stopping tolerances. If omitted,
    sqrt(fun_tol) preserves the historical residual threshold. ``patience``
    and ``eig_cutoff`` are retained as inactive legacy arguments.
    """
    def __init__(self, n_candidates=100, mode='exact', fun_tol=1e-12,
                 patience=100, stop_length=100, sigma_init_guess=.01,
                 eig_cutoff=1e-10, diff_cutoff=1e-7, seed=42, *,
                 speed_tol=None, xtol=None, maxfev=None, stability_tol=1e-6, state_scale=None):
        if mode not in ('exact', 'approx'):
            raise ValueError('mode must be exact or approx')
        _integer('n_candidates', n_candidates)
        _integer('stop_length', stop_length)
        for name, value in [('fun_tol', fun_tol), ('diff_cutoff', diff_cutoff)]:
            _positive(name, value)
        _positive('sigma_init_guess', sigma_init_guess, zero=True)
        speed_tol = np.sqrt(fun_tol) if speed_tol is None else speed_tol
        xtol = fun_tol if xtol is None else xtol
        maxfev = stop_length if maxfev is None else maxfev
        _positive('speed_tol', speed_tol)
        _positive('xtol', xtol)
        _integer('maxfev', maxfev)
        spectral_stability(None, stability_tol)
        self.n_candidates, self.mode, self.fun_tol = n_candidates, mode, fun_tol
        self.patience, self.stop_length, self.sigma_init_guess = patience, stop_length, sigma_init_guess
        self.eig_cutoff, self.diff_cutoff, self.seed = eig_cutoff, diff_cutoff, seed
        self.dedup_tol, self.speed_tol, self.xtol, self.maxfev = diff_cutoff, speed_tol, xtol, maxfev
        self.stability_tol, self.state_scale = stability_tol, state_scale

    def _rhs_numpy(self, z_np, model, task_input_np):
        p = next(model.parameters())
        z = torch.as_tensor(z_np, dtype=p.dtype, device=p.device)
        xin = _input(model, task_input_np)
        with torch.no_grad():
            return (model.recurrence(xin, z[None])[0] - z).cpu().numpy().astype(float)

    def _jacobian_numpy(self, z_np, model, task_input_np):
        p = next(model.parameters())
        z = torch.as_tensor(z_np, dtype=p.dtype, device=p.device)
        J = model.jacobian(z, inputs=_input(model, task_input_np)).detach().cpu().numpy()
        return J - np.eye(model.state_dim)

    @deterministic_analysis
    def find(self, model, *, task_input=None, init_states=None):
        from scipy.optimize import fsolve, minimize
        p = next(model.parameters())
        xin = _input(model, task_input)
        ti = None if xin is None else xin[0].detach().cpu().numpy()
        _scale(self.state_scale, model.state_dim)
        if init_states is None:
            initial = model.init_state(self.n_candidates, p.device).detach().clone().to(p.dtype)
            activity = np.random.RandomState(self.seed).randn(self.n_candidates, model.config.latent_dim)
            initial[:, :model.config.latent_dim] = torch.as_tensor(
                activity * self.sigma_init_guess, device=p.device, dtype=p.dtype)
        else:
            initial = _states(model, init_states)
        metadata = _metadata(self, model, xin, 'Gaussian activity; auxiliary init_state' if init_states is None
                             else 'all supplied complete states', initial)
        metadata.update(solver_dtype='float64', inactive_legacy_parameters=['patience', 'eig_cutoff'])
        solutions, speeds, records = [], [], []
        for z0 in initial.cpu().numpy():
            record = {}
            try:
                if self.mode == 'exact':
                    sol, info, ier, msg = fsolve(self._rhs_numpy, z0, args=(model, ti),
                        fprime=self._jacobian_numpy, full_output=True, xtol=self.xtol, maxfev=self.maxfev)
                    record.update(success=ier == 1, nfev=info['nfev'], message=str(msg))
                else:
                    def objective(z):
                        residual = self._rhs_numpy(z, model, ti)
                        return .5 * residual @ residual
                    result = minimize(objective, z0, method='Powell', options=dict(
                        maxiter=self.stop_length, maxfev=self.maxfev, ftol=self.fun_tol, xtol=self.xtol))
                    sol = result.x
                    record.update(success=bool(result.success), nfev=result.nfev,
                                  iterations=result.nit, message=str(result.message))
                # Store and verify the state actually representable by the model.
                sol = torch.as_tensor(sol, device=p.device, dtype=p.dtype).cpu().numpy()
                speed = np.linalg.norm(self._rhs_numpy(sol, model, ti))
            except Exception as exc:
                sol, speed = z0.copy(), float('inf')
                record.update(error=True, success=False, message=f'{type(exc).__name__}: {exc}')
            solutions.append(sol)
            speeds.append(speed)
            records.append(record)
        z = torch.as_tensor(np.stack(solutions), device=p.device, dtype=p.dtype)
        return _finish(model, z, speeds, initial, self, xin, metadata, None, records)


def find_fixed_points(model: NeuralDynamicsModel, *, backend='auto', task_input=None,
                      init_states=None, max_order=1, **kwargs) -> FixedPointSet:
    """Dispatch to analytic, numeric, golub or scipy; reject unknown options.

    Auto selects analytic only when the model declares that capability.
    init_states are complete states or trajectories. Only analytic supports
    max_order > 1, and it uses activation regions rather than init_states.
    """
    if backend not in ('auto', 'analytic', 'numeric', 'golub', 'scipy'):
        raise ValueError(f'Unknown fixed-point backend: {backend}')
    _integer('max_order', max_order)
    if backend == 'auto':
        backend = 'analytic' if model.supports_analytic_fixed_points else 'numeric'
    if backend == 'analytic':
        if init_states is not None:
            raise ValueError('analytic uses activation-region seeds, not init_states')
        return AnalyticPLRNNFixedPointFinder(max_order=max_order, **kwargs).find(model, task_input=task_input)
    if max_order != 1:
        raise ValueError('max_order > 1 is only supported by analytic')
    cls = {'numeric': NumericFixedPointFinder, 'golub': GolubFixedPointFinder, 'scipy': ScipyFixedPointFinder}[backend]
    return cls(**kwargs).find(model, task_input=task_input, init_states=init_states)
