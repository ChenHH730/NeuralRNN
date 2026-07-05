"""Fixed point / k-cycle analysis (dual backend).

Golden rule (PORTING_GUIDE contract D): the analysis layer only works through the model's public contract
(recurrence / jacobian / supports_analytic_fixed_points / analytic_parameters),
and **never** imports any concrete model classes. Thus any model satisfying the contract can be analyzed by the same analyzer.

Two backends:
  1) Numeric backend NumericFixedPointFinder — ported from RNN_DynamicalSystemAnalysis.ipynb:
     initializes a batch of candidate states in parallel, minimizes ‖F(z) − z‖² (velocity-field norm) with Adam,
     filters by a speed threshold, and removes duplicates.
     Applicable to any model (including the discrete step of continuous CTRNNs).
  2) Analytic backend AnalyticPLRNNFixedPointFinder — ported from CNS2023_tutorial.ipynb's
     scy_fi / main: exploits the piecewise-linear structure of PLRNNs to solve fixed points and k-cycles exactly,
     together with their eigenvalues.
     Only available when model.supports_analytic_fixed_points and analytic_parameters() are implemented.

The unified entry point find_fixed_points(model, ...) automatically picks the best available backend (analytic preferred, numeric fallback).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from ..modeling_utils import NeuralDynamicsModel


@dataclass
class FixedPoint:
    z: np.ndarray                       # Fixed-point coordinate (M,)
    speed: float                        # ‖F(z) − z‖ (numeric backend); 0 for analytic backend
    eigenvalues: np.ndarray | None = None  # Jacobian eigenvalues
    is_stable: bool | None = None       # Discrete system: max|eig| < 1
    order: int = 1                      # 1 = fixed point; k>1 = k-cycle (analytic backend)
    cycle: np.ndarray | None = None     # All points of the k-cycle (order, M)


@dataclass
class FixedPointSet:
    points: list[FixedPoint] = field(default_factory=list)

    def coords(self) -> np.ndarray:
        return np.stack([p.z for p in self.points]) if self.points else np.empty((0,))

    def __len__(self): return len(self.points)
    def __iter__(self): return iter(self.points)


# =========================================================================
# Numeric backend (gradient-based, model-agnostic)
# =========================================================================
class NumericFixedPointFinder:
    """Search for fixed points by minimizing the velocity-field norm ‖F(z) − z‖² (nn-brain style)."""

    def __init__(self, n_candidates: int = 64, n_iters: int = 10000, lr: float = 1e-3,
                 speed_tol: float = 1e-1, dedup_tol: float = 1e-2,
                 init_scale: float = 3.0, init_positive: bool = True):
        self.n_candidates = n_candidates
        self.n_iters = n_iters
        self.lr = lr
        self.speed_tol = speed_tol
        self.dedup_tol = dedup_tol
        self.init_scale = init_scale
        self.init_positive = init_positive

    @torch.no_grad()
    def _speed(self, model, z, task_input):
        f = model.recurrence(task_input, z)
        return (f - z).norm(dim=-1)

    def find(self, model: NeuralDynamicsModel, *, task_input: torch.Tensor | None = None,
             init_states: torch.Tensor | None = None) -> FixedPointSet:
        """task_input: (input_dim,) constant input condition during search (e.g., mean input at 0-coherence in a decision task)."""
        was_training = model.training
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        M = model.config.latent_dim
        device = next(model.parameters()).device

        if init_states is None:
            base = torch.rand(self.n_candidates, M, device=device) * self.init_scale
            z = base if self.init_positive else (base - self.init_scale / 2)
        else:
            z = init_states.clone().to(device)
        z = z.detach().requires_grad_(True)

        if task_input is not None:
            xin = task_input.to(device).unsqueeze(0).expand(z.shape[0], -1)
        else:
            xin = None

        opt = torch.optim.Adam([z], lr=self.lr)
        for _ in range(self.n_iters):
            opt.zero_grad()
            f = model.recurrence(xin, z)
            loss = ((f - z) ** 2).mean()
            loss.backward()
            opt.step()

        # Speed filtering (sort by speed and keep the best candidates first)
        speeds = self._speed(model, z.detach(), xin)
        order = speeds.argsort()
        z_sorted = z.detach()[order]
        speeds_sorted = speeds[order]
        keep = speeds_sorted < self.speed_tol
        cand = z_sorted[keep].cpu().numpy()
        cand_speed = speeds_sorted[keep].cpu().numpy()

        # Fallback: if nothing passes the filter, keep the best candidate
        if len(cand) == 0:
            best_idx = speeds.argmin()
            cand = z.detach()[best_idx:best_idx+1].cpu().numpy()
            cand_speed = speeds[best_idx:best_idx+1].cpu().numpy()

        # Deduplicate and compute eigenvalues/stability (using the model's jacobian contract)
        fps = FixedPointSet()
        for c, sp in zip(cand, cand_speed):
            if any(np.linalg.norm(c - p.z) < self.dedup_tol for p in fps.points):
                continue
            J = model.jacobian(torch.as_tensor(c, dtype=torch.float32, device=device),
                               inputs=(xin if xin is not None else None)).cpu().numpy()
            eig = np.linalg.eigvals(J)
            fps.points.append(FixedPoint(
                z=c, speed=float(sp), eigenvalues=eig,
                is_stable=bool(np.max(np.abs(eig)) < 1.0)))

        if was_training:
            model.train()
        return fps


# =========================================================================
# Analytic backend (PLRNN: scy_fi / main, exact FP + k-cycle solver)
# =========================================================================
def _construct_relu_matrix(number_quadrant: int, dim: int) -> np.ndarray:
    bits = format(number_quadrant, f"0{dim}b")[::-1]
    return np.diag(np.array([bool(int(b)) for b in bits]))


def _relu_matrix_list(dim: int, order: int) -> np.ndarray:
    out = np.empty((dim, dim, order))
    for i in range(order):
        n = int(np.floor(np.random.rand() * (2 ** dim)))
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
        inv = np.linalg.inv(np.eye(A.shape[0]) - z_f)
        return inv.dot(h1_f.dot(h1) + h2_f.dot(h2))
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


def _scy_fi(A, W1, W2, h1, h2, order, found_lower, outer_it=300, inner_it=100):
    """Heuristic exact solver for order-k cycles (ported from CNS2023 scy_fi). A is an (M,M) diagonal matrix."""
    hidden_dim = h2.shape[0]
    latent_dim = h1.shape[0]
    cycles, eigvals = [], []
    i = -1
    while i < outer_it:
        i += 1
        D = _relu_matrix_list(hidden_dim, order)
        diff = 1
        c = 0
        while c < inner_it:
            c += 1
            zc = _cycle_point_candidate(A, W1, W2, h1, h2, D, order)
            if zc is None:
                D = _relu_matrix_list(hidden_dim, order)
                continue
            traj = _latent_series(order, A, W1, W2, h1, h2, latent_dim, z0=zc)
            traj_D = np.empty((hidden_dim, hidden_dim, order))
            for j in range(order):
                traj_D[:, :, j] = np.diag((W2.dot(traj[j]) + h2) > 0)
            for j in range(order):
                diff = np.sum(np.abs(traj_D[:, :, j] - D[:, :, j]))
                if diff != 0:
                    break
                if found_lower and np.round(traj[0], 2) in np.round(
                        np.array(found_lower).flatten(), 2):
                    diff = 1
                    break
            if diff == 0 and not np.any(np.isin(np.round(traj[0], 2),
                                                np.round(cycles, 2) if cycles else np.array([]))):
                e = _get_eigvals(A, W1, W2, D, order)
                cycles.append(traj)
                eigvals.append(e)
                i = 0
                c = 0
            D = _relu_matrix_list(hidden_dim, order) if np.array_equal(D, traj_D) else traj_D
    return cycles, eigvals


class AnalyticPLRNNFixedPointFinder:
    """PLRNN analytic backend: enumerate cycles of order 1..max_order exactly, with eigenvalues (CNS2023 main)."""

    def __init__(self, max_order: int = 1, outer_it: int = 300, inner_it: int = 100):
        self.max_order = max_order
        self.outer_it = outer_it
        self.inner_it = inner_it

    def find(self, model: NeuralDynamicsModel, *,
             task_input: torch.Tensor | None = None) -> FixedPointSet:
        if not model.supports_analytic_fixed_points:
            raise RuntimeError(f"{type(model).__name__} does not support analytic fixed points; use a numeric backend.")
        p = model.analytic_parameters(task_input=task_input)  # Allow folding a constant input into the bias
        required = {"A", "W1", "W2", "h1", "h2"}
        if not required.issubset(p):
            raise RuntimeError(
                f"analytic_parameters() must return {required}, but got {set(p.keys())}"
            )
        A = p["A"].cpu().numpy()               # (M,M), diagonal or full matrix
        W1 = p["W1"].cpu().numpy()
        W2 = p["W2"].cpu().numpy()
        h1 = p["h1"].cpu().numpy()
        h2 = p["h2"].cpu().numpy()

        fps = FixedPointSet()
        found_lower = []
        for order in range(1, self.max_order + 1):
            cycles, eigvals = _scy_fi(A, W1, W2, h1, h2, order, found_lower,
                                      self.outer_it, self.inner_it)
            found_lower.append(cycles)
            for traj, e in zip(cycles, eigvals):
                traj = np.asarray(traj)
                fps.points.append(FixedPoint(
                    z=traj[0], speed=0.0, eigenvalues=e,
                    is_stable=bool(np.max(np.abs(e)) < 1.0),
                    order=order, cycle=traj if order > 1 else None))
        return fps


# =========================================================================
# scipy backend (ported from trainRNNbrain DynamicSystemAnalyzer)
# =========================================================================
class ScipyFixedPointFinder:
    """scipy backend: exact root-finding with fsolve + approximate minimization with Powell (ported from trainRNNbrain).

    ⚠️ WARNING: This backend is less stable on CTRNNs (Euler discrete steps) and is not recommended.
    fsolve's exact mode (mode='exact') often converges to points that are not fixed points, because the
    numerical error introduced by Euler discretization can make an exact RHS(z)=0 solution nonexistent.
    The approx mode (Powell) is more robust but still less reliable than NumericFixedPointFinder
    (PyTorch Adam gradient descent).

    Recommendation: prefer NumericFixedPointFinder (backend='numeric'),
    and only try mode='approx' of this backend when the numeric backend fails to find fixed points.

    Differences from NumericFixedPointFinder:
    - fsolve uses the Levenberg-Marquardt algorithm to solve RHS(z)=0 directly
    - Powell minimizes ‖RHS‖² as a fallback (mode='approx')
    - The model's jacobian can be supplied to fsolve as an analytic Jacobian
    """

    def __init__(self, n_candidates: int = 100, mode: str = "exact",
                 fun_tol: float = 1e-12, patience: int = 100,
                 stop_length: int = 100, sigma_init_guess: float = 0.01,
                 eig_cutoff: float = 1e-10, diff_cutoff: float = 1e-7,
                 seed: int = 42):
        """
        Args:
            n_candidates: number of candidate initial points
            mode: "exact" (fsolve) or "approx" (Powell minimization)
            fun_tol: function-value tolerance (points below this are treated as fixed points)
            patience: maximum number of iterations without improvement
            stop_length: hard upper iteration limit for stopping the search
            sigma_init_guess: standard deviation of Gaussian noise for initial guesses
            eig_cutoff: threshold for treating an eigenvalue as zero
            diff_cutoff: L2 distance threshold for treating two points as duplicates
            seed: random seed
        """
        self.n_candidates = n_candidates
        self.mode = mode
        self.fun_tol = fun_tol
        self.patience = patience
        self.stop_length = stop_length
        self.sigma_init_guess = sigma_init_guess
        self.eig_cutoff = eig_cutoff
        self.diff_cutoff = diff_cutoff
        self.seed = seed

    def _rhs_numpy(self, model, z_np, task_input_np):
        """Compute RHS(z) = F(z) - z, returning a numpy array."""
        device = next(model.parameters()).device
        z_t = torch.as_tensor(z_np, dtype=torch.float32, device=device).unsqueeze(0)
        xin = None
        if task_input_np is not None:
            xin = torch.as_tensor(task_input_np, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            f = model.recurrence(xin, z_t).squeeze(0)
        return (f - z_t.squeeze(0)).cpu().numpy()

    def _jacobian_numpy(self, model, z_np, task_input_np):
        """Compute the Jacobian ∂RHS/∂z, returning a numpy array."""
        device = next(model.parameters()).device
        z_t = torch.as_tensor(z_np, dtype=torch.float32, device=device)
        xin = None
        if task_input_np is not None:
            xin = torch.as_tensor(task_input_np, dtype=torch.float32, device=device).unsqueeze(0)
        J = model.jacobian(z_t, inputs=xin).cpu().numpy()
        return J - np.eye(J.shape[0])  # ∂(F(z)-z)/∂z = J_F - I

    def find(self, model: NeuralDynamicsModel, *, task_input: torch.Tensor | None = None,
             init_states: torch.Tensor | None = None) -> FixedPointSet:
        """Search for fixed points. task_input: (input_dim,) constant input condition."""
        from scipy.optimize import fsolve, minimize

        was_training = model.training
        model.eval()

        M = model.config.latent_dim
        device = next(model.parameters()).device
        rng = np.random.RandomState(self.seed)

        # Prepare task_input in numpy format
        ti_np = task_input.cpu().numpy() if task_input is not None else None

        # Generate initial guesses
        if init_states is not None:
            candidates = init_states.cpu().numpy()
        else:
            candidates = rng.randn(self.n_candidates, M) * self.sigma_init_guess

        found_points: list[np.ndarray] = []
        found_speeds: list[float] = []
        found_eigs: list[np.ndarray] = []

        for z0 in candidates:
            if self.mode == "exact":
                try:
                    z_sol, info, ier, msg = fsolve(
                        self._rhs_numpy, z0,
                        args=(model, ti_np),
                        fprime=self._jacobian_numpy,
                        full_output=True, xtol=self.fun_tol
                    )
                except Exception:
                    continue
                speed = np.linalg.norm(self._rhs_numpy(model, z_sol, ti_np))
                if speed > np.sqrt(self.fun_tol):
                    continue
            else:  # approx
                def obj(z):
                    r = self._rhs_numpy(model, z, ti_np)
                    return 0.5 * np.dot(r, r)

                try:
                    res = minimize(obj, z0, method='Powell',
                                   options={'maxiter': self.stop_length,
                                            'ftol': self.fun_tol})
                except Exception:
                    continue
                z_sol = res.x
                speed = np.sqrt(2 * res.fun)
                if speed > np.sqrt(self.fun_tol):
                    continue

            # Remove duplicates
            is_dup = False
            for fp in found_points:
                if np.linalg.norm(z_sol - fp) < self.diff_cutoff:
                    is_dup = True
                    break
            if is_dup:
                continue

            # Compute Jacobian eigenvalues and stability
            try:
                J = self._jacobian_numpy(model, z_sol, ti_np) + np.eye(M)
                eig = np.linalg.eigvals(J)
            except Exception:
                eig = None

            found_points.append(z_sol)
            found_speeds.append(speed)
            found_eigs.append(eig)

        # Build FixedPointSet
        fps = FixedPointSet()
        for z, sp, eig in zip(found_points, found_speeds, found_eigs):
            is_stable = None
            if eig is not None:
                is_stable = bool(np.max(np.abs(eig)) < 1.0)
            fps.points.append(FixedPoint(
                z=z, speed=sp, eigenvalues=eig, is_stable=is_stable))

        if was_training:
            model.train()
        return fps


# =========================================================================
# Unified entry point
# =========================================================================
def find_fixed_points(model: NeuralDynamicsModel, *, backend: str = "auto",
                      task_input: torch.Tensor | None = None,
                      max_order: int = 1, **kwargs) -> FixedPointSet:
    """Automatic backend selection: analytic preferred (if the model supports it), otherwise numeric.

    backend: "auto" / "numeric" / "analytic" / "scipy".
    task_input: constant external input condition. For the analytic backend it is folded into the effective bias
                (e.g., h1 + C*s) before solving; for the numeric/scipy backends it is held fixed while searching
                for fixed points.
    max_order: maximum cycle order scanned by the analytic backend.

    Backend comparison:
    - numeric:  PyTorch Adam gradient descent on ‖F(z)−z‖² (general, GPU-friendly, recommended)
    - analytic: exact PLRNN solver (PLRNN models only; supports constant task_input)
    - scipy:    scipy fsolve / Powell (⚠️ less stable, not recommended;
                fsolve often converges to non-fixed-points under Euler discretization;
                only try mode='approx' when the numeric backend returns no results)
    """
    if backend == "analytic" or (backend == "auto" and model.supports_analytic_fixed_points):
        return AnalyticPLRNNFixedPointFinder(max_order=max_order, **kwargs).find(
            model, task_input=task_input
        )
    if backend == "scipy":
        return ScipyFixedPointFinder(**kwargs).find(model, task_input=task_input)
    return NumericFixedPointFinder(**kwargs).find(model, task_input=task_input)
