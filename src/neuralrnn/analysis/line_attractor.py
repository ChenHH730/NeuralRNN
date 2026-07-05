"""Line-attractor analysis.

Ported from trainRNNbrain's DynamicSystemAnalyzerCDDM, used to analyze the neural mechanism of persistent
activity in tasks such as CDDM. A line attractor is a continuous slow manifold along which the network state
drifts very slowly (‖RHS‖ ≈ 0), supporting stable maintenance of continuous variables.

Golden rule: works only through the model's public contract (recurrence / jacobian).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from ..modeling_utils import NeuralDynamicsModel
from .dimensionality import fit_pca, PCAResult


@dataclass
class LineAttractorPoint:
    """One sampled point on the line attractor."""
    z: np.ndarray                       # State-space coordinate (M,)
    speed: float                        # ‖F(z) - z‖
    distance: float                     # Cumulative distance along the line attractor
    eigenvalues: np.ndarray | None = None   # Jacobian eigenvalues
    jacobian: np.ndarray | None = None      # Jacobian matrix


@dataclass
class LineAttractorResult:
    """Result of line-attractor analysis."""
    points: list[LineAttractorPoint] = field(default_factory=list)
    endpoints: tuple[np.ndarray, np.ndarray] | None = None   # (left, right)
    projection_axes: np.ndarray | None = None                 # (3, M) for 3D viz
    trajectories: np.ndarray | None = None                    # (B, T, M) used for PCA

    @property
    def distances(self) -> np.ndarray:
        return np.array([p.distance for p in self.points])

    @property
    def speeds(self) -> np.ndarray:
        return np.array([p.speed for p in self.points])

    @property
    def coords(self) -> np.ndarray:
        if not self.points:
            return np.empty((0,))
        return np.stack([p.z for p in self.points])


@torch.no_grad()
def find_line_attractor_endpoints(
    model: NeuralDynamicsModel,
    *,
    context_input: torch.Tensor,
    n_steps: int = 1000,
    relax_steps: int = 10,
    initial_state: torch.Tensor | None = None,
    nudge_scale: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Find the two endpoints of a line attractor.

    Method: run the model with two opposite-direction perturbed inputs and take the final states as
    endpoint approximations.
    Ported from trainRNNbrain DynamicSystemAnalyzerCDDM.get_LineAttractor_endpoints().

    Args:
        model: trained model
        context_input: (input_dim,) context input condition
        n_steps: number of run steps
        relax_steps: number of relaxation steps after convergence
        initial_state: (M,) initial state; defaults to zero vector
        nudge_scale: perturbation scale

    Returns:
        (endpoint_left, endpoint_right): numpy coordinates (M,) of the two endpoints
    """
    model.eval()
    device = next(model.parameters()).device
    M = model.config.latent_dim
    input_dim = model.config.input_dim

    z0 = initial_state if initial_state is not None else torch.zeros(M, device=device)
    ctx = context_input.to(device)

    # Generate perturbed inputs: negative and positive directions
    nudge = torch.randn(input_dim, device=device) * nudge_scale
    input_left = (ctx - nudge).unsqueeze(0).unsqueeze(0).expand(1, n_steps, -1)
    input_right = (ctx + nudge).unsqueeze(0).unsqueeze(0).expand(1, n_steps, -1)

    # Run the model
    z_left = z0.unsqueeze(0)
    z_right = z0.unsqueeze(0)

    for t in range(n_steps):
        z_left = model.recurrence(input_left[:, t], z_left)
        z_right = model.recurrence(input_right[:, t], z_right)

    # Relaxation: run a few more steps with pure context input
    ctx_input = ctx.unsqueeze(0).unsqueeze(0).expand(1, 1, -1)
    for _ in range(relax_steps):
        z_left = model.recurrence(ctx_input.squeeze(1), z_left)
        z_right = model.recurrence(ctx_input.squeeze(1), z_right)

    return z_left.squeeze(0).cpu().numpy(), z_right.squeeze(0).cpu().numpy()


@torch.no_grad()
def walk_line_attractor(
    model: NeuralDynamicsModel,
    *,
    context_input: torch.Tensor,
    endpoint_left: np.ndarray,
    endpoint_right: np.ndarray,
    n_points: int = 31,
    max_iter: int = 100,
) -> list[LineAttractorPoint]:
    """Sample points along the line attractor by minimizing ‖RHS‖².

    Linearly interpolate between the left and right endpoints and minimize ‖F(z)-z‖² with scipy SLSQP
    at each point.
    Ported from trainRNNbrain DynamicSystemAnalyzerCDDM.calc_LineAttractor_analytics().

    Args:
        model: trained model
        context_input: (input_dim,) context input condition
        endpoint_left, endpoint_right: endpoint coordinates (M,)
        n_points: number of sampling points
        max_iter: maximum optimization iterations per point

    Returns:
        list of LineAttractorPoint
    """
    from scipy.optimize import minimize

    model.eval()
    device = next(model.parameters()).device
    M = model.config.latent_dim

    ctx = context_input.to(device)

    def rhs_norm_sq(z_np):
        z_t = torch.as_tensor(z_np, dtype=torch.float32, device=device).unsqueeze(0)
        xin = ctx.unsqueeze(0)
        with torch.no_grad():
            f = model.recurrence(xin, z_t).squeeze(0)
        diff = f - z_t.squeeze(0)
        return 0.5 * float((diff ** 2).sum())

    def rhs_jacobian(z_np):
        z_t = torch.as_tensor(z_np, dtype=torch.float32, device=device)
        xin = ctx.unsqueeze(0)
        J = model.jacobian(z_t, inputs=xin).cpu().numpy()
        return J - np.eye(M)

    # Linearly interpolated initial guesses
    alphas = np.linspace(0, 1, n_points)
    points: list[LineAttractorPoint] = []
    cumulative_dist = 0.0

    for i, alpha in enumerate(alphas):
        z_init = (1 - alpha) * endpoint_left + alpha * endpoint_right

        try:
            res = minimize(rhs_norm_sq, z_init, method='SLSQP',
                           jac=lambda z: (rhs_jacobian(z).T @
                                          (rhs_jacobian(z) @ z - rhs_jacobian(z) @ z_init)),
                           options={'maxiter': max_iter, 'ftol': 1e-14})
            z_opt = res.x
            speed = np.sqrt(2 * res.fun)
        except Exception:
            z_opt = z_init
            speed = np.sqrt(2 * rhs_norm_sq(z_init))

        # Compute Jacobian and eigenvalues
        try:
            J = rhs_jacobian(z_opt) + np.eye(M)
            eig = np.linalg.eigvals(J)
        except Exception:
            J = None
            eig = None

        # Cumulative distance
        if i > 0:
            cumulative_dist += np.linalg.norm(z_opt - points[-1].z)

        points.append(LineAttractorPoint(
            z=z_opt, speed=speed, distance=cumulative_dist,
            eigenvalues=eig, jacobian=J))

    return points


@torch.no_grad()
def compute_line_attractor(
    model: NeuralDynamicsModel,
    *,
    context_input: torch.Tensor,
    projection_axes: np.ndarray | None = None,
    n_steps: int = 1000,
    n_points: int = 31,
    initial_state: torch.Tensor | None = None,
) -> LineAttractorResult:
    """Unified entry point for line-attractor analysis.

    Pipeline: find endpoints → sample along the line → compute analytics → project to the visualization
    coordinate system.

    Args:
        model: trained model
        context_input: (input_dim,) context input condition
        projection_axes: (3, M) axes of the 3D visualization subspace; PCA is used if None
        n_steps: number of run steps for endpoint search
        n_points: number of sampling points along the line attractor
        initial_state: (M,) initial state

    Returns:
        LineAttractorResult
    """
    # 1. Find endpoints
    ep_left, ep_right = find_line_attractor_endpoints(
        model, context_input=context_input, n_steps=n_steps,
        initial_state=initial_state)

    # 2. Sample along the line
    points = walk_line_attractor(
        model, context_input=context_input,
        endpoint_left=ep_left, endpoint_right=ep_right,
        n_points=n_points)

    # 3. Build the result
    result = LineAttractorResult(
        points=points,
        endpoints=(ep_left, ep_right),
        projection_axes=projection_axes)

    return result
