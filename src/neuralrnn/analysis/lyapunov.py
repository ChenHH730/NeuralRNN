"""Maximum Lyapunov exponent (chaos criterion).

Ported from SCYFI's max_lyapunov_exponent: accumulate the Jacobian along a trajectory,
periodically re-orthogonalize with QR, and accumulate log|R[0,0]| / T. λ_max > 0 indicates chaos
(Lorenz63 ≈ 0.9).

Model-agnostic: only uses model.generate (free rollout) and model.jacobian (contract).
"""
from __future__ import annotations

import torch

from ..modeling_utils import NeuralDynamicsModel


@torch.no_grad()
def max_lyapunov_exponent(model: NeuralDynamicsModel, z1: torch.Tensor, T: int = 10000,
                          T_trans: int = 1000, ons: int = 1,
                          dt: float | None = None) -> float:
    """z1: (M,) initial value. Evolve for T_trans steps to discard transients, then accumulate the
    maximum exponent over T steps.

    Args:
        dt: sampling time interval. If provided, the returned value is divided by dt to convert the
            discrete-time exponent into a continuous-time exponent. For example, CNS2023's Lorenz-63
            data has a sampling interval of 0.01; the original tutorial divides the discrete exponent
            by 0.01 to obtain ~0.906. If dt is None and model.config.dt exists and is positive, it is
            used automatically.
    """
    model.eval()
    M = model.config.latent_dim
    device = z1.device

    # Evolve transients: use generate to obtain the final state (generate returns (1, T+1, M))
    z = z1.unsqueeze(0)
    traj = model.generate(z, T_trans)
    z = traj[:, -1]                          # (1,M)

    lyap = 0.0
    Q = torch.eye(M, device=device)
    for t in range(T):
        z = model.recurrence(None, z)        # Autonomous single step (1,M)
        J = model.jacobian(z.squeeze(0))     # (M,M)
        Q = J @ Q
        if t % ons == 0:
            Q, R = torch.linalg.qr(Q)
            lyap += torch.log(torch.abs(R[0, 0])).item()
    lam = lyap / T
    if dt is None:
        dt = getattr(model.config, "dt", None)
    if dt is not None and dt > 0:
        return lam / dt
    return lam
