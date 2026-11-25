"""Maximum Lyapunov exponent (chaos criterion).

Ported from SCYFI's max_lyapunov_exponent: accumulate the Jacobian along a trajectory,
periodically re-orthogonalize with QR, and accumulate log|R[0,0]| / T. λ_max > 0 indicates chaos
(Lorenz63 ≈ 0.9).

Model-agnostic: only uses model.generate (free rollout) and model.jacobian (contract).
"""
from __future__ import annotations

import torch

from ..modeling_utils import NeuralDynamicsModel
from ._context import deterministic_analysis


@torch.no_grad()
@deterministic_analysis
def max_lyapunov_exponent(model: NeuralDynamicsModel, z1: torch.Tensor, T: int = 10000,
                          T_trans: int = 1000, ons: int = 1,
                          dt: float | None = None, task_input: torch.Tensor | None = None) -> float:
    """z1: (M,) initial value. Evolve for T_trans steps to discard transients, then accumulate the
    maximum exponent over T steps.

    Args:
        dt: sampling time interval. If provided, the returned value is divided by dt to convert the
            discrete-time exponent into a continuous-time exponent. For example, CNS2023's Lorenz-63
            data has a sampling interval of 0.01; the original tutorial divides the discrete exponent
            by 0.01 to obtain ~0.906. If dt is None and model.config.dt exists and is positive, it is
            used automatically.
    """
    if T <= 0 or T_trans < 0 or ons <= 0:
        raise ValueError("T and ons must be positive; T_trans must be nonnegative")
    if dt is not None and dt <= 0:
        raise ValueError("dt must be positive (use dt=1 for per-step units)")
    model.validate_state(z1)
    M = model.state_dim
    device = z1.device

    # Evolve transients: use generate to obtain the final state (generate returns (1, T+1, M))
    z = z1.unsqueeze(0)
    xin = None if task_input is None else task_input.to(z1).reshape(1, -1)
    for _ in range(T_trans):
        z = model.recurrence(xin, z)

    lyap = torch.zeros(M, device=device, dtype=z1.dtype)
    Q = torch.eye(M, device=device, dtype=z1.dtype)
    for t in range(T):
        J = model.jacobian(z.squeeze(0), inputs=xin)
        z = model.recurrence(xin, z)
        Q = J @ Q
        if (t + 1) % ons == 0 or t == T - 1:
            Q, R = torch.linalg.qr(Q)
            lyap += torch.log(torch.abs(torch.diag(R)))
    lam = float(lyap.max() / T)
    if dt is None:
        dt = getattr(model.config, "dt", None)
    if dt is not None and dt > 0:
        return lam / dt
    return lam
