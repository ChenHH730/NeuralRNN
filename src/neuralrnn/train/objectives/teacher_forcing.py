"""Generalized teacher-forcing objective GTF (Paradigm B: dynamics reconstruction).

Ported from predict_sequence_using_gtf in CNS2023_tutorial.ipynb:

    z_0      = F(z_prev=x_obs[0], s[0])
    z_forced = alpha * x_obs[t] + (1 - alpha) * z_pred        # generalized teacher forcing
    z_t      = F(z_prev=z_forced, s[t])
    loss     = MSE( readout(Z), targets )

Forcing strength alpha in [0,1]: alpha=1 reduces to pure teacher forcing, alpha=0 to free
running; DSR often uses a small alpha (e.g. 0.1) for "sparse forcing" to stabilize training of
chaotic systems.

Key rewrite (to align with framework contract, see PORTING_GUIDE recipe 2 / Contract C):
  - The original code's `model(z_prev, s)` is a single-step transition → here we call
    `model.recurrence(x_t=s, z_prev=...)`;
  - The original code assumes latent_dim == observation_dim (identity readout) and blends the
    whole vector; here we generalize to "force only the first obs_dim dimensions of the observed
    subspace", which is identical to the original when latent_dim == obs_dim.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import Objective
from ...modeling_utils import NeuralDynamicsModel


def generalized_teacher_forcing(z_pred: torch.Tensor, z_obs: torch.Tensor,
                                alpha: float) -> torch.Tensor:
    """z = alpha * z_obs + (1 - alpha) * z_pred (element-wise blending)."""
    return alpha * z_obs + (1.0 - alpha) * z_pred


class TeacherForcingObjective(Objective):
    def __init__(self, alpha: float = 0.1, forcing_interval: int | None = None):
        """
        Args:
            alpha: Teacher forcing blending strength in [0, 1].
            forcing_interval: If None (default), apply GTF at every step.
                If an integer tau > 0, apply forcing only when t % tau == 0,
                matching sparse teacher forcing used by ALRNN-DSR.
        """
        self.alpha = float(alpha)
        if forcing_interval is not None and forcing_interval <= 0:
            raise ValueError("forcing_interval must be None or a positive integer")
        self.forcing_interval = forcing_interval

    def set_forcing(self, alpha: float) -> None:
        self.alpha = float(alpha)

    def _force(self, z_pred: torch.Tensor, x_obs_t: torch.Tensor) -> torch.Tensor:
        """Inject observation into predicted latent state. Blend whole vector when latent_dim == obs_dim; otherwise only blend the first obs_dim dimensions."""
        M = z_pred.shape[-1]
        N = x_obs_t.shape[-1]
        if M == N:
            return generalized_teacher_forcing(z_pred, x_obs_t, self.alpha)
        forced = z_pred.clone()
        forced[..., :N] = generalized_teacher_forcing(z_pred[..., :N], x_obs_t, self.alpha)
        return forced

    def compute_loss(self, model: NeuralDynamicsModel, batch):
        X = batch["inputs"]                 # (B,T,N) observations (= latent trajectory, DSR identity)
        Y = batch["targets"]                # (B,T,N) observations shifted right by one
        S = batch.get("external_inputs")    # (B,T,K) or None
        B, T, N = X.shape

        s0 = S[:, 0] if S is not None else None
        M = model.config.latent_dim
        device = next(model.parameters()).device
        # When latent dim M == observation dim N, initialize previous state with the first observation;
        # when M != N, initialize with model.init_state and apply teacher forcing only to the first N dims.
        if M == N:
            z = X[:, 0].to(device)
        else:
            z = model.init_state(B, device)
            z = self._force(z, X[:, 0])
        z = model.recurrence(s0, z)
        preds = [z]
        for t in range(1, T):
            apply_force = (
                self.forcing_interval is None or t % self.forcing_interval == 0
            )
            if apply_force:
                z_forced = self._force(z, X[:, t])
            else:
                z_forced = z
            s_t = S[:, t] if S is not None else None
            z = model.recurrence(s_t, z_forced)
            preds.append(z)

        Z = torch.stack(preds, dim=1)        # (B,T,M)
        Yhat = model.readout(Z)              # identity readout returns Z
        # When latent dim M differs from observation dim N, compute loss only on the first N dimensions
        if Yhat.shape[-1] != Y.shape[-1]:
            Yhat = Yhat[..., :Y.shape[-1]]
        loss = F.mse_loss(Yhat, Y)
        return loss, {"loss": loss.item(), "alpha": self.alpha}
