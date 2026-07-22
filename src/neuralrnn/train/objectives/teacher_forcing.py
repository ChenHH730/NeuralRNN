"""Generalized teacher-forcing objective GTF (Paradigm B: dynamics reconstruction).

Ported from predict_sequence_using_gtf in CNS2023_tutorial.ipynb:

    z_0      = F(z_prev=x_obs[0], s[0])
    z_forced = alpha * x_obs[t] + (1 - alpha) * z_pred        # generalized teacher forcing
    z_t      = F(z_prev=z_forced, s[t])
    loss     = MSE( readout(Z), x_obs shifted by one step )

Forcing strength alpha in [0,1]: alpha=1 reduces to pure (hard) teacher forcing, alpha=0 to
free running. The ALRNN / dendPLRNN DSR references combine alpha=1 with a sparse
forcing_interval tau (force only every tau steps); the GTF-shPLRNN reference uses a small
alpha (e.g. 0.1) at every step.

Unified batch schema (ReconstructionDataset):
  - batch["activity"]: (B,T,N) observed trajectory (DSR identity observation)
  - batch["inputs"]:   (B,T,K) external inputs, or absent/None for autonomous systems
  One-step-ahead targets are constructed internally by shifting "activity" right by one.

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
from .registry import register_objective
from ...modeling_utils import NeuralDynamicsModel


def generalized_teacher_forcing(z_pred: torch.Tensor, z_obs: torch.Tensor,
                                alpha: float) -> torch.Tensor:
    """z = alpha * z_obs + (1 - alpha) * z_pred (element-wise blending)."""
    return alpha * z_obs + (1.0 - alpha) * z_pred


@register_objective("teacher_forcing")
class TeacherForcingObjective(Objective):
    def __init__(self, alpha: float = 0.1, forcing_interval: int | None = None,
                 tf_noise: float = 0.0):
        """
        Args:
            alpha: Teacher forcing blending strength in [0, 1].
            forcing_interval: If None (default), apply GTF at every step.
                If an integer tau > 0, apply forcing only when t % tau == 0,
                matching sparse teacher forcing used by ALRNN-DSR / dendPLRNN.
            tf_noise: Std of Gaussian noise added to the teacher signal
                (dendPLRNN BPTT-TF reference uses 0.05); 0 disables.
        """
        self.alpha = float(alpha)
        if forcing_interval is not None and forcing_interval <= 0:
            raise ValueError("forcing_interval must be None or a positive integer")
        self.forcing_interval = forcing_interval
        self.tf_noise = float(tf_noise)

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
        X = batch["activity"]               # (B,T,N) observations (= latent trajectory, DSR identity)
        S = batch.get("inputs")             # (B,T,K) external inputs or None (autonomous)
        B, T, N = X.shape
        # Noise is added to the teacher signal only; targets stay clean
        # (dendPLRNN BPTT-TF reference: inp += noise, target untouched).
        X_tf = X + torch.randn_like(X) * self.tf_noise if self.tf_noise > 0 else X

        s0 = S[:, 0] if S is not None else None
        M = model.config.latent_dim
        device = next(model.parameters()).device
        # When latent dim M == observation dim N, initialize previous state with the first observation;
        # when M != N, use the model's observation->latent lift (default: zero-hidden init with the
        # first N dims hard-set; ALRNN/dendPLRNN with learn_z0 use a learned B matrix).
        if M == N:
            z = X_tf[:, 0].to(device)
        else:
            z = model.init_state_from_obs(X_tf[:, 0].to(device))
        z = model.recurrence(s0, z)
        preds = [z]
        for t in range(1, T):
            apply_force = (
                self.forcing_interval is None or t % self.forcing_interval == 0
            )
            if apply_force:
                z_forced = self._force(z, X_tf[:, t])
            else:
                z_forced = z
            s_t = S[:, t] if S is not None else None
            z = model.recurrence(s_t, z_forced)
            preds.append(z)

        Z = torch.stack(preds, dim=1)        # (B,T,M)
        Yhat = model.readout(Z)              # (B,T,output_dim); identity readout observes first N units
        Y = X[:, 1:]                         # one-step-ahead targets
        Yhat = Yhat[:, :-1]                  # predictions for steps 1..T-1
        if Yhat.shape[-1] != Y.shape[-1]:
            Yhat = Yhat[..., :Y.shape[-1]]
        loss = F.mse_loss(Yhat, Y)
        return loss, {"loss": loss.item(), "alpha": self.alpha}
