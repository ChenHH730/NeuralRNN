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
import math

from .base import Objective
from .registry import register_objective
from ...modeling_utils import NeuralDynamicsModel


def generalized_teacher_forcing(z_pred: torch.Tensor, z_obs: torch.Tensor,
                                alpha: float) -> torch.Tensor:
    """z = alpha * z_obs + (1 - alpha) * z_pred (element-wise blending)."""
    return alpha * z_obs + (1.0 - alpha) * z_pred


@register_objective("teacher_forcing")
class TeacherForcingObjective(Objective):
    """Generalized teacher forcing (GTF) objective for DSR (Paradigm B, PLRNN family).

    Rolls the model step by step, blending each predicted state with the
    observation (z = α·x_obs + (1−α)·z_pred), and minimizes MSE between the
    readout of the rollout and the observations.
    """

    def __init__(self, alpha: float = 0.1, forcing_interval: int | None = None,
                 tf_noise: float = 0.0, observation_to_state=None):
        """
        Args:
            alpha: Teacher forcing blending strength in [0, 1].
            forcing_interval: If None (default), apply GTF at every step.
                If an integer tau > 0, apply forcing only when t % tau == 0,
                matching sparse teacher forcing used by ALRNN-DSR / dendPLRNN.
            observation_to_state: Optional callable (model, observation, predicted_state)
                returning the full lifted state. Required for non-identity observations.
            tf_noise: Std of Gaussian noise added to the teacher signal
                (dendPLRNN BPTT-TF reference uses 0.05); 0 disables.
        """
        self.set_forcing(alpha)
        if forcing_interval is not None and (isinstance(forcing_interval, bool) or
                not isinstance(forcing_interval, int) or forcing_interval <= 0):
            raise ValueError("forcing_interval must be None or a positive integer")
        self.forcing_interval = forcing_interval
        self.tf_noise = float(tf_noise)
        if not math.isfinite(self.tf_noise) or self.tf_noise < 0:
            raise ValueError("tf_noise must be finite and nonnegative")
        self.observation_to_state = observation_to_state

    def set_forcing(self, alpha: float) -> None:
        """Set the GTF blending strength (called by Trainer for annealing)."""
        self.alpha = float(alpha)
        if not math.isfinite(self.alpha) or not 0 <= self.alpha <= 1:
            raise ValueError("alpha must be finite and in [0,1]")

    def _force(self, z_pred: torch.Tensor, x_obs_t: torch.Tensor) -> torch.Tensor:
        """Inject observation into predicted latent state. Blend whole vector when latent_dim == obs_dim; otherwise only blend the first obs_dim dimensions."""
        M = z_pred.shape[-1]
        N = x_obs_t.shape[-1]
        if N > M:
            raise ValueError("Observation width exceeds state width")
        if M == N:
            return generalized_teacher_forcing(z_pred, x_obs_t, self.alpha)
        forced = z_pred.clone()
        forced[..., :N] = generalized_teacher_forcing(z_pred[..., :N], x_obs_t, self.alpha)
        return forced

    def compute_loss(self, model: NeuralDynamicsModel, batch):
        """Batch keys: "activity" (B,T,N) observations, optional "inputs"
        (B,T,K) external inputs. Returns (loss, {"loss", "alpha"}) where loss
        is the one-step-ahead MSE between readout(rollout) and observations."""
        X = batch["activity"]               # (B,T,N) observations (= latent trajectory, DSR identity)
        S = batch.get("inputs")             # (B,T,K) external inputs or None (autonomous)
        B, T, N = X.shape
        if T < 2:
            raise ValueError("Teacher forcing requires at least two time steps")
        if S is not None and S.shape[:2] != X.shape[:2]:
            raise ValueError("inputs and activity must have matching batch/time dimensions")
        if self.observation_to_state is None:
            if not getattr(model, "identity_observation", False):
                raise ValueError("Teacher forcing assumes identity observation of the first N state "
                                 "coordinates. Supply observation_to_state(model, observation, "
                                 "predicted_state) for a non-identity observation model.")
            if N > model.config.latent_dim or N != model.config.output_dim:
                raise ValueError("Observed width must equal output_dim and not exceed latent_dim")
        # Noise is added to the teacher signal only; targets stay clean
        # (dendPLRNN BPTT-TF reference: inp += noise, target untouched).
        X_tf = X + torch.randn_like(X) * self.tf_noise if self.tf_noise > 0 else X

        s0 = S[:, 0] if S is not None else None
        M = model.config.latent_dim
        device = next(model.parameters()).device
        # When latent dim M == observation dim N, initialize previous state with the first observation;
        # when M != N, use the model's observation->latent lift (default: zero-hidden init with the
        # first N dims hard-set; ALRNN/dendPLRNN with learn_z0 use a learned B matrix).
        if self.observation_to_state is not None:
            z = model.init_state(B, device)
            z = self.observation_to_state(model, X_tf[:, 0], z)
            model.validate_state(z)
        elif M == N:
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
                if self.observation_to_state is None:
                    z_forced = self._force(z, X_tf[:, t])
                else:
                    lifted = self.observation_to_state(model, X_tf[:, t], z)
                    model.validate_state(lifted)
                    if lifted.shape != z.shape:
                        raise ValueError("observation_to_state must preserve the complete state shape")
                    z_forced = generalized_teacher_forcing(z, lifted, self.alpha)
            else:
                z_forced = z
            s_t = S[:, t] if S is not None else None
            z = model.recurrence(s_t, z_forced)
            preds.append(z)

        Z = torch.stack(preds, dim=1)        # (B,T,M)
        Yhat = model.readout(Z)              # (B,T,output_dim); identity readout observes first N units
        Y = X[:, 1:]                         # one-step-ahead targets
        Yhat = Yhat[:, :-1]                  # predictions for steps 1..T-1
        if Yhat.shape != Y.shape:
            raise ValueError("Readout and observation shapes must match; outputs are not truncated")
        loss = F.mse_loss(Yhat, Y)
        return loss, {"loss": loss.item(), "alpha": self.alpha}
