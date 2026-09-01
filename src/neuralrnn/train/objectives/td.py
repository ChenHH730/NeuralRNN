"""TD value-learning objective (critic-only RL paradigm).

Trains a model's scalar readout as a state-value estimate V(s_t) with the
semi-gradient TD(0) error

    delta_t = r_{t+1} + gamma * V(s_{t+1}).detach() - V(s_t),
    loss    = mean over valid steps of delta_t^2,

where ``r`` is the batch's ``targets`` channel (reward). This is the training
rule of the value-RNN of Qian & Burrell (2024): the RNN receives the same
observations as the animal (cues and rewards) and learns to predict value,
with the TD error as the only feedback — no policy / actor is involved.

Batch contract (batch-first):
    inputs:  (B, T, input_dim) observation stream (cues + reward).
    targets: (B, T, 1) reward stream r_t.
    mask:    (B, T, 1) or (B, T) valid-step mask (1 = valid). The TD error at
             step t is counted only when both t and t+1 are valid, so padded
             tails and episode boundaries never leak into the bootstrap.

The model must return a scalar output per step (``output_dim == 1``), e.g.
``gated_rnn`` with a GRU core, or ``actor_critic`` with
``action_type="none"`` (critic-only agent).

Reference default (Qian & Burrell 2024): gamma = 0.83 at dt = 0.5 s
(discount rate 0.67 / s).
"""
from __future__ import annotations

import torch

from .base import Objective
from .registry import register_objective
from ..losses import masked_mse
from ...modeling_utils import NeuralDynamicsModel


@register_objective("td")
class TDObjective(Objective):
    """Semi-gradient TD(0) value-learning loss (see module docstring).

    Args:
        gamma: temporal discount factor.
    """

    def __init__(self, gamma: float = 0.99):
        self.gamma = gamma

    def compute_loss(self, model: NeuralDynamicsModel, batch):
        """Batch keys: "inputs" (B,T,K), "targets" (B,T[,1]), optional "mask".
        Returns (loss, {"loss", "mean_value"})."""
        out = model(batch["inputs"])           # DynamicsModelOutput
        V = out.outputs                        # (B,T,1) or (B,T)
        if V.dim() == 2:
            V = V.unsqueeze(-1)
        if V.shape[-1] != 1:
            raise ValueError(
                f"TDObjective requires a scalar readout (output_dim == 1), "
                f"got output shape {tuple(V.shape)}")

        r = batch["targets"]
        if r.dim() == 2:
            r = r.unsqueeze(-1)

        V_hat = V[:, :-1]                                   # V(s_t)
        V_next = V[:, 1:].detach()                          # V(s_{t+1}), no grad
        target = r[:, 1:] + self.gamma * V_next             # TD target

        mask = batch.get("mask")
        if mask is not None:
            if mask.dim() == 3:
                mask = mask[..., 0]                         # (B,T)
            mask = (mask[:, :-1] * mask[:, 1:]).unsqueeze(-1)

        loss = masked_mse(V_hat, target, mask, reduction="global")
        with torch.no_grad():
            mean_v = (V_hat * mask).sum() / mask.sum().clamp(min=1e-8) \
                if mask is not None else V_hat.mean()
        return loss, {"loss": loss.item(), "mean_value": mean_v.item()}
