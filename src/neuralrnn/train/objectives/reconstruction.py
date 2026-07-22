"""Reconstruction objective — fits a student model to recorded teacher data.

The loss combines up to two terms:

1. Behavioral reconstruction: MSE between the student readout and the
   recorded teacher behavioral output (``behavior_weight``; skipped when 0).
2. Activity reconstruction: MSE or NMSE between a map of the student hidden
   states and the recorded teacher activity (``activity_weight``).

The student-to-teacher state map (``state_map``) covers the common cases:
- "identity": student states live in the same space as the recorded activity
  (e.g. LINT trajectory fitting, connectome teacher/student with matched
  units);
- "embedding": student latent states are lifted into the teacher space via
  ``model.embedding_matrix`` (latent circuit, Langdon & Engel 2025);
- "auto" (default): "embedding" when the model exposes ``embedding_matrix``,
  else "identity".

A subset of the recorded dimensions can be selected via ``recorded_dims``
(partial recording, e.g. only M of N units are observed). With
``state_map="embedding", activity_loss="nmse"`` and default weights this
objective is numerically identical to the former ``LatentCircuitObjective``.
"""
from __future__ import annotations

from typing import Callable, Sequence

import torch

from ..losses.loss_functions import masked_mse
from .base import Objective
from .registry import register_objective


@register_objective("reconstruction")
class ReconstructionObjective(Objective):
    """Objective for reconstructing teacher activity and/or behavior.

    loss = behavior_weight * MSE(outputs, targets)
         + activity_weight * ACT_LOSS(map(states), activity)

    Args:
        behavior_weight: Weight of the behavioral MSE term; 0 disables it
            (batch need not contain "targets" then).
        activity_weight: Weight of the activity term; 0 disables it.
        state_map: "auto" | "identity" | "embedding" — how student states
            are mapped into the teacher activity space (see module docstring).
        activity_fn: Optional transform applied to the student states BEFORE
            state_map: None (use states), "firing_rates"
            (``model.get_firing_rates``), or callable ``(model, states)``.
        recorded_dims: Optional indices selecting a subset of the activity
            dimensions, applied to BOTH sides after the state map (partial
            recording).
        activity_loss: "nmse" (mean-centered teacher, clamped denominator)
            or "mse".
        eps: Lower clamp for the NMSE denominator.
    """

    def __init__(
        self,
        behavior_weight: float = 1.0,
        activity_weight: float = 1.0,
        state_map: str = "auto",
        activity_fn: str | Callable | None = None,
        recorded_dims: Sequence[int] | torch.Tensor | None = None,
        activity_loss: str = "nmse",
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if state_map not in ("auto", "identity", "embedding"):
            raise ValueError(f"Unknown state_map: {state_map!r}")
        if activity_loss not in ("nmse", "mse"):
            raise ValueError(f"Unknown activity_loss: {activity_loss!r}")
        self.behavior_weight = behavior_weight
        self.activity_weight = activity_weight
        self.state_map = state_map
        self.activity_fn = activity_fn
        self.recorded_dims = recorded_dims
        self.activity_loss = activity_loss
        self.eps = eps

    def _map_states(self, model, states: torch.Tensor) -> torch.Tensor:
        """Apply activity_fn then state_map to the student states."""
        s = states
        if self.activity_fn == "firing_rates":
            s = model.get_firing_rates(s)
        elif callable(self.activity_fn):
            s = self.activity_fn(model, s)
        state_map = self.state_map
        if state_map == "auto":
            state_map = "embedding" if hasattr(model, "embedding_matrix") else "identity"
        if state_map == "embedding":
            if not hasattr(model, "embedding_matrix"):
                raise ValueError(
                    "state_map='embedding' requires model.embedding_matrix"
                )
            s = s @ model.embedding_matrix
        return s

    def compute_loss(self, model, batch: dict) -> tuple:
        """Compute the reconstruction loss.

        Args:
            model: Student model (NeuralDynamicsModel).
            batch: dict with keys "inputs" and "activity"; optionally
                "targets" (required when behavior_weight > 0) and "mask".

        Returns:
            (loss, logs_dict). Logs contain "loss" plus "mse_z" when the
            behavior term is active and "nmse_y"/"mse_y" when the activity
            term is active.
        """
        out = model(batch["inputs"])
        mask = batch.get("mask")

        loss = 0.0
        logs: dict[str, float] = {}

        # Behavioral reconstruction term
        if self.behavior_weight > 0:
            if "targets" not in batch:
                raise KeyError(
                    "behavior_weight > 0 requires 'targets' in the batch"
                )
            mse_z = masked_mse(out.outputs, batch["targets"], mask)
            loss = loss + self.behavior_weight * mse_z
            logs["mse_z"] = mse_z.item()

        # Activity reconstruction term
        if self.activity_weight > 0:
            if "activity" not in batch:
                raise KeyError(
                    "activity_weight > 0 requires 'activity' in the batch"
                )
            s = self._map_states(model, out.states)
            a = batch["activity"]
            if s.shape[-1] != a.shape[-1]:
                raise ValueError(
                    f"Mapped student states (dim {s.shape[-1]}) do not match "
                    f"recorded activity (dim {a.shape[-1]})"
                )
            if self.recorded_dims is not None:
                idx = torch.as_tensor(
                    self.recorded_dims, dtype=torch.long, device=s.device
                )
                s = s[..., idx]
                a = a[..., idx]
            err = (s - a) ** 2
            # A mask applies to the activity term only if it broadcasts over
            # the activity dim (per-timestep mask). Per-output-dim masks
            # (e.g. from cognitive tasks) belong to the behavior term.
            act_mask = mask
            if act_mask is not None and act_mask.dim() == err.dim() - 1:
                act_mask = act_mask.unsqueeze(-1)
            if act_mask is not None and act_mask.shape[-1] not in (1, err.shape[-1]):
                act_mask = None
            if act_mask is not None:
                m = act_mask.to(err.dtype)
                act_num = (err * m).sum() / (m.sum() * err.shape[-1]).clamp_min(self.eps)
                if self.activity_loss == "nmse":
                    a_bar = a - a.mean(dim=[0, 1], keepdim=True)
                    act_den = ((a_bar ** 2) * m).sum() / (m.sum() * a.shape[-1]).clamp_min(self.eps)
                    act_den = act_den.clamp_min(self.eps)
                    act_loss = act_num / act_den
                else:
                    act_loss = act_num
            else:
                if self.activity_loss == "nmse":
                    a_bar = a - a.mean(dim=[0, 1], keepdim=True)
                    act_den = torch.mean(a_bar ** 2).clamp_min(self.eps)
                    act_loss = torch.mean(err) / act_den
                else:
                    act_loss = torch.mean(err)
            loss = loss + self.activity_weight * act_loss
            logs["nmse_y" if self.activity_loss == "nmse" else "mse_y"] = act_loss.item()

        if not isinstance(loss, torch.Tensor):
            raise ValueError(
                "Both behavior_weight and activity_weight are 0 — nothing to optimize"
            )
        logs["loss"] = loss.item()
        return loss, logs
