"""Regularized supervised objective.

Combines the standard supervised task loss with optional, composable regularizers:
L2 activity penalty, L2 weight penalty, and an input/output orthogonality penalty.
This absorbs the common patterns found in notebook-defined objectives such as
``OrthogonalityObjective`` (notebook 02) and ``MultitaskObjective`` (notebook 13).
"""
from __future__ import annotations

from .supervised import SupervisedObjective
from .registry import register_objective
from ..losses import activity_l2, weight_l2, model_orthogonality_penalty


@register_objective("regularized_supervised")
class RegularizedSupervisedObjective(SupervisedObjective):
    """Supervised task loss + optional activity / weight / orthogonality regularizers.

    Args:
        task_type: "classification" or "regression" (passed to SupervisedObjective).
        activity_weight: coefficient for ``activity_l2(states)``.
        weight_weight: coefficient for ``weight_l2(model, weight_patterns)``.
        weight_patterns: optional regex patterns to restrict ``weight_l2`` to a
            subset of parameters (e.g. ["h2h", "readout"] for recurrent + readout).
        weight_reduce: "mean" or "sum".  Use "sum" to match objectives that apply
            the coefficient directly to ``sum(p**2)`` (e.g. flexible-multitask).
        ortho_weight: coefficient for the input/output orthogonality penalty.
        ortho_input_name: attribute name for the input weight module (default "input2h").
        ortho_output_name: attribute name for the output weight module (default "readout_layer").
        mse_reduce: "per_trial" (default) or "global".  Use "global" to match
            objectives that compute a single masked MSE across the whole batch
            (e.g. latent-circuit and flexible-multitask notebooks).
        activity_reduce: "per_trial" (default) or "global".  Use "global" to
            match objectives that regularize global mean firing rate independent
            of the loss mask.

    Notes:
        - The orthogonality penalty is skipped (returns 0) if the model does not
          expose the requested attributes, so it is safe to use with non-EIRNN models.
        - All regularizer weights default to 0, so the default behavior is identical
          to ``SupervisedObjective``.
    """

    def __init__(
        self,
        task_type: str = "classification",
        activity_weight: float = 0.0,
        weight_weight: float = 0.0,
        weight_patterns: list[str] | None = None,
        weight_reduce: str = "mean",
        ortho_weight: float = 0.0,
        ortho_input_name: str = "input2h",
        ortho_output_name: str = "readout_layer",
        mse_reduce: str = "per_trial",
        activity_reduce: str = "per_trial",
    ):
        super().__init__(task_type=task_type)
        self.activity_weight = float(activity_weight)
        self.weight_weight = float(weight_weight)
        self.weight_patterns = weight_patterns
        self.weight_reduce = weight_reduce
        self.ortho_weight = float(ortho_weight)
        self.ortho_input_name = ortho_input_name
        self.ortho_output_name = ortho_output_name
        self.mse_reduce = mse_reduce
        self.activity_reduce = activity_reduce

    def compute_loss(self, model, batch):
        task_loss, logs = super().compute_loss(model, batch)
        # SupervisedObjective returns task_loss computed with its own reduction.
        # For regression with mse_reduce="global", recompute the task loss so the
        # coefficient and reduction exactly match the reference notebooks.
        if self.task_type == "regression" and self.mse_reduce == "global":
            from ..losses import masked_mse
            out = model(batch["inputs"])
            target = batch["targets"]
            if target.dim() == 2:
                target = target.unsqueeze(-1)
            task_loss = masked_mse(out.outputs, target, batch.get("mask"), reduction="global")
            logs["task_loss"] = task_loss.item()

        loss = task_loss

        if self.activity_weight > 0:
            out = model(batch["inputs"])
            act_pen = activity_l2(out.states, batch.get("mask"), reduction=self.activity_reduce)
            loss = loss + self.activity_weight * act_pen
            logs["activity_loss"] = act_pen.item()

        if self.weight_weight > 0:
            w_pen = weight_l2(model, self.weight_patterns, reduction=self.weight_reduce)
            loss = loss + self.weight_weight * w_pen
            logs["weight_loss"] = w_pen.item()

        if self.ortho_weight > 0:
            o_pen = model_orthogonality_penalty(
                model,
                input_name=self.ortho_input_name,
                output_name=self.ortho_output_name,
            )
            loss = loss + self.ortho_weight * o_pen
            logs["ortho_loss"] = o_pen.item()

        logs["loss"] = loss.item()
        if "task_loss" not in logs:
            logs["task_loss"] = task_loss.item()
        return loss, logs
