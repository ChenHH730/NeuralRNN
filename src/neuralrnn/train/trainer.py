"""Unified trainer (≈ transformers.Trainer).

Fully generic: only depends on model.forward / objective.compute_loss / dataset.sample_batch,
and is agnostic to paradigms such as "task optimization vs. dynamics reconstruction vs. behavior
fitting vs. variational". Paradigm differences are handled by the passed Objective (ARCHITECTURE §4).

Minimal usage:
    trainer = Trainer(model, dataset, objective, args)
    trainer.train()
    model.save_pretrained("ckpt/")          # config.json + model.safetensors

Design notes:
  - Step-based (one batch = one step), suitable for both DSR and task paradigms;
  - Supports gradient clipping, optional lr scheduling, optional GTF forcing annealing,
    periodic logging/evaluation/checkpointing;
  - No paradigm logic is hard-coded, guaranteeing "change Objective, change paradigm";
  - Supports hidden-state dropout (dropout_rate > 0 is implemented via forward_with_dropout,
    task loss is computed on clean outputs, consistent with trainRNNbrain).
"""
from __future__ import annotations

import json
import numbers
import os
from typing import Callable

import torch
from tqdm.auto import tqdm

from .training_args import TrainingArguments
from .objectives.base import Objective
from ..modeling_utils import NeuralDynamicsModel, DynamicsModelOutput


def _build_optimizer(params, args: TrainingArguments):
    name = args.optimizer.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=args.learning_rate, weight_decay=args.weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=args.weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=args.learning_rate, weight_decay=args.weight_decay)
    raise ValueError(f"Unknown optimizer: {args.optimizer}")


def _build_scheduler(optimizer, args: TrainingArguments):
    if args.lr_scheduler is None:
        return None
    if args.lr_scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_steps)
    if args.lr_scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(args.max_steps // 3, 1))
    raise ValueError(f"Unknown lr_scheduler: {args.lr_scheduler}")


class Trainer:
    def __init__(self, model: NeuralDynamicsModel, dataset, objective: Objective,
                 args: TrainingArguments | None = None,
                 eval_fn: Callable[[NeuralDynamicsModel], dict] | None = None,
                 post_step_hook: Callable[[NeuralDynamicsModel], None] | None = None):
        self.model = model
        self.dataset = dataset
        self.objective = objective
        self.args = args or TrainingArguments()
        self.eval_fn = eval_fn               # optional: returns an evaluation metric dict (e.g. D_stsp/D_H)
        self.post_step_hook = post_step_hook  # optional: called after each gradient update (e.g., constraint projection)
        self.history: list[dict] = []

        # Checkpoint dir: explicit output_dir, else "./outputs"
        self.output_dir = self.args.output_dir or "./outputs"
        # Log artifacts (curve figure + history files) go directly into output_dir;
        # "./temp/log" when no output dir is specified
        self.log_dir = self.args.output_dir or os.path.join(".", "temp", "log")

        self._pbar = None                    # active tqdm bar (None when not training)
        self._last_eval: dict = {}           # last eval_fn metrics (shown in postfix between evals)
        self._eval_history: list[dict] = []  # [{"step": ..., **metrics}] for the curve figure
        self._log_file_ready = False

        torch.manual_seed(self.args.seed)
        self.device = torch.device(self.args.device)
        self.model.to(self.device)
        self.optimizer = _build_optimizer(self.model.parameters(), self.args)
        self.scheduler = _build_scheduler(self.optimizer, self.args)

        # Dropout: participation tracking (EMA) for non-uniform sampling
        self._participation: torch.Tensor | None = None
        self._participation_eta: float = 0.1  # EMA decay

        # Early-stop & keep-best tracking
        self._best_loss: float = float('inf')
        self._best_state_dict: dict | None = None

        # Metric-based early stopping / best model (requires eval_fn + eval_every)
        self._best_metric = float('inf')
        self._best_metric_sign = 1.0
        self._eval_no_improve = 0
        self._best_state_dict_eval: dict | None = None
        if self.args.eval_metric is not None:
            self._best_metric_sign = -1.0 if self.args.greater_is_better else 1.0
            self._best_metric = float('inf')

    @staticmethod
    def _compute_participation(states: torch.Tensor, q: float = 0.9) -> torch.Tensor:
        """Compute per-neuron participation metric (quantile + std of |states|).

        Matches trainRNNbrain's get_participation_():
            participation[i] = quantile(|states[:, :, i]|, q) + std(|states[:, :, i]|)

        Args:
            states: (B, T, M) hidden states
            q: quantile level (default 0.9)

        Returns:
            (M,) participation vector
        """
        abs_s = states.detach().abs()  # (B, T, M)
        flat = abs_s.reshape(-1, abs_s.shape[-1])  # (N, M)
        quant = torch.quantile(flat, q, dim=0)
        std = flat.std(dim=0)
        return quant + std

    # ---- Data: always use sample_batch (supports both DSR and task datasets) and move to device ----
    def _next_batch(self) -> dict:
        batch = self.dataset.sample_batch()
        return {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    def _maybe_anneal_forcing(self, step: int) -> None:
        if not self.args.anneal_forcing:
            return
        frac = step / max(self.args.max_steps - 1, 1)
        alpha = self.args.forcing_start + frac * (self.args.forcing_end - self.args.forcing_start)
        self.objective.set_forcing(alpha)

    # ---- Progress display & log artifacts ----
    def _emit(self, msg: str) -> None:
        """Print a rare event (early stop etc.) without breaking the progress bar."""
        if self._pbar is not None:
            self._pbar.write(msg)
        else:
            print(msg)

    def _postfix(self, logs: dict) -> dict:
        """Progress-bar postfix: train metrics from this step + last eval metrics."""
        pf = {}
        for k, v in logs.items():
            if k == "step":
                continue
            pf[k] = f"{v:.4f}" if isinstance(v, float) else v
        for k, v in self._last_eval.items():
            if isinstance(v, float):
                pf[f"eval/{k}"] = f"{v:.4f}"
        return pf

    @staticmethod
    def _jsonable(d: dict) -> dict:
        return {k: (float(v) if isinstance(v, numbers.Real) else str(v)) for k, v in d.items()}

    def _write_log_line(self, record: dict) -> None:
        """Append one JSON line to <log_dir>/history.jsonl."""
        os.makedirs(self.log_dir, exist_ok=True)
        self._log_file_ready = True
        with open(os.path.join(self.log_dir, "history.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(self._jsonable(record)) + "\n")

    def _update_plot(self) -> None:
        """Redraw the training curve figure into <log_dir>/training_curves.png.

        Left y-axis: loss (log scale); right y-axis: any other numeric train metrics
        (e.g. accuracy) and eval_fn metrics (plotted at their eval steps).
        """
        if not self.history:
            return
        import matplotlib.pyplot as plt

        with plt.rc_context({"font.size": plt.rcParams["font.size"] + 3}):
            fig, ax1 = plt.subplots(figsize=(8, 4.5))
            if any("loss" in h for h in self.history):
                loss_steps = [h["step"] for h in self.history if "loss" in h]
                loss_vals = [h["loss"] for h in self.history if "loss" in h]
                ax1.plot(loss_steps, loss_vals, color="tab:blue", label="loss")
            ax1.set_xlabel("step")
            ax1.set_ylabel("loss")
            ax1.set_yscale("log")

            # Right axis: all non-loss numeric metrics (train logs + eval metrics)
            ax2 = ax1.twinx()
            cmap = plt.get_cmap("tab10")
            handles, labels = ax1.get_legend_handles_labels()
            ci = 1  # skip tab10[0] (tab:blue), reserved for the loss curve
            train_keys = sorted({k for h in self.history for k, v in h.items()
                                 if k not in ("step", "loss") and isinstance(v, numbers.Real)})
            for k in train_keys:
                xs = [h["step"] for h in self.history if k in h]
                ys = [h[k] for h in self.history if k in h]
                hdl, = ax2.plot(xs, ys, color=cmap(ci % 10), alpha=0.8, label=k)
                handles.append(hdl); labels.append(k); ci += 1
            if self._eval_history:
                eval_keys = sorted({k for h in self._eval_history for k, v in h.items()
                                    if k != "step" and isinstance(v, numbers.Real)})
                for k in eval_keys:
                    xs = [h["step"] for h in self._eval_history if k in h]
                    ys = [h[k] for h in self._eval_history if k in h]
                    hdl, = ax2.plot(xs, ys, "o--", color=cmap(ci % 10), label=f"eval/{k}")
                    handles.append(hdl); labels.append(f"eval/{k}"); ci += 1
            ax2.set_ylabel("metrics")
            if handles:
                ax1.legend(handles, labels, loc="best", fontsize=plt.rcParams["font.size"] - 5)
            fig.tight_layout()
            os.makedirs(self.log_dir, exist_ok=True)
            fig.savefig(os.path.join(self.log_dir, "training_curves.png"), dpi=120)
            plt.close(fig)

    def _flush_logs(self) -> None:
        """Final flush: full history JSON + final curve figure."""
        if not self._log_file_ready:
            return
        with open(os.path.join(self.log_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump([self._jsonable(h) for h in self.history], f, indent=2)
        self._update_plot()

    def train(self) -> list[dict]:
        self.model.train()
        use_dropout = self.args.dropout_rate > 0
        do_file_log = bool(self.args.log_every)
        if do_file_log:
            # Fresh run: reset the JSONL log (final full history goes to history.json)
            os.makedirs(self.log_dir, exist_ok=True)
            open(os.path.join(self.log_dir, "history.jsonl"), "w", encoding="utf-8").close()

        self._pbar = tqdm(range(self.args.max_steps), desc="train", dynamic_ncols=True,
                          disable=self.args.disable_progress_bar)
        try:
            for step in self._pbar:
                self._maybe_anneal_forcing(step)
                batch = self._next_batch()

                self.optimizer.zero_grad()

                if use_dropout:
                    # Dropout: task loss on clean output, matching trainRNNbrain strategy.
                    # forward_with_dropout returns (states_clean, outputs_clean,
                    # states_dropped, outputs_dropped) in one pass.
                    # For participation sampling, initialize with uniform on first step
                    if self.args.dropout_sampling == "participation" and self._participation is None:
                        # First step: use uniform to bootstrap participation
                        _boot_sampling = "uniform"
                        part = None
                    else:
                        _boot_sampling = self.args.dropout_sampling
                        part = self._participation if self.args.dropout_sampling == "participation" else None
                    sc, oc, sd, od = self.model.forward_with_dropout(
                        batch["inputs"],
                        dropout_rate=self.args.dropout_rate,
                        dropout_sampling=_boot_sampling,
                        dropout_beta=self.args.dropout_beta,
                        participation=part,
                    )
                    # Update participation via EMA
                    if self.args.dropout_sampling == "participation":
                        M = self.model.config.latent_dim
                        new_part = self._compute_participation(sc)
                        if self._participation is None:
                            self._participation = new_part
                        else:
                            self._participation = ((1 - self._participation_eta) * self._participation
                                                   + self._participation_eta * new_part)

                    # Patch model.forward temporarily so objective sees clean output
                    _orig_fwd = self.model.forward
                    self.model.forward = lambda *a, **kw: DynamicsModelOutput(
                        outputs=oc, states=sc)
                    try:
                        loss, logs = self.objective.compute_loss(self.model, batch)
                    finally:
                        self.model.forward = _orig_fwd
                else:
                    loss, logs = self.objective.compute_loss(self.model, batch)

                # ── Keep best model ──
                if self.args.keep_best:
                    current_loss = loss.item()
                    if current_loss < self._best_loss:
                        self._best_loss = current_loss
                        import copy
                        self._best_state_dict = copy.deepcopy(self.model.state_dict())

                # ── Early stop ──
                if self.args.early_stop_loss is not None and loss.item() < self.args.early_stop_loss:
                    self._emit(f"[train] step={step}  early stop: "
                               f"loss={loss.item():.4f} < {self.args.early_stop_loss}")
                    logs = {"step": step, **logs}
                    self.history.append(logs)
                    self._pbar.set_postfix(self._postfix(logs), refresh=True)
                    if do_file_log:
                        self._write_log_line(logs)
                    break

                loss.backward()
                if self.args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip_norm)
                self.optimizer.step()
                if self.post_step_hook is not None:
                    self.post_step_hook(self.model)
                if self.scheduler is not None:
                    self.scheduler.step()

                logs = {"step": step, **logs}
                self.history.append(logs)
                self._pbar.set_postfix(self._postfix(logs), refresh=False)

                if do_file_log and step % self.args.log_every == 0:
                    self._write_log_line(logs)
                    self._update_plot()

                if (self.args.eval_every and self.eval_fn is not None
                        and step > 0 and step % self.args.eval_every == 0):
                    self.model.eval()
                    metrics = self.eval_fn(self.model)
                    self.model.train()
                    self._last_eval = metrics
                    self._eval_history.append({"step": step, **metrics})
                    self._pbar.set_postfix(self._postfix(logs), refresh=True)
                    if do_file_log:
                        self._write_log_line({"step": step, "eval": True, **metrics})
                        self._update_plot()

                    # Metric-based early stopping / best model
                    if self.args.eval_metric is not None:
                        if self.args.eval_metric not in metrics:
                            raise ValueError(f"eval_metric '{self.args.eval_metric}' not found in eval_fn output. "
                                             f"Available keys: {list(metrics.keys())}")
                        current_metric = float(metrics[self.args.eval_metric])
                        signed_current = self._best_metric_sign * current_metric
                        tol = 1e-6
                        if signed_current < self._best_metric - tol:
                            self._best_metric = signed_current
                            import copy
                            self._best_state_dict_eval = copy.deepcopy(self.model.state_dict())
                            self._eval_no_improve = 0
                        else:
                            self._eval_no_improve += 1

                        if (self.args.early_stopping_patience is not None
                                and self._eval_no_improve >= self.args.early_stopping_patience):
                            self._emit(f"[eval ] early stop at step {step}: no improvement for "
                                       f"{self._eval_no_improve} evals")
                            break

                if self.args.save_every and step > 0 and step % self.args.save_every == 0:
                    self.save_checkpoint(step)
        finally:
            self._pbar.close()
            self._pbar = None
            self._flush_logs()

        # ── Restore best model ──
        if self.args.eval_metric is not None:
            if (self.args.keep_best or self.args.early_stopping_patience is not None) \
                    and self._best_state_dict_eval is not None:
                self.model.load_state_dict(self._best_state_dict_eval)
        elif self.args.keep_best and self._best_state_dict is not None:
            self.model.load_state_dict(self._best_state_dict)

        return self.history

    # ---- Save/load: reuse the model's save_pretrained (safetensors + json) ----
    def save_checkpoint(self, step: int) -> str:
        path = os.path.join(self.output_dir, f"checkpoint-{step}")
        self.model.save_pretrained(path, metadata={"step": step,
                                                    "training_args": self.args.to_dict()})
        return path

    @torch.no_grad()
    def evaluate(self) -> dict:
        if self.eval_fn is None:
            raise RuntimeError("eval_fn not provided")
        self.model.eval()
        return self.eval_fn(self.model)
