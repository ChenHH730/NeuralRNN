"""统一训练器（≈ transformers.Trainer）。

完全通用：只依赖 model.forward / objective.compute_loss / dataset.sample_batch，
对"任务优化 vs 动力学重构 vs 行为拟合 vs 变分"等范式无感。范式差异由传入的
Objective 决定（ARCHITECTURE §4）。

最小用法：
    trainer = Trainer(model, dataset, objective, args)
    trainer.train()
    model.save_pretrained("ckpt/")          # config.json + model.safetensors

设计要点：
  - 以 step 为单位（一个 batch = 一 step），契合 DSR/任务两种范式；
  - 支持梯度裁剪、可选 lr 调度、可选 GTF forcing 退火、定期日志/评估/存档；
  - 不把任何范式逻辑写进来，保证"换 Objective 即换范式"；
  - 支持隐藏状态 dropout（dropout_rate > 0 时通过 forward_with_dropout 实现，
    task loss 在 clean 输出上计算，与 trainRNNbrain 策略一致）。
"""
from __future__ import annotations

import os
from typing import Callable

import torch

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
    raise ValueError(f"未知 optimizer: {args.optimizer}")


def _build_scheduler(optimizer, args: TrainingArguments):
    if args.lr_scheduler is None:
        return None
    if args.lr_scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_steps)
    if args.lr_scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(args.max_steps // 3, 1))
    raise ValueError(f"未知 lr_scheduler: {args.lr_scheduler}")


class Trainer:
    def __init__(self, model: NeuralDynamicsModel, dataset, objective: Objective,
                 args: TrainingArguments | None = None,
                 eval_fn: Callable[[NeuralDynamicsModel], dict] | None = None,
                 post_step_hook: Callable[[NeuralDynamicsModel], None] | None = None):
        self.model = model
        self.dataset = dataset
        self.objective = objective
        self.args = args or TrainingArguments()
        self.eval_fn = eval_fn               # 可选：返回评估指标 dict（如 D_stsp/D_H）
        self.post_step_hook = post_step_hook  # 可选：每步梯度更新后调用（如约束投影）
        self.history: list[dict] = []

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

    # ---- 数据：统一走 sample_batch（DSR/任务都支持）并搬到 device ----
    def _next_batch(self) -> dict:
        batch = self.dataset.sample_batch()
        return {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    def _maybe_anneal_forcing(self, step: int) -> None:
        if not self.args.anneal_forcing:
            return
        frac = step / max(self.args.max_steps - 1, 1)
        alpha = self.args.forcing_start + frac * (self.args.forcing_end - self.args.forcing_start)
        self.objective.set_forcing(alpha)

    def train(self) -> list[dict]:
        self.model.train()
        use_dropout = self.args.dropout_rate > 0
        for step in range(self.args.max_steps):
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
                if self.args.log_every:
                    print(f"[train] step={step}  early stop: loss={loss.item():.4f} < {self.args.early_stop_loss}")
                logs = {"step": step, **logs}
                self.history.append(logs)
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

            if self.args.log_every and step % self.args.log_every == 0:
                msg = "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                for k, v in logs.items())
                print(f"[train] {msg}")

            if (self.args.eval_every and self.eval_fn is not None
                    and step > 0 and step % self.args.eval_every == 0):
                self.model.eval()
                metrics = self.eval_fn(self.model)
                print(f"[eval ] step={step}  " +
                      "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
                self.model.train()

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
                        print(f"[eval ] early stop at step {step}: no improvement for "
                              f"{self._eval_no_improve} evals")
                        break

            if self.args.save_every and step > 0 and step % self.args.save_every == 0:
                self.save_checkpoint(step)

        # ── Restore best model ──
        if self.args.eval_metric is not None:
            if (self.args.keep_best or self.args.early_stopping_patience is not None) \
                    and self._best_state_dict_eval is not None:
                self.model.load_state_dict(self._best_state_dict_eval)
        elif self.args.keep_best and self._best_state_dict is not None:
            self.model.load_state_dict(self._best_state_dict)

        return self.history

    # ---- 存读：直接复用模型的 save_pretrained（safetensors + json）----
    def save_checkpoint(self, step: int) -> str:
        path = os.path.join(self.args.output_dir, f"checkpoint-{step}")
        self.model.save_pretrained(path, metadata={"step": step,
                                                    "training_args": self.args.to_dict()})
        return path

    @torch.no_grad()
    def evaluate(self) -> dict:
        if self.eval_fn is None:
            raise RuntimeError("未提供 eval_fn")
        self.model.eval()
        return self.eval_fn(self.model)
