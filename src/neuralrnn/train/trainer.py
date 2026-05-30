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
  - 不把任何范式逻辑写进来，保证"换 Objective 即换范式"。
"""
from __future__ import annotations

import os
from typing import Callable

import torch

from .training_args import TrainingArguments
from .objectives.base import Objective
from ..modeling_utils import NeuralDynamicsModel


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
                 eval_fn: Callable[[NeuralDynamicsModel], dict] | None = None):
        self.model = model
        self.dataset = dataset
        self.objective = objective
        self.args = args or TrainingArguments()
        self.eval_fn = eval_fn               # 可选：返回评估指标 dict（如 D_stsp/D_H）
        self.history: list[dict] = []

        torch.manual_seed(self.args.seed)
        self.device = torch.device(self.args.device)
        self.model.to(self.device)
        self.optimizer = _build_optimizer(self.model.parameters(), self.args)
        self.scheduler = _build_scheduler(self.optimizer, self.args)

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
        for step in range(self.args.max_steps):
            self._maybe_anneal_forcing(step)
            batch = self._next_batch()

            self.optimizer.zero_grad()
            loss, logs = self.objective.compute_loss(self.model, batch)
            loss.backward()
            if self.args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip_norm)
            self.optimizer.step()
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

            if self.args.save_every and step > 0 and step % self.args.save_every == 0:
                self.save_checkpoint(step)

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
