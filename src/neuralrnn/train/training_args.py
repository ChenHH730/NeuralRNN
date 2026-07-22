"""Training hyperparameter container (≈ transformers.TrainingArguments).

All training-related hyperparameters are centralized in this dataclass and read by Trainer.
Paradigm-specific hyperparameters (e.g., GTF forcing strength alpha, LFADS KL weight) belong
in the corresponding Objective's config, keeping Trainer generic.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class TrainingArguments:
    # -- Optimization --
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    max_steps: int = 1000              # DSR / tasks are usually step-based (one batch = one step)
    batch_size: int = 16
    grad_clip_norm: float | None = 1.0  # gradient clipping; None disables
    optimizer: str = "adam"            # adam / adamw / sgd / radam

    # -- Scheduling --
    lr_scheduler: str | None = None    # None / "cosine" / "step" / "exponential"
    lr_end: float | None = None        # required for "exponential": decay learning_rate -> lr_end over max_steps
    warmup_steps: int = 0

    # -- Logging / evaluation / checkpointing --
    log_every: int = 50
    eval_every: int | None = None      # None means no evaluation during training
    save_every: int | None = None
    # Log artifacts (training curve figure + history files) go directly into output_dir;
    # None → checkpoints fall back to "./outputs", logs to "./temp/log"
    output_dir: str | None = None
    disable_progress_bar: bool = False  # set True to silence the tqdm progress bar
    device: str = "cpu"                # "cpu" / "cuda" / "cuda:0"
    seed: int = 0

    # -- Dropout (hidden-state regularization during training) --
    # Inspired by trainRNNbrain: sample a mask once and reuse it across the whole rollout
    # ("dead neuron" style). dropout_rate=0 disables (default). Recommended range: 0.05-0.2.
    # dropout_sampling: "uniform" (equal probability) / "participation" (weighted by participation) /
    #                   "output_weights" (weighted by output weights)
    dropout_rate: float = 0.0
    dropout_sampling: str = "uniform"
    dropout_beta: float = 1.0           # softmax temperature (controls concentration for non-uniform sampling)

    # -- Early stopping & best-model saving --
    early_stop_loss: float | None = None   # stop when training loss falls below this (mutually exclusive with eval_metric)
    keep_best: bool = False               # keep the model weights with the lowest training loss
    # Metric-based early stopping / best model (requires eval_fn and eval_every > 0)
    eval_metric: str | None = None         # key in eval_fn's returned dict used for early stopping
    greater_is_better: bool = False        # whether a larger eval_metric is better
    early_stopping_patience: int | None = None  # number of evals without improvement before stopping; None disables

    # -- Curriculum forcing (GTF / teacher-forcing annealing, optional) --
    # If the Objective supports alpha annealing, Trainer reads and updates it during training
    anneal_forcing: bool = False
    forcing_start: float = 1.0
    forcing_end: float = 0.0

    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)
