"""训练超参容器（≈ transformers.TrainingArguments）。

所有训练相关超参集中在此 dataclass，由 Trainer 读取。范式特异的超参（如 GTF 的
forcing 强度 alpha、LFADS 的 KL 权重）放在对应 Objective 的配置里，保持 Trainer 通用。
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class TrainingArguments:
    # —— 优化 ——
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    max_steps: int = 1000              # DSR/任务常按 step 计（一个 batch 一 step）
    batch_size: int = 16
    grad_clip_norm: float | None = 1.0  # 梯度裁剪；None 关闭
    optimizer: str = "adam"            # adam / adamw / sgd

    # —— 调度 ——
    lr_scheduler: str | None = None    # None / "cosine" / "step"
    warmup_steps: int = 0

    # —— 日志 / 评估 / 存储 ——
    log_every: int = 50
    eval_every: int | None = None      # None 表示不在训练中评估
    save_every: int | None = None
    output_dir: str = "./outputs"
    device: str = "cpu"                # "cpu" / "cuda" / "cuda:0"
    seed: int = 0

    # —— Dropout（训练时隐藏状态正则化）——
    # 灵感来自 trainRNNbrain：mask 采样一次，整个 rollout 复用（"dead neuron" 方式）。
    # dropout_rate=0 关闭（默认）。推荐范围：0.05–0.2。
    # dropout_sampling: "uniform"（等概率）/ "participation"（按参与度加权）/ "output_weights"（按输出权重加权）
    dropout_rate: float = 0.0
    dropout_sampling: str = "uniform"
    dropout_beta: float = 1.0           # softmax 温度（非 uniform 采样时控制集中度）

    # —— 课程式 forcing（GTF / teacher forcing 退火，可选）——
    # 若 Objective 支持 alpha 退火，可由 Trainer 在训练中读取并更新
    anneal_forcing: bool = False
    forcing_start: float = 1.0
    forcing_end: float = 0.0

    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)
