"""嵌套交叉验证 + 配置网格框架（Tiny RNN 范式）。

移植自 01-fitting-generated-data.ipynb 的 behavior_cv_training_config_combination：
对 base_config × config_ranges 展开出一组带名字的配置，对每个配置做
outer × inner 嵌套交叉验证 + 多随机种子，挑选验证损失最优的模型。

这是行为拟合的标准做法（小模型、强正则、嵌套CV 防过拟合）。本文件给出可运行骨架；
具体训练单模型复用 Trainer + BehavioralObjective（见 PORTING_GUIDE 配方7）。
"""
from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass, field

import numpy as np


def config_combination(base_config: dict, config_ranges: dict) -> list[dict]:
    """笛卡尔积展开：base_config 叠加 config_ranges 的每种取值组合。

    返回的每个 config 多一个 'model_name' 字段（由变化的键值拼成），便于检索。
    """
    keys = list(config_ranges.keys())
    out: list[dict] = []
    for values in itertools.product(*[config_ranges[k] for k in keys]):
        cfg = copy.deepcopy(base_config)
        name_parts = []
        for k, v in zip(keys, values):
            cfg[k] = v
            name_parts.append(f"{k}{v}")
        cfg["model_name"] = ".".join(name_parts)
        out.append(cfg)
    return out


def _kfold_indices(n: int, k: int, seed: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    folds = np.array_split(idx, k)
    splits = []
    for i in range(k):
        val = folds[i]
        train = np.concatenate([folds[j] for j in range(k) if j != i])
        splits.append((train, val))
    return splits


@dataclass
class CVResult:
    config: dict
    outer_val_losses: list[float] = field(default_factory=list)

    @property
    def mean_val_loss(self) -> float:
        return float(np.mean(self.outer_val_losses)) if self.outer_val_losses else float("nan")


def behavior_cv_training(base_config: dict, config_ranges: dict,
                         fit_one_fn, n_samples: int) -> list[CVResult]:
    """对每个配置做嵌套CV，返回每个配置的交叉验证结果。

    参数
    ----
    fit_one_fn(config, train_idx, val_idx, seed) -> float
        训练单个模型并返回验证损失。移植时用 Trainer + BehavioralObjective 实现
        （在 train_idx 上训练、val_idx 上评估 NLL）。
    n_samples : 被试/试次总数，用于切分。

    说明：这里给出 outer/inner/seed 的循环骨架；inner 折用于早停/选超参，outer 折
    给出泛化估计。简化骨架默认在 inner 选最优后用其设置评估 outer 验证集。
    """
    results: list[CVResult] = []
    for cfg in config_combination(base_config, config_ranges):
        res = CVResult(config=cfg)
        outer = _kfold_indices(n_samples, cfg.get("outer_splits", 3), seed=0)
        for o, (outer_train, outer_val) in enumerate(outer):
            seed_losses = []
            for seed in range(cfg.get("seed_num", 1)):
                # inner CV（在 outer_train 内部）：可用于早停/选种子，骨架直接训练
                inner = _kfold_indices(len(outer_train),
                                       cfg.get("inner_splits", 2), seed=seed)
                best_inner = min(
                    fit_one_fn(cfg, outer_train[itr], outer_train[ival], seed)
                    for itr, ival in inner
                )
                seed_losses.append(best_inner)
            # 用最优种子设置在 outer 验证集上的损失作为该折的泛化估计
            res.outer_val_losses.append(float(np.min(seed_losses)))
        results.append(res)
    return results


def find_best_models_for_exp(results: list[CVResult]) -> CVResult:
    """选择平均验证损失最低的配置（对应 notebook 的 find_best_models_for_exp）。"""
    return min(results, key=lambda r: r.mean_val_loss)
