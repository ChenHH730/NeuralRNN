"""开源数据集注册表 + 统一加载入口 load_dataset()。

每纳入一篇论文，往 DATASET_REGISTRY 加一条 DatasetSpec（URL 直接抄该论文
notebook 里的 wget / Dataverse 链接），并指定一个 loader（"模块:函数"）。

设计见 ARCHITECTURE §3.2。下载与缓存逻辑在 data/download.py（按需实现）。
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DatasetSpec:
    kind: str                                   # neurogym / timeseries / behavioral / trained_rnn / trajectory
    loader: str | None = None                   # "module:function"，产出数据集对象
    url: str | None = None                      # 下载地址（裸 URL / Dataverse / Zenodo / OSF）
    files: dict[str, str] | None = None         # 逻辑名 -> 文件名
    filename: str | None = None                 # 单文件下载名
    unpack: str | None = None                   # None / "zip" / "tar"
    sha256: str | None = None                   # 校验
    task: str | None = None                     # neurogym 任务名
    extra: dict[str, Any] = field(default_factory=dict)


# ------- 条目示例（URL 来自各 notebook）-------
DATASET_REGISTRY: dict[str, DatasetSpec] = {
    # CNS2023：Lorenz63 重构基准（含 train/test/预训练模型）
    "lorenz63": DatasetSpec(
        kind="timeseries",
        url="https://github.com/DurstewitzLab/CNS-2023/raw/main/lorenz-datasets.zip",
        unpack="zip",
        files={"train": "lorenz63_train.npy", "test": "lorenz63_test.npy"},
        loader="neuralrnn.data.timeseries_dataset:TimeSeriesDataset.from_npy",
    ),
    # nn-brain：neurogym 任务（无需下载）
    "perceptual_decision_making": DatasetSpec(
        kind="neurogym", task="PerceptualDecisionMaking-v0",
        loader="neuralrnn.data.neurogym_dataset:NeurogymDataset.from_task",
    ),
    # nn-brain：ParametricWorkingMemory 任务（DelayComparison，无需下载）
    "delay_comparison": DatasetSpec(
        kind="neurogym", task="DelayComparison-v0",
        loader="neuralrnn.data.neurogym_dataset:NeurogymDataset.from_task",
    ),
    # 低秩 RNN（Harvard Dataverse；移植配方 3 时启用 loader）
    "dms_lowrank_rank2": DatasetSpec(
        kind="trained_rnn",
        url="https://dataverse.harvard.edu/api/access/datafile/6963161",
        filename="dms_rank2_500.pt",
        # loader="neuralrnn.models.lowrank.modeling_lowrank:load_network",
    ),
    # 移植新论文时在此追加……
}


def _resolve(spec_loader: str):
    """解析 "module:attr" 或 "module:Class.method" 为可调用对象。"""
    module_path, attr = spec_loader.split(":")
    obj = importlib.import_module(module_path)
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


def load_dataset(name: str, **overrides):
    """统一入口：查 registry → （按需下载/缓存）→ 实例化数据集。

    overrides 透传给 loader（如 sequence_length / batch_size / dt / seq_len）。
    """
    if name not in DATASET_REGISTRY:
        raise KeyError(f"未注册数据集 '{name}'。已有: {sorted(DATASET_REGISTRY)}")
    spec = DATASET_REGISTRY[name]

    if spec.kind == "neurogym":
        loader = _resolve(spec.loader)
        return loader(task=spec.task, **overrides)

    # 需要下载的本地文件型数据集
    from .download import ensure_files  # 按需实现：返回 {逻辑名: 本地路径}
    local = ensure_files(spec)          # 处理下载/解压/校验/缓存
    loader = _resolve(spec.loader)
    if spec.files:
        # 约定：files 的逻辑名（train/test...）作为 *_path 关键字传入
        path_kwargs = {f"{k}_path": v for k, v in local.items()}
        return loader(**path_kwargs, **overrides)
    return loader(local.get("file"), **overrides)
