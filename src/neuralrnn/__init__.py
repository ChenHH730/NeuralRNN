"""NeuralRNN：认知神经科学 RNN 方法的统一框架。

把两类前沿范式统一到同一套 transformers 风格的接口下：
  范式A —— 在认知任务上优化训练 RNN，并做可解释性分析（不动点/向量场/降维）；
  范式B —— 直接从神经/行为数据重构动力系统（PLRNN/LFADS/低秩/Tiny RNN）。

核心抽象：所有模型都是"带读出的离散动力系统"，只需实现 recurrence/readout 两个方法，
即可接入统一的 Trainer（范式差异由 Objective 决定）与 analysis 分析器。

快速上手
--------
统一构造与存读（≈ transformers）：
    from neuralrnn import AutoConfig, AutoModel
    cfg = AutoConfig.for_model("shallow_plrnn", input_dim=0, latent_dim=3, output_dim=3,
                               hidden_dim=50, autonomous=True)
    model = AutoModel.from_config(cfg)
    model.save_pretrained("ckpt/"); AutoModel.from_pretrained("ckpt/")

训练（换 Objective 即换范式）：
    from neuralrnn import Trainer, TrainingArguments, TeacherForcingObjective, load_dataset
    ds = load_dataset("lorenz63", sequence_length=200, batch_size=16)
    Trainer(model, ds, TeacherForcingObjective(alpha=0.1),
            TrainingArguments(max_steps=2000)).train()

分析（模型无关）：
    from neuralrnn.analysis import find_fixed_points, max_lyapunov_exponent
    fps = find_fixed_points(model)            # 解析优先，自动回退数值

详见 docs/ARCHITECTURE.md 与 docs/PORTING_GUIDE.md。
"""
from __future__ import annotations

__version__ = "0.1.0.dev0"

# —— 核心基类 / 输出容器 ——
from .configuration_utils import NeuralRNNConfig
from .modeling_utils import NeuralDynamicsModel, DynamicsModelOutput

# —— Auto 工厂 ——
from .auto import (
    AutoConfig, AutoModel,
    register_config, register_model,
    CONFIG_REGISTRY, MODEL_REGISTRY,
)

# —— 数据 ——
from .data import (
    BaseDataset, StandardScaler, TimeSeriesDataset, TrialTimeseriesDataset, CustomDataset,
    CognitiveTaskDataset, LatentCircuitDataset,
    DATASET_REGISTRY, DatasetSpec, load_dataset,
)

# —— 训练 ——
from .train import (
    Trainer, TrainingArguments,
    Objective, SupervisedObjective, TeacherForcingObjective,
    BehavioralObjective, VariationalObjective,
)

# —— 可视化 ——
from . import visualization

__all__ = [
    "__version__",
    "NeuralRNNConfig", "NeuralDynamicsModel", "DynamicsModelOutput",
    "AutoConfig", "AutoModel", "register_config", "register_model",
    "CONFIG_REGISTRY", "MODEL_REGISTRY",
    "BaseDataset", "StandardScaler", "TimeSeriesDataset", "TrialTimeseriesDataset", "CustomDataset",
    "CognitiveTaskDataset", "LatentCircuitDataset",
    "DATASET_REGISTRY", "DatasetSpec", "load_dataset",
    "Trainer", "TrainingArguments",
    "Objective", "SupervisedObjective", "TeacherForcingObjective",
    "BehavioralObjective", "VariationalObjective",
]
