"""NeuralRNN: a unified framework for RNN methods in cognitive neuroscience.

Brings two front-line paradigms under one transformers-style API:
  Paradigm A -- optimize and train RNNs on cognitive tasks, with interpretability analysis
  (fixed points / vector fields / dimensionality reduction);
  Paradigm B -- reconstruct dynamics directly from neural / behavioral data
  (PLRNN / LFADS / low-rank / Tiny RNN).

Core abstraction: every model is a "discrete dynamical system with readout". Implementing only
`recurrence` and `readout` is enough to plug into the unified Trainer (paradigm differences are
handled by Objective) and analysis modules.

Quick start
-----------
Unified construction and serialization (≈ transformers):
    from neuralrnn import AutoConfig, AutoModel
    cfg = AutoConfig.for_model("shallow_plrnn", input_dim=0, latent_dim=3, output_dim=3,
                               hidden_dim=50, autonomous=True)
    model = AutoModel.from_config(cfg)
    model.save_pretrained("ckpt/"); AutoModel.from_pretrained("ckpt/")

Training (change Objective to change paradigm):
    from neuralrnn import Trainer, TrainingArguments, TeacherForcingObjective, load_dataset
    ds = load_dataset("lorenz63", sequence_length=200, batch_size=16)
    Trainer(model, ds, TeacherForcingObjective(alpha=0.1),
            TrainingArguments(max_steps=2000)).train()

Analysis (model-agnostic):
    from neuralrnn.analysis import find_fixed_points, max_lyapunov_exponent
    fps = find_fixed_points(model)            # analytic first, falls back to numeric

See docs/ARCHITECTURE.md and docs/PORTING_GUIDE.md for details.
"""
from __future__ import annotations

__version__ = "0.3.2.dev0"

# Core base classes / output container
from .configuration_utils import (
    NeuralRNNConfig, resolve_euler_alpha,
    SUPPORTED_NONLINEARITY_MODES, validate_nonlinearity_mode,
)
from .modeling_utils import NeuralDynamicsModel, DynamicsModelOutput

# Unified activation factory
from .activations import get_activation, SUPPORTED_ACTIVATIONS

# Auto factories
from .auto import (
    AutoConfig, AutoModel,
    register_config, register_model,
    CONFIG_REGISTRY, MODEL_REGISTRY,
)

# Data
from .data import (
    BaseDataset, StandardScaler, TimeSeriesDataset, TrialTimeseriesDataset, CustomDataset,
    CognitiveTaskDataset, ReconstructionDataset,
    DATASET_REGISTRY, DatasetSpec, load_dataset,
)

# Training
from .train import (
    Trainer, TrainingArguments,
    Objective, SupervisedObjective, RegularizedSupervisedObjective,
    TeacherForcingObjective, BehavioralObjective, VariationalObjective,
    ReconstructionObjective, ConstrainedSupervisedObjective,
    build_objective, register_objective, OBJECTIVE_REGISTRY, AutoObjective,
    masked_mse, masked_cross_entropy, masked_nll, loss_mse,
    activity_l2, weight_l2, weight_l1,
    orthogonality_penalty, model_orthogonality_penalty,
    accuracy_classification, accuracy_general,
)

# Visualization
from . import visualization

__all__ = [
    "__version__",
    "NeuralRNNConfig", "NeuralDynamicsModel", "DynamicsModelOutput",
    "resolve_euler_alpha",
    "SUPPORTED_NONLINEARITY_MODES", "validate_nonlinearity_mode",
    "get_activation", "SUPPORTED_ACTIVATIONS",
    "AutoConfig", "AutoModel", "register_config", "register_model",
    "CONFIG_REGISTRY", "MODEL_REGISTRY",
    "BaseDataset", "StandardScaler", "TimeSeriesDataset", "TrialTimeseriesDataset", "CustomDataset",
    "CognitiveTaskDataset", "ReconstructionDataset",
    "DATASET_REGISTRY", "DatasetSpec", "load_dataset",
    "Trainer", "TrainingArguments",
    "Objective", "SupervisedObjective", "RegularizedSupervisedObjective",
    "TeacherForcingObjective", "BehavioralObjective", "VariationalObjective",
    "ReconstructionObjective", "ConstrainedSupervisedObjective",
    "build_objective", "register_objective", "OBJECTIVE_REGISTRY", "AutoObjective",
    "masked_mse", "masked_cross_entropy", "masked_nll", "loss_mse",
    "activity_l2", "weight_l2", "weight_l1",
    "orthogonality_penalty", "model_orthogonality_penalty",
    "accuracy_classification", "accuracy_general",
]

