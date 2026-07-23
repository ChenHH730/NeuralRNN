"""NeuralRNN: a unified framework for RNN methods in cognitive neuroscience.

Brings two front-line paradigms under one transformers-style API:
  Paradigm A -- optimize and train RNNs on cognitive tasks, with interpretability analysis
  (fixed points / vector fields / dimensionality reduction);
  Paradigm B -- reconstruct dynamics directly from neural / behavioral data
  (PLRNN / LFADS / low-rank / Tiny RNN).

Core abstraction: every model is a "discrete dynamical system with readout". Implementing only
`recurrence` and `readout` is enough to plug into the unified Trainer (paradigm differences are
handled by Objective) and analysis modules.
"""

from __future__ import annotations

__version__ = "0.3.5"

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
    BaseDataset, StandardScaler, CustomDataset,
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
    "BaseDataset", "StandardScaler", "CustomDataset",
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

