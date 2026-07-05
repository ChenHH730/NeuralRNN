"""Training layer: generic Trainer + paradigm-specific Objectives + nested cross-validation."""
from .trainer import Trainer
from .training_args import TrainingArguments
from .objectives import (
    Objective,
    SupervisedObjective,
    TeacherForcingObjective,
    BehavioralObjective,
    VariationalObjective,
)
from .cv import (
    config_combination,
    behavior_cv_training,
    find_best_models_for_exp,
    CVResult,
)

__all__ = [
    "Trainer",
    "TrainingArguments",
    "Objective",
    "SupervisedObjective",
    "TeacherForcingObjective",
    "BehavioralObjective",
    "VariationalObjective",
    "config_combination",
    "behavior_cv_training",
    "find_best_models_for_exp",
    "CVResult",
]
