"""Collection of objectives: paradigm differences are encapsulated here (see ARCHITECTURE §4)."""
from .base import Objective
from .registry import (
    OBJECTIVE_REGISTRY,
    register_objective,
    build_objective,
    AutoObjective,
)
from .supervised import SupervisedObjective
from .regularized_supervised import RegularizedSupervisedObjective
from .teacher_forcing import TeacherForcingObjective, generalized_teacher_forcing
from .behavioral import BehavioralObjective
from .variational import VariationalObjective
from .latent_circuit import LatentCircuitObjective
from .constrained import ConstrainedSupervisedObjective

__all__ = [
    "Objective",
    "OBJECTIVE_REGISTRY",
    "register_objective",
    "build_objective",
    "AutoObjective",
    "SupervisedObjective",
    "RegularizedSupervisedObjective",
    "TeacherForcingObjective",
    "generalized_teacher_forcing",
    "BehavioralObjective",
    "VariationalObjective",
    "LatentCircuitObjective",
    "ConstrainedSupervisedObjective",
]
