"""目标函数集合：范式差异封装在此（见 ARCHITECTURE §4）。"""
from .base import Objective
from .supervised import SupervisedObjective
from .teacher_forcing import TeacherForcingObjective, generalized_teacher_forcing
from .behavioral import BehavioralObjective
from .variational import VariationalObjective

__all__ = [
    "Objective",
    "SupervisedObjective",
    "TeacherForcingObjective",
    "generalized_teacher_forcing",
    "BehavioralObjective",
    "VariationalObjective",
]
