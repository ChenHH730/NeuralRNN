"""Cognitive task library for Paradigm A (task-optimized RNNs).

Unified interface (see docs/DATA_REFACTOR.md): every task is a ``Task``
subclass registered in ``TASK_CLASSES`` by canonical name. Aliases
(``siegel_miller``, ``two_afc``, ``parametric_wm``, ``romo``, ``lr_mante``)
resolve via ``resolve_task_name`` with a DeprecationWarning.

``Task.generate_trials()`` returns:
    (inputs, targets, mask, conditions)

where:
    inputs:     (n_trials, n_t, input_dim) — task input tensor
    targets:    (n_trials, n_t, output_dim) — full-length target sequence
    mask:       (n_trials, n_t, output_dim) float — loss mask
    conditions: list of dicts — per-trial metadata. Unified keys: ``epochs``
        ({phase: (start, end)}), ``n_steps``, ``is_catch``; plus task-specific
        legacy keys (coherence/context/choice/...).

Backward compatibility: module-level ``*_trials`` shim functions and the
``TASK_REGISTRY`` dict (name -> callable returning the 4-tuple, aliases
included) are kept unchanged for existing code.
"""
import warnings

from .task_base import Task
from .mante_task import ManteTask, generate_trials as mante_trials
from .mante2_task import Mante2Task, generate_trials as mante2_trials
from .dms_continuous_task import DMSContinuousTask, generate_trials as dms_continuous_trials
from .wm_angle_task import WMAngleTask, generate_trials as wm_angle_trials
from .wm_frequency_task import WMFrequencyTask, generate_trials as wm_frequency_trials
from .rdm_task import RDMTask, generate_trials as rdm_trials
from .raposo_task import RaposoTask, generate_trials as raposo_trials
from .dms_task import DMSTask, generate_trials as dms_trials
from .lr_mante_task import generate_trials as lr_mante_trials  # deprecated alias shim
from .go_nogo_task import GoNogoTask, generate_trials as go_nogo_trials
from .multitask_yang_task import MultitaskYangTask, generate_trials as multitask_yang_trials
from .multitask_flexible_task import MultitaskFlexibleTask, generate_trials as multitask_flexible_trials
from .multitask_flexible_dataset import MultitaskFlexibleDataset
from .checkerboard_task import CheckerboardTask, generate_trials as checkerboard_trials

# Canonical registry: name -> Task subclass (single source of truth)
TASK_CLASSES = {
    cls.name: cls
    for cls in (
        ManteTask, Mante2Task, RDMTask, RaposoTask, DMSTask, DMSContinuousTask,
        WMAngleTask, WMFrequencyTask, GoNogoTask, CheckerboardTask,
        MultitaskYangTask, MultitaskFlexibleTask,
    )
}

# Alias -> canonical name (declared on each Task subclass)
TASK_ALIASES = {
    alias: cls.name
    for cls in TASK_CLASSES.values()
    for alias in cls.aliases
}


def resolve_task_name(name: str) -> str:
    """Resolve a task name (canonical or alias) to the canonical name.

    Raises ValueError for unknown names; warns on alias use.
    """
    if name in TASK_CLASSES:
        return name
    if name in TASK_ALIASES:
        canonical = TASK_ALIASES[name]
        warnings.warn(
            f"Task name '{name}' is deprecated, use '{canonical}' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return canonical
    raise ValueError(
        f"Unknown task '{name}'. Available: {sorted(TASK_CLASSES)} "
        f"(aliases: {sorted(TASK_ALIASES)})"
    )


# Backward-compatible registry: name -> generate_trials shim (aliases included)
TASK_REGISTRY = {
    # Context-dependent decision making
    "mante": mante_trials,
    "mante2": mante2_trials,
    "siegel_miller": mante_trials,  # deprecated alias
    "lr_mante": lr_mante_trials,    # deprecated alias of mante2
    # Perceptual decision making / evidence accumulation
    "rdm": rdm_trials,
    "two_afc": rdm_trials,  # deprecated alias
    # Delayed match-to-sample
    "dms": dms_trials,
    "dms_continuous": dms_continuous_trials,
    # Parametric working memory
    "wm_angle": wm_angle_trials,
    "parametric_wm": wm_angle_trials,  # deprecated alias
    "wm_frequency": wm_frequency_trials,
    "romo": wm_frequency_trials,  # deprecated alias
    # Multisensory / context-dependent decision making
    "raposo": raposo_trials,
    # Go/NoGo (ActivationMattersRNN)
    "go_nogo": go_nogo_trials,
    # Multitask families
    "multitask_yang": multitask_yang_trials,
    "multitask_flexible": multitask_flexible_trials,
    # Checkerboard decision making (Kleinman et al. 2025)
    "checkerboard": checkerboard_trials,
}

__all__ = [
    "Task",
    "TASK_CLASSES",
    "TASK_ALIASES",
    "resolve_task_name",
    "TASK_REGISTRY",
    "mante_trials",
    "mante2_trials",
    "dms_continuous_trials",
    "wm_angle_trials",
    "wm_frequency_trials",
    "rdm_trials",
    "raposo_trials",
    "dms_trials",
    "lr_mante_trials",
    "go_nogo_trials",
    "multitask_yang_trials",
    "multitask_flexible_trials",
    "MultitaskFlexibleDataset",
    "checkerboard_trials",
]
