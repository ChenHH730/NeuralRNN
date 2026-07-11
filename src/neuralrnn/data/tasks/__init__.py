"""Cognitive task generators for Paradigm A (task-optimized RNNs).

Each task module exposes a generate_trials() function that returns:
    (inputs, targets, mask, conditions)

where:
    inputs:     (n_trials, n_t, input_dim) — task input tensor
    targets:    (n_trials, n_t, output_dim) — full-length target sequence
    mask:       (n_trials, n_t, output_dim) boolean/float — loss mask
    conditions: list of dicts — trial condition metadata

All tasks now return full-length targets and a boolean/float mask of the same
shape, so they can be consumed uniformly by CognitiveTaskDataset and Trainer.
"""
from .mante_task import generate_trials as mante_trials
from .dms_continuous_task import generate_trials as dms_continuous_trials
from .wm_angle_task import generate_trials as wm_angle_trials
from .wm_frequency_task import generate_trials as wm_frequency_trials
from .rdm_task import generate_trials as rdm_trials
from .raposo_task import generate_trials as raposo_trials
from .dms_task import generate_trials as dms_trials
from .lr_mante_task import generate_trials as lr_mante_trials
from .go_nogo_task import generate_trials as go_nogo_trials
from .multitask_yang_task import generate_trials as multitask_yang_trials
from .multitask_flexible_task import generate_trials as multitask_flexible_trials
from .multitask_flexible_dataset import MultitaskFlexibleDataset

TASK_REGISTRY = {
    # Context-dependent decision making
    "mante": mante_trials,
    "siegel_miller": mante_trials,  # backward-compatible alias
    # Perceptual decision making / evidence accumulation
    "rdm": rdm_trials,
    "two_afc": rdm_trials,  # backward-compatible alias
    # Delayed match-to-sample
    "dms": dms_trials,
    "dms_continuous": dms_continuous_trials,
    # Parametric working memory
    "wm_angle": wm_angle_trials,
    "parametric_wm": wm_angle_trials,  # backward-compatible alias
    "wm_frequency": wm_frequency_trials,
    "romo": wm_frequency_trials,  # backward-compatible alias
    # Multisensory / context-dependent decision making
    "raposo": raposo_trials,
    "lr_mante": lr_mante_trials,
    # Go/NoGo and Memory Number (ActivationMattersRNN)
    "go_nogo": go_nogo_trials,
    # Multitask families
    "multitask_yang": multitask_yang_trials,
    "multitask_flexible": multitask_flexible_trials,
}

__all__ = [
    "TASK_REGISTRY",
    "mante_trials",
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
]
