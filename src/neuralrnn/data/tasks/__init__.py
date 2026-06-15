"""Cognitive task generators for Paradigm A (task-optimized RNNs).

Each task module exposes a generate_trials() function that returns:
    (inputs, targets, mask, conditions)

where:
    inputs:     (n_trials, n_t, input_dim) — task input tensor
    targets:    (n_trials, n_t, output_dim) or (n_trials, n_masked, output_dim) — targets
    mask:       (n_trials, n_t, output_dim) boolean or (n_trials, n_masked, output_dim) — loss mask
    conditions: list of dicts — trial condition metadata

Some tasks (ManteTask, DelayMatchToSample) return targets pre-sliced by a training_mask.
The CognitiveTaskDataset wrapper handles both formats.
"""
from .siegel_miller_task import generate_trials as siegel_miller_trials
from .mante_task import generate_trials as mante_trials
from .mante_short_task import generate_trials as mante_short_trials
from .two_afc_task import generate_trials as two_afc_trials
from .delay_match_to_sample_task import generate_trials as delay_match_to_sample_trials
from .parametric_wm_task import generate_trials as parametric_wm_trials
from .rdm_task import generate_trials as rdm_trials
from .romo_task import generate_trials as romo_trials
from .raposo_task import generate_trials as raposo_trials
from .dms_task import generate_trials as dms_trials
from .lr_mante_task import generate_trials as lr_mante_trials

TASK_REGISTRY = {
    "siegel_miller": siegel_miller_trials,
    "mante": mante_trials,
    "mante_short": mante_short_trials,
    "two_afc": two_afc_trials,
    "delay_match_to_sample": delay_match_to_sample_trials,
    "parametric_wm": parametric_wm_trials,
    "rdm": rdm_trials,
    "romo": romo_trials,
    "raposo": raposo_trials,
    "dms": dms_trials,
    "lr_mante": lr_mante_trials,
}

__all__ = [
    "TASK_REGISTRY",
    "siegel_miller_trials",
    "mante_trials",
    "mante_short_trials",
    "two_afc_trials",
    "delay_match_to_sample_trials",
    "parametric_wm_trials",
    "rdm_trials",
    "romo_trials",
    "raposo_trials",
    "dms_trials",
    "lr_mante_trials",
]
