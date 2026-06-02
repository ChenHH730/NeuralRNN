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

TASK_REGISTRY = {
    "siegel_miller": siegel_miller_trials,
    "mante": mante_trials,
    "mante_short": mante_short_trials,
    "two_afc": two_afc_trials,
    "delay_match_to_sample": delay_match_to_sample_trials,
    "parametric_wm": parametric_wm_trials,
}

__all__ = [
    "TASK_REGISTRY",
    "siegel_miller_trials",
    "mante_trials",
    "mante_short_trials",
    "two_afc_trials",
    "delay_match_to_sample_trials",
    "parametric_wm_trials",
]
