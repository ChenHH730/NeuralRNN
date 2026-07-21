"""Tests for the neurogym dataset layer (NeurogymDataset + load_dataset passthrough).

Covers the neurogym 1.x / 2.x compatibility layer: unwrapped-env construction (no gymnasium
wrapper-attribute deprecation warnings), Discrete vs Box action spaces, dynamic passthrough of
arbitrary env ids through load_dataset, and seed reproducibility.
"""
import warnings

import pytest
import torch

pytest.importorskip("neurogym")

from neuralrnn import load_dataset
from neuralrnn.data import list_neurogym_datasets, neurogym_version
from neuralrnn.data.neurogym_dataset import NeurogymDataset, _resolve_task_id

WRAPPER_ATTR_MSG = "to get variables from other wrappers is deprecated"


def _assert_clean(fn):
    """Run fn, asserting no gymnasium wrapper-attribute deprecation warning is emitted."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        result = fn()
    bad = [w for w in rec if WRAPPER_ATTR_MSG in str(w.message)]
    assert not bad, f"wrapper-attribute warnings leaked: {[str(w.message) for w in bad]}"
    return result


class TestHelpers:
    def test_neurogym_version(self):
        assert isinstance(neurogym_version(), str)

    def test_list_neurogym_datasets(self):
        envs = list_neurogym_datasets()
        assert "PerceptualDecisionMaking-v0" in envs
        assert "DelayComparison-v0" in envs

    def test_resolve_task_id(self):
        assert _resolve_task_id("PerceptualDecisionMaking-v0") == "PerceptualDecisionMaking-v0"
        assert _resolve_task_id("PerceptualDecisionMaking") == "PerceptualDecisionMaking-v0"
        assert _resolve_task_id("gonogo") == "GoNogo-v0"
        # Unknown names pass through unchanged (error is raised at make time)
        assert _resolve_task_id("no_such_env") == "no_such_env"


class TestFromTask:
    def test_discrete_batch_and_unwrapped_env(self):
        def run():
            ds = NeurogymDataset.from_task("PerceptualDecisionMaking-v0",
                                           batch_size=4, seq_len=50, dt=100)
            batch = ds.sample_batch()
            # Analysis-time env access pattern used by the notebooks
            env = ds.env
            env.new_trial()
            _ = env.ob, env.gt, env.dt, env.trial
            _ = env.start_ind, env.end_ind
            return ds, batch

        ds, batch = _assert_clean(run)
        assert batch["inputs"].shape == (4, 50, ds.input_dim)
        assert batch["targets"].shape == (4, 50)
        assert batch["targets"].dtype == torch.long
        assert batch["mask"] is None
        assert ds.output_type == "discrete"
        # self.env must be the bare TrialEnv, not a gymnasium wrapper
        assert type(ds.env).__name__ == "PerceptualDecisionMaking"

    def test_continuous_action_space(self):
        ds = NeurogymDataset.from_task("ReachingDelayResponse-v0", batch_size=2, seq_len=30)
        batch = ds.sample_batch()
        assert ds.output_type == "continuous"
        assert batch["targets"].dtype == torch.float32
        assert batch["targets"].shape == (2, 30, ds.output_dim)

    def test_seed_reproducibility(self):
        a = load_dataset("perceptual_decision_making", batch_size=4, seq_len=50, seed=0)
        b = load_dataset("perceptual_decision_making", batch_size=4, seq_len=50, seed=0)
        for _ in range(3):
            ba, bb = a.sample_batch(), b.sample_batch()
            assert torch.equal(ba["inputs"], bb["inputs"])
            assert torch.equal(ba["targets"], bb["targets"])


class TestLoadDatasetPassthrough:
    def test_passthrough_exact_id(self):
        ds = load_dataset("GoNogo-v0", batch_size=2, seq_len=30)
        assert isinstance(ds, NeurogymDataset)
        assert ds.sample_batch()["inputs"].shape[0] == 2

    def test_passthrough_case_insensitive(self):
        ds = load_dataset("gonogo", batch_size=2, seq_len=30)
        assert isinstance(ds, NeurogymDataset)

    def test_registered_names_win_over_passthrough(self):
        # 'go_nogo' is a built-in cognitive_task and must not be shadowed by neurogym's GoNogo-v0
        from neuralrnn.data import CognitiveTaskDataset
        ds = load_dataset("go_nogo", n_steps=30, batch_size=2)
        assert isinstance(ds, CognitiveTaskDataset)

    def test_unknown_name_raises_keyerror(self):
        with pytest.raises(KeyError, match="not registered"):
            load_dataset("definitely_not_a_task_xyz")

    def test_existing_entries_still_work(self):
        ds = load_dataset("perceptual_decision_making", batch_size=2, seq_len=30)
        assert isinstance(ds, NeurogymDataset)
        ds2 = load_dataset("delay_comparison", batch_size=2, seq_len=30)
        assert isinstance(ds2, NeurogymDataset)


class TestTrialAlignedMode:
    """n_trials=... gives the CognitiveTaskDataset-style interface (whole trials, padded)."""

    def test_attributes_shapes_and_mask(self):
        ds = load_dataset("perceptual_decision_making", n_trials=20, batch_size=4, seed=0)
        n, t_max = ds.inputs.shape[0], ds.inputs.shape[1]
        assert (n, len(ds.conditions), len(ds)) == (20, 20, 20)
        assert ds.inputs.shape == (20, t_max, ds.input_dim)
        assert ds.targets.shape == (20, t_max)
        assert ds.targets.dtype == torch.long
        assert ds.mask.shape == (20, t_max)
        assert ds.dt is not None
        # mask marks exactly the valid steps of each trial
        for i, cond in enumerate(ds.conditions):
            assert ds.mask[i].sum().item() == cond["n_steps"]
            assert ds.mask[i, cond["n_steps"]:].sum().item() == 0

    def test_conditions_fields(self):
        ds = load_dataset("perceptual_decision_making", n_trials=5, seed=0)
        cond = ds.conditions[0]
        # native neurogym trial keys are preserved, plus unified extras
        assert "ground_truth" in cond and "coh" in cond
        assert cond["n_steps"] > 0
        assert "stimulus" in cond["epochs"]
        start, end = cond["epochs"]["stimulus"]
        assert 0 <= start <= end <= cond["n_steps"]

    def test_variable_timing_pads_to_max(self):
        # default PDM timing is stochastic -> trials have different lengths
        ds = load_dataset("perceptual_decision_making", n_trials=30, seed=1)
        assert max(c["n_steps"] for c in ds.conditions) == ds.inputs.shape[1]

    def test_constant_timing_uniform_length(self):
        ds = load_dataset("perceptual_decision_making", n_trials=10, dt=100,
                          timing={"fixation": ("constant", 500), "stimulus": ("constant", 500)})
        lengths = {c["n_steps"] for c in ds.conditions}
        assert len(lengths) == 1
        assert lengths.pop() == ds.inputs.shape[1]

    def test_sample_batch_whole_trials(self):
        ds = load_dataset("perceptual_decision_making", n_trials=20, batch_size=4, seed=0)
        batch = ds.sample_batch()
        t_max = ds.inputs.shape[1]
        assert batch["inputs"].shape == (4, t_max, ds.input_dim)
        assert batch["targets"].shape == (4, t_max)
        assert batch["mask"].shape == (4, t_max)
        # batch rows come from the stored trials
        for row in batch["inputs"]:
            assert any(torch.equal(row, trial) for trial in ds.inputs)

    def test_get_all_trials(self):
        ds = load_dataset("perceptual_decision_making", n_trials=8, seed=0)
        all_trials = ds.get_all_trials()
        assert sorted(all_trials.keys()) == ["inputs", "mask", "targets"]
        assert torch.equal(all_trials["inputs"], ds.inputs)

    def test_continuous_action_space(self):
        ds = load_dataset("ReachingDelayResponse-v0", n_trials=5)
        assert ds.output_type == "continuous"
        assert ds.targets.dtype == torch.float32
        assert ds.targets.shape == (5, ds.inputs.shape[1], ds.output_dim)

    def test_seed_reproducibility(self):
        a = load_dataset("perceptual_decision_making", n_trials=8, seed=3)
        b = load_dataset("perceptual_decision_making", n_trials=8, seed=3)
        assert torch.equal(a.inputs, b.inputs)
        assert a.conditions == b.conditions

    def test_mask_works_with_masked_cross_entropy(self):
        from neuralrnn import masked_cross_entropy
        ds = load_dataset("perceptual_decision_making", n_trials=10, batch_size=4, seed=0)
        batch = ds.sample_batch()
        logits = torch.randn(4, ds.inputs.shape[1], ds.output_dim)
        loss = masked_cross_entropy(logits, batch["targets"], batch["mask"])
        assert loss.isfinite()

    def test_streaming_mode_rejects_trial_interface(self):
        ds = load_dataset("perceptual_decision_making", batch_size=2, seq_len=30)
        with pytest.raises(RuntimeError, match="n_trials"):
            ds.get_all_trials()
        with pytest.raises(RuntimeError, match="n_trials"):
            len(ds)

    def test_trial_aligned_no_wrapper_warnings(self):
        _assert_clean(lambda: load_dataset("perceptual_decision_making", n_trials=10, seed=0))
