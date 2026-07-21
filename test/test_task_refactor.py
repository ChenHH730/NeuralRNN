"""Tests for the data-layer refactor (docs/DATA_REFACTOR.md):

* unified Task base class + registries (TASK_CLASSES / aliases / shims)
* deprecated task names & kwargs (warn but keep working)
* unified condition schema (epochs / n_steps / is_catch)
* rdm catch-trial bug fix
* CognitiveTaskDataset streaming mode
* sample_trials (cognitive + neurogym datasets)
* visualization.plot_trials
"""
import warnings

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest
import torch

from neuralrnn.data import CognitiveTaskDataset, Trials, load_dataset
from neuralrnn.data.tasks import (
    TASK_CLASSES, TASK_REGISTRY, TASK_ALIASES, resolve_task_name,
)
from neuralrnn import visualization as viz

UNIFIED_KEYS = {"epochs", "n_steps", "is_catch"}


# --------------------------------------------------------------------------- registry

class TestTaskRegistry:
    def test_canonical_names(self):
        assert set(TASK_CLASSES) == {
            "mante", "mante2", "rdm", "raposo", "dms", "dms_continuous",
            "wm_angle", "wm_frequency", "go_nogo", "checkerboard",
            "multitask_yang", "multitask_flexible",
        }

    def test_aliases(self):
        assert TASK_ALIASES == {
            "siegel_miller": "mante", "lr_mante": "mante2", "two_afc": "rdm",
            "parametric_wm": "wm_angle", "romo": "wm_frequency",
        }

    def test_resolve_alias_warns(self):
        with pytest.warns(DeprecationWarning, match="lr_mante"):
            assert resolve_task_name("lr_mante") == "mante2"

    def test_resolve_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown task"):
            resolve_task_name("not_a_task")

    def test_task_registry_backward_compatible(self):
        # old name -> callable returning the 4-tuple
        for name in ("mante", "siegel_miller", "rdm", "two_afc", "lr_mante",
                     "mante2", "dms", "dms_continuous", "wm_angle",
                     "parametric_wm", "wm_frequency", "romo", "raposo",
                     "go_nogo", "checkerboard"):
            assert callable(TASK_REGISTRY[name])


# --------------------------------------------------------------------------- task classes

class TestTaskClasses:
    def test_unified_condition_keys_all_tasks(self):
        for name, cls in TASK_CLASSES.items():
            kwargs = {"rule": "fdgo"} if name.startswith("multitask") else {}
            if name in ("mante", "dms_continuous"):
                kwargs["n_reps"] = 1
            elif name == "go_nogo":
                kwargs["n_values"] = 4
            else:
                kwargs["n_trials"] = 4
            kwargs["seed"] = 0
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                inputs, targets, mask, conditions = cls(**kwargs).generate_trials()
            assert isinstance(inputs, torch.Tensor), name
            assert inputs.shape[0] == targets.shape[0] == mask.shape[0] == len(conditions), name
            assert all(UNIFIED_KEYS <= set(c) for c in conditions), name

    def test_task_dims_match_class_attrs(self):
        for name, cls in TASK_CLASSES.items():
            kwargs = {"rule": "fdgo"} if name.startswith("multitask") else {}
            if name in ("mante", "dms_continuous"):
                kwargs["n_reps"] = 1
            elif name == "go_nogo":
                kwargs["n_values"] = 2
            else:
                kwargs["n_trials"] = 2
            kwargs["seed"] = 0
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                inputs, targets, _, _ = cls(**kwargs).generate_trials()
            assert inputs.shape[-1] == cls.input_dim, name
            assert targets.shape[-1] == cls.output_dim, name

    def test_deprecated_kwargs_warn_and_map(self):
        with pytest.warns(DeprecationWarning):
            out = TASK_REGISTRY["rdm"](num_trials=5, std=0.1,
                                       fraction_catch_trials=0.0, seed=0)
        assert out[0].shape[0] == 5

    def test_conflicting_kwargs_raise(self):
        with pytest.raises(TypeError, match="both"):
            TASK_REGISTRY["rdm"](num_trials=5, n_trials=5)

    def test_mante_n_reps_semantics(self):
        with pytest.warns(DeprecationWarning, match="n_trials"):
            inputs, _, _, conditions = TASK_REGISTRY["mante"](n_trials=2, n_coh=3)
        # legacy n_trials = per-condition reps: 2 reps x 2 contexts x 3x3 cohs
        assert inputs.shape[0] == 2 * 2 * 9
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            inputs2, _, _, _ = TASK_REGISTRY["mante"](n_reps=2, n_coh=3)
        assert inputs2.shape == inputs.shape

    def test_rdm_catch_bug_fixed(self):
        # Pre-refactor, fraction_catch_trials>0 could crash (UnboundLocalError)
        # and produced conditions inconsistent with the actual trial.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            _, targets, _, conditions = TASK_REGISTRY["rdm"](
                num_trials=300, fraction_catch_trials=0.4, seed=3)
        n_catch = sum(c["is_catch"] for c in conditions)
        assert n_catch > 0
        for i, c in enumerate(conditions):
            if c["is_catch"]:
                assert c["coherence"] == 0.0
                assert float(targets[i].abs().sum()) == 0.0
            else:
                assert c["coherence"] != 0.0

    def test_mante2_alias_equivalent(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            a = TASK_REGISTRY["lr_mante"](num_trials=10, seed=0)
            torch.manual_seed(0)
            b = TASK_REGISTRY["mante2"](n_trials=10, seed=0)
        # same seed: numpy-driven parts identical (torch noise differs unless
        # the global torch RNG is aligned; conditions/signals match)
        assert [c.get("coh_color") for c in a[3]] == [c.get("coh_color") for c in b[3]]


# --------------------------------------------------------------------------- dataset modes

class TestCognitiveTaskDatasetModes:
    def test_aligned_default_unchanged(self):
        ds = CognitiveTaskDataset.from_task("mante2", batch_size=8, n_trials=20, seed=0)
        assert ds.mode == "aligned"
        assert ds.inputs.shape == (20, 68, 4)
        assert len(ds) == 20
        b = ds.sample_batch()
        assert b["inputs"].shape == (8, 68, 4) and b["mask"] is not None
        all_t = ds.get_all_trials()
        assert sorted(all_t.keys()) == ["inputs", "mask", "targets"]

    def test_streaming_batch(self):
        ds = CognitiveTaskDataset.from_task("mante2", batch_size=8, n_trials=50,
                                            mode="streaming", seq_len=150, seed=0)
        b = ds.sample_batch()
        assert b["inputs"].shape == (8, 150, 4)
        assert b["targets"].shape == (8, 150, 1)
        assert b["mask"] is None

    def test_streaming_blocks_aligned_interface(self):
        ds = CognitiveTaskDataset.from_task("rdm", n_trials=10, mode="streaming")
        for attr in ("inputs", "targets", "mask", "conditions"):
            with pytest.raises(RuntimeError, match="streaming"):
                getattr(ds, attr)
        with pytest.raises(RuntimeError, match="streaming"):
            ds.get_all_trials()
        with pytest.raises(RuntimeError, match="streaming"):
            len(ds)

    def test_streaming_reproducible(self):
        torch.manual_seed(0)
        ds1 = CognitiveTaskDataset.from_task("mante2", batch_size=4, n_trials=50,
                                             mode="streaming", seq_len=100, seed=0)
        b1 = ds1.sample_batch()
        torch.manual_seed(0)
        ds2 = CognitiveTaskDataset.from_task("mante2", batch_size=4, n_trials=50,
                                             mode="streaming", seq_len=100, seed=0)
        b2 = ds2.sample_batch()
        assert torch.equal(b1["inputs"], b2["inputs"])

    def test_streaming_variable_length_pool(self):
        # checkerboard train mode has variable-length trials; streaming must
        # trim to true lengths (n_steps) so no padding zeros inside windows.
        ds = CognitiveTaskDataset.from_task("checkerboard", n_trials=20,
                                            mode="streaming", seq_len=500,
                                            batch_size=2, seed=0)
        b = ds.sample_batch()
        assert b["inputs"].shape == (2, 500, 4)

    def test_bad_mode_raises(self):
        with pytest.raises(ValueError, match="mode"):
            CognitiveTaskDataset.from_task("rdm", n_trials=5, mode="bogus")


# --------------------------------------------------------------------------- sample_trials

class TestSampleTrials:
    def test_cognitive_aligned(self):
        ds = CognitiveTaskDataset.from_task("mante2", n_trials=20, seed=0)
        tr = ds.sample_trials(5)
        assert isinstance(tr, Trials)
        assert tr.inputs.shape == (5, 68, 4)
        assert len(tr.conditions) == 5
        assert torch.equal(tr.inputs, ds.inputs[:5])

    def test_cognitive_aligned_seeded(self):
        ds = CognitiveTaskDataset.from_task("mante2", n_trials=20, seed=0)
        assert torch.equal(ds.sample_trials(5, seed=1).inputs,
                           ds.sample_trials(5, seed=1).inputs)

    def test_cognitive_streaming(self):
        ds = CognitiveTaskDataset.from_task("checkerboard", n_trials=20,
                                            mode="streaming", seq_len=200, seed=0)
        tr = ds.sample_trials(3, seed=0)
        assert tr.inputs.shape[0] == 3 and len(tr.conditions) == 3
        # mask marks exactly n_steps valid steps per trial (any output channel)
        for i in range(3):
            valid = tr.mask[i].reshape(tr.mask.shape[1], -1).any(dim=-1).sum()
            assert int(valid.item()) == tr.conditions[i]["n_steps"]

    def test_base_dataset_raises(self):
        from neuralrnn.data import BaseDataset
        with pytest.raises(NotImplementedError):
            BaseDataset().sample_trials(1)


# --------------------------------------------------------------------------- plot_trials

class TestPlotTrials:
    def test_plot_trials_inputs(self):
        ds = CognitiveTaskDataset.from_task("mante2", n_trials=10, seed=0)
        tr = ds.sample_trials(4)
        fig, axes = viz.plot_trials(tr, n=2, dt=20)
        assert len(axes) == 2
        fig2, axes2 = viz.plot_trials(ds, n=3)
        assert len(axes2) == 3
        fig3, axes3 = viz.plot_trials(
            {"inputs": tr.inputs, "targets": tr.targets, "conditions": tr.conditions})
        assert len(axes3) == 4  # default: all 4 trials

    def test_plot_trials_rejects_bad_input(self):
        with pytest.raises(TypeError):
            viz.plot_trials(42)


# --------------------------------------------------------------------------- neurogym

ngym = pytest.importorskip("neurogym", reason="neurogym not installed")


class TestNeurogymSampleTrials:
    def test_streaming_mode(self):
        ds = load_dataset("perceptual_decision_making", batch_size=4, seq_len=50, dt=100)
        tr = ds.sample_trials(3, seed=0)
        assert isinstance(tr, Trials)
        assert tr.inputs.shape[0] == 3
        assert all(UNIFIED_KEYS <= set(c) for c in tr.conditions)
        for i in range(3):
            assert int(tr.mask[i].sum().item()) == tr.conditions[i]["n_steps"]
        assert torch.equal(tr.inputs, ds.sample_trials(3, seed=0).inputs)
        # streaming still works afterwards
        assert ds.sample_batch()["inputs"].shape[0] == 4

    def test_aligned_mode_subset(self):
        ds = load_dataset("perceptual_decision_making", batch_size=4, n_trials=8,
                          dt=100, seed=0)
        tr = ds.sample_trials(4)
        assert torch.equal(tr.inputs, ds.inputs[:4])
        assert tr.conditions == ds.conditions[:4]

    def test_plot_trials_neurogym(self):
        ds = load_dataset("perceptual_decision_making", batch_size=4, seq_len=50, dt=100)
        fig, axes = viz.plot_trials(ds.sample_trials(2, seed=0), n=2, dt=100)
        assert len(axes) == 2
