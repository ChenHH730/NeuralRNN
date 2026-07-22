"""Tests for ReconstructionDataset and ReconstructionObjective.

Covers:
- Numerical equivalence with the legacy LatentCircuitObjective (guarded;
  auto-skips once the legacy module is removed)
- behavior_weight=0 (activity-only) and activity_weight=0 paths
- identity / embedding / auto state maps
- recorded_dims subset selection (partial recording)
- NMSE vs MSE activity loss math
- mask handling
- activity_fn="firing_rates" (gain_rnn)
- gradient flow
- ReconstructionDataset construction, sampling, and registry wiring
"""
import pytest

torch = pytest.importorskip("torch")

from neuralrnn import AutoConfig, AutoModel
from neuralrnn.data.reconstruction_dataset import ReconstructionDataset
from neuralrnn.train.objectives.reconstruction import ReconstructionObjective

try:
    from neuralrnn.train.objectives.latent_circuit import LatentCircuitObjective
except ImportError:  # legacy module removed
    LatentCircuitObjective = None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_rnn():
    """Create a small CTRNN teacher."""
    cfg = AutoConfig.for_model(
        "ctrnn", input_dim=6, latent_dim=20, output_dim=2,
        dt=40.0, tau=200.0, dale=True, sigma_rec=0.15,
    )
    model = AutoModel.from_config(cfg)
    model.eval()
    return model


def _make_lc():
    """Create a small latent circuit student."""
    cfg = AutoConfig.for_model(
        "latent_circuit", input_dim=6, latent_dim=4, output_dim=2,
        embedding_dim=20, dt=40.0, tau=200.0, sigma_rec=0.0,
    )
    return AutoModel.from_config(cfg)


def _make_gain_rnn():
    """Create a small gain_rnn (rate mode) teacher/student."""
    cfg = AutoConfig.for_model(
        "gain_rnn", input_dim=4, latent_dim=16, output_dim=2,
        dt=0.05, tau=1.0, nonlinearity_mode="rate", sigma_rec=0.0,
    )
    model = AutoModel.from_config(cfg)
    model.eval()
    return model


def _make_task_ds():
    from neuralrnn.data.cognitive_task_dataset import CognitiveTaskDataset
    return CognitiveTaskDataset.from_task(
        "mante", n_reps=2, n_coh=2, batch_size=4
    )


def _make_recon_ds(record_targets=True, **kwargs):
    rnn = _make_rnn()
    task_ds = _make_task_ds()
    return ReconstructionDataset.from_rnn_and_task(
        rnn, task_ds, batch_size=4, record_targets=record_targets, **kwargs
    )


def _same_batch(ds):
    torch.manual_seed(0)
    return ds.sample_batch()


# ---------------------------------------------------------------------------
# Equivalence with the legacy LatentCircuitObjective
# ---------------------------------------------------------------------------

@pytest.mark.skipif(LatentCircuitObjective is None, reason="legacy objective removed")
def test_matches_legacy_latent_circuit_objective():
    torch.manual_seed(42)
    model = _make_lc()
    # Use a mask-free dataset: the legacy objective never applied a mask.
    rnn = _make_rnn()
    inputs = torch.randn(8, 15, 6)

    class ToyDS:
        def __init__(self, inputs):
            self.inputs = inputs

    ds = ReconstructionDataset.from_rnn_and_task(
        rnn, ToyDS(inputs), batch_size=4
    )
    batch = _same_batch(ds)
    assert "mask" not in batch
    legacy_batch = {
        "inputs": batch["inputs"],
        "targets": batch["targets"],
        "rnn_states": batch["activity"],
    }

    new_obj = ReconstructionObjective(state_map="embedding", activity_weight=1.0)
    old_obj = LatentCircuitObjective(l_y=1.0)

    loss_new, logs_new = new_obj.compute_loss(model, batch)
    loss_old, logs_old = old_obj.compute_loss(model, legacy_batch)

    assert torch.allclose(loss_new, loss_old, atol=0)
    assert logs_new["mse_z"] == pytest.approx(logs_old["mse_z"])
    assert logs_new["nmse_y"] == pytest.approx(logs_old["nmse_y"])


# ---------------------------------------------------------------------------
# Objective behavior
# ---------------------------------------------------------------------------

class TestReconstructionObjective:
    def test_default_matches_latent_circuit_recipe(self):
        model = _make_lc()
        ds = _make_recon_ds()
        obj = ReconstructionObjective(state_map="embedding")
        loss, logs = obj.compute_loss(model, _same_batch(ds))
        assert loss.isfinite()
        assert logs["mse_z"] >= 0
        assert logs["nmse_y"] >= 0
        assert logs["loss"] == pytest.approx(logs["mse_z"] + logs["nmse_y"])

    def test_behavior_weight_zero_skips_targets(self):
        model = _make_lc()
        ds = _make_recon_ds(record_targets=False)
        batch = _same_batch(ds)
        assert "targets" not in batch
        obj = ReconstructionObjective(
            behavior_weight=0.0, state_map="embedding"
        )
        loss, logs = obj.compute_loss(model, batch)
        assert "mse_z" not in logs
        assert "nmse_y" in logs

    def test_behavior_weight_requires_targets(self):
        model = _make_lc()
        ds = _make_recon_ds(record_targets=False)
        obj = ReconstructionObjective(behavior_weight=1.0)
        with pytest.raises(KeyError):
            obj.compute_loss(model, _same_batch(ds))

    def test_identity_mse_matches_manual(self):
        torch.manual_seed(0)
        student = _make_rnn()
        teacher = _make_rnn()
        inputs = torch.randn(3, 10, 6)
        with torch.no_grad():
            activity = teacher(inputs).states
        ds = ReconstructionDataset(inputs=inputs, activity=activity, batch_size=3)
        torch.manual_seed(1)
        batch = ds.sample_batch()
        obj = ReconstructionObjective(
            behavior_weight=0.0, state_map="identity", activity_loss="mse"
        )
        loss, logs = obj.compute_loss(student, batch)
        with torch.no_grad():
            manual = ((student(batch["inputs"]).states - batch["activity"]) ** 2).mean()
        assert loss.item() == pytest.approx(manual.item())
        assert "mse_y" in logs

    def test_recorded_dims_subset(self):
        torch.manual_seed(0)
        student = _make_rnn()
        teacher = _make_rnn()
        inputs = torch.randn(3, 10, 6)
        with torch.no_grad():
            activity = teacher(inputs).states
        batch = {"inputs": inputs, "activity": activity}
        idx = [0, 3, 7, 11]
        obj = ReconstructionObjective(
            behavior_weight=0.0, state_map="identity",
            activity_loss="mse", recorded_dims=idx,
        )
        loss, _ = obj.compute_loss(student, batch)
        with torch.no_grad():
            states = student(inputs).states
            manual = ((states[..., idx] - activity[..., idx]) ** 2).mean()
        assert loss.item() == pytest.approx(manual.item())

    def test_nmse_denominator_math(self):
        torch.manual_seed(0)
        student = _make_rnn()
        inputs = torch.randn(3, 10, 6)
        activity = torch.randn(3, 10, 20) * 5.0
        batch = {"inputs": inputs, "activity": activity}
        obj = ReconstructionObjective(
            behavior_weight=0.0, state_map="identity", activity_loss="nmse"
        )
        loss, _ = obj.compute_loss(student, batch)
        with torch.no_grad():
            states = student(inputs).states
            a_bar = activity - activity.mean(dim=[0, 1], keepdim=True)
            denom = (a_bar ** 2).mean().clamp_min(1e-8)
            manual = ((states - activity) ** 2).mean() / denom
        assert loss.item() == pytest.approx(manual.item())

    def test_state_map_auto(self):
        lc = _make_lc()
        rnn = _make_rnn()
        obj = ReconstructionObjective(state_map="auto")
        assert obj._map_states.__self__ is obj  # bound
        # auto -> embedding for latent circuit
        s = torch.randn(2, 5, 4)
        mapped = obj._map_states(lc, s)
        assert mapped.shape[-1] == 20  # embedding_dim
        # auto -> identity for ctrnn
        s2 = torch.randn(2, 5, 20)
        assert obj._map_states(rnn, s2) is s2

    def test_embedding_requires_embedding_matrix(self):
        rnn = _make_rnn()
        obj = ReconstructionObjective(
            behavior_weight=0.0, state_map="embedding"
        )
        batch = {"inputs": torch.randn(2, 5, 6), "activity": torch.randn(2, 5, 20)}
        with pytest.raises(ValueError):
            obj.compute_loss(rnn, batch)

    def test_dim_mismatch_raises(self):
        rnn = _make_rnn()
        obj = ReconstructionObjective(behavior_weight=0.0, state_map="identity")
        batch = {"inputs": torch.randn(2, 5, 6), "activity": torch.randn(2, 5, 7)}
        with pytest.raises(ValueError):
            obj.compute_loss(rnn, batch)

    def test_mask_activity_term(self):
        torch.manual_seed(0)
        student = _make_rnn()
        inputs = torch.randn(3, 10, 6)
        activity = torch.randn(3, 10, 20)
        mask = torch.ones(3, 10)
        mask[:, 5:] = 0.0
        batch = {"inputs": inputs, "activity": activity, "mask": mask}
        obj = ReconstructionObjective(
            behavior_weight=0.0, state_map="identity", activity_loss="mse"
        )
        loss, _ = obj.compute_loss(student, batch)
        with torch.no_grad():
            states = student(inputs).states
            err = (states - activity) ** 2  # (B, T, N)
            manual = err[:, :5].mean()
        assert loss.item() == pytest.approx(manual.item(), rel=1e-5)

    def test_firing_rates_activity_fn(self):
        torch.manual_seed(0)
        teacher = _make_gain_rnn()
        student = _make_gain_rnn()
        inputs = torch.randn(2, 8, 4)
        with torch.no_grad():
            activity = teacher.get_firing_rates(teacher(inputs).states)
        batch = {"inputs": inputs, "activity": activity}
        obj = ReconstructionObjective(
            behavior_weight=0.0, state_map="identity",
            activity_fn="firing_rates", activity_loss="mse",
        )
        loss, _ = obj.compute_loss(student, batch)
        with torch.no_grad():
            states = student(inputs).states
            manual = ((student.get_firing_rates(states) - activity) ** 2).mean()
        assert loss.item() == pytest.approx(manual.item())

    def test_gradient_flows(self):
        model = _make_lc()
        ds = _make_recon_ds()
        obj = ReconstructionObjective(state_map="embedding")
        loss, _ = obj.compute_loss(model, _same_batch(ds))
        loss.backward()
        assert model.w_rec.weight.grad is not None
        assert model.w_in.weight.grad is not None
        assert model.w_out.weight.grad is not None

    def test_zero_weights_raise(self):
        obj = ReconstructionObjective(behavior_weight=0.0, activity_weight=0.0)
        with pytest.raises(ValueError):
            obj.compute_loss(_make_lc(), {"inputs": torch.randn(1, 2, 6)})

    def test_registry(self):
        from neuralrnn import AutoObjective, build_objective
        obj = build_objective("reconstruction", behavior_weight=0.0)
        assert isinstance(obj, ReconstructionObjective)
        obj2 = AutoObjective.from_name("reconstruction", state_map="identity")
        assert obj2.state_map == "identity"
        obj3 = build_objective({"name": "reconstruction", "activity_loss": "mse"})
        assert obj3.activity_loss == "mse"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TestReconstructionDataset:
    def test_from_rnn_and_task(self):
        ds = _make_recon_ds()
        assert ds.kind == "reconstruction"
        assert ds.inputs.ndim == 3
        assert ds.targets.ndim == 3
        assert ds.activity.ndim == 3
        assert ds.inputs.shape[0] == ds.activity.shape[0] == len(ds)
        assert ds.input_dim == ds.inputs.shape[-1]
        assert ds.output_dim == ds.targets.shape[-1]

    def test_sample_batch_keys(self):
        ds = _make_recon_ds()
        batch = ds.sample_batch()
        assert set(batch) >= {"inputs", "activity", "targets"}
        assert batch["inputs"].shape[0] == 4

    def test_activity_only(self):
        ds = _make_recon_ds(record_targets=False)
        assert ds.targets is None
        assert ds.output_dim == 0
        batch = ds.sample_batch()
        assert "targets" not in batch

    def test_mask_propagated(self):
        ds = _make_recon_ds()
        # CognitiveTaskDataset provides a mask via get_all_trials
        if ds.mask is not None:
            batch = ds.sample_batch()
            assert "mask" in batch
            assert batch["mask"].shape[:2] == batch["inputs"].shape[:2]

    def test_activity_fn_firing_rates(self):
        teacher = _make_gain_rnn()
        task_ds = _make_task_ds()
        # gain_rnn input_dim=4, but mante has 6 inputs; use raw arrays instead
        inputs = torch.randn(5, 12, 4)

        class ToyDS:
            def __init__(self, inputs):
                self.inputs = inputs

        ds = ReconstructionDataset.from_rnn_and_task(
            teacher, ToyDS(inputs), batch_size=2, activity_fn="firing_rates",
            record_targets=False,
        )
        with torch.no_grad():
            expected = teacher.get_firing_rates(teacher(inputs).states)
        assert torch.allclose(ds.activity, expected)
        assert ds.targets is None

    def test_fallback_without_get_all_trials(self):
        rnn = _make_rnn()
        inputs = torch.randn(5, 12, 6)

        class AttrOnlyDS:
            def __init__(self, inputs):
                self.inputs = inputs
                self.mask = torch.ones(inputs.shape[0], inputs.shape[1])

        ds = ReconstructionDataset.from_rnn_and_task(
            rnn, AttrOnlyDS(inputs), batch_size=2
        )
        assert ds.inputs.shape == inputs.shape
        assert ds.mask is not None
        batch = ds.sample_batch()
        assert "mask" in batch
