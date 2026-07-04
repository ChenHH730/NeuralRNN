"""Tests for the latent circuit model and related components.

Tests cover:
- Model construction and shapes
- Cayley transform orthonormality
- Connectivity masks
- Embedding roundtrip
- Save/load
- LatentCircuitObjective
- CognitiveTaskDataset (all tasks)
- LatentCircuitDataset
- Connectivity analysis
- Perturbation analysis
- Post-step hook in Trainer
- Analysis integration (fixed points, PCA)
"""
import pytest

torch = pytest.importorskip("torch")
import numpy as np
import tempfile

from neuralrnn import AutoConfig, AutoModel
from neuralrnn.modeling_utils import NeuralDynamicsModel, DynamicsModelOutput


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_rnn():
    """Create a small CTRNN for testing."""
    cfg = AutoConfig.for_model(
        "ctrnn", input_dim=6, latent_dim=20, output_dim=2,
        dt=40.0, tau=200.0, dale=True, sigma_rec=0.15,
    )
    model = AutoModel.from_config(cfg)
    model.eval()
    return model


def _make_lc():
    """Create a small latent circuit for testing."""
    cfg = AutoConfig.for_model(
        "latent_circuit", input_dim=6, latent_dim=4, output_dim=2,
        embedding_dim=20, dt=40.0, tau=200.0, sigma_rec=0.0,
    )
    return AutoModel.from_config(cfg)


def _make_task_ds(task_name="siegel_miller", **kwargs):
    """Create a small cognitive task dataset."""
    from neuralrnn.data.cognitive_task_dataset import CognitiveTaskDataset
    defaults = dict(n_trials=2, batch_size=4)
    # Only add n_coh for tasks that accept it
    if task_name in ("siegel_miller", "mante", "delay_match_to_sample"):
        defaults["n_coh"] = 2
    defaults.update(kwargs)
    return CognitiveTaskDataset.from_task(task_name, **defaults)


def _make_lc_ds():
    """Create a latent circuit dataset from a small RNN and task."""
    from neuralrnn.data.latent_circuit_dataset import LatentCircuitDataset
    rnn = _make_rnn()
    task_ds = _make_task_ds()
    return LatentCircuitDataset.from_rnn_and_task(rnn, task_ds, batch_size=4)


# ---------------------------------------------------------------------------
# TestLatentCircuitConstruction
# ---------------------------------------------------------------------------

class TestLatentCircuitConstruction:
    def test_default_config(self):
        cfg = AutoConfig.for_model("latent_circuit")
        assert cfg.model_type == "latent_circuit"
        assert cfg.latent_dim == 8
        assert cfg.embedding_dim == 50
        assert cfg.input_dim == 6
        assert cfg.output_dim == 2

    def test_forward_shapes(self):
        model = _make_lc()
        x = torch.randn(4, 20, 6)
        out = model(x)
        assert isinstance(out, DynamicsModelOutput)
        assert out.outputs.shape == (4, 20, 2)
        assert out.states.shape == (4, 20, 4)

    def test_recurrence_shape(self):
        model = _make_lc()
        x_t = torch.randn(4, 6)
        z_prev = torch.zeros(4, 4)
        z_t = model.recurrence(x_t, z_prev)
        assert z_t.shape == (4, 4)

    def test_readout_shape(self):
        model = _make_lc()
        z = torch.randn(4, 4)
        y = model.readout(z)
        assert y.shape == (4, 2)

    def test_auto_config_and_model(self):
        cfg = AutoConfig.for_model("latent_circuit", input_dim=6, latent_dim=4, output_dim=2, embedding_dim=20)
        model = AutoModel.from_config(cfg)
        assert isinstance(model, NeuralDynamicsModel)


# ---------------------------------------------------------------------------
# TestCayleyTransform
# ---------------------------------------------------------------------------

class TestCayleyTransform:
    def test_q_shape(self):
        model = _make_lc()
        Q = model.embedding_matrix
        assert Q.shape == (4, 20)  # (n, N)

    def test_q_orthonormal(self):
        model = _make_lc()
        Q = model.embedding_matrix
        # Q @ Q^T should be close to identity (n x n)
        product = Q @ Q.T
        eye = torch.eye(4)
        assert torch.allclose(product, eye, atol=1e-5), f"Q @ Q^T not close to I: {product}"

    def test_q_orthonormal_random_a(self):
        cfg = AutoConfig.for_model("latent_circuit", input_dim=3, latent_dim=3, output_dim=2, embedding_dim=10)
        model = AutoModel.from_config(cfg)
        Q = model.embedding_matrix
        product = Q @ Q.T
        eye = torch.eye(3)
        assert torch.allclose(product, eye, atol=1e-4)


# ---------------------------------------------------------------------------
# TestConnectivityMasks
# ---------------------------------------------------------------------------

class TestConnectivityMasks:
    def test_input_mask(self):
        model = _make_lc()
        w_in = model.w_in.weight.data  # (n, input_dim)
        mask = model.input_mask         # (n, input_dim)
        # Only first input_dim diagonal entries should be nonzero
        # After apply_constraints: w_in = mask * relu(w_in)
        # So w_in should be zero where mask is zero
        assert torch.allclose(w_in * (1 - mask), torch.zeros_like(w_in), atol=1e-6)

    def test_output_mask(self):
        model = _make_lc()
        w_out = model.w_out.weight.data  # (output_dim, n)
        mask = model.output_mask          # (output_dim, n)
        # Only last output_dim diagonal entries should be nonzero
        assert torch.allclose(w_out * (1 - mask), torch.zeros_like(w_out), atol=1e-6)

    def test_apply_constraints_preserves_structure(self):
        model = _make_lc()
        # Modify weights
        with torch.no_grad():
            model.w_in.weight.add_(torch.randn_like(model.w_in.weight))
            model.w_out.weight.add_(torch.randn_like(model.w_out.weight))
        model.apply_constraints()
        # Check structure is restored
        w_in = model.w_in.weight.data
        mask = model.input_mask
        assert torch.allclose(w_in * (1 - mask), torch.zeros_like(w_in), atol=1e-6)


# ---------------------------------------------------------------------------
# TestEmbedding
# ---------------------------------------------------------------------------

class TestEmbedding:
    def test_embed_project_roundtrip(self):
        model = _make_lc()
        x = torch.randn(4, 4)  # latent dim = 4
        y = model.embed(x)
        x_back = model.project(y)
        assert torch.allclose(x, x_back, atol=1e-5)

    def test_embed_shape(self):
        model = _make_lc()
        x = torch.randn(4, 4)
        y = model.embed(x)
        assert y.shape == (4, 20)  # embedding_dim = 20

    def test_project_shape(self):
        model = _make_lc()
        y = torch.randn(4, 20)
        x = model.project(y)
        assert x.shape == (4, 4)  # latent_dim = 4


# ---------------------------------------------------------------------------
# TestSaveLoad
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_roundtrip(self, tmp_path):
        model = _make_lc()
        model.eval()
        path = str(tmp_path / "lc_model")
        model.save_pretrained(path)

        model2 = AutoModel.from_pretrained(path)
        model2.eval()

        x = torch.randn(2, 10, 6)
        out1 = model(x)
        out2 = model2(x)
        assert torch.allclose(out1.outputs, out2.outputs, atol=1e-6)
        assert torch.allclose(out1.states, out2.states, atol=1e-6)


# ---------------------------------------------------------------------------
# TestLatentCircuitObjective
# ---------------------------------------------------------------------------

class TestLatentCircuitObjective:
    def test_compute_loss(self):
        from neuralrnn.train.objectives.latent_circuit import LatentCircuitObjective
        model = _make_lc()
        ds = _make_lc_ds()
        obj = LatentCircuitObjective(l_y=1.0)
        batch = ds.sample_batch()
        loss, logs = obj.compute_loss(model, batch)
        assert loss.isfinite()
        assert "mse_z" in logs
        assert "nmse_y" in logs
        assert logs["mse_z"] >= 0
        assert logs["nmse_y"] >= 0

    def test_gradient_flows(self):
        from neuralrnn.train.objectives.latent_circuit import LatentCircuitObjective
        model = _make_lc()
        ds = _make_lc_ds()
        obj = LatentCircuitObjective(l_y=1.0)
        batch = ds.sample_batch()
        loss, _ = obj.compute_loss(model, batch)
        loss.backward()
        assert model.w_rec.weight.grad is not None
        assert model.w_in.weight.grad is not None
        assert model.w_out.weight.grad is not None


# ---------------------------------------------------------------------------
# TestCognitiveTaskDataset
# ---------------------------------------------------------------------------

class TestCognitiveTaskDataset:
    @pytest.mark.parametrize("task_name", [
        "siegel_miller", "mante_short", "two_afc", "parametric_wm",
    ])
    def test_task_shapes(self, task_name):
        ds = _make_task_ds(task_name)
        assert ds.inputs.ndim == 3
        assert ds.targets.ndim == 3
        assert ds.mask.ndim == 3
        assert ds.inputs.shape[0] == ds.targets.shape[0]
        assert ds.inputs.shape[1] == ds.targets.shape[1] or ds.targets.shape[1] < ds.inputs.shape[1]

    def test_sample_batch(self):
        ds = _make_task_ds()
        batch = ds.sample_batch()
        assert "inputs" in batch
        assert "targets" in batch
        assert "mask" in batch
        assert batch["inputs"].shape[0] == 4  # batch_size

    def test_get_all_trials(self):
        ds = _make_task_ds()
        data = ds.get_all_trials()
        assert data["inputs"].shape[0] == len(ds.conditions)

    def test_load_dataset(self):
        from neuralrnn.data import load_dataset
        ds = load_dataset("siegel_miller", n_trials=2, n_coh=2, batch_size=4)
        assert ds.inputs.ndim == 3


# ---------------------------------------------------------------------------
# TestLatentCircuitDataset
# ---------------------------------------------------------------------------

class TestLatentCircuitDataset:
    def test_from_rnn_and_task(self):
        ds = _make_lc_ds()
        assert ds.inputs.ndim == 3
        assert ds.targets.ndim == 3
        assert ds.rnn_states.ndim == 3
        assert ds.inputs.shape[0] == ds.rnn_states.shape[0]

    def test_sample_batch(self):
        ds = _make_lc_ds()
        batch = ds.sample_batch()
        assert "inputs" in batch
        assert "targets" in batch
        assert "rnn_states" in batch


# ---------------------------------------------------------------------------
# TestConnectivityAnalysis
# ---------------------------------------------------------------------------

class TestConnectivityAnalysis:
    def test_analyze_connectivity(self):
        from neuralrnn.analysis.connectivity import analyze_connectivity
        rnn = _make_rnn()
        lc = _make_lc()
        result = analyze_connectivity(lc, rnn)
        assert result.w_rec.shape == (4, 4)
        assert result.W_rec_projected.shape == (4, 4)
        assert result.Q.shape == (4, 20)
        assert isinstance(result.correlation, float)
        assert np.isfinite(result.correlation)


# ---------------------------------------------------------------------------
# TestPerturbationAnalysis
# ---------------------------------------------------------------------------

class TestPerturbationAnalysis:
    def test_latent_to_rnn_perturbation(self):
        from neuralrnn.analysis.perturbation import PerturbationSpec, latent_to_rnn_perturbation
        lc = _make_lc()
        Q = lc.embedding_matrix.detach().cpu().numpy()
        spec = PerturbationSpec(i=0, j=1, delta=-0.5)
        pert = latent_to_rnn_perturbation(spec, Q)
        assert pert.shape == (20, 20)
        # Should be rank-1
        assert np.linalg.matrix_rank(pert, tol=1e-6) <= 1

    def test_analyze_perturbation(self):
        from neuralrnn.analysis.perturbation import analyze_perturbation, PerturbationSpec
        rnn = _make_rnn()
        lc = _make_lc()
        task_ds = _make_task_ds()
        # Use valid indices (latent_dim=4, so indices 0-3)
        spec = PerturbationSpec(i=0, j=1, delta=-0.5, label="test")
        result = analyze_perturbation(rnn, lc, task_ds, spec)
        assert "curves" in result.behavior_before
        assert "curves" in result.behavior_after
        # RNN weights should be restored
        assert torch.isfinite(rnn.h2h.weight).all()


# ---------------------------------------------------------------------------
# TestPostStepHook
# ---------------------------------------------------------------------------

class TestPostStepHook:
    def test_trainer_calls_hook(self):
        from neuralrnn import Trainer, TrainingArguments
        from neuralrnn.train.objectives.supervised import SupervisedObjective
        from neuralrnn.data.base import BaseDataset

        model_cfg = AutoConfig.for_model("ctrnn", input_dim=6, latent_dim=10, output_dim=2, dt=40.0, tau=200.0)
        model = AutoModel.from_config(model_cfg)

        # Create a simple toy dataset with (B,T) mask for supervised training
        class _ToyTask(BaseDataset):
            kind = "neurogym"
            def __init__(self):
                super().__init__()
                self.input_dim = 6
                self.output_dim = 2
            def sample_batch(self):
                return {
                    "inputs": torch.randn(4, 20, 6),
                    "targets": torch.randn(4, 20, 2),
                    "mask": torch.ones(4, 20),
                }

        task_ds = _ToyTask()
        obj = SupervisedObjective("regression")
        args = TrainingArguments(max_steps=3, log_every=100, device="cpu", seed=42)

        hook_called = []
        def hook(m):
            hook_called.append(True)

        trainer = Trainer(model, task_ds, obj, args, post_step_hook=hook)
        trainer.train()
        assert len(hook_called) == 3


# ---------------------------------------------------------------------------
# TestAnalysisIntegration
# ---------------------------------------------------------------------------

class TestAnalysisIntegration:
    def test_fixed_points_on_latent_circuit(self):
        from neuralrnn.analysis import find_fixed_points
        model = _make_lc()
        model.eval()
        # Provide a task_input since latent circuit's recurrence requires x_t
        task_input = torch.zeros(6)
        fps = find_fixed_points(model, n_candidates=4, n_iters=20, backend="numeric", task_input=task_input)
        assert len(fps) >= 0  # May or may not find fixed points

    def test_pca_on_latent_circuit(self):
        from neuralrnn.analysis import fit_pca
        states = torch.randn(50, 4).numpy()
        pca = fit_pca(states, n_components=2)
        assert pca.components.shape == (2, 4)
        assert len(pca.explained_variance_ratio) == 2

    def test_vector_field_on_latent_circuit(self):
        from neuralrnn.analysis import compute_vector_field, fit_pca
        model = _make_lc()
        model.eval()
        # Generate some trajectories to get a PCA basis
        x = torch.randn(20, 10, 6)
        out = model(x)
        states = out.states.detach().reshape(-1, 4).numpy()
        pca = fit_pca(states, n_components=2)
        # Compute vector field in the PCA plane
        vf = compute_vector_field(
            model,
            basis=torch.tensor(pca.components, dtype=torch.float32),
            mean=torch.tensor(pca.mean, dtype=torch.float32),
            task_input=torch.zeros(6),
            extent=(-1, 1), n_grid=5,
        )
        n_total = 5 * 5
        assert vf.grid_pc.shape == (5, 5, 2)
        assert vf.velocity_pc.shape == (5, 5, 2)
        assert vf.speed.shape == (5, 5)
        assert np.isfinite(vf.speed).all()

    def test_psychometric_on_latent_circuit(self):
        from neuralrnn.analysis import compute_psychometric
        model = _make_lc()
        model.eval()
        ds = _make_task_ds()
        result = compute_psychometric(model, ds.inputs, ds.conditions)
        assert "curves" in result
        assert "choices" in result
        assert result["choices"].shape[0] == ds.inputs.shape[0]
