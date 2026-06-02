"""Tests for analysis fixes in 05_latent_circuit_paradigmA.ipynb.

Covers:
1. PCA coloring by context × response
2. linearize() handling (M,) and (1,M) shaped inputs
3. compute_vector_field() output reshaping to (n_grid, n_grid, 2)
4. Fixed-point analysis with context-specific task inputs
5. Model save/load roundtrip
"""
import os
import sys
import tempfile

import numpy as np
import torch
import pytest

# Ensure the package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from neuralrnn import AutoConfig, AutoModel
from neuralrnn.analysis import (
    find_fixed_points, fit_pca, linearize, compute_vector_field,
    compute_psychometric,
)
from neuralrnn.analysis.dimensionality import PCAResult
from neuralrnn.analysis.linearization import LinearizationResult
from neuralrnn.analysis.vector_field import VectorField


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture(scope='module')
def lc_model():
    """Create a small latent circuit model for testing."""
    cfg = AutoConfig.for_model(
        'latent_circuit', input_dim=6, latent_dim=8, output_dim=2,
        embedding_dim=20, dt=40.0, tau=200.0, sigma_rec=0.15,
    )
    model = AutoModel.from_config(cfg)
    model.eval()
    return model


@pytest.fixture(scope='module')
def task_inputs():
    """Context-specific task inputs."""
    return {
        'motion': torch.tensor([0.0, 1.2, 0.0, 0.0, 0.0, 0.0]),
        'color':  torch.tensor([1.2, 0.0, 0.0, 0.0, 0.0, 0.0]),
    }


@pytest.fixture(scope='module')
def sample_conditions():
    """Sample conditions list for testing build_colors."""
    conditions = []
    for ctx in ['motion', 'color']:
        for motion_coh in [-0.2, 0.0, 0.2]:
            for color_coh in [-0.2, 0.0, 0.2]:
                conditions.append({
                    'context': ctx,
                    'motion_coh': motion_coh,
                    'color_coh': color_coh,
                    'correct_choice': 1 if (
                        (ctx == 'motion' and motion_coh > 0) or
                        (ctx == 'color' and color_coh > 0)
                    ) else -1,
                })
    return conditions


# ======================================================================
# Test 1: PCA coloring
# ======================================================================

class TestPCAColoring:
    """Test the build_colors function for context × response coloring."""

    def _build_colors(self, conditions, choices):
        """Replicate the build_colors function from the notebook."""
        colors = []
        for i, c in enumerate(conditions):
            ctx = c['context']
            if ctx == 'motion':
                coh = abs(c['motion_coh'])
                if choices[i] > 0:
                    colors.append((0.8 * coh + 0.2, 0.0, 0.0, 0.3 + 0.5 * coh))
                else:
                    colors.append((0.0, 0.0, 0.8 * coh + 0.2, 0.3 + 0.5 * coh))
            else:
                coh = abs(c['color_coh'])
                if choices[i] > 0:
                    colors.append((0.0, 0.6 * coh + 0.2, 0.0, 0.3 + 0.5 * coh))
                else:
                    colors.append((0.5 * coh + 0.2, 0.0, 0.5 * coh + 0.2, 0.3 + 0.5 * coh))
        return np.array(colors)

    def test_output_shape(self, sample_conditions):
        choices = np.array([1, -1] * (len(sample_conditions) // 2) +
                          [1] * (len(sample_conditions) % 2))
        colors = self._build_colors(sample_conditions, choices)
        assert colors.shape == (len(sample_conditions), 4), f"Expected (N,4), got {colors.shape}"

    def test_repeat_expansion(self, sample_conditions):
        """np.repeat should expand per-trial colors to per-timestep."""
        choices = np.array([1, -1] * (len(sample_conditions) // 2) +
                          [1] * (len(sample_conditions) % 2))
        colors = self._build_colors(sample_conditions, choices)
        T = 75  # typical trial length
        expanded = np.repeat(colors, T, axis=0)
        assert expanded.shape == (len(sample_conditions) * T, 4)
        # First T rows should all be the same color
        assert np.allclose(expanded[0], expanded[T - 1])

    def test_motion_context_colors(self, sample_conditions):
        """Motion context: right=red, left=blue."""
        # Filter motion context
        motion_conds = [c for c in sample_conditions if c['context'] == 'motion']
        choices = np.array([1 if c['motion_coh'] > 0 else -1 for c in motion_conds])
        colors = self._build_colors(motion_conds, choices)

        # Right choices should be red (R > 0, B ≈ 0)
        right_mask = choices > 0
        assert np.all(colors[right_mask, 0] > 0), "Right choices in motion ctx should be red"
        assert np.all(colors[right_mask, 2] < 0.01), "Right choices in motion ctx should have no blue"

        # Left choices should be blue (R ≈ 0, B > 0)
        left_mask = choices <= 0
        assert np.all(colors[left_mask, 0] < 0.01), "Left choices in motion ctx should have no red"
        assert np.all(colors[left_mask, 2] > 0), "Left choices in motion ctx should be blue"

    def test_color_context_colors(self, sample_conditions):
        """Color context: right=green, left=purple."""
        color_conds = [c for c in sample_conditions if c['context'] == 'color']
        choices = np.array([1 if c['color_coh'] > 0 else -1 for c in color_conds])
        colors = self._build_colors(color_conds, choices)

        # Right choices should be green (G > 0, R ≈ 0, B ≈ 0)
        right_mask = choices > 0
        assert np.all(colors[right_mask, 1] > 0), "Right choices in color ctx should be green"
        assert np.all(colors[right_mask, 0] < 0.01), "Right choices in color ctx should have no red"

        # Left choices should be purple (R > 0, B > 0, G ≈ 0)
        left_mask = choices <= 0
        assert np.all(colors[left_mask, 0] > 0), "Left choices in color ctx should have red (purple)"
        assert np.all(colors[left_mask, 2] > 0), "Left choices in color ctx should have blue (purple)"
        assert np.all(colors[left_mask, 1] < 0.01), "Left choices in color ctx should have no green"

    def test_intensity_scales_with_coherence(self, sample_conditions):
        """Higher |coherence| → higher intensity (alpha and color value)."""
        conds = [
            {'context': 'motion', 'motion_coh': 0.0, 'color_coh': 0.0},
            {'context': 'motion', 'motion_coh': 0.2, 'color_coh': 0.0},
        ]
        choices = np.array([1, 1])  # both right
        colors = self._build_colors(conds, choices)
        # Zero coherence should have lower intensity than high coherence
        assert colors[0, 3] < colors[1, 3], "Alpha should increase with coherence"
        assert colors[0, 0] < colors[1, 0], "Red intensity should increase with coherence"

    def test_rgba_range(self, sample_conditions):
        """All RGBA values should be in [0, 1]."""
        choices = np.array([1, -1] * (len(sample_conditions) // 2) +
                          [1] * (len(sample_conditions) % 2))
        colors = self._build_colors(sample_conditions, choices)
        assert np.all(colors >= 0) and np.all(colors <= 1), \
            f"Colors out of [0,1] range: min={colors.min()}, max={colors.max()}"


# ======================================================================
# Test 2: linearize() shape handling
# ======================================================================

class TestLinearizeShape:
    """Test that linearize handles both (M,) and (1,M) shaped z."""

    def test_1d_input(self, lc_model, task_inputs):
        """z as (M,) should work."""
        z = np.random.randn(8).astype(np.float32)
        result = linearize(lc_model, z, task_input=task_inputs['motion'])
        assert isinstance(result, LinearizationResult)
        assert result.jacobian.shape == (8, 8)
        assert result.eigenvalues.shape == (8,)

    def test_2d_input(self, lc_model, task_inputs):
        """z as (1,M) should also work (auto-squeezed)."""
        z = np.random.randn(1, 8).astype(np.float32)
        result = linearize(lc_model, z, task_input=task_inputs['motion'])
        assert isinstance(result, LinearizationResult)
        assert result.jacobian.shape == (8, 8)

    def test_tensor_input(self, lc_model, task_inputs):
        """z as torch.Tensor (M,) should work."""
        z = torch.randn(8)
        result = linearize(lc_model, z, task_input=task_inputs['motion'])
        assert result.jacobian.shape == (8, 8)

    def test_tensor_2d_input(self, lc_model, task_inputs):
        """z as torch.Tensor (1,M) should work (auto-squeezed)."""
        z = torch.randn(1, 8)
        result = linearize(lc_model, z, task_input=task_inputs['motion'])
        assert result.jacobian.shape == (8, 8)

    def test_no_task_input(self, lc_model):
        """linearize with task_input=None — latent circuit requires input, use zero."""
        z = np.random.randn(8).astype(np.float32)
        zero_input = torch.zeros(6)
        result = linearize(lc_model, z, task_input=zero_input)
        assert result.jacobian.shape == (8, 8)

    def test_stability_classification(self, lc_model, task_inputs):
        """is_stable should be bool, n_unstable should be int."""
        z = np.random.randn(8).astype(np.float32)
        result = linearize(lc_model, z, task_input=task_inputs['motion'])
        assert isinstance(result.is_stable, (bool, np.bool_))
        assert isinstance(result.n_unstable, (int, np.integer))


# ======================================================================
# Test 3: compute_vector_field() reshaping
# ======================================================================

class TestVectorFieldReshape:
    """Test that compute_vector_field returns (n_grid, n_grid, 2) arrays."""

    def test_output_shapes(self, lc_model, task_inputs):
        """grid_pc and velocity_pc should be (n_grid, n_grid, 2)."""
        n_grid = 10
        basis = torch.randn(2, 8)
        mean = torch.zeros(8)
        vf = compute_vector_field(
            lc_model, basis, mean,
            task_input=task_inputs['motion'],
            extent=(-2, 2), n_grid=n_grid,
        )
        assert isinstance(vf, VectorField)
        assert vf.grid_pc.shape == (n_grid, n_grid, 2), \
            f"grid_pc shape {vf.grid_pc.shape} != ({n_grid}, {n_grid}, 2)"
        assert vf.velocity_pc.shape == (n_grid, n_grid, 2), \
            f"velocity_pc shape {vf.velocity_pc.shape} != ({n_grid}, {n_grid}, 2)"
        assert vf.speed.shape == (n_grid, n_grid), \
            f"speed shape {vf.speed.shape} != ({n_grid}, {n_grid})"

    def test_3d_indexing_works(self, lc_model, task_inputs):
        """The 3D indexing that caused IndexError should now work."""
        n_grid = 8
        basis = torch.randn(2, 8)
        mean = torch.zeros(8)
        vf = compute_vector_field(
            lc_model, basis, mean,
            task_input=task_inputs['motion'],
            extent=(-1, 1), n_grid=n_grid,
        )
        # This was the failing line
        speed = np.sqrt(vf.velocity_pc[:, :, 0]**2 + vf.velocity_pc[:, :, 1]**2)
        assert speed.shape == (n_grid, n_grid)

    def test_with_pca_basis(self, lc_model, task_inputs):
        """Test with PCA-derived basis (typical use case)."""
        # Generate some states to fit PCA
        with torch.no_grad():
            inputs = torch.randn(20, 75, 6)
            out = lc_model(inputs)
            states = out.states.detach().reshape(-1, 8).numpy()

        pca = fit_pca(states, n_components=2)
        vf = compute_vector_field(
            lc_model,
            basis=torch.tensor(pca.components, dtype=torch.float32),
            mean=torch.tensor(pca.mean, dtype=torch.float32),
            task_input=task_inputs['color'],
            extent=(-2, 2), n_grid=12,
        )
        assert vf.grid_pc.shape == (12, 12, 2)
        assert vf.velocity_pc.shape == (12, 12, 2)

    def test_no_task_input(self, lc_model):
        """Vector field with zero task input (latent circuit requires input)."""
        basis = torch.randn(2, 8)
        mean = torch.zeros(8)
        vf = compute_vector_field(
            lc_model, basis, mean,
            task_input=torch.zeros(6),
            extent=(-1, 1), n_grid=5,
        )
        assert vf.grid_pc.shape == (5, 5, 2)


# ======================================================================
# Test 4: Fixed-point analysis with context-specific inputs
# ======================================================================

class TestFixedPointContexts:
    """Test fixed-point analysis with both context inputs."""

    def test_find_with_motion_context(self, lc_model, task_inputs):
        """Finding fixed points with motion context input."""
        fps = find_fixed_points(
            lc_model, n_candidates=16, n_iters=500,
            backend='numeric', task_input=task_inputs['motion'],
        )
        assert len(fps) >= 0  # may find 0 or more
        for fp in fps:
            assert fp.z.shape == (8,)
            assert isinstance(fp.speed, float)

    def test_find_with_color_context(self, lc_model, task_inputs):
        """Finding fixed points with color context input."""
        fps = find_fixed_points(
            lc_model, n_candidates=16, n_iters=500,
            backend='numeric', task_input=task_inputs['color'],
        )
        assert len(fps) >= 0
        for fp in fps:
            assert fp.z.shape == (8,)

    def test_linearize_fixed_point(self, lc_model, task_inputs):
        """Linearize at a found fixed point — the (1,M) shape bug test."""
        fps = find_fixed_points(
            lc_model, n_candidates=16, n_iters=500,
            backend='numeric', task_input=task_inputs['motion'],
        )
        if len(fps) > 0:
            fp = fps.points[0]
            # This was the original failing call: fp.z with unsqueeze
            # Now linearize auto-handles both shapes
            lin = linearize(lc_model, fp.z, task_input=task_inputs['motion'])
            assert lin.jacobian.shape == (8, 8)
            assert isinstance(lin.is_stable, (bool, np.bool_))

            # Also test with explicit unsqueeze (should still work)
            lin2 = linearize(lc_model, torch.tensor(fp.z).unsqueeze(0),
                           task_input=task_inputs['motion'])
            assert lin2.jacobian.shape == (8, 8)

    def test_fixed_point_eigenvalues(self, lc_model, task_inputs):
        """Eigenvalues should be complex, 8 of them for 8-dim system."""
        fps = find_fixed_points(
            lc_model, n_candidates=16, n_iters=500,
            backend='numeric', task_input=task_inputs['motion'],
        )
        if len(fps) > 0:
            fp = fps.points[0]
            lin = linearize(lc_model, fp.z, task_input=task_inputs['motion'])
            assert lin.eigenvalues.shape == (8,)
            # Stability check: max |eigenvalue| vs 1.0
            max_eig = np.max(np.abs(lin.eigenvalues))
            assert lin.is_stable == (max_eig < 1.0)


# ======================================================================
# Test 5: Model save/load roundtrip
# ======================================================================

class TestModelSaveLoad:
    """Test save_pretrained / from_pretrained roundtrip."""

    def test_save_and_load_lc(self):
        """Latent circuit model roundtrip (fresh model to avoid mutation)."""
        torch.manual_seed(99)
        cfg = AutoConfig.for_model(
            'latent_circuit', input_dim=6, latent_dim=8, output_dim=2,
            embedding_dim=20, dt=40.0, tau=200.0, sigma_rec=0.15,
        )
        model = AutoModel.from_config(cfg)
        model.eval()
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, 'lc')
            model.save_pretrained(save_path, metadata={'test': True})

            # Check files exist
            assert os.path.exists(os.path.join(save_path, 'config.json'))
            assert (os.path.exists(os.path.join(save_path, 'model.safetensors')) or
                    os.path.exists(os.path.join(save_path, 'model.pt')))
            assert os.path.exists(os.path.join(save_path, 'metadata.json'))

            # Load and verify
            loaded = AutoModel.from_pretrained(save_path)
            assert loaded.config.latent_dim == model.config.latent_dim
            assert loaded.config.input_dim == model.config.input_dim
            assert loaded.config.output_dim == model.config.output_dim

            # Verify state dict roundtrip (exact)
            orig_sd = model.state_dict()
            loaded_sd = loaded.state_dict()
            for key in orig_sd:
                assert key in loaded_sd, f"Missing key: {key}"
                assert torch.allclose(orig_sd[key], loaded_sd[key], atol=1e-6), \
                    f"Weight mismatch at {key}: max diff {(orig_sd[key]-loaded_sd[key]).abs().max().item()}"

    def test_save_creates_directory(self, lc_model):
        """save_pretrained should create directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, 'new_dir', 'model')
            lc_model.save_pretrained(save_path)
            assert os.path.isdir(save_path)

    def test_roundtrip_weights(self):
        """All weights should be identical after roundtrip (fresh model)."""
        cfg = AutoConfig.for_model(
            'latent_circuit', input_dim=6, latent_dim=8, output_dim=2,
            embedding_dim=20, dt=40.0, tau=200.0, sigma_rec=0.15,
        )
        model = AutoModel.from_config(cfg)
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, 'lc')
            model.save_pretrained(save_path)
            loaded = AutoModel.from_pretrained(save_path)

            for (n1, p1), (n2, p2) in zip(
                model.named_parameters(), loaded.named_parameters()
            ):
                assert n1 == n2, f"Parameter name mismatch: {n1} vs {n2}"
                assert torch.allclose(p1, p2), f"Weight mismatch at {n1}"


# ======================================================================
# Test 6: PCA on latent circuit states
# ======================================================================

class TestLatentCircuitPCA:
    """Test PCA analysis on latent circuit activations."""

    def test_fit_pca_on_latent_states(self, lc_model):
        """PCA on 8-dim latent states should work."""
        with torch.no_grad():
            inputs = torch.randn(20, 75, 6)
            out = lc_model(inputs)
            states = out.states.detach().reshape(-1, 8).numpy()

        pca = fit_pca(states, n_components=3)
        assert pca.components.shape == (3, 8)
        assert pca.mean.shape == (8,)
        assert len(pca.explained_variance_ratio) == 3
        assert all(0 <= v <= 1 for v in pca.explained_variance_ratio)

    def test_transform_and_inverse(self, lc_model):
        """PCA transform → inverse_transform should approximately recover data."""
        with torch.no_grad():
            inputs = torch.randn(20, 75, 6)
            out = lc_model(inputs)
            states = out.states.detach().reshape(-1, 8).numpy()

        pca = fit_pca(states, n_components=3)
        projected = pca.transform(states)
        recovered = pca.inverse_transform(projected)
        # With 3 components out of 8, reconstruction won't be perfect
        # but should be reasonable
        mse = np.mean((states - recovered) ** 2)
        assert mse < 1.0, f"PCA reconstruction MSE {mse} too large"

    def test_project_fixed_points(self, lc_model, task_inputs):
        """Fixed points should be projectable into PCA space."""
        with torch.no_grad():
            inputs = torch.randn(20, 75, 6)
            out = lc_model(inputs)
            states = out.states.detach().reshape(-1, 8).numpy()

        pca = fit_pca(states, n_components=3)
        fps = find_fixed_points(
            lc_model, n_candidates=16, n_iters=500,
            backend='numeric', task_input=task_inputs['motion'],
        )
        if len(fps) > 0:
            fp_coords = np.array([fp.z for fp in fps])
            fp_pca = pca.transform(fp_coords)
            assert fp_pca.shape == (len(fps), 3)


# ======================================================================
# Main
# ======================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
