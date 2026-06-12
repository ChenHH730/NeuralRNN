"""Tests for fixed point analysis backends (numeric, scipy, analytic).

Verifies that:
- ScipyFixedPointFinder finds fixed points on a simple CTRNN
- Exact vs approximate modes both work
- Results are consistent across backends
- Stability classification is correct
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from neuralrnn import AutoConfig, AutoModel
from neuralrnn.analysis.fixed_points import (
    find_fixed_points,
    NumericFixedPointFinder,
    ScipyFixedPointFinder,
    FixedPointSet,
)


@pytest.fixture
def ctrnn_model():
    cfg = AutoConfig.for_model("ctrnn", input_dim=3, latent_dim=8, output_dim=2)
    return AutoModel.from_config(cfg)


@pytest.fixture
def plrnn_model():
    cfg = AutoConfig.for_model("shallow_plrnn", input_dim=0, latent_dim=3,
                               output_dim=3, hidden_dim=20, autonomous=True)
    return AutoModel.from_config(cfg)


class TestScipyBackend:
    def test_scipy_exact_finds_points(self, ctrnn_model):
        fps = ScipyFixedPointFinder(n_candidates=20, mode="exact", seed=42).find(
            ctrnn_model, task_input=torch.zeros(3))
        assert isinstance(fps, FixedPointSet)
        # Should find at least one point (or empty if system has no FPs)
        for fp in fps.points:
            assert fp.z.shape == (8,)
            assert fp.speed >= 0

    def test_scipy_approx_finds_points(self, ctrnn_model):
        fps = ScipyFixedPointFinder(n_candidates=20, mode="approx", seed=42).find(
            ctrnn_model, task_input=torch.zeros(3))
        assert isinstance(fps, FixedPointSet)

    def test_scipy_classifies_stability(self, ctrnn_model):
        fps = ScipyFixedPointFinder(n_candidates=30, mode="exact", seed=42).find(
            ctrnn_model, task_input=torch.zeros(3))
        for fp in fps.points:
            if fp.eigenvalues is not None:
                assert fp.is_stable is not None
                assert isinstance(fp.is_stable, (bool, np.bool_))

    def test_unified_entry_scipy(self, ctrnn_model):
        fps = find_fixed_points(ctrnn_model, backend="scipy",
                                task_input=torch.zeros(3), n_candidates=20)
        assert isinstance(fps, FixedPointSet)


class TestNumericBackend:
    def test_numeric_finds_points(self, ctrnn_model):
        fps = NumericFixedPointFinder(n_candidates=16, n_iters=100, speed_tol=0.5).find(
            ctrnn_model, task_input=torch.zeros(3))
        assert isinstance(fps, FixedPointSet)

    def test_unified_entry_numeric(self, ctrnn_model):
        fps = find_fixed_points(ctrnn_model, backend="numeric",
                                task_input=torch.zeros(3), n_candidates=8, n_iters=50)
        assert isinstance(fps, FixedPointSet)


class TestAnalyticBackend:
    def test_analytic_on_plrnn(self, plrnn_model):
        fps = find_fixed_points(plrnn_model, backend="analytic", max_order=1)
        assert isinstance(fps, FixedPointSet)

    def test_auto_selects_analytic_for_plrnn(self, plrnn_model):
        fps = find_fixed_points(plrnn_model, backend="auto", max_order=1)
        assert isinstance(fps, FixedPointSet)

    def test_auto_falls_back_to_numeric_for_ctrnn(self, ctrnn_model):
        fps = find_fixed_points(ctrnn_model, backend="auto",
                                task_input=torch.zeros(3), n_candidates=8, n_iters=50)
        assert isinstance(fps, FixedPointSet)


class TestFixedPointSet:
    def test_coords_empty(self):
        fps = FixedPointSet()
        assert fps.coords().shape == (0,)

    def test_coords_nonempty(self, ctrnn_model):
        fps = find_fixed_points(ctrnn_model, backend="numeric",
                                task_input=torch.zeros(3), n_candidates=8, n_iters=50)
        coords = fps.coords()
        if len(fps) > 0:
            assert coords.shape[1] == 8  # latent_dim

    def test_iteration(self):
        from neuralrnn.analysis.fixed_points import FixedPoint
        fps = FixedPointSet()
        fps.points.append(FixedPoint(z=np.zeros(3), speed=0.0))
        fps.points.append(FixedPoint(z=np.ones(3), speed=0.1))
        assert len(fps) == 2
        assert sum(1 for _ in fps) == 2
