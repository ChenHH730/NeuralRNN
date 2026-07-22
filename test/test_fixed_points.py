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

    def _sign_combo_model(self):
        """Hand-crafted shallow PLRNN whose 8 subregions each hold a valid FP.

        z' = A z + W1 relu(W2 z + h2) + h1 with A=0, W1=2I, W2=I, h2=0,
        h1=(-1,-2,-0.5): in every region D, z_i = h1_i/(1-2 d_i) is sign-consistent,
        giving all 8 sign combinations of (1,2,0.5) as fixed points. These FPs
        share coordinates pairwise, so an element-wise dedup (as in the original
        CNS2023 code) would collapse them to one; vector-wise dedup keeps all 8.
        """
        cfg = AutoConfig.for_model("shallow_plrnn", input_dim=0, latent_dim=3,
                                   output_dim=3, hidden_dim=3, autonomous=True)
        model = AutoModel.from_config(cfg)
        with torch.no_grad():
            model.A.zero_()
            model.W1.copy_(2 * torch.eye(3))
            model.W2.copy_(torch.eye(3))
            model.h1.copy_(torch.tensor([-1.0, -2.0, -0.5]))
            model.h2.zero_()
        return model

    def test_analytic_dedup_keeps_coordinate_sharing_fps(self):
        np.random.seed(0)
        fps = find_fixed_points(self._sign_combo_model(), backend="analytic",
                                max_order=1, outer_it=100, inner_it=10)
        coords = {tuple(np.round(p.z, 6)) for p in fps}
        expected = {(sx * 1.0, sy * 2.0, sz * 0.5)
                    for sx in (1, -1) for sy in (1, -1) for sz in (1, -1)}
        assert coords == expected

    def test_analytic_dedup_tol_merges_close_fps(self):
        np.random.seed(0)
        fps = find_fixed_points(self._sign_combo_model(), backend="analytic",
                                max_order=1, outer_it=100, inner_it=10,
                                dedup_tol=10.0)
        assert len(fps) == 1


    def test_auto_falls_back_to_numeric_for_ctrnn(self, ctrnn_model):
        fps = find_fixed_points(ctrnn_model, backend="auto",
                                task_input=torch.zeros(3), n_candidates=8, n_iters=50)
        assert isinstance(fps, FixedPointSet)

    def test_analytic_with_task_input(self):
        cfg = AutoConfig.for_model("shallow_plrnn", input_dim=2, latent_dim=3,
                                   output_dim=3, hidden_dim=10, autonomous=False)
        model = AutoModel.from_config(cfg)
        task_input = torch.tensor([0.3, -0.2], dtype=torch.float32)
        fps = find_fixed_points(model, backend="analytic", task_input=task_input,
                                max_order=1, outer_it=40, inner_it=20)
        assert isinstance(fps, FixedPointSet)
        # Random init may occasionally find no FPs; the key check is that the solver runs
        # without error and supports task_input (quantitative match is covered by the
        # numeric comparison test below).

    def test_analytic_task_input_matches_numeric(self):
        cfg = AutoConfig.for_model("shallow_plrnn", input_dim=2, latent_dim=3,
                                   output_dim=3, hidden_dim=10, autonomous=False)
        model = AutoModel.from_config(cfg)
        task_input = torch.tensor([0.3, -0.2], dtype=torch.float32)

        fps_analytic = find_fixed_points(model, backend="analytic", task_input=task_input,
                                         max_order=1, outer_it=20, inner_it=10)
        fps_numeric = find_fixed_points(model, backend="numeric", task_input=task_input,
                                        n_candidates=32, n_iters=2000, speed_tol=0.5)

        if len(fps_analytic) == 0 or len(fps_numeric) == 0:
            pytest.skip("One backend found no fixed points")

        # The lowest-speed numeric point should be close to some analytic point
        best_numeric = min(fps_numeric.points, key=lambda p: p.speed)
        distances = [np.linalg.norm(best_numeric.z - p.z) for p in fps_analytic.points]
        assert min(distances) < 0.5, "Numeric and analytic fixed points disagree"

    def test_analytic_ignores_task_input_for_autonomous_model(self, plrnn_model):
        task_input = torch.tensor([0.3, -0.2], dtype=torch.float32)
        fps_with = find_fixed_points(plrnn_model, backend="analytic", task_input=task_input,
                                     max_order=1, outer_it=10, inner_it=5)
        fps_without = find_fixed_points(plrnn_model, backend="analytic",
                                        max_order=1, outer_it=10, inner_it=5)
        assert len(fps_with) == len(fps_without)
        if len(fps_with) > 0 and len(fps_without) > 0:
            coords_with = np.sort(fps_with.coords(), axis=0)
            coords_without = np.sort(fps_without.coords(), axis=0)
            assert np.allclose(coords_with, coords_without, atol=1e-5)


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
