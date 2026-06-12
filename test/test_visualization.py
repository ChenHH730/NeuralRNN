"""Tests for visualization module.

Verifies that all plotting functions:
- Accept synthetic data without errors
- Return (fig, ax) tuples
- Handle optional parameters correctly
- Work with matplotlib Agg backend (headless)
"""
import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")  # headless backend for testing
import matplotlib.pyplot as plt

from neuralrnn.visualization import (
    plot_trajectories_2d,
    plot_trajectories_3d,
    plot_fixed_points,
    plot_fixed_points_3d,
    plot_vector_field,
    plot_phase_portrait,
    plot_weight_matrix,
    plot_connectivity,
    plot_averaged_responses,
    plot_line_attractor,
    plot_line_attractor_3d,
    animate_trajectories_3d,
    plot_psychometric_curves,
    plot_trial_predictions,
)
from neuralrnn.analysis.dimensionality import fit_pca
from neuralrnn.analysis.fixed_points import FixedPoint, FixedPointSet
from neuralrnn.analysis.line_attractor import LineAttractorPoint, LineAttractorResult


# --- Fixtures ---

@pytest.fixture(autouse=True)
def close_figs():
    yield
    plt.close("all")


@pytest.fixture
def trajectories():
    np.random.seed(42)
    return [np.random.randn(50, 8) for _ in range(3)]


@pytest.fixture
def pca_result(trajectories):
    all_s = np.concatenate(trajectories, axis=0)
    return fit_pca(all_s, n_components=3)


@pytest.fixture
def fixed_points():
    fps = FixedPointSet()
    fps.points.append(FixedPoint(z=np.zeros(8), speed=0.0, is_stable=True))
    fps.points.append(FixedPoint(z=np.ones(8), speed=0.5, is_stable=False))
    fps.points.append(FixedPoint(z=np.ones(8) * 0.5, speed=0.1, is_stable=None))
    return fps


@pytest.fixture
def vector_field_data():
    from neuralrnn.analysis.vector_field import VectorField
    n = 10
    grid = np.stack(np.meshgrid(np.linspace(-1, 1, n), np.linspace(-1, 1, n)), axis=-1)
    vel = np.random.randn(n, n, 2) * 0.1
    speed = np.sqrt((vel ** 2).sum(axis=-1))
    return VectorField(grid_pc=grid, velocity_pc=vel, speed=speed)


@pytest.fixture
def line_attractor_result():
    np.random.seed(42)
    points = []
    for i in range(11):
        z = np.random.randn(8) * 0.5
        points.append(LineAttractorPoint(
            z=z, speed=float(i * 0.01), distance=float(i * 0.1),
            eigenvalues=np.random.randn(8) + 0j, jacobian=np.random.randn(8, 8)))
    return LineAttractorResult(
        points=points,
        endpoints=(np.random.randn(8), np.random.randn(8)))


class TestTrajectoryPlots:
    def test_plot_2d_basic(self, trajectories):
        fig, ax = plot_trajectories_2d(trajectories)
        assert fig is not None and ax is not None

    def test_plot_2d_with_pca(self, trajectories, pca_result):
        fig, ax = plot_trajectories_2d(trajectories, pca_result)
        assert fig is not None

    def test_plot_2d_with_colors_and_labels(self, trajectories):
        colors = ['red', 'green', 'blue']
        labels = ['A', 'B', 'C']
        fig, ax = plot_trajectories_2d(trajectories, colors=colors, labels=labels)
        assert fig is not None

    def test_plot_2d_on_existing_ax(self, trajectories):
        _, ax = plt.subplots()
        fig, ret_ax = plot_trajectories_2d(trajectories, ax=ax)
        assert ret_ax is ax

    def test_plot_3d_basic(self, trajectories):
        fig, ax = plot_trajectories_3d(trajectories)
        assert fig is not None

    def test_plot_3d_camera_angles(self, trajectories):
        fig, ax = plot_trajectories_3d(trajectories, elev=30, azim=60)
        assert fig is not None


class TestFixedPointPlots:
    def test_plot_2d(self, fixed_points, pca_result):
        fig, ax = plot_fixed_points(fixed_points, pca_result)
        assert fig is not None

    def test_plot_3d(self, fixed_points, pca_result):
        fig, ax = plot_fixed_points_3d(fixed_points, pca_result)
        assert fig is not None

    def test_custom_colors(self, fixed_points):
        fig, ax = plot_fixed_points(fixed_points, colors={"stable": "green"})
        assert fig is not None


class TestVectorFieldPlots:
    def test_basic(self, vector_field_data):
        fig, ax = plot_vector_field(vector_field_data)
        assert fig is not None

    def test_with_speed_color(self, vector_field_data):
        fig, ax = plot_vector_field(vector_field_data, show_speed=True)
        assert fig is not None


class TestPhasePortrait:
    def test_trajectories_only(self, trajectories, pca_result):
        fig, ax = plot_phase_portrait(trajectories=trajectories, pca_result=pca_result)
        assert fig is not None

    def test_all_combined(self, trajectories, fixed_points, vector_field_data, pca_result):
        fig, ax = plot_phase_portrait(
            trajectories=trajectories,
            fixed_points=fixed_points,
            vector_field=vector_field_data,
            pca_result=pca_result,
            title="Test Phase Portrait")
        assert fig is not None


class TestWeightMatrixPlots:
    def test_single_matrix(self):
        W = np.random.randn(16, 6)
        fig, ax = plot_weight_matrix(W, title="W_inp")
        assert fig is not None

    def test_connectivity(self):
        W_inp = np.random.randn(16, 6)
        W_rec = np.random.randn(16, 16)
        W_out = np.random.randn(2, 16)
        fig, axes = plot_connectivity(W_inp, W_rec, W_out)
        assert fig is not None
        assert len(axes) == 3

    def test_connectivity_with_dale(self):
        W_inp = np.random.randn(16, 6)
        W_rec = np.random.randn(16, 16)
        W_out = np.random.randn(2, 16)
        dale_mask = np.array([1] * 12 + [0] * 4)
        fig, axes = plot_connectivity(W_inp, W_rec, W_out, dale_mask=dale_mask)
        assert fig is not None


class TestAveragedResponses:
    def test_basic(self):
        responses = np.random.randn(5, 50)
        fig, ax = plot_averaged_responses(responses)
        assert fig is not None

    def test_with_labels(self):
        responses = np.random.randn(4, 50)
        fig, ax = plot_averaged_responses(responses, labels=["C1", "C2", "C3", "C4"])
        assert fig is not None


class TestLineAttractorPlots:
    def test_plot_2d(self, line_attractor_result, pca_result):
        fig, ax = plot_line_attractor(line_attractor_result, pca_result)
        assert fig is not None

    def test_plot_2d_no_rhs(self, line_attractor_result, pca_result):
        fig, ax = plot_line_attractor(line_attractor_result, pca_result, show_rhs=False)
        assert fig is not None

    def test_plot_3d(self, line_attractor_result):
        axes_3d = np.random.randn(3, 8)
        fig, ax = plot_line_attractor_3d(line_attractor_result, axes_3d)
        assert fig is not None

    def test_plot_3d_with_trajectories(self, line_attractor_result, trajectories):
        axes_3d = np.random.randn(3, 8)
        fig, ax = plot_line_attractor_3d(
            line_attractor_result, axes_3d, trajectories=trajectories)
        assert fig is not None


class TestAnimation:
    def test_animate_basic(self, trajectories):
        anim = animate_trajectories_3d(trajectories, fps=10, duration=1.0, step=10)
        assert anim is not None

    def test_animate_with_pca(self, trajectories, pca_result):
        anim = animate_trajectories_3d(trajectories, pca_result, fps=10, duration=1.0, step=10)
        assert anim is not None


class TestPsychometricPlots:
    def test_basic(self):
        from neuralrnn.analysis.psychometric import PsychometricCurve
        curves = [
            PsychometricCurve(coherences=np.linspace(-1, 1, 7),
                              prob_right=np.linspace(0.1, 0.9, 7),
                              sigmoid_params=None, label="Motion"),
            PsychometricCurve(coherences=np.linspace(-1, 1, 7),
                              prob_right=np.linspace(0.2, 0.8, 7),
                              sigmoid_params=None, label="Color"),
        ]
        fig, ax = plot_psychometric_curves(curves)
        assert fig is not None


class TestTrialPredictions:
    def test_basic(self):
        B, T, O = 6, 50, 2
        pred = np.random.randn(B, T, O)
        target = np.random.randn(B, T, O)
        fig, axes = plot_trial_predictions(pred, target, n_trials=4)
        assert fig is not None

    def test_with_mask(self):
        B, T, O = 4, 50, 2
        pred = np.random.randn(B, T, O)
        target = np.random.randn(B, T, O)
        mask = np.ones((B, T))
        mask[:, 30:] = 0
        fig, axes = plot_trial_predictions(pred, target, mask=mask, n_trials=2)
        assert fig is not None
