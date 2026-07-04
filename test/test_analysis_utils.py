"""Smoke tests for analysis utilities.

Covers:
  - linear algebra helpers (Gram-Schmidt, overlap, correlation, angles,
    trajectory flattening, device mapping)
  - population structure helpers (connectivity vectors, GMM, means/covariances)
  - manifold trajectory-to-position/velocity conversion
  - reusable loss functions
  - perturbation application and choice computation
  - DSR state-space divergence metrics
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from neuralrnn import AutoConfig, AutoModel
from neuralrnn.analysis import linalg_utils, population_structure, manifold
from neuralrnn.losses import loss_functions
from neuralrnn.analysis import perturbation, stsp_metrics


class _MockLowRankNet:
    """Duck-typed model for population_structure.make_vecs."""

    def __init__(self, N=10, rank=2, input_size=2, output_size=2):
        self.rank = rank
        self.input_size = input_size
        self.output_size = output_size
        self.m = torch.randn(N, rank)
        self.n = torch.randn(N, rank)
        self.wi = torch.randn(input_size, N)
        self.wo = torch.randn(N, output_size)


class TestLinalgUtils:
    def test_gram_schmidt_orthonormalizes(self):
        vecs = [np.random.randn(8) for _ in range(3)]
        ortho = linalg_utils.gram_schmidt(vecs)
        assert len(ortho) == 3
        for i in range(3):
            for j in range(i + 1, 3):
                assert np.abs(ortho[i] @ ortho[j]) < 1e-8

    def test_overlap_matrix_shape(self):
        vecs = [np.random.randn(12) for _ in range(4)]
        ov = linalg_utils.overlap_matrix(vecs)
        assert ov.shape == (4, 4)
        assert np.allclose(np.diag(ov), 0.0)

    def test_corrvecs_and_angle_helpers(self):
        v = np.array([1.0, 0.0, 0.0])
        w = np.array([0.0, 1.0, 0.0])
        assert np.isclose(linalg_utils.corrvecs(v, w), 0.0, atol=1e-8)
        assert np.isclose(linalg_utils.angle_vectors(v, w), np.pi / 2, atol=1e-8)
        # Subspace angle: [1,1,0] against span([1,0,0]) should be 45 degrees.
        u = np.array([1.0, 1.0, 0.0])
        assert np.isclose(
            linalg_utils.angle_vec_subsp(u, [np.array([1.0, 0.0, 0.0])]),
            np.pi / 4,
            atol=1e-8,
        )

    def test_project_onto_subspace(self):
        v = np.array([1.0, 1.0, 0.0])
        proj = linalg_utils.project(v, [np.array([1.0, 0.0, 0.0])])
        assert np.allclose(proj, [1.0, 0.0, 0.0], atol=1e-8)

    def test_flatten_unflatten_roundtrip(self):
        X = np.random.randn(5, 20, 7)
        flat = linalg_utils.flatten_trajectory(X)
        recon = linalg_utils.unflatten_trajectory(flat, n_trials=5)
        assert np.allclose(X, recon)

    def test_map_device_to_model_device(self):
        cfg = AutoConfig.for_model("ctrnn", input_dim=2, latent_dim=8, output_dim=2)
        model = AutoModel.from_config(cfg)
        tensors = [torch.randn(3, 3), {"a": torch.randn(2)}]
        moved = linalg_utils.map_device(tensors, model)
        device = next(model.parameters()).device
        assert moved[0].device == device
        assert moved[1]["a"].device == device


class TestPopulationStructure:
    def test_make_vecs_lengths(self):
        net = _MockLowRankNet(N=12, rank=2, input_size=2, output_size=3)
        vecs = population_structure.make_vecs(net)
        # 2 (m) + 2 (n) + 2 (wi) + 3 (wo) = 9 vectors, each length 12
        assert len(vecs) == 9
        assert all(v.shape == (12,) for v in vecs)

    def test_gmm_fit_and_population_summaries(self):
        pytest.importorskip("sklearn")
        N, d, k = 50, 4, 2
        X = np.random.randn(N, d)
        labels, model = population_structure.gmm_fit(X, n_components=k, random_state=0)
        assert labels.shape == (N,)
        assert len(np.unique(labels)) <= k

        means = population_structure.compute_population_means(X, labels)
        assert means.shape[0] <= k
        assert means.shape[1] == d

        covs = population_structure.compute_population_covariances(X, labels)
        assert len(covs) == means.shape[0]
        assert all(c.shape == (d, d) for c in covs)


class TestManifold:
    def test_trajectories_to_pos_vel(self):
        traj = np.random.randn(30, 4)
        pos, vel = manifold.trajectories_to_pos_vel(traj)
        assert pos.shape == (29, 4)
        assert vel.shape == (29, 4)
        assert np.allclose(vel, np.diff(traj, axis=0))

    def test_trajectories_to_pos_vel_3d(self):
        traj = np.random.randn(5, 10, 4)
        pos, vel = manifold.trajectories_to_pos_vel(traj)
        assert pos.shape == (49, 4)
        assert vel.shape == (49, 4)


class TestLossFunctions:
    def test_loss_mse_returns_scalar(self):
        B, T, O = 4, 10, 2
        output = torch.randn(B, T, O)
        target = torch.randn(B, T, O)
        mask = torch.ones(B, T, 1)
        loss = loss_functions.loss_mse(output, target, mask)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0

    def test_accuracy_general_returns_scalar_between_zero_and_one(self):
        B, T, O = 8, 12, 1
        output = torch.randn(B, T, O)
        target = torch.randn(B, T, O)
        target[target == 0] = 1.0  # avoid all-zero trials
        mask = torch.ones(B, T, 1)
        acc = loss_functions.accuracy_general(output, target, mask)
        assert isinstance(acc, torch.Tensor)
        assert 0.0 <= acc.item() <= 1.0 or np.isnan(acc.item())


class TestPerturbation:
    def test_apply_perturbation_changes_weights(self):
        cfg = AutoConfig.for_model("ctrnn", input_dim=2, latent_dim=8, output_dim=2)
        model = AutoModel.from_config(cfg)
        original = model.h2h.weight.data.clone()
        delta = np.eye(8) * 0.01
        perturbation.apply_perturbation(model, delta)
        assert not torch.allclose(model.h2h.weight.data, original)

    def test_compute_choice_shape(self):
        cfg = AutoConfig.for_model("ctrnn", input_dim=2, latent_dim=8, output_dim=2)
        model = AutoModel.from_config(cfg)
        inputs = torch.randn(6, 15, 2)
        choices = perturbation.compute_choice(model, inputs)
        assert choices.shape == (6,)
        assert np.all(choices >= 0)


class TestStspMetrics:
    def test_binning_divergence_finite(self):
        x_true = np.random.randn(200, 2)
        x_gen = x_true + 0.1 * np.random.randn(200, 2)
        d = stsp_metrics.state_space_divergence_binning(x_gen, x_true, n_bins=10)
        assert np.isfinite(d)

    def test_gmm_divergence_finite(self):
        x_true = np.random.randn(200, 3)
        x_gen = x_true + 0.1 * np.random.randn(200, 3)
        d = stsp_metrics.state_space_divergence_gmm(x_gen, x_true, mc_n=100)
        assert np.isfinite(d)

    def test_hellinger_distance_finite(self):
        p = np.random.rand(20)
        p = p / p.sum()
        q = np.random.rand(20)
        q = q / q.sum()
        h = stsp_metrics.hellinger_distance(p, q)
        assert np.isfinite(h)
        assert 0.0 <= h <= 1.0

    def test_state_space_divergence_auto_routes_to_binning(self):
        x_true = np.random.randn(200, 2)
        x_gen = x_true + 0.1 * np.random.randn(200, 2)
        d = stsp_metrics.state_space_divergence(x_gen, x_true)
        assert np.isfinite(d)
