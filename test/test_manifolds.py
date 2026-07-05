"""Tests for PLRNN manifold tracing."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from neuralrnn import AutoConfig, AutoModel
from neuralrnn.analysis.manifolds import (
    PLRNNManifoldTracer,
    compute_manifold,
    _local_manifold_basis,
    _region_id,
)
from neuralrnn.analysis.fixed_points import find_fixed_points


def test_region_id_consistency():
    """Region ID should be deterministic for a given activation pattern."""
    M, L = 3, 4
    A = np.eye(M) * 0.5
    W1 = np.random.randn(M, L)
    W2 = np.random.randn(L, M)
    h1 = np.zeros(M)
    h2 = np.zeros(L)
    z = np.random.randn(M)
    id1 = _region_id(z, W2, h2)
    id2 = _region_id(z, W2, h2)
    assert id1 == id2


def test_local_manifold_basis_dimensions():
    """Local basis should have correct shape and dimension."""
    M, L = 4, 5
    A = np.eye(M) * 0.5
    W1 = np.random.randn(M, L) * 0.1
    W2 = np.random.randn(L, M) * 0.1
    h1 = np.zeros(M)
    h2 = np.zeros(L)
    z = np.random.randn(M)
    basis, eigvals, dim = _local_manifold_basis(z, A, W1, W2, h2, stable=True)
    assert basis.shape == (dim, M)
    assert eigvals.shape == (dim,)


def test_tracer_returns_segments():
    """Tracer should return at least the initial segment on a simple PLRNN."""
    cfg = AutoConfig.for_model("shallow_plrnn", input_dim=0, latent_dim=3,
                               output_dim=3, hidden_dim=8, autonomous=True)
    model = AutoModel.from_config(cfg)
    fps = find_fixed_points(model, backend="analytic", max_order=1, outer_it=10, inner_it=5)
    if len(fps) == 0:
        pytest.skip("No fixed points found for this random model")
    fp = fps.points[0]

    trace = compute_manifold(
        model, fixed_point=fp.z, stable=False,
        n_samples=50, factor=0.05, max_iter=1, propagation_steps=10
    )
    assert len(trace.segments) >= 1
    assert trace.fixed_point is not None
    assert trace.fixed_point.shape == (3,)


def test_tracer_task_input():
    """Tracer should work with a non-autonomous model and constant task input."""
    cfg = AutoConfig.for_model("shallow_plrnn", input_dim=2, latent_dim=3,
                               output_dim=3, hidden_dim=8, autonomous=False)
    model = AutoModel.from_config(cfg)
    task_input = torch.tensor([0.5, -0.2], dtype=torch.float32)

    fps = find_fixed_points(
        model, backend="analytic", task_input=task_input,
        max_order=1, outer_it=10, inner_it=5
    )
    if len(fps) == 0:
        pytest.skip("No fixed points found for this random model")
    fp = fps.points[0]

    trace = compute_manifold(
        model, fixed_point=fp.z, task_input=task_input, stable=False,
        n_samples=50, factor=0.05, max_iter=1, propagation_steps=10
    )
    assert len(trace.segments) >= 1


def test_tracer_stable_unstable_different():
    """Stable and unstable traces should differ for a saddle."""
    # Construct a simple 2D saddle: z1 unstable, z2 stable
    cfg = AutoConfig.for_model("shallow_plrnn", input_dim=0, latent_dim=2,
                               output_dim=2, hidden_dim=2, autonomous=True)
    model = AutoModel.from_config(cfg)
    with torch.no_grad():
        model.A.fill_(0.0)
        model.W1.zero_()
        model.W2.zero_()
        model.h1.zero_()
        model.h2.fill_(-10.0)  # inactive ReLUs
        # Set A diag manually: A[0]=1.5 (unstable), A[1]=0.5 (stable)
        model.A[0] = 1.5
        model.A[1] = 0.5

    z0 = np.zeros(2)
    trace_unstable = compute_manifold(
        model, fixed_point=z0, stable=False,
        n_samples=30, factor=0.1, max_iter=1
    )
    trace_stable = compute_manifold(
        model, fixed_point=z0, stable=True,
        n_samples=30, factor=0.1, max_iter=1
    )

    # Unstable basis should be 1D along z0; stable basis 1D along z1
    unstable_basis = trace_unstable.segments[0].basis
    stable_basis = trace_stable.segments[0].basis
    assert unstable_basis.shape[0] == 1
    assert stable_basis.shape[0] == 1
    # Bases should be roughly orthogonal
    dot = abs(np.dot(unstable_basis[0], stable_basis[0]))
    assert dot < 0.5  # loose due to random init perturbations
