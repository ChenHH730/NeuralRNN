"""Tests for Lyapunov exponent computation."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from neuralrnn import AutoConfig, AutoModel
from neuralrnn.analysis.lyapunov import max_lyapunov_exponent


def test_lyapunov_without_dt():
    """For a deterministic contraction A=0.5, inactive ReLUs, exponent = log(0.5)."""
    cfg = AutoConfig.for_model("shallow_plrnn", input_dim=0, latent_dim=2,
                               output_dim=2, hidden_dim=2, autonomous=True)
    model = AutoModel.from_config(cfg)
    # Make ReLUs inactive and A = 0.5
    with torch.no_grad():
        model.A.fill_(0.5)
        model.W1.zero_()
        model.h1.zero_()
        model.h2.fill_(10.0)  # pre-activation positive -> all ReLUs on? Wait, we want inactive.
        # ReLU is inactive when W2 z + h2 <= 0. With h2 very negative, inactive for small z.
        model.h2.fill_(-10.0)
    z1 = torch.zeros(2)
    lam = max_lyapunov_exponent(model, z1, T=1000, T_trans=100, ons=1)
    assert np.isclose(lam, np.log(0.5), atol=0.05)


def test_lyapunov_with_dt():
    """Passing dt scales the discrete-time exponent by 1/dt."""
    cfg = AutoConfig.for_model("shallow_plrnn", input_dim=0, latent_dim=2,
                               output_dim=2, hidden_dim=2, autonomous=True)
    model = AutoModel.from_config(cfg)
    with torch.no_grad():
        model.A.fill_(0.5)
        model.W1.zero_()
        model.h1.zero_()
        model.h2.fill_(-10.0)
    z1 = torch.zeros(2)
    lam_discrete = max_lyapunov_exponent(model, z1, T=1000, T_trans=100, ons=1)
    lam_cont = max_lyapunov_exponent(model, z1, T=1000, T_trans=100, ons=1, dt=0.01)
    assert np.isclose(lam_cont, lam_discrete / 0.01, atol=1e-6)


def test_lyapunov_uses_config_dt():
    """If dt is not passed but model.config.dt exists, use it."""
    cfg = AutoConfig.for_model("shallow_plrnn", input_dim=0, latent_dim=2,
                               output_dim=2, hidden_dim=2, autonomous=True, dt=0.02)
    model = AutoModel.from_config(cfg)
    with torch.no_grad():
        model.A.fill_(0.5)
        model.W1.zero_()
        model.h1.zero_()
        model.h2.fill_(-10.0)
    z1 = torch.zeros(2)
    lam = max_lyapunov_exponent(model, z1, T=1000, T_trans=100, ons=1)
    assert np.isclose(lam, np.log(0.5) / 0.02, atol=0.05)


def test_lyapunov_dt_none_preserves_discrete():
    """If dt is None and config has no dt, return discrete-time value."""
    cfg = AutoConfig.for_model("shallow_plrnn", input_dim=0, latent_dim=2,
                               output_dim=2, hidden_dim=2, autonomous=True)
    model = AutoModel.from_config(cfg)
    with torch.no_grad():
        model.A.fill_(0.5)
        model.W1.zero_()
        model.h1.zero_()
        model.h2.fill_(-10.0)
    z1 = torch.zeros(2)
    lam = max_lyapunov_exponent(model, z1, T=1000, T_trans=100, ons=1, dt=None)
    assert np.isclose(lam, np.log(0.5), atol=0.05)
