"""Tests for the unified activation factory."""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from neuralrnn.activations import get_activation, SUPPORTED_ACTIVATIONS


@pytest.mark.parametrize(
    "name",
    ["relu", "tanh", "sigmoid", "selu", "gelu", "silu", "swish"],
)
def test_base_activations_run(name: str):
    fn = get_activation(name)
    x = torch.randn(4, 8)
    y = fn(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_supported_activations_includes_parameterized():
    assert "softplus" in SUPPORTED_ACTIVATIONS
    assert "leaky_relu" in SUPPORTED_ACTIVATIONS
    assert "leakyrelu" in SUPPORTED_ACTIVATIONS
    assert "elu" in SUPPORTED_ACTIVATIONS


def test_softplus_beta():
    x = torch.randn(4, 8)
    fn = get_activation("softplus", beta=2.0)
    assert torch.allclose(fn(x), F.softplus(x, beta=2.0))


def test_softplus_default_beta_matches_reference():
    x = torch.randn(4, 8)
    fn = get_activation("softplus")
    assert torch.allclose(fn(x), F.softplus(x, beta=1.0))


def test_leaky_relu_aliases_and_slope():
    x = torch.randn(4, 8)
    fn1 = get_activation("leaky_relu", negative_slope=0.1)
    fn2 = get_activation("leakyrelu", negative_slope=0.1)
    expected = F.leaky_relu(x, negative_slope=0.1)
    assert torch.allclose(fn1(x), expected)
    assert torch.allclose(fn2(x), expected)


def test_leaky_relu_alpha_alias():
    x = torch.randn(4, 8)
    fn = get_activation("leaky_relu", alpha=0.2)
    assert torch.allclose(fn(x), F.leaky_relu(x, negative_slope=0.2))


def test_elu_alpha():
    x = torch.randn(4, 8)
    fn = get_activation("elu", alpha=1.5)
    assert torch.allclose(fn(x), F.elu(x, alpha=1.5))


def test_unknown_activation_raises():
    with pytest.raises(ValueError, match="Unknown activation"):
        get_activation("not_an_activation")


def test_unexpected_kwargs_raise():
    with pytest.raises(ValueError, match="does not accept kwargs"):
        get_activation("relu", beta=1.0)


def test_case_and_dash_insensitive():
    x = torch.randn(4, 8)
    fn1 = get_activation("Leaky-ReLU")
    fn2 = get_activation("leaky_relu")
    assert torch.allclose(fn1(x), fn2(x))
