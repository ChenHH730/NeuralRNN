"""Tests for the unified Euler-step resolution (alpha > dt/tau > family default/1.0).

See ``neuralrnn.configuration_utils.resolve_euler_alpha``.
"""
import json
import os
import warnings

import pytest

from neuralrnn import AutoConfig, AutoModel
from neuralrnn.configuration_utils import resolve_euler_alpha


# ---------------- resolve_euler_alpha unit tests ----------------

def test_alpha_explicit_wins():
    with pytest.warns(UserWarning, match="alpha takes precedence"):
        alpha, dt = resolve_euler_alpha(10.0, 100.0, 0.3)  # inconsistent pair
    assert alpha == 0.3 and dt == 10.0


def test_dt_tau_fallback():
    alpha, dt = resolve_euler_alpha(10.0, 50.0, None)
    assert alpha == pytest.approx(0.2) and dt == 10.0


def test_discrete_default():
    alpha, dt = resolve_euler_alpha(None, 100.0, None, default_dt=None)
    assert alpha == 1.0 and dt is None


def test_family_default_dt():
    alpha, dt = resolve_euler_alpha(None, 100.0, None, default_dt=20.0)
    assert alpha == pytest.approx(0.2) and dt == 20.0


def test_conflict_warns_alpha_wins():
    with pytest.warns(UserWarning, match="alpha takes precedence"):
        alpha, _ = resolve_euler_alpha(50.0, 100.0, 0.2, model_type="test")
    assert alpha == 0.2


def test_consistent_pair_no_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        alpha, _ = resolve_euler_alpha(20.0, 100.0, 0.2)
    assert alpha == 0.2


def test_invalid_values():
    with pytest.raises(ValueError, match="alpha"):
        resolve_euler_alpha(None, 100.0, 0.0)
    with pytest.raises(ValueError, match="tau"):
        resolve_euler_alpha(10.0, 0.0, None)
    with pytest.warns(UserWarning, match="unstable"):
        resolve_euler_alpha(None, 100.0, 1.5)


# ---------------- per-family defaults ----------------

@pytest.mark.parametrize("model_type,exp_alpha,exp_dt", [
    ("ctrnn", 1.0, 100.0),
    ("ei_rnn", 1.0, 100.0),
    ("constrained_rnn", 1.0, 100.0),
    ("latent_circuit", 0.2, 40.0),
    ("connectome_rnn", 1.0, None),
    ("lowrank_rnn", 0.2, 20.0),
])
def test_family_defaults(model_type, exp_alpha, exp_dt):
    cfg = AutoConfig.for_model(model_type)
    assert cfg.alpha == pytest.approx(exp_alpha)
    assert cfg.dt == exp_dt


@pytest.mark.parametrize("model_type", ["ctrnn", "ei_rnn", "constrained_rnn", "latent_circuit"])
def test_explicit_alpha_everywhere(model_type):
    """Explicit alpha is honored by every continuous-time family (previously
    silently ignored by ctrnn-family models)."""
    cfg = AutoConfig.for_model(model_type, input_dim=2, latent_dim=8, output_dim=1, alpha=0.3)
    assert cfg.alpha == 0.3
    model = AutoModel.from_config(cfg)
    assert model.alpha == 0.3


def test_latent_circuit_dt_none_no_crash():
    """Previously raised TypeError: unsupported operand for NoneType / float."""
    cfg = AutoConfig.for_model("latent_circuit", dt=None)
    assert cfg.alpha == pytest.approx(0.2)  # family default dt=40, tau=200
    model = AutoModel.from_config(cfg)
    assert model.alpha == pytest.approx(0.2)


def test_lowrank_explicit_alpha_not_overridden_by_dt():
    """Regression: old sentinel logic silently let dt win when alpha == 0.2."""
    with pytest.warns(UserWarning, match="alpha takes precedence"):
        cfg = AutoConfig.for_model("lowrank_rnn", alpha=0.2, dt=50.0, tau=100.0)
    assert cfg.alpha == 0.2


# ---------------- serialization / old-checkpoint compatibility ----------------

def test_old_ctrnn_checkpoint_without_alpha(tmp_path):
    old = {"model_type": "ctrnn", "input_dim": 2, "latent_dim": 8, "output_dim": 1,
           "dt": 10.0, "tau": 50.0, "activation": "relu"}
    (tmp_path / "config.json").write_text(json.dumps(old))
    cfg = AutoConfig.from_pretrained(str(tmp_path))
    assert cfg.alpha == pytest.approx(0.2)


def test_old_lowrank_checkpoint_with_noise_std(tmp_path):
    old = {"model_type": "lowrank_rnn", "input_dim": 1, "latent_dim": 10, "output_dim": 1,
           "rank": 1, "alpha": 0.2, "noise_std": 0.07, "dt": None, "tau": 100.0}
    (tmp_path / "config.json").write_text(json.dumps(old))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        cfg = AutoConfig.from_pretrained(str(tmp_path))
    assert cfg.alpha == 0.2
    assert cfg.sigma_rec == 0.07


def test_noise_std_deprecation_warning():
    with pytest.warns(DeprecationWarning, match="sigma_rec"):
        cfg = AutoConfig.for_model("lowrank_rnn", noise_std=0.07)
    assert cfg.sigma_rec == 0.07


def test_sigma_rec_explicit_not_overridden_by_deprecated():
    with pytest.warns(DeprecationWarning):
        cfg = AutoConfig.for_model("lowrank_rnn", sigma_rec=0.02, noise_std=0.07)
    assert cfg.sigma_rec == 0.02  # non-default sigma_rec keeps precedence


def test_new_checkpoint_roundtrip_no_warnings(tmp_path):
    cfg = AutoConfig.for_model("lowrank_rnn")
    cfg.to_json_file(str(tmp_path))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cfg2 = AutoConfig.from_pretrained(str(tmp_path))
    assert cfg2.alpha == 0.2 and cfg2.dt == 20.0 and cfg2.sigma_rec == 0.05
