"""Tests for the stp_rnn model (StpRNNModel): short-term plasticity as a
dynamic gain parameterization of the gain_rnn family.

Covers family defaults (notebook-11 parity), STP dynamics (hand-computed),
init modes (constant / alternating / random), the neuromodulator stp_alpha,
freeze behavior, save-load roundtrips, analysis-tool fallbacks, and a full
notebook-11 parity integration test against an independent re-implementation
of the reference math.
"""
import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F

from neuralrnn import AutoConfig, AutoModel, Trainer, TrainingArguments, SupervisedObjective
from neuralrnn.data import BaseDataset
from neuralrnn.models.gain_rnn import StpRNNConfig, StpRNNModel, make_stp_masks


class _ToyRegressionTask(BaseDataset):
    kind = "neurogym"

    def __init__(self, input_dim=3, output_dim=2, T=12, B=6):
        self.input_dim, self.output_dim, self.T, self.B = input_dim, output_dim, T, B

    def sample_batch(self):
        return {
            "inputs": torch.randn(self.B, self.T, self.input_dim),
            "targets": torch.randn(self.B, self.T, self.output_dim),
            "mask": None,
        }

    def task_input(self):
        return torch.zeros(self.input_dim)


def _make_stp(**overrides):
    kw = dict(input_dim=3, latent_dim=6, output_dim=2)
    kw.update(overrides)
    return AutoModel.from_config(AutoConfig.for_model("stp_rnn", **kw))


# ============================ registration / defaults ============================

class TestRegistration:
    def test_shapes_and_extras(self):
        model = _make_stp()
        out = model(torch.randn(4, 10, 3))
        assert out.outputs.shape == (4, 10, 2)
        assert out.states.shape == (4, 10, 6)  # states = h only
        assert out.extras["syn_x"].shape == (4, 10, 6)
        assert out.extras["syn_u"].shape == (4, 10, 6)
        assert isinstance(model, StpRNNModel)

    def test_family_defaults_nb11(self):
        cfg = AutoConfig.for_model("stp_rnn", input_dim=3, latent_dim=6, output_dim=2)
        assert cfg.nonlinearity_mode == "post_blend"
        assert cfg.activation == "relu"
        assert cfg.noise_position == "post"
        assert cfg.noise_alpha_scaling is True
        assert cfg.positive_input_weights is True
        assert cfg.positive_output_weights is True
        assert cfg.freeze_gain is True and cfg.freeze_bias is True
        assert cfg.freeze_stp is True
        assert cfg.dt == 10.0 and cfg.tau == 100.0
        assert cfg.h0_init == 0.1
        assert cfg.stp_init == "constant"

    def test_dt_none_raises(self):
        with pytest.raises(ValueError, match="physical dt"):
            _make_stp(dt=None, alpha=0.5)

    def test_default_freeze_pattern(self):
        model = _make_stp()
        grads = {n: p.requires_grad for n, p in model.named_parameters()}
        for n in ("stp_tau_x", "stp_tau_u", "stp_U", "gain", "bias"):
            assert not grads[n], f"{n} should be frozen by default"
        for n in ("h2h.weight", "h2h.bias", "input2h.weight", "readout_layer.weight"):
            assert grads[n], f"{n} should be trainable"
        # notebook-11 parity: input bias zeroed and frozen (no input bias in the reference).
        assert not grads["input2h.bias"]
        assert model.input2h.bias.abs().sum().item() == 0.0


# ============================ STP dynamics ============================

class TestStpDynamics:
    def test_stp_step_handcomputed(self):
        model = _make_stp(stp_tau_x=200.0, stp_tau_u=1500.0, stp_U=0.2)
        r = torch.rand(5, 6) * 30  # ~ spikes/s
        x = torch.rand(5, 6)
        u = torch.rand(5, 6)
        x_new, u_new = model._stp_step(r, x, u)
        dt, dt_sec = 10.0, 0.01
        exp_x = x + (dt / 200.0) * (1 - x) - dt_sec * u * x * r
        exp_u = u + (dt / 1500.0) * (0.2 - u) + dt_sec * 0.2 * (1 - u) * r
        assert torch.allclose(x_new, exp_x.clamp(0, 1), atol=1e-6)
        assert torch.allclose(u_new, exp_u.clamp(0, 1), atol=1e-6)

    def test_stp_step_clamps(self):
        model = _make_stp(stp_tau_x=200.0, stp_tau_u=1500.0, stp_U=0.2)
        r = torch.full((2, 6), 1e4)
        x_new, u_new = model._stp_step(r, torch.ones(2, 6), torch.ones(2, 6))
        assert (x_new >= 0).all() and (x_new <= 1).all()
        assert (u_new >= 0).all() and (u_new <= 1).all()

    def test_stp_alpha_scales_U(self):
        model = _make_stp(stp_U=0.4)
        model.set_stp_alpha(0.5)
        assert torch.allclose(model._effective_U(), torch.full((6,), 0.2))
        model.set_stp_alpha(4.0)
        assert torch.allclose(model._effective_U(), torch.ones(6))  # clamped

    def test_full_state_single_step_handcomputed(self):
        torch.manual_seed(0)
        M, K = 4, 3
        model = _make_stp(latent_dim=M, input_dim=K, stp_tau_x=200.0,
                          stp_tau_u=1500.0, stp_U=0.2, sigma_rec=0.0,
                          positive_input_weights=False)
        with torch.no_grad():
            model.gain.copy_(torch.ones(M))  # frozen anyway; make explicit
            model.bias.zero_()
        model.eval()
        x_t = torch.randn(2, K)
        h = torch.rand(2, M)
        syn_x = torch.rand(2, M)
        syn_u = torch.rand(2, M)
        z_prev = torch.cat([h, syn_x, syn_u], dim=-1)
        z_new = model.recurrence(x_t, z_prev)

        dt, dt_sec, alpha = 10.0, 0.01, 0.1
        x1 = (syn_x + (dt / 200.0) * (1 - syn_x) - dt_sec * syn_u * syn_x * h).clamp(0, 1)
        u1 = (syn_u + (dt / 1500.0) * (0.2 - syn_u) + dt_sec * 0.2 * (1 - syn_u) * h).clamp(0, 1)
        h_post = h * x1 * u1
        pre = F.linear(x_t, model.input2h.weight, model.input2h.bias) + F.linear(
            h_post, model.h2h.weight, model.h2h.bias)
        exp_h = torch.relu((1 - alpha) * h + alpha * pre)
        assert torch.allclose(z_new[..., :M], exp_h, atol=1e-6)
        assert torch.allclose(z_new[..., M:2 * M], x1, atol=1e-6)
        assert torch.allclose(z_new[..., 2 * M:], u1, atol=1e-6)

    def test_m_dim_fallback_equals_unit_efficacy(self):
        model = _make_stp()
        model.eval()
        h = torch.rand(3, 6)
        x_t = torch.randn(3, 3)
        z_full = torch.cat([h, torch.ones(3, 6), torch.ones(3, 6)], dim=-1)
        # With x=u=1 the STP step changes x/u but h_post = r*x'*u' is computed
        # from the updated variables in both paths; the M-fallback uses the same
        # rule with x=u=1 as the previous step values.
        got_m = model.recurrence(x_t, h)
        got_full = model.recurrence(x_t, z_full)[..., :6]
        assert torch.allclose(got_m, got_full, atol=1e-6)
        assert got_m.shape == (3, 6)

    def test_stp_alpha_changes_trajectory(self):
        model = _make_stp()
        model.eval()
        x = torch.randn(2, 15, 3)
        out1 = model(x).states
        model.set_stp_alpha(0.5)
        out2 = model(x).states
        assert not torch.allclose(out1, out2)


# ============================ init modes ============================

class TestStpInit:
    def test_alternating(self):
        model = _make_stp(latent_dim=6, stp_init="alternating")
        tau_x, tau_u, U = model.stp_tau_x, model.stp_tau_u, model.stp_U
        # even = facilitating, odd = depressing (notebook 11)
        assert torch.allclose(tau_x[0::2], torch.full((3,), 1500.0))
        assert torch.allclose(tau_u[0::2], torch.full((3,), 200.0))
        assert torch.allclose(U[0::2], torch.full((3,), 0.15))
        assert torch.allclose(tau_x[1::2], torch.full((3,), 200.0))
        assert torch.allclose(tau_u[1::2], torch.full((3,), 1500.0))
        assert torch.allclose(U[1::2], torch.full((3,), 0.45))

    def test_random_reproducible_and_bounded(self):
        m1 = _make_stp(latent_dim=32, stp_init="random", stp_seed=42)
        m2 = _make_stp(latent_dim=32, stp_init="random", stp_seed=42)
        for n in ("stp_tau_x", "stp_tau_u", "stp_U"):
            assert torch.allclose(getattr(m1, n), getattr(m2, n))
        assert (m1.stp_U >= 0.001).all() and (m1.stp_U <= 0.99).all()
        assert (m1.stp_tau_x >= 100.0).all() and (m1.stp_tau_x <= 3000.0).all()
        # tau_x and tau_u sampled independently (Zhou & Buonomano 2024).
        assert not torch.allclose(m1.stp_tau_x, m1.stp_tau_u)

    def test_explicit_arrays_take_precedence(self):
        tau_x = np.linspace(100, 900, 6)
        model = _make_stp(stp_init="alternating", stp_tau_x=tau_x)
        assert torch.allclose(model.stp_tau_x, torch.tensor(tau_x, dtype=torch.float32))

    def test_constant_broadcast(self):
        model = _make_stp(stp_tau_x=333.0, stp_U=0.33)
        assert torch.allclose(model.stp_tau_x, torch.full((6,), 333.0))
        assert torch.allclose(model.stp_U, torch.full((6,), 0.33))


# ============================ init_state / stp_alpha ============================

class TestStateAndAlpha:
    def test_init_state_steady(self):
        model = _make_stp(stp_U=0.3)
        s0 = model.init_state(4)
        assert s0.shape == (4, 18)
        assert torch.allclose(s0[:, :6], torch.full((4, 6), 0.1))
        assert torch.allclose(s0[:, 6:12], torch.ones(4, 6))
        assert torch.allclose(s0[:, 12:], torch.full((4, 6), 0.3), atol=1e-6)

    def test_init_state_uses_alpha_scaled_U(self):
        model = _make_stp(stp_U=0.4, stp_alpha=0.5)
        s0 = model.init_state(2)
        assert torch.allclose(s0[:, 12:], torch.full((2, 6), 0.2), atol=1e-6)

    def test_set_stp_alpha(self):
        model = _make_stp()
        model.set_stp_alpha(0.8)
        assert torch.allclose(model.get_stp_alpha(), torch.full((6,), 0.8))
        model.set_stp_alpha(torch.linspace(0.6, 1.0, 6))
        assert torch.allclose(model.get_stp_alpha(), torch.linspace(0.6, 1.0, 6))
        with pytest.raises(ValueError, match="non-negative"):
            model.set_stp_alpha(-0.1)
        with pytest.raises(ValueError, match="scalar or have shape"):
            model.set_stp_alpha(torch.ones(5))

    def test_forward_m_dim_initial_state_raises(self):
        model = _make_stp()
        with pytest.raises(ValueError, match="last dim"):
            model(torch.randn(2, 5, 3), initial_state=torch.zeros(2, 6))


# ============================ notebook-11 parity integration ============================

def _nb11_reference_rollout(model, inputs):
    """Independent re-implementation of the notebook-11 reference math
    (Masse et al. 2019 style, copied from notebook 11 cell 5) using the
    parameters of `model`. Used to cross-check StpRNNModel end to end."""
    cfg = model.config
    M = cfg.latent_dim
    alpha = cfg.dt / cfg.tau
    dt_sec = cfg.dt / 1000.0
    e_size = int(round(M * cfg.ei_ratio))
    sign = torch.ones(M)
    sign[e_size:] = -1.0
    dale = torch.diag(sign)
    w_rnn_mask = 1.0 - torch.eye(M)
    w_out_mask = torch.ones(cfg.output_dim, M)
    w_out_mask[:, e_size:] = 0.0

    W_rec = model.h2h.weight.detach().abs() * w_rnn_mask @ dale
    w_in = F.relu(model.input2h.weight.detach())
    w_out = F.relu(model.readout_layer.weight.detach()) * w_out_mask
    alpha_x = cfg.dt / model.stp_tau_x.detach()
    alpha_u = cfg.dt / model.stp_tau_u.detach()
    U = model.stp_U.detach()

    h = model.h0.detach().expand(inputs.shape[0], -1).clone()
    sx, su = torch.ones_like(h), U.expand_as(h).clone()
    hs, outs = [], []
    for t in range(inputs.shape[1]):
        x_new = (sx + alpha_x * (1 - sx) - dt_sec * su * sx * h).clamp(0, 1)
        u_new = (su + alpha_u * (U - su) + dt_sec * U * (1 - su) * h).clamp(0, 1)
        sx, su = x_new, u_new
        h_post = su * sx * h
        pre = F.linear(inputs[:, t], w_in) + F.linear(h_post, W_rec, model.h2h.bias.detach())
        h = torch.relu((1 - alpha) * h + alpha * pre)
        hs.append(h)
        outs.append(F.linear(h, w_out, model.readout_layer.bias.detach()))
    return torch.stack(hs, 1), torch.stack(outs, 1)


def test_notebook11_parity():
    """Full-rollout parity against an independent re-implementation of the
    notebook-11 reference math (noise off)."""
    torch.manual_seed(7)
    M, K, O = 8, 3, 2
    masks = make_stp_masks(M, O, ei_ratio=0.8)
    model = _make_stp(latent_dim=M, input_dim=K, output_dim=O,
                      stp_init="alternating", dale=True, ei_ratio=0.8,
                      sigma_rec=0.0, init_method="gamma", init_seed=3,
                      rec_mask=masks["rec_mask"], out_mask=masks["out_mask"])
    model.eval()
    x = torch.randn(3, 30, K)
    out = model(x)
    ref_h, ref_y = _nb11_reference_rollout(model, x)
    assert torch.allclose(out.states, ref_h, atol=1e-5)
    assert torch.allclose(out.outputs, ref_y, atol=1e-5)


# ============================ freeze / training ============================

class TestFreezeAndTraining:
    def test_stp_group(self):
        model = _make_stp()
        assert model._match_parameters(["stp"], None) == {"stp_U", "stp_tau_u", "stp_tau_x"}

    def test_freeze_stp_false_unfreezes(self):
        model = _make_stp(freeze_stp=False)
        grads = {n: p.requires_grad for n, p in model.named_parameters()}
        assert grads["stp_tau_x"] and grads["stp_tau_u"] and grads["stp_U"]
        # gain/bias stay frozen (their own flags default True for stp_rnn).
        assert not grads["gain"] and not grads["bias"]

    def test_training_keeps_stp_frozen(self):
        model = _make_stp()
        before = copy.deepcopy(model.state_dict())
        Trainer(model, _ToyRegressionTask(), SupervisedObjective("regression"),
                TrainingArguments(max_steps=2, log_every=0)).train()
        after = model.state_dict()
        for n in ("stp_tau_x", "stp_tau_u", "stp_U", "gain", "bias", "input2h.bias"):
            assert torch.allclose(before[n], after[n], atol=1e-10), f"{n} changed"
        assert not torch.allclose(before["h2h.weight"], after["h2h.weight"], atol=1e-10)

    def test_freeze_roundtrip(self, tmp_path):
        model = _make_stp(freeze_stp=False, freeze_recurrent=True)
        before = {n: p.requires_grad for n, p in model.named_parameters()}
        model.save_pretrained(str(tmp_path))
        reloaded = AutoModel.from_pretrained(str(tmp_path))
        after = {n: p.requires_grad for n, p in reloaded.named_parameters()}
        assert before == after


# ============================ serialization ============================

def test_save_load_roundtrip_with_alpha(tmp_path):
    model = _make_stp(stp_init="random", stp_seed=5)
    model.set_stp_alpha(0.7)
    model.eval()
    x = torch.randn(2, 9, 3)
    out_before = model(x)
    model.save_pretrained(str(tmp_path))
    loaded = AutoModel.from_pretrained(str(tmp_path))
    loaded.eval()
    for (n1, p1), (n2, p2) in zip(model.state_dict().items(), loaded.state_dict().items()):
        assert n1 == n2 and torch.allclose(p1, p2), n1
    # The runtime stp_alpha buffer is restored from the checkpoint.
    assert torch.allclose(loaded.get_stp_alpha(), torch.full((6,), 0.7))
    out_after = loaded(x)
    assert torch.allclose(out_before.outputs, out_after.outputs, atol=1e-6)
    assert torch.allclose(out_before.extras["syn_u"], out_after.extras["syn_u"], atol=1e-6)


def test_config_roundtrip_arrays():
    cfg = StpRNNConfig(input_dim=3, latent_dim=4, output_dim=2,
                       stp_tau_x=np.linspace(100, 400, 4), stp_U=[0.1, 0.2, 0.3, 0.4],
                       stp_alpha=np.full(4, 0.9))
    import json
    json.dumps(cfg.to_dict())
    cfg2 = StpRNNConfig.from_dict(cfg.to_dict())
    assert cfg2.stp_tau_x == [100.0, 200.0, 300.0, 400.0]
    assert cfg2.stp_U == [0.1, 0.2, 0.3, 0.4]
    model = AutoModel.from_config(cfg2)
    assert torch.allclose(model.stp_U, torch.tensor([0.1, 0.2, 0.3, 0.4]))


# ============================ analysis fallbacks ============================

class TestAnalysis:
    def test_jacobian_m_dim(self):
        model = _make_stp()
        model.eval()
        J = model.jacobian(torch.rand(6) * 0.5)
        assert J.shape == (6, 6)
        assert torch.isfinite(J).all()

    def test_fixed_point_finder_smoke(self):
        from neuralrnn.analysis.fixed_points import NumericFixedPointFinder

        model = _make_stp()
        finder = NumericFixedPointFinder(n_candidates=4, n_iters=5)
        fps = finder.find(model)
        assert len(fps.points) >= 1  # fallback keeps the best candidate

    def test_dropout_h_only(self):
        model = _make_stp()
        model.train()
        x = torch.randn(2, 8, 3)
        sc, oc, sd, od = model.forward_with_dropout(x, dropout_rate=0.3)
        assert sc.shape == (2, 8, 6) and sd.shape == (2, 8, 6)
        assert oc.shape == (2, 8, 2) and od.shape == (2, 8, 2)
        sc0, oc0, sd0, od0 = model.forward_with_dropout(x, dropout_rate=0.0)
        assert torch.allclose(sc0, sd0) and torch.allclose(oc0, od0)


# ============================ gamma init / masks helper ============================

class TestGammaInit:
    def test_shapes_diag_and_seed(self):
        m1 = _make_stp(latent_dim=16, init_method="gamma", init_seed=11)
        m2 = _make_stp(latent_dim=16, init_method="gamma", init_seed=11)
        assert torch.allclose(m1.h2h.weight, m2.h2h.weight)
        assert (m1.h2h.weight >= 0).all()
        assert m1.h2h.weight.diagonal().abs().sum().item() == 0.0
        e_size = int(round(16 * 0.8))
        assert (m1.readout_layer.weight[:, e_size:] == 0).all()
        assert (m1.input2h.weight >= 0).all()

    def test_make_stp_masks(self):
        masks = make_stp_masks(10, 3, ei_ratio=0.8)
        assert masks["rec_mask"].shape == (10, 10)
        assert masks["out_mask"].shape == (10, 3)
        assert masks["rec_mask"].diagonal().sum() == 0.0
        assert masks["out_mask"][8:, :].sum() == 0.0
        assert masks["out_mask"][:8, :].sum() == 8 * 3


# ============================ noise behavior ============================

def test_noise_only_in_training():
    model = _make_stp(sigma_rec=0.5)
    x_t, z = torch.randn(2, 3), model.init_state(2)
    model.eval()
    a, b = model.recurrence(x_t, z), model.recurrence(x_t, z)
    assert torch.allclose(a, b)
    model.train()
    torch.manual_seed(0)
    c = model.recurrence(x_t, z)
    torch.manual_seed(1)
    d = model.recurrence(x_t, z)
    assert not torch.allclose(c, d)


def test_synaptic_efficacy_helper():
    model = _make_stp()
    out = model(torch.randn(2, 6, 3))
    eff = StpRNNModel.synaptic_efficacy(out.extras)
    assert torch.allclose(eff, out.extras["syn_x"] * out.extras["syn_u"])
