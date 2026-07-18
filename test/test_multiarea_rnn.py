"""Tests for the multiarea_rnn family, the dale_signs CTRNN extension, the
checkerboard task, and the demixed analysis module.

Covers Contract-A acceptance (AutoConfig/AutoModel/save-load), mask structure
(block cascade, E-only inter-area sources, densities, per-area Dale), gradient
absence at masked positions, parity between MultiAreaRNNModel and a manually
masked ConstrainedRNNModel, checkerboard task shapes/semantics, and dPCA
recovery on synthetic data.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from neuralrnn import AutoConfig, AutoModel
from neuralrnn.models.multiarea_rnn import (
    MultiAreaRNNConfig,
    MultiAreaRNNModel,
    build_multiarea_masks,
    area_slices,
    area_ei_indices,
)
from neuralrnn.models.constrained_rnn import (
    ConstrainedRNNConfig,
    ConstrainedRNNModel,
)
from neuralrnn.data.tasks import checkerboard_trials


def _make_multi(**overrides):
    kw = dict(input_dim=4, output_dim=2, area_sizes=[20, 20, 20], dt=10.0, tau=50.0)
    kw.update(overrides)
    return AutoModel.from_config(AutoConfig.for_model("multiarea_rnn", **kw))


# ============================ Contract A acceptance ============================

def test_registration_and_forward_shapes():
    m = _make_multi()
    assert isinstance(m, MultiAreaRNNModel)
    assert m.config.model_type == "multiarea_rnn"
    out = m(torch.randn(2, 15, 4))
    assert out.states.shape == (2, 15, 60)
    assert out.outputs.shape == (2, 15, 2)


def test_save_load_roundtrip(tmp_path):
    m = _make_multi()
    m.save_pretrained(tmp_path)
    m2 = AutoModel.from_pretrained(tmp_path)
    assert isinstance(m2, MultiAreaRNNModel)
    assert m2.config.area_sizes == [20, 20, 20]
    assert m2.config.mask_seed == m.config.mask_seed
    sd1, sd2 = m.state_dict(), m2.state_dict()
    assert set(sd1) == set(sd2)
    for k in sd1:
        assert torch.allclose(sd1[k], sd2[k]), k


def test_latent_dim_inferred_from_areas():
    cfg = MultiAreaRNNConfig(input_dim=4, output_dim=2, area_sizes=[10, 30])
    assert cfg.latent_dim == 40
    with pytest.raises(ValueError):
        MultiAreaRNNConfig(input_dim=4, output_dim=2, area_sizes=[10, 10], latent_dim=15)


# ============================ mask structure ============================

def test_mask_block_structure_and_densities():
    rng_seed = 7
    rec, in_m, out_m, signs = build_multiarea_masks(
        area_sizes=[100, 100, 100], input_dim=4, output_dim=2, mask_seed=rng_seed)
    s = area_slices([100, 100, 100])
    e, i = area_ei_indices([100, 100, 100], 0.8)[1:]

    # Intra-area blocks dense
    for a in range(3):
        assert rec[s[a], s[a]].mean() == 1.0
    # Feedforward densities (10% E->E, 2% E->I), feedback 5% E->E
    assert abs(rec[np.ix_(e[1], e[0])].mean() - 0.10) < 0.03
    assert abs(rec[np.ix_(i[1], e[0])].mean() - 0.02) < 0.02
    assert abs(rec[np.ix_(e[0], e[1])].mean() - 0.05) < 0.03
    # Inter-area sources are E only (I-source columns empty)
    assert rec[s[1], i[0]].sum() == 0
    assert rec[s[0], i[1]].sum() == 0
    # No skipping connections (area 0 <-> area 2)
    assert rec[s[2], s[0]].sum() == 0
    assert rec[s[0], s[2]].sum() == 0
    # Input only into area 0, readout only from area 2 E units
    assert in_m[:, s[0]].sum() == 4 * 100
    assert in_m[:, s[1]].sum() == 0 and in_m[:, s[2]].sum() == 0
    assert out_m[s[2], :].sum() == 80 * 2
    assert out_m[s[0], :].sum() == 0 and out_m[s[1], :].sum() == 0
    # Dale signs: per-area 80/20
    for a in range(3):
        assert signs[s[a]].sum() == 80 - 20


def test_feedforward_only_and_single_area():
    rec, _, _, _ = build_multiarea_masks(
        area_sizes=[50, 50], input_dim=2, output_dim=1, fb_density=0.0)
    s = area_slices([50, 50])
    assert rec[s[0], s[1]].sum() == 0  # no feedback
    assert rec[s[1], s[0]].sum() > 0   # feedforward remains
    # single area: no inter-area blocks at all
    rec1, _, _, _ = build_multiarea_masks(area_sizes=[40], input_dim=2, output_dim=1)
    assert rec1.shape == (40, 40) and rec1.mean() == 1.0


def test_mask_seed_deterministic():
    a = build_multiarea_masks([30, 30], 2, 1, mask_seed=3)
    b = build_multiarea_masks([30, 30], 2, 1, mask_seed=3)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_masked_positions_receive_no_gradient():
    m = _make_multi()
    out = m(torch.randn(2, 10, 4)).outputs
    out.sum().backward()
    grad = m.h2h.weight.grad
    assert torch.all(grad[m.rec_mask == 0] == 0)
    assert torch.all(m.input2h.weight.grad[m.in_mask.t() == 0] == 0)
    assert torch.all(m.readout_layer.weight.grad[m.out_mask.t() == 0] == 0)


def test_dale_column_signs():
    m = _make_multi()
    W = m._recurrent_weight().detach()
    signs = np.diag(m.dale_mask.numpy())
    for j in range(W.shape[1]):
        col = W[:, j]
        nz = col[col != 0]
        if len(nz):
            assert torch.all(nz * signs[j] >= 0)


# ============================ parity with manual constrained_rnn ============================

def test_parity_with_manual_constrained_rnn():
    """MultiAreaRNNModel == ConstrainedRNNModel with hand-built masks (same weights)."""
    kw = dict(input_dim=4, output_dim=2, area_sizes=[20, 20, 20], dt=10.0, tau=50.0,
              mask_seed=11)
    ma = _make_multi(**kw)
    rec, in_m, out_m, signs = build_multiarea_masks(
        area_sizes=[20, 20, 20], input_dim=4, output_dim=2, mask_seed=11)
    manual = ConstrainedRNNModel(ConstrainedRNNConfig(
        input_dim=4, latent_dim=60, output_dim=2, dt=10.0, tau=50.0,
        dale=True, dale_signs=signs.tolist(),
        rec_mask=rec, in_mask=in_m, out_mask=out_m))
    with torch.no_grad():
        manual.input2h.weight.copy_(ma.input2h.weight)
        manual.input2h.bias.copy_(ma.input2h.bias)
        manual.h2h.weight.copy_(ma.h2h.weight)
        manual.h2h.bias.copy_(ma.h2h.bias)
        manual.readout_layer.weight.copy_(ma.readout_layer.weight)
        manual.readout_layer.bias.copy_(ma.readout_layer.bias)
    x = torch.randn(3, 25, 4)
    o1, o2 = ma(x), manual(x)
    assert torch.allclose(o1.states, o2.states, atol=1e-6)
    assert torch.allclose(o1.outputs, o2.outputs, atol=1e-6)


def test_dale_signs_ctrnn_extension():
    """dale_signs implies dale=True and overrides the global ei_ratio split."""
    from neuralrnn.models.ctrnn import CTRNNConfig, CTRNNModel
    signs = [1.0, 1.0, -1.0, 1.0, -1.0, -1.0]
    m = CTRNNModel(CTRNNConfig(input_dim=2, latent_dim=6, output_dim=1,
                               dale_signs=signs))
    assert m.dale_mask is not None
    assert torch.allclose(torch.diag(m.dale_mask), torch.tensor(signs))
    with pytest.raises(ValueError):
        CTRNNConfig(input_dim=2, latent_dim=6, output_dim=1, dale_signs=[1.0, -1.0])


# ============================ checkerboard task ============================

def test_checkerboard_eval_shapes_and_alignment():
    inp, tgt, msk, cond = checkerboard_trials(
        n_trials=56, mode="eval", balanced=True, seed=0)
    assert inp.shape == (56, 260, 4)
    assert tgt.shape[:2] == (56, 260) and tgt.shape[2] == 2
    assert len(set(c["epoch_bounds"][3] for c in cond)) == 1  # aligned
    assert len(set((c["coherence"], c["left_color"]) for c in cond)) == 28
    # ramp excluded from loss: mask zero for first 20 steps of decision
    b = cond[0]["epoch_bounds"]
    assert msk[0, b[1]:b[1] + 20].sum() == 0
    assert msk[0, b[1] + 20:b[2]].mean() == 1.0


def test_checkerboard_targets_and_catch():
    inp, tgt, msk, cond = checkerboard_trials(
        n_trials=200, mode="eval", balanced=True, seed=1, catch_fraction=0.1)
    for i, c in enumerate(cond):
        b = c["epoch_bounds"]
        if c["catch"]:
            assert c["correct_choice"] == -1
            assert tgt[i].sum() == 0
        else:
            # exactly one DV is 1 during the decision epoch
            assert tgt[i, b[1]:b[2]].sum() == (b[2] - b[1])
            # coherence channels on only during decision
            assert inp[i, :b[1], 2:].sum() == 0
            assert inp[i, b[1]:b[2], 2].abs().sum() > 0


# ============================ demixed analysis ============================

def test_fit_dpca_recovers_synthetic_axes():
    from neuralrnn.analysis import fit_dpca, axis_overlap_matrix
    rng = np.random.default_rng(0)
    M, T = 60, 40
    w_d = rng.normal(size=M); w_d /= np.linalg.norm(w_d)
    w_c = rng.normal(size=M); w_c -= w_d * (w_c @ w_d); w_c /= np.linalg.norm(w_c)
    trials, conds = [], []
    for d in (-1, 1):
        for c in (-1, 1):
            for _ in range(8):
                t = np.linspace(0, 1, T)
                sig = (np.outer(t, w_d) * d * 2 + np.outer(np.sin(3 * t), w_c) * c
                       + rng.normal(scale=0.2, size=(T, M)))
                trials.append(sig); conds.append({"direction": d, "color": c})
    res = fit_dpca(np.stack(trials), conds, variables=("direction", "color"))
    assert abs(res.axes["direction"][0] @ w_d) > 0.95
    assert abs(res.axes["color"][0] @ w_c) > 0.95
    assert res.variance_ratio["direction"] > res.variance_ratio["color"]
    assert abs(axis_overlap_matrix(res.axes["direction"], res.axes["color"])[0, 0]) < 0.2
