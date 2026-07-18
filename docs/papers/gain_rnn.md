# Gain RNN Family (gain_rnn / stp_rnn)

> **Paradigm**: A (task optimization) + B (teacher–student inference)
> **Original repositories**: `reference_project/constrained_rnn/connectome-constrained-RNN/`,
> `reference_project/motor/gain_modulation/`, `reference_project/motor/stp_modulation/`
> **Framework target**: `models/gain_rnn` (`GainRNNModel` + `StpRNNModel`)
> **Status**: ✅ Ready (model layer + tests; notebook 16 reproduction available)

## 1. What problem it solves

Three lines of work share one paradigm: **freeze (or constrain) the synaptic weight
matrix and put the degrees of freedom on neuron-level modulation parameters**:

- **Beiran & Litwin-Kumar (2025)**, *connectome-constrained RNN*: the connectome
  provides the weight matrix $J$ but not single-neuron excitability, so a student network
  shares $J$ with the teacher and learns only per-neuron gains and biases (2N unknowns
  instead of N²). Used for system identification.
- **Stroud et al. (2018)**, *gain modulation*: neuromodulators reconfigure a trained
  motor network by scaling per-neuron input–output slopes, leaving weights, readout and
  initial conditions untouched. Used for control (motor primitives, speed manifolds).
- **Zhou & Buonomano (2024)** / **Masse et al. (2019)**, *STP*: short-term synaptic
  plasticity makes the effective presynaptic gain *dynamic* — the release probability /
  vesicle variables act as activity-dependent gain factors. Used for activity-silent
  working memory and for temporal/spatial generalization.

The gain_rnn family unifies all three in one CTRNN-lineage module, superseding the
legacy `connectome_rnn` model (removed once this family covered its functionality).

## 2. Core method

### 2.1 Rate map and gain placement

Every nonlinearity application goes through the parameterized rate map (bias always
inside $\phi$):

| `gain_position` | `rate_map(u)` | Paper |
|---|---|---|
| `"outside"` (default) | $r_i = g_i\,\phi(u_i + b_i)$ | Beiran & Litwin-Kumar (output gain ≡ presynaptic column scaling $J\,\mathrm{diag}(g)$) |
| `"inside"` | $r_i = \phi(g_i u_i + b_i)$ | Stroud et al. (input gain ≡ slope modulation; with `activation="piecewise_tanh"` this is exactly their eq. 2) |

Linearized, all placements reduce to $W\,\mathrm{diag}(g)$; the distinction matters only
in the nonlinear regime (amplitude vs. sensitivity). For ReLU with $g>0$ the two
placements are exactly equivalent (scale degeneracy).

The rate map composes with the unified `nonlinearity_mode` vocabulary:
`"rate"` (family default; state = current, recurrence and readout consume `rate_map(z)`),
`"post_blend"`, `"pre_activation"`. Noise placement is controlled by `noise_position`:
`"pre"` (CTRNN behavior) or `"post"` (state-level noise on the leaked blend before the
final nonlinearity; with `noise_alpha_scaling=True` the std is $\sqrt{2\alpha}\,\sigma$,
matching both the Masse-style post_blend model and Zhou & Buonomano's rate model).

### 2.2 STP as a dynamic gain (stp_rnn)

The static gain/bias are frozen at identity by default and the effective presynaptic gain
becomes the dynamic factor $\text{syn}_x \cdot \text{syn}_u$ (Tsodyks–Markram rate form,
cell-specific — shared by all synapses of the same presynaptic neuron):

$$
\begin{aligned}
x' &= x + \tfrac{dt}{\tau_x}(1-x) - dt_{sec}\, u\, x\, r \\
u' &= u + \tfrac{dt}{\tau_u}(U_{\text{eff}}-u) + dt_{sec}\, U_{\text{eff}}(1-u)\, r \\
U_{\text{eff}} &= \mathrm{clamp}(\alpha_{\text{stp}} \cdot U, 0, 1), \qquad
\text{rec\_in} = r \odot x' \odot u'
\end{aligned}
$$

with $dt$ in ms and $dt_{sec} = dt/1000$ (release terms assume rates in spikes/s,
following both reference implementations). Old values update simultaneously; $x, u$ are
clamped to $[0,1]$. The neuromodulator $\alpha_{\text{stp}}$ (Zhou & Buonomano's abstract
dopamine) is a runtime buffer set per trial via `model.set_stp_alpha(...)` — never trained.

- Transition $F_\theta$: gain_rnn Euler step with `rec_in` above (shared `_euler_step`).
- Readout $G_\phi$: rate-mode reads `rate_map(z)`; stp forward returns `states = h`
  (B,T,M) with `extras = {"syn_x", "syn_u"}`.
- Key hyperparameters: `stp_tau_x/stp_tau_u/stp_U` (per-neuron), `stp_init`
  (constant / alternating / random), `stp_alpha`, `freeze_stp` (default True — neither
  reference paper trains STP constants).

## 3. How to use this method in our framework

| Paper element | Framework API | Note |
|---|---|---|
| $r_i = g_i \phi(x_i + b_i)$, fixed $J$ (Beiran) | `GainRNNConfig(nonlinearity_mode="rate", gain_position="outside", activation="softplus", freeze_input/recurrent/output=True)`; copy $J$ into `model.h2h.weight` | cross-model parity with `ConnectomeRNNModel` is unit-tested |
| Piecewise-tanh gain function (Stroud eq. 2) | `gain_position="inside", activation="piecewise_tanh", activation_params={"r0": 20.0, "rmax": 100.0}` | slope at the origin = gain |
| Masse 2019 STP-RNN (notebook 11) | `StpRNNConfig(stp_init="alternating", dale=True, init_method="gamma", **make_stp_masks(...))` | family defaults already match nb11 (post_blend, noise post, positive weights, h0=0.1) |
| Zhou & Buonomano 2024 | `StpRNNConfig(nonlinearity_mode="rate", stp_init="random", positive_output_weights=False)` + `model.set_stp_alpha(a)` per trial | truncated-normal STP constants; α is a trial-level cue |

- New config fields: `gain_position`, `gain_init/bias_init`, `freeze_gain/freeze_bias`,
  `noise_position`, `positive_input_weights/positive_output_weights`,
  `activation_params`, `h0_init` (gain_rnn); `stp_tau_x/stp_tau_u/stp_U`, `stp_init`,
  alternating/random sampler fields, `stp_alpha`, `freeze_stp`, `init_method="gamma"`,
  gamma fields, `init_seed` (stp_rnn).
- Analytic fixed points: not supported; numeric/scipy backends work via the M-dim
  analysis fallback (`recurrence` with `syn_x = syn_u = 1` frozen).
- Freeze groups: base four + `gains`, `biases` (gain_rnn), plus `stp` (stp_rnn).

### 3.1 Quick start

```python
from neuralrnn import AutoConfig, AutoModel
from neuralrnn.models.gain_rnn import make_stp_masks

# --- Connectome-constrained student (Beiran & Litwin-Kumar 2025) ---
cfg = AutoConfig.for_model(
    "gain_rnn", input_dim=N, latent_dim=N, output_dim=2,
    dt=0.1, tau=1.0, activation="softplus",
    freeze_input=True, freeze_recurrent=True, freeze_output=True,
)
student = AutoModel.from_config(cfg)
with torch.no_grad():
    student.h2h.weight.copy_(J)          # the shared connectome
    student.input2h.weight.copy_(torch.eye(N))   # per-neuron teacher input

# --- STP network (notebook-11 defaults) ---
masks = make_stp_masks(100, 3, ei_ratio=0.8)
cfg = AutoConfig.for_model(
    "stp_rnn", input_dim=24, latent_dim=100, output_dim=3,
    stp_init="alternating", dale=True, init_method="gamma", init_seed=0,
    rec_mask=masks["rec_mask"], out_mask=masks["out_mask"],
)
model = AutoModel.from_config(cfg)
out = model(inputs)          # out.states = h; out.extras["syn_x"/"syn_u"]
```

## 4. Consistency with the original implementations

- The connectome scenario (`rate` + `outside` + `softplus`, shared J copied into
  `h2h.weight` and frozen) was verified by exact cross-model trajectory parity
  (`atol=1e-6`) against the legacy `connectome_rnn` class before that class was
  removed; notebook 16 now serves as the end-to-end reproduction.
- `test/test_stp_rnn.py::test_notebook11_parity`: full-rollout parity against an
  independent re-implementation of the notebook-11 reference math (alternating STP,
  gamma init, Dale, no-self-connection, E-only non-negative readout; noise off).
- Hand-computed single-step tests pin the rate map (both placements × three
  nonlinearity modes), both noise positions, and the STP update (simultaneous old-value
  update, clamping, α-scaled U).
- Zhou & Buonomano's `"random"` STP init is tested for seed reproducibility, truncation
  bounds, and tau_x/tau_u independence; `set_stp_alpha` changes trajectories as expected.

## 5. Reproduction experiments

- Connectome-constrained reproduction (teacher–student, partial recording):
  `notebook/16_connectome_rnn_paradigmB.ipynb`, built on `gain_rnn` (Fig. 1 task-output
  student + Fig. 2 recorded-activity students with M = 30/60/90/180).
- Notebook 11 (Masse 2019) remains the validated STP task reproduction; its inline
  model is superseded by `stp_rnn` (old checkpoints are incompatible and were already
  marked for deletion before any re-run).
- Stroud et al. 2018 and Zhou & Buonomano 2024 full reproductions are future work; the
  module provides the required model features (inside gain + piecewise_tanh; random STP
  init + per-trial α + rate mode).

## 6. Design choices and caveats

- **Single family directory** `models/gain_rnn/` hosts both model types
  (`gain_rnn`, `stp_rnn`), following the constrained_rnn multi-variant precedent.
- **No `fixed_recurrent_weight` field** (the old connectome API): fixed weights are
  handled uniformly by copying weights + `freeze_recurrent`; save/load goes through
  safetensors like every other family.
- **STP constants are `nn.Parameter`s with `freeze_stp=True` by default** — trainable in
  principle (unfreeze to explore), frozen by default to match both reference papers.
  Forward-time clamps keep tau positive and U in [0, 1].
- **Analysis caveat**: fixed-point / linearization tools analyze the frozen-efficacy
  M-dim map (syn_x = syn_u = 1), not the full 3M system.
- **`stp_alpha` dual identity**: config field = initial value; runtime buffer = actual
  value; checkpoints restore the buffer.
- Row scaling (`diag(g) J`, postsynaptic gain) is not implemented (not used by any of
  the three reference papers); inside/outside cover the presynaptic placements.
