# Short-Term Plasticity RNN

> Masse, N. Y., Yang, G. R., Song, H. F., Wang, X. J. & Freedman, D. J. (2019).
> *Circuit mechanisms for the maintenance and manipulation of information in working memory.*
> Nature Neuroscience.

## Problem

Working-memory (WM) information has classically been assumed to reside in persistent
neuronal activity, but experiments show that persistent activity varies markedly across
tasks. The paper asks:

1. Can short-term synaptic plasticity (STP) support **activity-silent manipulation** of WM?
2. Can STP explain why different tasks evoke different amounts of persistent activity?

## Core method

The authors train recurrent neural networks with:

- **80 excitatory + 20 inhibitory recurrent neurons** (Dale constraints: E weights ≥ 0,
  I weights ≤ 0).
- **Per-neuron short-term synaptic plasticity** on recurrent projections, with half the
  neurons facilitating and half depressing.
- A **soft L2 firing-rate penalty** (β = 0.02) to encourage low-activity solutions.
- Euler-discretized continuous-time ReLU dynamics with τ = 100 ms and Δt = 10 ms.

STP variables (available neurotransmitter fraction *x* and utilization *u*) evolve per
Mongillo et al. (2008):

```
dx/dt = (1 - x)/τ_x - u·x·r·Δt
du/dt = (U - u)/τ_u + U(1 - u)·r·Δt
```

Facilitating synapses: τ_x = 1500 ms, τ_u = 200 ms, U = 0.15.
Depressing synapses:  τ_x = 200 ms,  τ_u = 1500 ms, U = 0.45.

The postsynaptic input uses the STP-modulated presynaptic activity `h_post = x·u·r`.

## Key findings

- **DMS** (pure maintenance): networks store sample information primarily in synaptic
  efficacies; neuronal decoding decays during the delay. Shuffling synaptic efficacies at
  test onset hurts performance more than shuffling firing rates.
- **DRMS** (manipulation): when the test must match a rotated sample, networks develop
  persistent sample-selective firing-rate activity. Both neuronal and synaptic shuffles
  impair behavior.
- **ABBA** (representation control): repeated non-matching distractors force the network
  to maintain distinct sample/test representations, again accompanied by persistent activity.
- Across tasks, the degree of manipulation correlates with the strength of persistent
  activity (Spearman R ≈ 0.93).

## Framework mapping

The model layer is now part of the gain_rnn family in `src/neuralrnn/models/gain_rnn/`
(registered as `model_type="stp_rnn"`; see `docs/papers/gain_rnn.md` for the unified
design). The task-level reproduction remains in:

- `notebook/11_STP_RNN_paradigmA.ipynb` (inline model class; kept as the validated
  task-level reproduction)

Mapping from the notebook-11 inline model to the src `stp_rnn`:

| Notebook 11 (inline) | src `stp_rnn` |
|---|---|
| `synapse_config="full"` (alternating fac/dep) | `stp_init="alternating"` (default fac/dep triples identical) |
| `alpha_x/alpha_u/U` buffers | `stp_tau_x/stp_tau_u/stp_U` parameters (ms), `freeze_stp=True` |
| noise `sqrt(2*alpha)*sigma_rec` after blend | `noise_position="post"` + `noise_alpha_scaling=True` (defaults) |
| `F.relu(w_in)` / `F.relu(w_out) * w_out_mask` | `positive_input_weights` / `positive_output_weights` (defaults True) + `out_mask` |
| `w_rnn_mask` (zero diagonal) | `make_stp_masks(...)` → `rec_mask` |
| gamma init (`gamma_shape_exc/inh/scale`) | `init_method="gamma"` + same field names |
| `h0 = 0.1`, `syn_u_init = U` | `h0_init=0.1` (default); `init_state` = `[h0, 1, U_eff]` |
| `bias_rnn` parameter | `h2h.bias` |
| no input bias | `input2h.bias` zeroed and frozen |

The full state is `[h, syn_x, syn_u]` (3M); `forward` returns `states = h` and
`extras = {"syn_x", "syn_u"}`, so the decoding/shuffle analyses in the notebook port
directly. Old `./models/11/stp_rnn_*` checkpoints were trained by the buggy pre-fix
code and are incompatible with the new class — delete them before any re-run.

The notebook matches the reference TensorFlow implementation on the following details,
which are required to reproduce the paper's decoding results (all preserved in the src
model):

1. **STP steady-state initialization**: `syn_u_init = U` (0.15 fac, 0.45 dep), not 1.0.
2. **Input noise**: scaled by `sqrt(2/alpha)` so the effective std is ~0.447.
3. **Recurrent noise**: `sqrt(2*alpha)*sigma_rec` added to the blended update before the final ReLU.
4. **Non-negative input/output weights**: ReLU applied to `w_in` and `w_out`.
5. **No self-connections**: recurrent diagonal masked to zero.
6. **Excitatory-only readout**: output weights from inhibitory neurons are zero.
7. **Initial firing rate**: `h0 = 0.1`.
8. **Decoding normalization**: per-neuron min-max scaling on the training split before SVM fitting.

The src model additionally generalizes to Zhou & Buonomano (2024)-style neuromodulated
STP (`stp_init="random"` + per-trial `model.set_stp_alpha(...)` +
`nonlinearity_mode="rate"`); see `docs/papers/gain_rnn.md`.

## Diff-test points against the original code

- Original code is TensorFlow 1.x; this port is PyTorch.
- The port keeps the same gamma initialization, STP constants, E/I ratio, trial timing,
  and masked loss as the reference `parameters.py` / `model.py`.
- The analysis utilities (`fit_pca`, `find_fixed_points`, `linearize`) are reused from
  NeuralRNN's analysis module.

## Reference

- Original code: https://github.com/nmasse/Short-term-plasticity-RNN
- Paper: https://doi.org/10.1038/s41593-019-0414-3
