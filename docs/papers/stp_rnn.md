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

The current port is implemented inline in:

- `notebook/11_STP_RNN_paradigmA.ipynb`

It defines:

- `STPRNNConfig` / `STPRNNModel` — inline STP-RNN with Dale constraints.
- `STPSupervisedObjective` — masked cross-entropy + L2 firing-rate penalty.
- `DMSDataset` / `ABBADataset` — motion-direction task generators.
- SVM decoding, shuffle analysis, PCA, and fixed-point analysis using `neuralrnn.analysis`.

The notebook now matches the reference TensorFlow implementation on the following details,
which are required to reproduce the paper's decoding results:

1. **STP steady-state initialization**: `syn_u_init = U` (0.15 fac, 0.45 dep), not 1.0.
2. **Input noise**: scaled by `sqrt(2/alpha)` so the effective std is ~0.447.
3. **Recurrent noise**: `sqrt(2*alpha)*sigma_rec` added to the blended update before the final ReLU.
4. **Non-negative input/output weights**: ReLU applied to `w_in` and `w_out`.
5. **No self-connections**: recurrent diagonal masked to zero.
6. **Excitatory-only readout**: output weights from inhibitory neurons are zero.
7. **Initial firing rate**: `h0 = 0.1`.
8. **Decoding normalization**: per-neuron min-max scaling on the training split before SVM fitting.

Once the port is validated, the model will be moved to `src/neuralrnn/models/stp_rnn/` and
registered with `AutoConfig` / `AutoModel` following Contract A.

## Diff-test points against the original code

- Original code is TensorFlow 1.x; this port is PyTorch.
- The port keeps the same gamma initialization, STP constants, E/I ratio, trial timing,
  and masked loss as the reference `parameters.py` / `model.py`.
- The analysis utilities (`fit_pca`, `find_fixed_points`, `linearize`) are reused from
  NeuralRNN's analysis module.

## Reference

- Original code: https://github.com/nmasse/Short-term-plasticity-RNN
- Paper: https://doi.org/10.1038/s41593-019-0414-3
