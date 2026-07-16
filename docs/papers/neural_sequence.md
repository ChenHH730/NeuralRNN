# Neural Sequence Models of Short-Term Memory

This note covers work that studies **neural sequences** — time-varying population activity patterns — in short-term / working memory tasks.

- **Orhan & Ma (2019)** — *A diverse range of factors affect the nature of neural representations underlying short-term memory*. Nature Neuroscience 22, 275–283.
- **Zhou et al. (2023)** — *Multiplexing working memory and time in the trajectories of neural networks*. Nature Human Behaviour 7, 1056–1070.

Both use task-trained RNNs as in-silico models and quantify when sequential versus persistent activity emerges.

## What problem these papers address

Short-term memory (STM) can be implemented in recurrent circuits either by **persistent activity** of individual neurons (fixed-point attractors) or by **sequential population activity** in which each neuron is active only transiently but the population pattern evolves over time. Experimental evidence exists for both regimes, and it has been unclear what determines which one a circuit adopts.

These two papers tackle complementary questions:

1. **Orhan & Ma (2019)** ask: *what task and circuit factors push a trained network toward sequential versus persistent representations?*
2. **Zhou et al. (2023)** ask: *can the same neural trajectory simultaneously encode a remembered item and elapsed time, effectively multiplexing working memory and timing?*

## Current status in NeuralRNN

- **Orhan & Ma (2019)** and **Zhou et al. (2023)** Figure 3 are reproduced in a single notebook: `notebook/15_neural_sequence_paradigmA.ipynb`.

## Orhan & Ma (2019)

### Model

A vanilla discrete-time ReLU rate RNN:

$$
r_t = \text{ReLU}(W_r r_{t-1} + W_x x_t + b)
$$

- In NeuralRNN this is `AutoConfig.for_model('ctrnn', dt=None, activation='relu')`, which sets `alpha = 1.0`.
- Recurrent weight initialized after model creation as $\lambda_0 I + \sigma_0 \Sigma_{\text{off}}$, where off-diagonal entries of $\Sigma_{\text{off}}$ are Gaussian with SD $1/\sqrt{M}$ and $M$ is the hidden size.
- Paper uses 500 recurrent units; the notebook defaults to 100 for fast validation and can be increased.

### Tasks

All tasks share the same input encoding: 25 input channels with Gaussian tuning curves on a linear feature axis, plus Poisson sampling. Time step `dt = 10 ms`. Tasks are defined **inline in the notebook** (no `src` modules):

- **2AFC** (low temporal complexity): stimulus `s \in \{-15, +15\}` for 250 ms, delay 1000 ms, response 250 ms.
- **COMP** (high temporal complexity): two stimuli `s_1, s_2 \in [-40, 40]` shown sequentially (250 ms each) with a 1000 ms delay; report whether `s_2 > s_1`.
- **Variable 2AFC**: same as 2AFC, but delay duration is drawn from `{100, 400, 700, 1000}` ms per trial; response occurs at a fixed absolute time.

### Sequentiality Index (SI)

`analysis.sequentiality.compute_sequentiality_index` implements the paper's SI:

1. Select units with mean activity above a threshold.
2. Compute each unit's peak time and the entropy of the peak-time distribution.
3. Compute a ridge-to-background ratio around each peak.
4. SI = mean log ridge-to-background ratio + entropy.

### Main findings reproduced

| Manipulation | Prediction | Result in notebook (MAX_STEPS=4000, hidden=100) |
|---|---|---|
| COMP vs. 2AFC | higher temporal complexity → higher SI | COMP SI = 3.36, 2AFC SI = 3.22 |
| Stronger `sigma_0` (0.15 vs. 0.05) | stronger coupling → higher SI | 2AFC strong SI = 3.76, baseline SI = 3.22 |
| Stronger L2 (1e-3 vs. 1e-5) | stronger regularization → lower SI | COMP strong-L2 SI = 3.30, baseline SI = 3.36 |
| Recurrent weight profile | high-SI networks show forward-asymmetric connectivity | COMP network shows larger weights from earlier-peaking to later-peaking neurons |

All networks also achieve high task accuracy (>0.88).

### Mechanism

Trained networks develop a **non-normal, asymmetric recurrent connectivity**. Sorting units by peak time, weights from earlier-peaking to later-peaking neurons are larger than the reverse. `analysis.sequentiality.weight_profile_by_peak_order` computes this profile.

## Zhou et al. (2023) (T+WM)

**Model.** Dale-constrained leaky ReLU RNN:

$$
\tau \frac{d\mathbf{r}}{dt} = -\mathbf{r} + [W^{\text{RNN}} \mathbf{r} + W^{\text{In}} \mathbf{u} + b^{\text{RNN}} + \text{noise}]_+
$$

Discretized with $\tau=50$ ms, $dt=10$ ms ($\alpha=0.2$), 256 units, 80% excitatory / 20% inhibitory, recurrent noise. Input weights and recurrent bias are frozen during training; only $W^{\text{RNN}}$, $W^{\text{Out}}$ and the output bias are trained.

**Tasks.** The T+WM (timing + working memory) variant of the differential-delay DMS family:

- Cue A/B predicts a short (1 s) or long (2.2 s) delay.
- The network reports match/non-match via a motor output and produces a temporal-expectation half-ramp.
- 10% invalid/reversed trials and variable onset/delay jitter during training.

**Reproduced analyses (Figure 3 only).**

- Ramp-to-sequence transition across training checkpoints: effective dimensionality, sequentiality index, and elapsed-time decoding correlation.
- Sorted neurograms at selected checkpoints.

Connectivity analyses (Figure 5) are omitted in the merged notebook. Decoding is implemented inline in the notebook rather than as reusable public API functions.

## Mapping to NeuralRNN

| Component | NeuralRNN module |
|---|---|
| Models | `models/ctrnn` (`ei_rnn`) |
| Tasks | inline in `notebook/15_neural_sequence_paradigmA.ipynb` |
| Objective | `RegularizedSupervisedObjective(task_type='regression')` |
| Sequentiality index | `analysis.sequentiality.compute_sequentiality_index` |
| Peak-time sorting / weight profiles | `analysis.sequentiality.sort_neurons_by_peak_time`, `weight_profile_by_peak_order` |
| E/I splitting | `analysis.sequentiality.split_ei_weight_submatrices` (general helper, not used in the merged notebook) |
| Dimensionality | `analysis.dimensionality.effective_dimensionality` |
| Training & saving | `Trainer`, `TrainingArguments`, `model.save_pretrained` |

## Reproduction notebooks

- `notebook/15_neural_sequence_paradigmA.ipynb` — Orhan & Ma (2019) and Zhou et al. (2023) Figure 3.

## Key diff-test points

- SI values are sensitive to the exact window/bin parameters and the activity threshold; both notebooks use the paper's defaults (`threshold=0.1`, `window=5`, `n_bins=20`).
- The Orhan notebook uses **Poisson-sampled inputs** (`torch.poisson`) during training; the Zhou notebook uses deterministic Gaussian-bump inputs on a ring.
- The Zhou T+WM network is trained with Dale's law (80/20 E/I), frozen input weights and recurrent bias, and recurrent-noise during training only.
- No full hyperparameter grid is run; representative settings are used to show the core qualitative trends.
- Decoding is implemented inline in the Zhou section of the merged notebook; no `analysis.decoding` module is added.

## References

- Orhan, A.E. & Ma, W.J. (2019). A diverse range of factors affect the nature of neural representations underlying short-term memory. *Nature Neuroscience*, 22, 275–283.
- Zhou, S., Seay, M., Taxidis, J., Golshani, P. & Buonomano, D.V. (2023). Multiplexing working memory and time in the trajectories of neural networks. *Nature Human Behaviour*, 7, 1056–1070.
