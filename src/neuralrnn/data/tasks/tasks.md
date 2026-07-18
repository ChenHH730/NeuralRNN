# NeuralRNN Built-in Task Catalog

This document inventories the task generators shipped in `src/neuralrnn/data/tasks/`. It covers (1) what each task does and how it is parameterized, (2) which tasks are functionally redundant and could be merged, and (3) which notebooks currently consume each task.

## Conventions

- All generators expose a `generate_trials(**kwargs)` function.
- Returned tuples are `(inputs, targets, mask, conditions)`:
  - `inputs`:     `(n_trials, n_t, input_dim)`
  - `targets`:    `(n_trials, n_t, output_dim)` — full length for all current tasks.
  - `mask`:       `(n_trials, n_t, output_dim)` boolean or float tensor.
  - `conditions`: list of per-trial dicts with metadata (coherence, context, choice, etc.).
- Tasks are registered in `tasks/__init__.py` under `TASK_REGISTRY` so they can be loaded through `CognitiveTaskDataset.from_task(task_name, ...)` or `load_dataset(task_name, ...)`.

## Task Catalog

| Registry name | File | Cognitive paradigm | Input dim | Output dim | Notes |
|---|---|---|---|---|---|
| `mante` | `mante_task.py` | Context-dependent DM (Mante et al. 2013) | 6 | 2 | — |
| `siegel_miller` | alias | Context-dependent DM (Siegel et al. 2015) | 6 | 2 | alias for `mante` |
| `rdm` | `rdm_task.py` | Random-dot-motion perceptual DM | 1 | 1 | — |
| `two_afc` | alias | Two-alternative forced choice | 2 | 2 | alias for `rdm` |
| `dms_continuous` | `dms_continuous_task.py` | Delayed match-to-sample (continuous) | 4 | 2 | — |
| `wm_angle` | `wm_angle_task.py` | Parametric WM on circular angle | 2 | 2 | — |
| `parametric_wm` | alias | Parametric WM on circular angle | 2 | 2 | alias for `wm_angle` |
| `wm_frequency` | `wm_frequency_task.py` | Parametric WM on frequency | 1 | 1 | — |
| `romo` | alias | Parametric WM on frequency (Romo) | 1 | 1 | alias for `wm_frequency` |
| `raposo` | `raposo_task.py` | Multisensory decision making | 4 | 1 | — |
| `dms` | `dms_task.py` | Delayed match-to-sample (discrete A/B) | 2 | 1 | — |
| `lr_mante` | `lr_mante_task.py` | Context-dependent DM (low-rank format) | 4 | 1 | — |
| `checkerboard` | `checkerboard_task.py` | Checkerboard majority-color DM (Kleinman et al. 2025) | 4 | 2 | 10% catch trials; train/eval timing modes |

### 1. Context-dependent decision-making family

#### `mante` — Mante et al. (2013) context-dependent DM

- **Also called**: Siegel-Miller task (Siegel et al., 2015).
- **Objective**: Attend to either motion or color coherence and report its sign.
- **Inputs**: `(N, n_t, 6)` — `[motion_ctx, color_ctx, motion_r, motion_l, color_r, color_l]`.
- **Targets**: `(N, n_t, 2)` — two readout units active during decision.
- **Mask**: `(N, n_t, 2)` — 1 during decision, 0 otherwise.
- **Defaults**: `n_trials=25`, `n_t=75`, `alpha=0.2`, `sigma_in=0.01`, `baseline=0.2`, `n_coh=6`.
- **Coherences**: `np.linspace(-0.2, 0.2, n_coh)` by default; override via `cohs`.
- **Trial timing** (relative to `n_t`):
  - cue:   `[0.10*n_t, 0.33*n_t]`
  - stim:  `[0.40*n_t, n_t]`
  - dec:   `[0.75*n_t, n_t]`
- **Noise**: `sqrt(2/alpha * sigma_in^2) * N(0, I_6)`, rectified.

#### `lr_mante` — Low-rank RNN Mante

- **Objective**: Same cognitive task as `mante` but using the low-rank RNN convention.
- **Inputs**: `(N, total_duration, 4)` — `[color_stim, motion_stim, color_ctx, motion_ctx]`.
- **Targets**: `(N, total_duration, 1)` — scalar `+1`/`-1`.
- **Mask**: `(N, total_duration, 1)` — 1 during response.
- **Defaults**: `num_trials=1000`, `coherences=[-4,-2,-1,1,2,4]`, `std=0.1`.
- **Timing** (ms, `dt=20ms`): fixation 100, ctx_pre 350, stimulus 800, delay 100, decision 20.

### 2. Match-to-sample family

#### `dms_continuous` — Continuous-coherence DMS

- **Objective**: Two sequential noisy stimuli; report whether they have the same sign.
- **Inputs**: `(N, n_t, 4)` — `[test_r, test_l, sample_r, sample_l]`.
- **Targets**: `(N, n_t, 2)` — full-length.
- **Mask**: `(N, n_t, 2)` — 1 during pre-sample and decision, 0 during delay.
- **Defaults**: `n_trials=25`, `n_t=75`, `alpha=0.2`, `sigma_in=0.01`, `baseline=0.2`, `n_coh=6`.
- **Coherences**: `np.linspace(-0.2, 0.2, n_coh)`.

#### `dms` — Discrete-symbol DMS

- **Objective**: Delayed match-to-sample with one-hot A/B symbols.
- **Inputs**: `(N, total_duration, 2)` — one-hot A/B.
- **Targets**: `(N, total_duration, 1)` — `+1` match, `-1` non-match.
- **Mask**: `(N, total_duration, 1)` — 1 during decision.
- **Defaults**: `num_trials=1000`, `std=0.03`.
- **Timing** (ms, `dt=20ms`): fixation 100, stim1 500, delay 500-3000, stim2 500, decision 1000.

### 3. Parametric working-memory family

#### `wm_angle` — Circular-angle WM

- **Also called**: `parametric_wm` (legacy name).
- **Objective**: Remember a continuous angle over a delay and reproduce it.
- **Inputs**: `(N, n_t, 2)` — `[cos(theta), sin(theta)]` during stimulus.
- **Targets**: `(N, n_t, 2)` — same during decision.
- **Mask**: `(N, n_t, 2)` — 1 during stimulus and decision, 0 during delay.
- **Defaults**: `n_trials=100`, `n_t=75`.

#### `wm_frequency` — Frequency-comparison WM

- **Also called**: Romo task (Romo et al., 1999).
- **Objective**: Compare two sequential vibration frequencies `f1` vs `f2` after a delay.
- **Inputs**: `(N, total_duration, 1)` — normalized frequency.
- **Targets**: `(N, total_duration, 1)` — `(f1 - f2)/FSPAN`.
- **Mask**: `(N, total_duration, 1)` — 1 during decision.
- **Defaults**: `num_trials=1000`, `std=0.01`.
- **Timing** (ms, `dt=20ms`): fixation 100, stim1 100, delay 500-1000, stim2 100, decision 100.

### 4. Perceptual decision-making family

#### `rdm` — Random-dot-motion DM

- **Also called**: `two_afc` in the 2-output/noiseless formulation.
- **Objective**: Report sign of a noisy motion coherence.
- **Inputs**: `(N, total_duration, 1)` — coherence + noise.
- **Targets**: `(N, total_duration, 1)` — `+1` or `-1`.
- **Mask**: `(N, total_duration, 1)` — 1 during response.
- **Defaults**: `num_trials=1000`, `coherences=[-4,-2,-1,1,2,4]`, `std=0.1`.
- **Timing** (ms, `dt=20ms`): fixation 100, stimulus 800, delay 100, decision 20.

### 5. Multisensory decision-making family

#### `raposo` — Multisensory decision making

- **Objective**: Attend to visual, auditory, or both modalities and report dominant direction.
- **Inputs**: `(N, total_duration, 4)` — `[visual_stim, auditory_stim, visual_ctx, auditory_ctx]`.
- **Targets**: `(N, total_duration, 1)` — choice sign.
- **Mask**: `(N, total_duration, 1)` — 1 during response.
- **Defaults**: `num_trials=1000`, `coherences=[-4,-2,-1,1,2,4]`, `std=0.1`.
- **Timing** (ms, `dt=20ms`): fixation 100, ctx_pre 350, stimulus 800, delay 100, decision 20.

### 6. Checkerboard decision-making family

#### `checkerboard` — Kleinman et al. (2025) checkerboard DM

- **Objective**: Report the majority color of a red/green checkerboard by reaching to the matching target; target colors are randomized across left/right so color and direction decisions are statistically independent.
- **Inputs**: `(N, n_t, 4)` — `[left_target_color (-1 red/+1 green), right_target_color, red_coh, green_coh]`; target channels on during the targets epoch, coherence channels (+ independent N(0, 0.1^2) noise) on during decision.
- **Targets**: `(N, n_t, 2)` — `[left_dv, right_dv]`, correct DV = 1 during decision, else 0.
- **Mask**: `(N, n_t, 2)` — 0 during the first 200 ms of the decision epoch (integration ramp) and padding, 1 elsewhere.
- **Defaults**: `n_trials=64`, `dt=10.0`, `cohs=linspace(-0.9, 0.9, 14)`, `catch_fraction=0.1`, `input_noise_std=0.1`, `loss_ramp_ms=200`.
- **Timing** (ms): center hold ~N(200, 50^2) -> targets ~U[600, 1000] -> decision 1500 -> off 100; `mode="eval"` fixes (200, 800, 1500, 100) for trial alignment. Train-mode trials are zero-padded to the batch max.
- **Conditions**: `coherence`, `left_color`, `correct_choice` (0 left / 1 right / -1 catch), `catch`, `epoch_bounds`. Catch trials: half no input, half targets only.

## Notebook Usage Mapping

| Task | Used in notebook(s) | How |
|---|---|---|
| `mante` | `10_constrained_RNN_paradigmA.ipynb`, `02_latent_circuit_paradigmB.ipynb` | SparseRNN / ModularRNN / CTRNN comparison; latent circuit training |
| `siegel_miller` (alias) | `02_latent_circuit_paradigmB.ipynb` | Latent circuit training & analysis |
| `rdm` | `07_lowrank_RNN_paradigmA.ipynb` | Rank-1 RDM tutorial |
| `dms_continuous` | none | — |
| `wm_angle` | none | — |
| `wm_frequency` | `07_lowrank_RNN_paradigmA.ipynb` | 5-task accuracy check |
| `romo` (alias) | `07_lowrank_RNN_paradigmA.ipynb` (legacy) | Same as `wm_frequency` |
| `dms` | `07_lowrank_RNN_paradigmA.ipynb`, `test/07a_DMS_test.ipynb` | Learnability check |
| `raposo` | `07_lowrank_RNN_paradigmA.ipynb` | 5-task accuracy check |
| `lr_mante` | `07_lowrank_RNN_paradigmA.ipynb`, `08_lowrank_RNN_paradigmB.ipynb` | Mante analysis / LINT |
| `checkerboard` | `17_multi_area_rnn_paradigmA.ipynb` | Multi-area RNN training (Kleinman 2025) |

## Related Registry Entries

- `src/neuralrnn/data/neurogym_dataset.py` — wrappers for `neurogym` environments.
- `src/neuralrnn/data/timeseries_dataset.py` — `lorenz63` and other non-cognitive timeseries.
- `src/neuralrnn/data/bartolo_monkey_dataset.py` — probabilistic reversal-learning data.
- `src/neuralrnn/data/custom_dataset.py` — user-provided arrays.

See `other_task.md` (repo root) for a cross-project comparison with `trainRNNbrain`, `neurogym`, and `multitask`.
