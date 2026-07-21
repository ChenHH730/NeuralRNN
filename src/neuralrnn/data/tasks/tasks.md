# NeuralRNN Built-in Task Catalog

This document inventories the unified task library in `src/neuralrnn/data/tasks/`
(design: `docs/DATA_REFACTOR.md`).

## Conventions

- Every task is a `Task` subclass registered in `TASK_CLASSES`
  (`tasks/__init__.py`). Instantiate directly or via
  `CognitiveTaskDataset.from_task(name, ...)` / `load_dataset(name, ...)`.
- Unified constructor parameters (per-task extras documented below):
  - `n_trials` — **total** number of trials (grid tasks `mante` /
    `dms_continuous` use `n_reps` = repetitions per condition cell instead);
  - `sigma_in` — input-noise std (0 = noiseless);
  - `catch_fraction` — fraction of catch trials;
  - `seed` — reproducibility (group-A tasks seed numpy; input noise follows
    the global torch RNG — seed torch too for exact reproduction);
  - `dt` — ms per step where the task is timed in ms.
- `generate_trials()` returns `(inputs, targets, mask, conditions)`:
  - `inputs`:     `(n_trials, n_t, input_dim)` torch tensor
  - `targets`:    `(n_trials, n_t, output_dim)` torch tensor — full length
  - `mask`:       `(n_trials, n_t, output_dim)` float tensor
  - `conditions`: list of per-trial dicts. Unified keys: `epochs`
    (`{phase: (start, end)}` in steps), `n_steps` (true length before
    padding), `is_catch`; plus task-specific keys below.
- Deprecated names still work with a `DeprecationWarning`: task aliases
  (`siegel_miller`, `two_afc`, `parametric_wm`, `romo`, `lr_mante` → `mante2`)
  and parameter names (`num_trials` → `n_trials`, `std` → `sigma_in`,
  `fraction_catch_trials` → `catch_fraction`, mante-family `n_trials` →
  `n_reps`, checkerboard `input_noise_std` → `sigma_in`).
- Module-level `generate_trials(**kwargs)` shims and `TASK_REGISTRY`
  (name → callable) are kept for backward compatibility.
- Dataset modes: `load_dataset(name, mode="aligned"|"streaming")` — streaming
  concatenates pool trials into `seq_len` windows (see
  `docs/DATA_REFACTOR.md` §2.2). `ds.sample_trials(n)` returns a few complete
  trials from any dataset without creating a second one.

## Task Catalog

| Name | File | Cognitive paradigm | In | Out | Aliases |
|---|---|---|---|---|---|
| `mante` | `mante_task.py` | Context-dependent DM (Mante et al. 2013) | 6 | 2 | `siegel_miller` |
| `mante2` | `mante2_task.py` | Context-dependent DM (low-rank RNN convention) | 4 | 1 | `lr_mante` |
| `rdm` | `rdm_task.py` | Random-dot-motion perceptual DM | 1 | 1 | `two_afc` |
| `raposo` | `raposo_task.py` | Multisensory decision making | 4 | 1 | — |
| `dms` | `dms_task.py` | Delayed match-to-sample (discrete A/B) | 2 | 1 | — |
| `dms_continuous` | `dms_continuous_task.py` | Delayed match-to-sample (continuous) | 4 | 2 | — |
| `wm_angle` | `wm_angle_task.py` | Parametric WM on circular angle | 2 | 2 | `parametric_wm` |
| `wm_frequency` | `wm_frequency_task.py` | Parametric WM on frequency (Romo) | 1 | 1 | `romo` |
| `go_nogo` | `go_nogo_task.py` | Go/NoGo (Tolmachev & Engel 2025) | 3 | 1 | — |
| `checkerboard` | `checkerboard_task.py` | Checkerboard majority-color DM (Kleinman et al. 2025) | 4 | 2 | — |
| `multitask_yang` | `multitask_yang_task.py` | Yang et al. 2019, 20 rules | 85 | 33 | — |
| `multitask_flexible` | `multitask_flexible_task.py` | Driscoll et al. 2024, 15 rules | 20 | 3 | — |

### 1. Context-dependent decision-making family

#### `mante` — Mante et al. (2013) context-dependent DM

- **Also called**: Siegel-Miller task (Siegel et al., 2015); alias `siegel_miller`.
- **Objective**: Attend to either motion or color coherence and report its sign.
- **Inputs**: `(N, n_t, 6)` — `[motion_ctx, color_ctx, motion_r, motion_l, color_r, color_l]`.
- **Targets**: `(N, n_t, 2)` — two readout units active during decision.
- **Mask**: `(N, n_t, 2)` — 1 during decision, 0 otherwise.
- **Params**: `n_reps=25` (per condition cell; total = `n_reps * 2 * n_coh^2`),
  `n_t=75`, `alpha=0.2`, `sigma_in=0.01`, `baseline=0.2`, `n_coh=6`,
  `cohs=None` (default `linspace(-0.2, 0.2, n_coh)`), `seed=None`.
- **Trial timing** (relative to `n_t`): cue `[0.10, 0.33]`, stim `[0.40, 1]`,
  decision `[0.75, 1]`.
- **Noise**: `sqrt(2/alpha * sigma_in^2) * N(0, I_6)`, rectified.
- **Condition keys**: `context` (str), `motion_coh`, `color_coh`, `correct_choice` (±1).

#### `mante2` — Mante, low-rank RNN convention

- **Formerly**: `lr_mante` (deprecated alias).
- **Objective**: Same paradigm as `mante` in the Dubreuil et al. (2022) /
  Valente et al. (2022) convention (signed scalar output).
- **Inputs**: `(N, total_duration, 4)` — `[color_stim, motion_stim, color_ctx, motion_ctx]`.
- **Targets**: `(N, total_duration, 1)` — scalar `+1`/`-1`.
- **Mask**: `(N, total_duration, 1)` — 1 during response.
- **Params**: `n_trials=1000`, `coherences=[-4,-2,-1,1,2,4]`, `sigma_in=0.1`,
  `catch_fraction=0.0`, `coh_color`/`coh_motion`/`context` (fixed values or
  None=random), `dt=20.0`, `seed=None`.
- **Timing** (ms): fixation 100, ctx_pre 350, stimulus 800, delay 100, decision 20.
- **Condition keys**: `context` (1=color, 2=motion), `coh_color`, `coh_motion`, `target`.
- The legacy module `lr_mante_task` (timing globals, `_setup()`, mutable
  `SCALE_CTX`) is an alias of `mante2_task` and keeps working.

### 2. Match-to-sample family

#### `dms_continuous` — Continuous-coherence DMS

- **Objective**: Two sequential noisy stimuli; report whether they have the same sign.
- **Inputs**: `(N, n_t, 4)` — `[test_r, test_l, sample_r, sample_l]`.
- **Targets**: `(N, n_t, 2)` — full-length.
- **Mask**: `(N, n_t, 2)` — 1 during pre-sample and decision, 0 during delay.
- **Params**: `n_reps=25` (per cell; total = `n_reps * n_coh^2`), `n_t=75`,
  `alpha=0.2`, `sigma_in=0.01`, `baseline=0.2`, `n_coh=6`, `seed=None`.
- **Condition keys**: `test_coh`, `sample_coh`, `match`, `correct_choice` (±1).

#### `dms` — Discrete-symbol DMS

- **Objective**: Delayed match-to-sample with one-hot A/B symbols.
- **Inputs**: `(N, total_duration, 2)` — one-hot A/B.
- **Targets**: `(N, total_duration, 1)` — `+1` match, `-1` non-match.
- **Mask**: `(N, total_duration, 1)` — 1 during decision.
- **Params**: `n_trials=1000`, `trial_type=None` (or 'A-A'/'A-B'/'B-A'/'B-B'),
  `sigma_in=0.03`, `catch_fraction=0.0`, `dt=20.0`, `seed=None`.
- **Timing** (ms): fixation 100, stim1 500, delay 500-3000 (per-trial),
  stim2 500, decision 1000. Epoch bounds are per trial (see `conditions[i]['epochs']`).
- **Condition keys**: `trial_type`, `input1`, `input2`, `choice`, `stim1_dur`,
  `stim2_dur`, `delay_dur`.

### 3. Parametric working-memory family

#### `wm_angle` — Circular-angle WM

- **Alias**: `parametric_wm`.
- **Objective**: Remember a continuous angle over a delay and reproduce it.
- **Inputs**: `(N, n_t, 2)` — `[cos(theta), sin(theta)]` during stimulus.
- **Targets**: `(N, n_t, 2)` — same during decision.
- **Mask**: `(N, n_t, 2)` — 1 during stimulus and decision, 0 during delay.
- **Params**: `n_trials=100`, `n_t=75`, `sigma_in=0.0`, `seed=None`.
- **Condition keys**: `theta`.

#### `wm_frequency` — Frequency-comparison WM

- **Alias**: `romo` (Romo et al., 1999).
- **Objective**: Compare two sequential vibration frequencies `f1` vs `f2` after a delay.
- **Inputs**: `(N, total_duration, 1)` — normalized frequency.
- **Targets**: `(N, total_duration, 1)` — `(f1 - f2)/FSPAN`.
- **Mask**: `(N, total_duration, 1)` — 1 during decision.
- **Params**: `n_trials=1000`, `sigma_in=0.01`, `fpairs=None`,
  `catch_fraction=0.0`, `delay_discrete=None`, `dt=20.0`, `seed=None`.
- **Timing** (ms): fixation 100, stim1 100, delay 500-1000, stim2 100, decision 100.
- **Condition keys**: `f1`, `f2`, `delay`.

### 4. Perceptual decision-making family

#### `rdm` — Random-dot-motion DM

- **Alias**: `two_afc`.
- **Objective**: Report sign of a noisy motion coherence.
- **Inputs**: `(N, total_duration, 1)` — coherence + noise.
- **Targets**: `(N, total_duration, 1)` — `+1` or `-1`.
- **Mask**: `(N, total_duration, 1)` — 1 during response.
- **Params**: `n_trials=1000`, `coherences=[-4,-2,-1,1,2,4]`, `sigma_in=0.1`,
  `catch_fraction=0.0`, `dt=20.0`, `seed=None`.
- **Timing** (ms): fixation 100, stimulus 800, delay 100, decision 20.
- **Condition keys**: `coherence`, `correct_choice`.
- **Bug fix (refactor)**: `catch_fraction > 0` previously crashed
  (`UnboundLocalError`) and produced inconsistent condition dicts; fixed.

### 5. Multisensory decision-making family

#### `raposo` — Multisensory decision making

- **Objective**: Attend to visual, auditory, or both modalities and report dominant direction.
- **Inputs**: `(N, total_duration, 4)` — `[visual_stim, auditory_stim, visual_ctx, auditory_ctx]`.
- **Targets**: `(N, total_duration, 1)` — choice sign.
- **Mask**: `(N, total_duration, 1)` — 1 during response.
- **Params**: `n_trials=1000`, `coherences=[-4,-2,-1,1,2,4]`,
  `catch_fraction=0.0`, `context=None` (1=visual, -1=auditory, 0=both),
  `sigma_in=0.1`, `dt=20.0`, `seed=None`.
- **Timing** (ms): fixation 100, ctx_pre 350, stimulus 800, delay 100, decision 20.
- **Condition keys**: `context` (-1/0/1), `choice`.

### 6. Go/NoGo family

#### `go_nogo` — Tolmachev & Engel (2025) Go/NoGo

- **Objective**: Report 1 / 0 / 0.5 at the Go cue depending on whether the
  held scalar input is above / below / at 0.5. Deterministic.
- **Inputs**: `(N, n_steps, 3)` — `[value, go_cue, bias]`.
- **Targets**: `(N, n_steps, 1)`.
- **Mask**: `(N, n_steps, 1)` — 1 in `mask_periods` (default `(10,30),(40,60)`).
- **Params**: `n_steps=60`, `stim_on/off`, `cue_on/off`, `n_values=11`,
  `n_reps=1` (total trials = `n_values * n_reps`), `input_dim=3`,
  `output_dim=1`, `mask_periods`, `seed=None`.
- **Condition keys**: `input_value`, `output_value`.

### 7. Checkerboard decision-making family

#### `checkerboard` — Kleinman et al. (2025) checkerboard DM

- **Objective**: Report the majority color of a red/green checkerboard by
  reaching to the matching target; target colors are randomized across
  left/right so color and direction decisions are statistically independent.
- **Inputs**: `(N, n_t, 4)` — `[left_target_color (-1 red/+1 green),
  right_target_color, red_coh, green_coh]`; target channels on during the
  targets epoch, coherence channels (+ independent `N(0, sigma_in^2)` noise)
  on during decision.
- **Targets**: `(N, n_t, 2)` — `[left_dv, right_dv]`, correct DV = 1 during decision.
- **Mask**: `(N, n_t, 2)` — 0 during the first `loss_ramp_ms` of the decision
  epoch and padding, 1 elsewhere.
- **Params**: `n_trials=64`, `dt=10.0`, `mode="train"|"eval"`,
  `cohs=linspace(-0.9, 0.9, 14)`, `catch_fraction=0.1`, `sigma_in=0.1`,
  `loss_ramp_ms=200`, `balanced=False`, `seed=None`.
- **Timing** (ms): center hold ~N(200, 50^2) -> targets ~U[600, 1000] ->
  decision 1500 -> off 100; `mode="eval"` fixes (200, 800, 1500, 100).
  Train-mode trials are zero-padded to the batch max.
- **Condition keys**: `coherence`, `left_color`, `correct_choice`
  (0 left / 1 right / -1 catch), `is_catch`, `epochs`; deprecated aliases
  `catch` / `epoch_bounds` are still present.

### 8. Multitask families

#### `multitask_yang` — Yang et al. (2019), 20 rules

- Requires `rule` (one of 20 rule names: fdgo, dm1, dmsgo, ...).
- **Params**: `rule`, `n_trials=64`, `mode="random"|"test"|"psychometric"`,
  `sigma_in=0.01` (noise std = `sigma_in * sqrt(2/alpha)`), `seed`, plus
  per-rule kwargs (`dt`, `alpha`, ...).
- Task class returns torch tensors; the legacy module-level
  `generate_trials(rule, ..., sigma_x=...)` shim returns numpy and is
  unchanged. Weighted masks (response ×5, fixation ×2).
- Consumed via `MultitaskYangDataset` (notebook 12).
- **Condition keys**: `rule`, `rule_name`, `response_loc`, `epochs`.

#### `multitask_flexible` — Driscoll et al. (2024), 15 rules

- Same structure as `multitask_yang` with low-dim (sin, cos) stimulus
  encoding: 20 inputs, 3 outputs. Consumed via `MultitaskFlexibleDataset`
  (notebook 13).

## Notebook Usage Mapping

| Task | Used in notebook(s) | How |
|---|---|---|
| `mante` | `10_constrained_RNN_paradigmA.ipynb`, `02_latent_circuit_paradigmB.ipynb`, `14_activation_paradigmA.ipynb` | SparseRNN / ModularRNN / CTRNN comparison; latent circuit training; CDDM (via `mante_task.generate_trials`) |
| `mante2` | `07_lowrank_RNN_paradigmA.ipynb`, `08_lowrank_RNN_paradigmB.ipynb` | Mante analysis / LINT |
| `rdm` | `07_lowrank_RNN_paradigmA.ipynb` | Rank-1 RDM tutorial |
| `dms_continuous` | `cognitive_tasks.ipynb` | demo |
| `wm_angle` | `cognitive_tasks.ipynb` | demo |
| `wm_frequency` | `07_lowrank_RNN_paradigmA.ipynb` | 5-task accuracy check |
| `dms` | `07_lowrank_RNN_paradigmA.ipynb`, `test/07a_DMS_test.ipynb` | Learnability check |
| `raposo` | `07_lowrank_RNN_paradigmA.ipynb` | 5-task accuracy check |
| `go_nogo` | `14_activation_paradigmA.ipynb` | Go/NoGo training |
| `checkerboard` | `17_multi_area_rnn_paradigmA.ipynb` | Multi-area RNN training (Kleinman 2025) |
| `multitask_yang` | `12_multitask_paradigmA.ipynb` | 20-task training |
| `multitask_flexible` | `13_flexible_multitask_paradigmA.ipynb` | Flexible multitask (Driscoll 2024) |

## Related Registry Entries

- `src/neuralrnn/data/neurogym_dataset.py` — wrappers for `neurogym` environments.
- `src/neuralrnn/data/timeseries_dataset.py` — `lorenz63` and other non-cognitive timeseries.
- `src/neuralrnn/data/bartolo_monkey_dataset.py` — probabilistic reversal-learning data.
- `src/neuralrnn/data/custom_dataset.py` — user-provided arrays.

See `other_task.md` (repo root) for a cross-project comparison with
`trainRNNbrain`, `neurogym`, and `multitask`.
