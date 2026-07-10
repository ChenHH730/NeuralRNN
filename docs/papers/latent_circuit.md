# Latent Circuit Inference from Heterogeneous Neural Responses

> **Paradigm**: A (Task-Based Optimization) + B (dynamics)
> **Original repo**: https://github.com/langdon-endeavors/latent-circuit-inference
> **Framework location**: `models/latent_circuit` + `LatentCircuitObjective` + `analysis/connectivity` + `analysis/perturbation` + `analysis/psychometric`
> **Status**: ✅ Ready

## 1. Problem Statement

Higher cortical areas like the prefrontal cortex (PFC) show complex heterogeneous tuning of single neurons to multiple task variables. Existing dimensionality reduction methods that correlate neural activity with task variables (regression-based "demixing" approaches) do not incorporate recurrent interactions among task variables, so they cannot reveal how computations arise from connectivity to drive behavior.

High-dimensional RNNs trained on cognitive tasks produce similar heterogeneity but their connectivity is opaque. The latent circuit model addresses this by fitting a low-dimensional recurrent circuit to neural responses, simultaneously inferring connectivity and an embedding matrix.

## 2. Core Method

### Latent Circuit Dynamics

The latent circuit has `n` nodes (default 8) embedded in an `N`-dimensional space (RNN dimension, default 50) via an orthonormal embedding matrix `Q`.

**Recurrence** (Euler discretization):
```
x_t = (1 - α) * x_{t-1} + α * ReLU(w_rec @ x_{t-1} + w_in @ u_t + noise)
```
where `α = dt/τ`, noise ~ N(0, 2α σ²_rec).

**Readout**: Linear map from latent space to task output:
```
y_t = w_out @ x_t
```

### Embedding Matrix Q (Cayley Transform)

Q is parameterized through the Cayley transform of a learnable N×N matrix `a`:
```
skew = (a - a^T) / 2
Q_full = (I - skew) @ (I + skew)^{-1}
Q = Q_full[:n, :]    # shape (n, N)
```

Q is orthonormal: Q @ Q^T = I_n.

### Connectivity Masks

- **Input mask**: Diagonal — each input connects to its designated node
  - Nodes 0-5: 6 task inputs (2 context, 2 motion, 2 color)
  - Nodes 6-7: 2 choice outputs (left/right)
- **Output mask**: Last `output_dim` diagonal entries — choice nodes connect to outputs

### Loss Function

```
L = MSE(readout(x), z) + λ_y * NMSE(x @ Q, y)
```

where:
- `z` = RNN behavioral output (target)
- `y` = RNN hidden states (target)
- `NMSE(a, b) = MSE(a, b) / MSE(b_bar, 0)` (normalized MSE)

### Training

- Optimizer: Adam (lr=0.02, weight_decay=0.001)
- Batch size: 128
- After each gradient step: recompute Q via Cayley transform, reapply connectivity masks
- Ensemble: fit 100 models with different random seeds, select top 10 by test fit quality

### Key Finding

When applied to RNNs trained on a context-dependent decision-making task, the latent circuit reveals a **suppression mechanism** where context representations inhibit irrelevant sensory representations.

## 3. Framework Mapping

| Original code | Framework API | Notes |
|---|---|---|
| `latent_net.py:LatentNet` | `models/latent_circuit/modeling_latent_circuit.py: LatentCircuitModel` | Core model |
| `latent_net.py:LatentNet.fit` | `train/objectives/latent_circuit.py: LatentCircuitObjective` + `Trainer` with `post_step_hook` | Training |
| `connectivity.py:init_connectivity` | Not ported (EIRNN uses different init) | Per user decision |
| `plotting_functions.py:psychometric` | `analysis/psychometric.py: compute_psychometric` | Psychometric analysis |
| `Tutorial.ipynb` | `notebook/05_latent_circuit_paradigmA.ipynb` | Full tutorial |
| Tasks (6 files) | `data/tasks/*.py` | All 6 tasks ported |

### New config fields

- `embedding_dim` (int): N — high-dimensional RNN size (default 50)
- `sigma_rec` (float): Recurrent noise (default 0.15)
- `tau` (float): Time constant in ms (default 200)
- `dt` (float): Discretization step in ms (default 40)

### Data source

Cognitive tasks are procedurally generated (no download needed). Registry entries:
- `mante`: Mante / Siegel-Miller context-dependent decision-making (6 inputs, 2 outputs)
- `siegel_miller`: Backward-compatible alias for `mante`
- `rdm`: Random-dot-motion perceptual decision making (1 input, 1 output)
- `two_afc`: Backward-compatible alias for `rdm`
- `dms_continuous`: Continuous delayed match-to-sample (4 inputs, 2 outputs)
- `wm_angle`: Parametric working memory / circular angle (2 inputs, 2 outputs)
- `parametric_wm`: Backward-compatible alias for `wm_angle`
- `wm_frequency`: Parametric working memory / frequency comparison (1 input, 1 output)
- `romo`: Backward-compatible alias for `wm_frequency`

## 4. Cross-validation with Reference Implementation

- **Model dynamics**: The recurrence matches the reference exactly (Euler discretization with ReLU)
- **Cayley transform**: Q is verified to be orthonormal (Q @ Q^T ≈ I)
- **Connectivity masks**: Verified — input mask is diagonal on first `input_dim` entries, output mask on last `output_dim` entries
- **EIRNN difference**: The reference uses `init_connectivity()` with E/I balance and spectral radius scaling. We use the framework's `ei_rnn` model which enforces Dale's law via sign-constrained weights — conceptually equivalent but different initialization.

## 5. Reproduction

**Notebook**: `notebook/05_latent_circuit_paradigmA.ipynb`

**Key metrics to verify**:
- RNN task accuracy: >90% (after training)
- Latent circuit embedding agreement: Qx vs y scatter on diagonal
- Connectivity correlation: w_rec vs Q^T W_rec Q (r > 0.7 for successful fits)
- Psychometric curves: context-dependent modulation (steep for attended modality, flat for unattended)

**Reference**: Langdon & Engel (2025), Nature Neuroscience. "Latent circuit inference from heterogeneous neural responses during cognitive tasks."
