# Durstewitz lab: PLRNN family for Dynamical Systems Reconstruction (DSR)

> **Paradigm**: B (dynamical systems reconstruction)
> **Original source**: Durstewitz lab PLRNN series (Durstewitz 2017; Brenner et al. 2022, 2024; Hess et al. 2023)
> **Framework target**: `models/plrnn` + `analysis` (analytic fixed points / Lyapunov / D_stsp / D_H / manifolds)
> **Status**: ✅ Reference implementation (`shallow_plrnn`, `dend_plrnn`, `alrnn`)

## 1. What problem it solves

Given an observed time series (a simulated dynamical system such as Lorenz-63, or measured neural/behavioral signals), train a generative RNN whose **autonomous** rollouts match the real system in long-term statistics—reproducing attractor shape, power spectrum, and even the largest Lyapunov exponent. This is **Dynamical Systems Reconstruction (DSR)**.

The framework implements three PLRNN variants that share the same piecewise-linear structure and therefore the same analytic toolbox:

- **shallowPLRNN** (CNS 2023 tutorial)
- **dendPLRNN** (Brenner et al., ICML 2022)
- **ALRNN** (Brenner et al., NeurIPS 2024)

## 2. Core method: shallowPLRNN

Shallow piecewise-linear RNN:

$$
z_t = A\,z_{t-1} + W_1\,\mathrm{ReLU}(W_2 z_{t-1} + h_2) + h_1\;(+\,C s_t),\qquad A=\mathrm{diag}.
$$

- Readout: identity (direct observation of latent states; in DSR the latent dimension $M$ usually equals the observation dimension $N$);
- Training objective: **Generalized Teacher Forcing (GTF)**—during the forward pass, observations are injected with strength $\alpha$ into the prediction $z \leftarrow \alpha z^{\text{obs}} + (1-\alpha) z^{\text{pred}}$, then MSE is computed. Small $\alpha$ (e.g. 0.1) is critical for gradient training of chaotic systems (`TeacherForcingObjective`);
- Analytic power: piecewise-linear structure ⇒ closed-form Jacobian
  $J(z)=\mathrm{diag}(A)+W_1\mathrm{diag}(\mathbb{1}[W_2 z+h_2>0])W_2$,
  and fixed points / $k$-cycles can be solved **linearly** from the cycle equations (`scy_fi`).

## 3. PLRNN family variants

All three variants live in `models/plrnn`, share the same config base, and expose `analytic_parameters()` so the analytic fixed-point backend can be reused.

### 3.1 dendPLRNN

State equation (linear spline basis expansion):

$$
\mathbf{z}_t = \mathbf{A}\mathbf{z}_{t-1}
+ \mathbf{W}\sum_{b=1}^{B}\alpha_b\,\mathrm{ReLU}(\mathbf{z}_{t-1}-\mathbf{h}_b)
+ \mathbf{h}_0\;(+\,\mathbf{C}\mathbf{s}_t).
$$

Key ideas:

- Each latent unit is equipped with $B$ ReLU basis functions at different thresholds $\mathbf{h}_b$;
- The sum is a **linear spline** over the latent state, giving the network more expressive power without increasing the latent dimension;
- The expansion is mathematically equivalent to a conventional PLRNN with $M \cdot B$ hidden units, so the analytic fixed-point machinery still applies;
- Clipping variants (e.g. mirrored basis) can guarantee bounded orbits when the spectral radius of $\mathbf{A}$ is below 1.

Config knobs:

| Field | Meaning |
|---|---|
| `n_bases` | Number $B$ of spline bases per latent unit |
| `use_clipping` | Enable hard clipping of the basis expansion |
| `clip_range` | Optional scalar clip range for latent states |

### 3.2 ALRNN

State equation (almost-linear RNN):

$$
\mathbf{z}_t = \mathbf{A}\mathbf{z}_{t-1} + \mathbf{W}\boldsymbol{\Phi}^*(\mathbf{z}_{t-1}) + \mathbf{h}\;(+\,\mathbf{C}\mathbf{s}_t),
$$

where

$$
\boldsymbol{\Phi}^*(\mathbf{z}) = [z_1,\dots,z_{M-P},\mathrm{ReLU}(z_{M-P+1}),\dots,\mathrm{ReLU}(z_M)]^\top.
$$

Key ideas:

- Only the last $P = M - \texttt{n_linear}$ latent units pass through ReLU; the remaining units are linear;
- The state space is therefore divided into only $2^P$ linear subregions, making post-hoc symbolic analysis tractable;
- Training typically uses **identity teacher forcing** (reset the first $N$ latents to observations every $\tau$ steps) plus MSE.

Config knobs:

| Field | Meaning |
|---|---|
| `n_linear` | Number of linear (non-ReLU) units; $P = M - \texttt{n_linear}$ |
| `use_clipping` | Enable hard clipping of latent states |
| `clip_range` | Optional scalar clip range |

## 4. SCYFI: analytic fixed points and $k$-cycles

SCYFI (Searcher for Cycles and Fixed points) is the algorithm from Eisenmann et al. (2023). It exploits the piecewise-linear structure of PLRNNs to locate fixed points and periodic cycles **analytically**, without gradient-based numerical search.

For a fixed ReLU activation pattern $D = \mathrm{diag}(\mathbb{1}[W_2 z + h_2 > 0])$, the shallowPLRNN map is affine:

$$
z_t = (A + W_1 D W_2) z_{t-1} + h_1 + W_1 D h_2.
$$

A fixed point in that region therefore satisfies

$$
z^* = (I - A - W_1 D W_2)^{-1} (h_1 + W_1 D h_2),
$$

provided the matrix is invertible. For a $k$-cycle with patterns $D_1,\dots,D_k$, SCYFI solves the composed affine map analytically and checks that the candidate actually lies in the assumed regions (self-consistency).

Because the number of possible activation patterns grows combinatorially, SCYFI uses a heuristic: after each failed candidate, it re-initializes the search in the region of the last "virtual" fixed point. This typically converges to all real fixed points and low-order cycles without exhaustive enumeration.

### 4.1 Constant task inputs

For a non-autonomous PLRNN with constant task input $s^*$:

$$
z_t = A z_{t-1} + W_1 \mathrm{ReLU}(W_2 z_{t-1} + h_2) + h_1 + C s^*.
$$

The fixed-point equation is still affine in each region with an effective bias

$$
h_{1,\text{eff}} = h_1 + C s^*.
$$

The framework therefore folds $C s^*$ into the bias inside `analytic_parameters(task_input=s*)` and runs the autonomous SCYFI solver. This avoids the previous fallback to numeric search for task-conditioned fixed points.

## 5. DetectingManifolds: stable and unstable manifolds

DetectingManifolds (Eisenmann et al., 2025) extends SCYFI from isolated fixed points/cycles to their **stable and unstable manifolds**. These manifolds are crucial for understanding state-space organization:

- **Stable manifolds** of saddle points separate basins of attraction in multistable systems.
- **Unstable manifolds** show where trajectories diverge from saddles.
- Intersections of stable and unstable manifolds create homoclinic/heteroclinic points and are a signature of chaos.

Algorithm 1 (DetectingManifolds) works as follows:

1. Use SCYFI to find a saddle fixed point $p$.
2. Compute the local Jacobian $J(p) = A + W_1 D(p) W_2$.
3. The local stable (resp. unstable) manifold is the eigenspace of eigenvalues with $|\lambda| < 1$ (resp. $|\lambda| > 1$).
4. Sample points along these eigenvectors, staying inside the same ReLU region.
5. Propagate sampled points forward (unstable) or backward (stable). The backward step inverts the affine map $z_{t-1} = (A + W_1 D W_2)^{-1}(z_t - h_1 - W_1 D h_2)$ and checks self-consistency of $D$.
6. When points enter a new region, fit a PCA segment to the support points in that region.
7. Repeat recursively to trace the manifold across multiple regions.

In NeuralRNN this is exposed as `compute_manifold(model, fixed_point, task_input=None, stable=True, ...)`. It operates on the effective shallowPLRNN parameters from `model.analytic_parameters(task_input)`, so it works for `shallow_plrnn`, `dend_plrnn`, and `alrnn`.

## 6. How it lands in the framework

| Original code | Framework API | Note |
|---|---|---|
| `shallowPLRNN.forward(z, s)` | `models/plrnn/modeling_plrnn.py:ShallowPLRNNModel.recurrence` | element-wise numerical alignment |
| `dendPLRNN/BPTT_TF/bptt/PLRNN_model.py` | `DendPLRNNModel` | `_basis` implements the spline expansion |
| `ALRNN-DSR/src/models/alrnn.jl` | `ALRNNModel` | `_phi_star` implements $\boldsymbol{\Phi}^*$ |
| Analytic Jacobian | `*.jacobian` | diff-test against autodiff (except at ReLU boundaries) |
| `predict_sequence_using_gtf` + `generalized_teacher_forcing` | `train/objectives/teacher_forcing.py` | generalized partial forcing for $M\neq N$ |
| `TimeSeriesDataset` + `sample_batch` | `data/timeseries_dataset.py` / `data/trial_dataset.py` | time-first → batch-first; `TrialTimeseriesDataset` preserves trial structure |
| `max_lyapunov_exponent` (QR re-orthonormalization) | `analysis/lyapunov.py` | uses `generate` + `jacobian` contract; divide by `dt` for continuous-time comparison |
| `scy_fi` / `main` / `construct_relu_matrix` | `analysis/fixed_points.py:AnalyticPLRNNFixedPointFinder` | supports constant `task_input` folded into bias |
| DetectingManifolds Algorithm 1 | `analysis/manifolds.py:PLRNNManifoldTracer` + `compute_manifold` | operates on effective shallowPLRNN params |

- New config fields: `latent_dim` (=$M$), `hidden_dim` (=$L$ for shallowPLRNN), `autonomous` (whether external input $C$ is included), plus variant-specific fields `n_bases` and `n_linear`.
- All three models expose `analytic_parameters(task_input=None)` returning `(A, W1, W2, h1, h2)` for the analytic solver.
- Data: `lorenz63` is registered in `data/registry.py` (CNS-2023 zip, includes train/test, `dt=0.01`). Task-state neural activity should use `TrialTimeseriesDataset.from_arrays(..., external_inputs=...)` to preserve trial structure.

## 7. Diff-test points

- **Numerical consistency**: with weights from the original PLRNN notebook, compare framework `recurrence` with original `forward` on the same $z$ (tolerance 1e-6);
- **Analytic vs. autodiff Jacobian**: `allclose` on random $z$ away from ReLU boundaries for all three variants;
- **Analytic fixed points**: `AnalyticPLRNNFixedPointFinder` results should match the original `main(np.diag(A), …)`
  (we keep the original `np.diag(A)` broadcasting convention to ensure reproducible eigenvalues);
- **Task-input fixed points**: for a non-autonomous PLRNN, the numeric backend's lowest-speed fixed point should lie near one of the analytic fixed points when the same `task_input` is passed;
- **Round-trip**: `save_pretrained` / `from_pretrained` preserves configs and weights for `dend_plrnn` and `alrnn`;
- **Overall reconstruction**: after training on Lorenz-63, $\lambda_{\max}/dt$ should be close to 0.9, and D_stsp / D_H should drop significantly.

## 8. Reproduction experiments

### 8.1 Lorenz-63 autonomous reconstruction

Corresponds to `notebook/02_plrnn_reconstruction_paradigmB.ipynb` (Part I). Pipeline:

```text
load_dataset("lorenz63")
→ ShallowPLRNNModel + TeacherForcingObjective(alpha=0.1) + Trainer
→ generate free run
→ compute D_stsp / D_H / λ_max
→ divide λ_max by dt=0.01 to compare with continuous-time ground truth
→ find_fixed_points(..., backend="analytic") to solve fixed points analytically
```

### 8.2 Task-state neural activity reconstruction

Corresponds to `notebook/02_plrnn_reconstruction_paradigmB.ipynb` (Part II). Pipeline:

```text
load_dataset("perceptual_decision_making")
→ CTRNNModel + SupervisedObjective(classification) + Trainer
→ collect 1000 trials of hidden states + task inputs
→ stack as (n_trial, trial_length, n_variable)
→ TrialTimeseriesDataset.from_arrays(activity, external_inputs=inputs)
→ ShallowPLRNNModel(input_dim=3, latent_dim=50, autonomous=False)
  + TeacherForcingObjective(alpha=0.1) + Trainer
→ evaluate free-run reconstruction with held-out task inputs
→ PCA, vector field, analytic fixed points with task input, Jacobian eigenvalues
→ compute_manifold(..., stable=False) to trace unstable manifold segments
```

This demonstrates that PLRNNs can reconstruct **non-autonomous, high-dimensional neural dynamics** driven by task inputs, and that the analytic toolbox (fixed points, manifolds) applies to task-conditioned systems once the constant input is folded into the bias.

### 8.3 Variant comparison in notebook 02

`notebook/02_plrnn_reconstruction_paradigmB.ipynb` now contains dedicated sections that repeat the shallowPLRNN analyses for `dend_plrnn` and `alrnn`:

- **Section 1.6** trains `dend_plrnn` and `alrnn` on the autonomous Lorenz-63 benchmark, evaluates free-run `D_stsp` / `D_H` / `λ_max`, and compares their analytic fixed points to the shallowPLRNN baseline.
- **Section 2.11** trains the same two variants on the 50-dimensional CTRNN task-state activity collected in Section 2.2, evaluates held-out reconstruction quality, and performs per-condition PCA and analytic fixed-point analysis under the zero-coherence task input.

For a focused Lorenz-63-only comparison of the three PLRNN-family variants, see `notebook/02a_plrnn_variants_dend_alrnn.ipynb`.
