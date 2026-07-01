# Durstewitz lab: shallowPLRNN Dynamical Systems Reconstruction (DSR)

> **Paradigm**: B (dynamical systems reconstruction)
> **Original source**: CNS 2023 tutorial, Durstewitz lab DSR series
> **Framework target**: `models/plrnn` + `analysis` (analytic fixed points / Lyapunov / D_stsp / D_H)
> **Status**: ✅ Reference implementation (the most complete Paradigm B + analytic analysis example in the framework)

## 1. What problem it solves

Given an observed time series (simulated system such as Lorenz63, or measured neural/behavioral signals), train a generative RNN so that its **autonomous** generated trajectories match the real system in long-term statistics—reproducing attractor shape, power spectrum, and even the largest Lyapunov exponent. This is called **Dynamical Systems Reconstruction (DSR)**.

## 2. Core method

Shallow piecewise-linear RNN (shallowPLRNN):

$$
z_t = A\,z_{t-1} + W_1\,\mathrm{ReLU}(W_2 z_{t-1} + h_2) + h_1\;(+\,C s_t),\qquad A=\mathrm{diag}.
$$

- Readout: identity (direct observation of latent states, the DSR standard setting; $M$ usually equals observation dimension $N$);
- Training objective: **Generalized Teacher Forcing (GTF)**—during the forward pass, inject observations with strength $\alpha$ into the prediction $z \leftarrow \alpha z^{\text{obs}} + (1-\alpha) z^{\text{pred}}$, then compute MSE. Small $\alpha$ (e.g. 0.1) is critical for gradient training of chaotic systems (`TeacherForcingObjective`);
- Analytic power: piecewise-linear structure ⇒ closed-form Jacobian
  $J(z)=\mathrm{diag}(A)+W_1\mathrm{diag}(\mathbb{1}[W_2 z+h_2>0])W_2$,
  and fixed points / $k$-cycles can be solved **linearly** from the cycle equations (`scy_fi`).

## 3. How it lands in the framework

| Original code | Framework API | Note |
|---|---|---|
| `shallowPLRNN.forward(z, s)` | `models/plrnn/modeling_plrnn.py: recurrence` | element-wise numerical alignment |
| Analytic Jacobian | `ShallowPLRNNModel.jacobian` | diff-test against autodiff (except at ReLU boundaries) |
| `predict_sequence_using_gtf` + `generalized_teacher_forcing` | `train/objectives/teacher_forcing.py` | generalized partial forcing for $M\neq N$ |
| `TimeSeriesDataset` + `sample_batch` | `data/timeseries_dataset.py` | time-first → batch-first, targets shifted right by one |
| `max_lyapunov_exponent` (QR re-orthonormalization) | `analysis/lyapunov.py` | uses `generate` + `jacobian` contract |
| `state_space_divergence_*` / `power_spectrum_error` | `analysis/stsp_metrics.py` | D_stsp (binning/gmm), D_H |
| `scy_fi` / `main` / `construct_relu_matrix` | `analysis/fixed_points.py: AnalyticPLRNNFixedPointFinder` | strictly keeps original `A=diag` convention |

- New config fields: `latent_dim` (=$M$), `hidden_dim` (=$L$), `autonomous` (whether external input $C$ is included).
- Exposes `analytic_parameters()` so the analytic backend can fetch $(A, W_1, W_2, h_1, h_2)$.
- Data: `lorenz63` in `data/registry.py` (CNS-2023 zip, includes train/test).

## 4. Diff-test points

- **Numerical consistency**: with weights from the original PLRNN notebook, compare framework `recurrence` with original `forward` on the same $z$ (tolerance 1e-6);
- **Analytic vs. autodiff Jacobian**: `allclose` on random $z$ away from ReLU boundaries;
- **Analytic fixed points**: `AnalyticPLRNNFixedPointFinder` results should match the original `main(np.diag(A), …)`
  (we keep the original `np.diag(A)` broadcasting convention to ensure reproducible eigenvalues);
- **Overall reconstruction**: after training, $\lambda_{\max}$ should be close to 0.9 (Lorenz63), and D_stsp / D_H should drop significantly.

## 5. Reproduction experiments

Corresponds to `notebook/` (Paradigm B Lorenz63 tutorial). Pipeline: `load_dataset("lorenz63")` →
`ShallowPLRNNModel` + `TeacherForcingObjective(alpha=0.1)` + `Trainer` training →
`generate` free run → compute D_stsp / D_H / λ_max, use `find_fixed_points` to analytically solve fixed points and judge stability.
