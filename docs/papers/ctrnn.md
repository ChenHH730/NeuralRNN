# nn-brain: CTRNN Task Training + Fixed-Point Dynamics Analysis

> **Paradigm**: A (task optimization + interpretable analysis)
> **Original source**: nn-brain tutorial (`RNN_DynamicalSystemAnalysis.ipynb`, based on neurogym)
> **Framework target**: `models/ctrnn` + `analysis` (numerical fixed points / linearization / dimensionality reduction / vector field)
> **Status**: ✅ Reference implementation

## 1. What problem it solves

Cognitive neuroscience often trains RNNs on cognitive tasks (e.g. PerceptualDecisionMaking) and then treats the trained network as a "model brain", using dynamical-systems tools to reverse-engineer **how it computes**—for example, whether evidence accumulation for a decision corresponds to a line attractor.

## 2. Core method

Continuous-time RNN (CTRNN), Euler-discretized from the ODE:

$$
z_t = (1-\alpha)\,z_{t-1} + \alpha\, f\!\big(W_{\text{rec}} z_{t-1} + W_{\text{in}} x_t + b\big),\qquad \alpha=\Delta t/\tau.
$$

- Readout $G_\phi(z)=W_{\text{out}} z + b_{\text{out}}$, argmax for classification tasks;
- Training objective: cross-entropy (supervised, `SupervisedObjective`);
- Analysis: under fixed input conditions, minimize $\|F(z)-z\|^2$ to search for fixed points, eigen-decompose the Jacobian to judge stability, and project activity / fixed points / line attractors onto the same PCA plane.

## 3. How it lands in the framework

| Original code | Framework API | Note |
|---|---|---|
| `CTRNN.recurrence(input, hidden)` | `models/ctrnn/modeling_ctrnn.py: recurrence` | Euler step, includes $\alpha=\Delta t/\tau$ |
| neurogym `Dataset(task)` | `data/neurogym_dataset.py: NeurogymDataset.from_task` | time-first → batch-first |
| Training loop (CE loss) | `SupervisedObjective` + `Trainer` | |
| `optim.Adam([hidden])` minimizing $\|F(z)-z\|^2$ | `analysis/fixed_points.py: NumericFixedPointFinder` | model-agnostic |
| `np.linalg.eig(jac)` + PCA plotting | `analysis/linearization.py` + `analysis/dimensionality.py` | use leading eigen-directions to draw line attractor |

- Config fields: `tau`, `dt`, `activation`, `dale`/`ei_ratio` (E-I version), `sigma_rec`, `trainable_h0`.
- Jacobian: automatic differentiation as the default fallback is sufficient (CTRNN needs no analytic form).
- Data: `perceptual_decision_making` in `data/registry.py` (neurogym, no download needed).

## 4. Diff-test points

Load the CTRNN weights trained by the original notebook into the framework `CTRNNModel`, feed the same `input`, and compare `recurrence(x, h)` with the original `net.rnn.recurrence(input, hidden)` for `allclose` (tolerance 1e-5). The converged fixed-point set should be of the same order of magnitude as in the original notebook and fall in the same PCA region.

## 5. Reproduction experiments

Corresponds to `notebook/` (Paradigm A tutorial). Expected behavior: accuracy on the perceptual decision task increases monotonically with coherence; fixed points line up roughly as a line attractor along PC1, and the leading eigenvalue direction aligns with the accumulation direction.
