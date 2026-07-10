# Multitask RNN (Yang et al., 2019; Driscoll et al., 2024)

> **Paradigm**: A (task optimization)
> **Original repositories**: https://github.com/gyyang/multitask (Yang), https://github.com/lauradriscoll/flexible_multitask (Driscoll)
> **Framework target**: `models/ctrnn` + `SupervisedObjective`
> **Status**: ✅ Ready (Yang), ✅ Ready (Driscoll flexible)

## 1. What problem it solves

The prefrontal cortex supports many cognitive functions, yet most experiments and models study one task at a time. Yang et al. (2019) trained a single recurrent neural network to perform 20 inter-related cognitive tasks and asked how a single circuit represents and supports many distinct tasks. The paper shows that, after training, units self-organize into clusters that are specialized for subsets of tasks, and that task representations can become compositional.

## 2. Core method

**Network dynamics (continuous-time leaky RNN):**

$$
\tau \frac{d\mathbf{r}}{dt} = -\mathbf{r} + f\left(W^{rec}\mathbf{r} + W^{in}\mathbf{u} + \mathbf{b} + \sqrt{2\tau\sigma_{rec}^2}\boldsymbol{\xi}\right)
$$

Euler-discretized with $\Delta t = 20$ ms, $\tau = 100$ ms, $\alpha = \Delta t / \tau = 0.2$.

**Reference hyperparameters:**
- Recurrent units: 256
- Activation: Softplus
- Recurrent noise: $\sigma_{rec} = 0.05$
- Input noise: $\sigma_{in} = 0.01$
- Recurrent initialization: $0.5 \cdot I$

**Inputs:**
- Fixation input (1 dim)
- Modality 1 ring (32 dims)
- Modality 2 ring (32 dims)
- Rule input (20 dims, one-hot)

Total input dim = 85.

**Outputs:**
- Fixation output (1 dim)
- Output ring (32 dims)

Total output dim = 33.

**Tasks (20):**
- Go family: `fdgo`, `reactgo`, `delaygo`
- Anti family: `fdanti`, `reactanti`, `delayanti`
- DM family: `dm1`, `dm2`, `contextdm1`, `contextdm2`, `multidm`
- Delayed DM family: `delaydm1`, `delaydm2`, `contextdelaydm1`, `contextdelaydm2`, `multidelaydm`
- Matching family: `dmsgo`, `dmsnogo`, `dmcgo`, `dmcnogo`

**Training:**
- Loss: masked MSE with fixation target 0.85/0.05 and Gaussian output-ring targets.
- Mask: pre-response weight 1, response weight 5, fixation output 2x.
- Optimizer: Adam, lr=0.001.
- Task interleaving with `contextdm1` and `contextdm2` oversampled 5x.

## 3. How to use this method in our framework

| Original code | Framework API | Note |
|---|---|---|
| `multitask/task.py` | `data/tasks/multitask_yang_task.py` | All 20 task generators |
| `multitask/network.py:LeakyRNNCell` | `models/ctrnn/modeling_ctrnn.py:CTRNNModel` | Same leaky CTRNN dynamics |
| `multitask/train.py` | `train/trainer.py:Trainer` + `SupervisedObjective` | Masked MSE regression |
| Task variance / FTV / clustering | Notebook inline | `12_multitask_paradigmA.ipynb` |

**New dataset API:**

```python
from neuralrnn.data.tasks.multitask_yang_dataset import MultitaskYangDataset

ds = MultitaskYangDataset(
    rules=RULES_ALL,
    rule_prob_map={"contextdm1": 5.0, "contextdm2": 5.0},
    batch_size=64,
    sigma_x=0.01,
    seed=0,
)
batch = ds.sample_batch()  # inputs, targets, mask of shape (B, T, D)
```

**Model config:**

```python
from neuralrnn.models.ctrnn.configuration_ctrnn import CTRNNConfig

config = CTRNNConfig(
    input_dim=85, latent_dim=256, output_dim=33,
    dt=20.0, tau=100.0, activation="softplus",
    sigma_rec=0.05, noise_alpha_scaling=True,
)
```

## 4. Consistency with the original implementation

- Trial timing, stimulus ring encoding, coherence values, and mask weighting are ported directly from `multitask/task.py`.
- The internal `(time, batch, dim)` layout is transposed to NeuralRNN's `(batch, time, dim)` at the dataset boundary.
- The recurrent noise scaling uses `sigma_rec * sqrt(2/alpha)` when `noise_alpha_scaling=True`, matching the original Euler discretization.

## 5. Reproduction experiments

Notebook: `NeuralRNN/notebook/12_multitask_paradigmA.ipynb`

Reproduces:
- Fig. 2: Task-variance clusters (heatmap, t-SNE, lesion effects).
- Fig. 4: Pairwise FTV distributions.
- Fig. 5: Context-dependent DM dissection.
- Fig. 6: Compositional task vectors.

Expected behavior: with a 256-unit network trained for enough steps, all tasks reach >95% performance and the analyses show clustered/compositional representations.

## 6. Flexible multitask extension (Driscoll et al., 2024)

The follow-up paper (Driscoll, Shenoy & Sussillo, 2024) studies a 15-task variant with lower-dimensional circular stimuli and identifies **dynamical motifs** as the substrate of flexible multitask computation. The framework now implements this paradigm as well.

- Paper: https://doi.org/10.1038/s41593-024-01668-6
- Original code: https://github.com/lauradriscoll/flexible_multitask
- Summary: `reference_project/multitask/summary.md`
- Reproduction plan: `reference_project/multitask/plan.md`

**Dataset API:**

```python
from neuralrnn.data.tasks.multitask_flexible_task import generate_trials, RULES_ALL
from neuralrnn.data.tasks.multitask_flexible_dataset import MultitaskFlexibleDataset

inputs, targets, mask, conditions = generate_trials("delaygo", n_trials=64, mode="random", seed=0)

ds = MultitaskFlexibleDataset(
    rules=RULES_ALL,
    rule_prob_map={"contextdelaydm1": 5.0, "contextdelaydm2": 5.0},
    batch_size=64,
    sigma_x=0.01,
    seed=0,
)
batch = ds.sample_batch()  # inputs, targets, mask of shape (B, T, D)
```

**Model config:**

```python
from neuralrnn.models.ctrnn.configuration_ctrnn import CTRNNConfig

config = CTRNNConfig(
    input_dim=20, latent_dim=128, output_dim=3,
    dt=20.0, tau=100.0, activation="softplus",
    sigma_rec=0.05, noise_alpha_scaling=True,
)
```

**Notebooks:**
- `NeuralRNN/notebook/12a_flexible_multitask_paradigmA.ipynb` — end-to-end training and analysis.
- `NeuralRNN/notebook/12b_flexible_multitask_paradigmA.ipynb` — step-by-step debugging panels for Figures 1–4; includes `OriginalStyleFixedPointFinder` comparison and per-period fixed-point seeding.
- `NeuralRNN/notebook/12c_flexible_multitask_paradigmA.ipynb` — refactored version of 12b using only global `NumericFixedPointFinder`, a single highlighted stimulus direction in Fig 1, and Pro/Anti hue inversion in Fig 2.
- `NeuralRNN/notebook/13_flexible_multitask_paradigmA.ipynb` — step-by-step tutorial that follows the paper structure in three parts (single-task / two-task / multi-task).  Shared helpers are defined in dedicated upfront cells, each figure is produced in its own cell, and task examples are visualized before the corresponding fixed-point analyses.  Fixed-point caches are saved/loaded next to each model checkpoint to avoid recomputation.

Reproduces (analysis cells are step-by-step):
- Fig. 1/2: Fixed points and input interpolation for single/two-task motifs.
- Fig. 3: Per-task-period variance matrix with hierarchical clustering.
- Fig. 4: Shared vs. distinct motif comparison.

**Consistency notes:**
- Stimuli and responses are 2D circular vectors `(A sin θ, A cos θ)` instead of 32-unit rings.
- Context-dependent delayed DM tasks (`contextdelaydm1`, `contextdelaydm2`) are oversampled 5× by default, matching the paper's training recipe.
- L2 activity and weight regularization (`1e-6`) are applied through a custom `MultitaskObjective`.
