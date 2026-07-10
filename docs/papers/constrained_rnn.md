# Constrained RNNs: seRNN, SparseRNN, and ModularRNN

> **Paradigm**: A (task optimization)
> **Original repositories**:
> - seRNN: `reference_project/constrained_rnn/spatially-embedded-RNN/`
> - Sparse / modular / local random RNN: `reference_project/randomRNN/critical_init/fig4/`
> **Framework target**: `models/constrained_rnn` + `ConstrainedSupervisedObjective`
> **Status**: ✅ Ready

## 1. What problem it solves

Recurrent networks trained for cognitive tasks are usually fully connected, yet biological neural circuits are constrained by physical space, metabolic cost, and sparse wiring. Constrained RNNs train task-optimized recurrent networks while restricting connectivity through hard masks or soft spatial regularizers. This makes it possible to ask how structure shapes dynamics and task performance, and to reproduce two distinct lines of work in a single framework:

1. **Spatially-embedded RNN (seRNN)** from Achterberg et al. (2023), *Nature Machine Intelligence*: units are placed on a 3-D grid and a distance-weighted regularizer biases recurrent weights toward short-range, topologically central connections.
2. **Sparse / modular random RNNs** inspired by Pachitariu et al. (critical initialization): recurrent connectivity is sampled with a fixed density or a block-modular structure, and only a subset of weights is trainable.

Both families are implemented under the `constrained_rnn` model family, which extends the CTRNN recurrence with hard structural masks and exposes specialized regularizers.

## 2. Core method

### 2.1 Base ConstrainedRNN recurrence

All constrained variants share the CTRNN Euler-discretized dynamics:

$$
z_{t+1} = (1 - \alpha) z_t + \alpha \phi\left( (M_{\text{rec}} \odot W_{\text{rec}}) z_t + (M_{\text{in}} \odot W_{\text{in}}) x_t + b_{\text{rec}} \right)
$$

$$
y_t = (M_{\text{out}} \odot W_{\text{out}}) z_t + b_{\text{out}}
$$

where:
- $\alpha = \Delta t / \tau$,
- $\phi$ is the activation (ReLU by default; any name supported by `neuralrnn.activations.get_activation` can be used),
- $M_{\text{rec}}$, $M_{\text{in}}$, $M_{\text{out}}$ are optional binary masks (default all ones),
- $\odot$ denotes elementwise multiplication.

A mask value of zero means **no connection**: the corresponding weight receives zero gradient and the effective weight is always zero. Masks are registered as non-trainable buffers and saved with the model state dict.

### 2.2 seRNN spatial regularizer

Units are placed on a regular grid. The recurrent weights are penalized by a distance-weighted L1 term, optionally combined with unbiased weighted communicability:

$$
\Omega(W) = \lambda \sum_{ij} |W_{ij}| \; d_{ij}^{\,p}
$$

with optional communicability term:

$$
C = \exp\left( D^{-1/2} |W| D^{-1/2} \right), \quad D_{ii} = \sum_j |W_{ij}|
$$

$$
\Omega_{\text{comm}}(W) = \lambda \sum_{ij} |W_{ij}| \; C_{ij}^{\,\alpha} \; d_{ij}^{\,p}
$$

Here $d_{ij}$ is the Euclidean distance between units $i$ and $j$ on the grid, $\lambda$ is `se1_weight`, $\alpha$ is `comms_factor`, and $p$ is `distance_power`. When `comms_factor=0` the regularizer reduces to pure distance-weighted L1.

### 2.3 Sparse RNN

A target fraction $s$ of recurrent connections is kept; the rest are hard-masked to zero:

$$
M_{ij} \sim \text{Bernoulli}(s)
$$

Autapses can be removed by setting `allow_self_connections=False`.

### 2.4 Modular RNN

Hidden units are partitioned into $K$ modules of equal size. Intra-module connections are sampled with density $\rho_{\text{intra}}$; inter-module connections exist with probability $p_{\text{inter}}$:

$$
M_{ij} = \begin{cases}
\text{Bernoulli}(\rho_{\text{intra}}) & \text{if module}(i) = \text{module}(j) \\
\text{Bernoulli}(p_{\text{inter}}) & \text{otherwise}
\end{cases}
$$

### 2.5 Training objective

For hard-masked variants the standard supervised loss is sufficient because masking is applied every forward pass. For seRNN the structural penalty is added to the task loss:

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \gamma \, \Omega(W_{\text{rec}})
$$

In the framework this is handled by `ConstrainedSupervisedObjective(task_type='classification', constraint_weight=1.0)`; the model exposes `constraint_loss()` and the objective adds it to the task loss.

### 2.6 Key hyperparameters

| Parameter | Meaning | Typical value |
|---|---|---|
| `latent_dim` | Hidden size $M$ | 100 |
| `dt`, `tau` | Euler step and time constant | `dt=20`, `tau=100` for Mante; `dt=None`, `tau=1.0` for seRNN |
| `sparsity` | Fraction of recurrent connections kept | 0.05 / 0.10 / 0.50 |
| `n_modules`, `p_inter`, `intra_density` | Modular structure | 4, 0.05, 1.0 |
| `grid_shape`, `se1_weight`, `comms_factor` | seRNN grid and regularizer | `(5,5,4)`, 0.5, 1.0 |

## 3. How to use this method in our framework

| Original code | Framework API | Note |
|---|---|---|
| `seRNN_demo_torch.ipynb`: maze generator + custom loop | `notebook/10_constrained_RNN_paradigmA.ipynb` | Uses `MazeDataset` wrapper + `Trainer` + `ConstrainedSupervisedObjective` |
| `seRNN_demo_torch.ipynb`: `SE1_sWc` regularizer | `models/constrained_rnn/modeling_constrained_rnn.py: SERNNModel.constraint_loss()` | Distance + communicability |
| `critical_init/fig4/sparse_clustered_local_sims.py::sparse_evals` | `SparseRNNModel` | Hard random sparse mask |
| `critical_init/fig4/sparse_clustered_local_sims.py::clustered_evals` | `ModularRNNModel` | Block-modular mask |
| Mante task generator | `data/tasks/mante_task.py: generate_trials()` | 6 inputs, 2 outputs, decision mask |
| Fixed-point search | `analysis.fixed_points.find_fixed_points()` | Model-agnostic |

### 3.1 Quick start

```python
from neuralrnn import AutoConfig, AutoModel

# Base constrained RNN with custom masks
cfg = AutoConfig.for_model(
    'constrained_rnn',
    input_dim=6, latent_dim=100, output_dim=2,
    rec_mask=rec_mask, in_mask=in_mask, out_mask=out_mask,
)
model = AutoModel.from_config(cfg)

# Spatially-embedded RNN
cfg = AutoConfig.for_model(
    'se_rnn',
    input_dim=8, latent_dim=100, output_dim=4,
    grid_shape=(5, 5, 4), se1_weight=0.5, comms_factor=1.0,
)
model = AutoModel.from_config(cfg)

# Sparse RNN
cfg = AutoConfig.for_model(
    'sparse_rnn',
    input_dim=6, latent_dim=100, output_dim=2,
    sparsity=0.10, allow_self_connections=False, seed=42,
)
model = AutoModel.from_config(cfg)

# Modular RNN
cfg = AutoConfig.for_model(
    'modular_rnn',
    input_dim=6, latent_dim=100, output_dim=2,
    n_modules=4, p_inter=0.05, intra_density=1.0,
)
model = AutoModel.from_config(cfg)
```

### 3.2 Training

For seRNN:
```python
from neuralrnn.train.objectives.constrained import ConstrainedSupervisedObjective

objective = ConstrainedSupervisedObjective(
    task_type='classification', constraint_weight=1.0
)
```

For hard-masked sparse/modular networks the standard `SupervisedObjective` is sufficient.

### 3.3 Analysis

All analysis tools that work through the `NeuralDynamicsModel` interface (PCA, fixed points, vector field, linearization) can be used directly on constrained RNNs.

## 4. Consistency with the original implementation

- **seRNN**: The maze task generator, network size (100 units), 3-D $5 \times 5 \times 4$ grid, orthogonal recurrent init, input noise (std=0.05), Adam optimizer (lr=1e-3), and 10-epoch training protocol match the original PyTorch demo. The final task loss/accuracy and modularity values are comparable; small-worldness can differ because of framework-specific random-number streams and null-model sampling.
- **Sparse / modular masks**: Masks are sampled once at model construction from a seeded RNG. The same seed reproduces the same mask. The effective recurrent weight is `W * mask` every forward pass, so masked positions stay zero and receive no gradient, identical to setting weights to zero manually.
- **Dale constraints**: Can be combined with structural masks. The structural mask is applied first, then the Dale sign transform.

## 5. Reproduction experiments

- **Notebook**: `notebook/10_constrained_RNN_paradigmA.ipynb`
  - **Part I**: seRNN maze first-choice demo. Expected final validation accuracy ~0.60–0.65; modularity ~0.15–0.20; negative weight-distance correlation.
  - **Part II**: CTRNN vs. SparseRNN (5%, 10%, 50%) on Mante. Expected test accuracy $\geq$ 90% for CTRNN and 50% sparse; lower sparsity leads to graceful degradation.
  - **Part III**: CTRNN vs. SparseRNN on DelayComparison. Expected high accuracy; CTRNN shows a clear line attractor in delay-period PCA; sparse networks show similar geometry when enough connectivity remains.

## 6. Key design choices and caveats

- **Hard masks are buffers**, not parameters, so they are saved/loaded with the model state dict.
- **Masks multiply weights at every forward pass**, ensuring zero effective weight and zero gradient at masked positions.
- **seRNN regularizer is a soft penalty**, not a hard constraint; it encourages but does not enforce short-range connectivity.
- **Sparse/Modular masks are sampled once at model construction** from a seeded RNG; the same seed reproduces the same mask.
- **For large hidden sizes**, the communicability `matrix_exp` in seRNN can be expensive.
