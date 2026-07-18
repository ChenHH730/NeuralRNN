# Connectome-Constrained Recurrent Networks

> **Paradigm**: B (dynamical systems reconstruction / teacher-student inference)
> **Original repository**: `reference_project/constrained_rnn/connectome-constrained-RNN/`
> **Framework target**: `models/gain_rnn` (`GainRNNModel`) + notebook-local objectives
> **Status**: ✅ Reproduced in `notebook/16_connectome_rnn_paradigmB.ipynb` (Fig. 1 & 2; the legacy `models/connectome_rnn` class was removed in favor of the gain_rnn family)

## 1. What problem it solves

Recent connectomics datasets provide comprehensive synaptic wiring diagrams of large neural circuits. A natural question is: *how much do such connectivity maps constrain the possible dynamics and function of the circuit?* Beiran & Litwin-Kumar (2025) study this with a teacher–student paradigm:

- A recurrent network (**teacher**) with known connectivity generates task-related activity.
- A **student** network shares the teacher's synaptic weight matrix (the connectome) but has different, unknown single-neuron parameters (gains and biases).
- The student is trained either on task output or on recordings of a subset of neurons.

The paper shows that connectivity alone is generally **not sufficient** to predict single-neuron activity, because many combinations of single-neuron parameters solve the same task. However, recordings from a small number of neurons—on the order of the dynamical dimensionality—can break this degeneracy and enable accurate predictions of unrecorded neurons.

## 2. Core method

### 2.1 Firing-rate RNN with per-neuron gain and bias

Both teacher and student are firing-rate RNNs:

$$
\tau \frac{d x_i}{d t} = -x_i + \sum_j J_{ij} r_j + I_i(t),
\qquad
r_i = g_i \, \phi(x_i + b_i),
$$

where $x$ is the input current, $r$ is the firing rate, $J$ is the shared recurrent weight matrix, and $g_i$, $b_i$ are the single-neuron parameters. Euler discretization gives:

$$
x_t = (1 - \alpha) x_{t-1} + \alpha \left( J r_{t-1} + W_{in} u_t + b_{rec} \right),
\qquad
r_t = g \odot \phi(x_t + b),
$$

with $\alpha = \Delta t / \tau$. The nonlinearity $\phi$ can be any activation supported by `neuralrnn.activations.get_activation` (the paper and the default example use `softplus`).

### 2.2 Connectome-constrained vs. unconstrained students

- **Connectome-constrained**: $J$ is fixed and equal to the teacher's. Only gains and biases are trained.
- **Unconstrained connectivity**: gains and biases are fixed to the teacher's; $J$ is trained. Because there is no identity map between unrecorded neurons, a linear assignment is used after training to align student and teacher neurons.

### 2.3 Loss functions

- **Task-output training**: MSE between student and teacher readout.
- **Activity-recording training**: MSE between student and teacher firing rates on the $M$ recorded neurons.

### 2.4 Linear theory

For a linear network at equilibrium, $r = A b$ with:

$$
A = (I - J)^{\dagger} J.
$$

The SVD of $A$ reveals **stiff** parameter modes (large singular values) and **sloppy** modes (small singular values). Recording $M$ neurons corresponds to selecting $M$ rows of $A$; when $M$ reaches the rank of $A$, the unrecorded activity can be perfectly predicted.

### 2.5 Optimal neuron selection

For the linear model, the most informative neuron is the one whose row $A_{i,:}$ overlaps most with the weighted left singular vectors. A greedy algorithm selects neurons, projects out their contribution, and repeats.

## 3. How to use this method in our framework

| Original code | Framework API | Note |
|---|---|---|
| Teacher/student RNN with per-neuron gain/bias | `models/gain_rnn/modeling_gain_rnn.py: GainRNNModel` (`rate` + `outside` + `softplus`) | State `z` = current `x`; `get_firing_rates` returns `r` |
| Fixed/shared recurrent weight | Copy J into `h2h.weight` + `freeze_recurrent=True` | Uniform freeze vocabulary; safetensors save/load |
| Task-output loss | `SupervisedObjective(task_type='regression')` | Standard framework objective |
| Recording-constrained loss | Notebook-local `RecordedActivityObjective` | MSE on recorded neuron firing rates only |
| Linear assignment for unrecorded neurons | `scipy.optimize.linear_sum_assignment` | Used for unconstrained-connectivity baseline |
| Linear theory (SVD) | `numpy.linalg.svd` of $A = (I-J)^+J$ | Stiff/sloppy mode analysis |
| Optimal selection | Notebook-local greedy selection | Maximizes weighted row-singular-vector overlap |

### 3.1 Quick start

```python
from neuralrnn import AutoConfig, AutoModel
import numpy as np

N = 300
J = np.random.randn(N, N).astype(np.float32) * 0.5 / np.sqrt(N)

cfg = AutoConfig.for_model(
    'gain_rnn',
    input_dim=2,
    latent_dim=N,
    output_dim=1,
    dt=0.05,
    tau=1.0,
    activation='softplus',
    activation_params={'beta': 1.0},
    gain_position='outside',
    nonlinearity_mode='rate',
    freeze_input=True,
    freeze_recurrent=True,
    freeze_output=True,
)
model = AutoModel.from_config(cfg)
with torch.no_grad():
    model.h2h.weight.copy_(torch.from_numpy(J))  # the shared connectome
```

### 3.2 Training on recorded neurons

```python
from neuralrnn import SupervisedObjective

class RecordedActivityObjective(SupervisedObjective):
    def __init__(self, teacher, recorded_idx):
        super().__init__(task_type='regression')
        self.teacher = teacher
        self.recorded_idx = recorded_idx

    def compute_loss(self, model, batch):
        out = model(batch['inputs'])
        rates_s = model.get_firing_rates(out.states)
        with torch.no_grad():
            out_t = self.teacher(batch['inputs'])
            rates_t = self.teacher.get_firing_rates(out_t.states)
        diff = rates_s[:, :, self.recorded_idx] - rates_t[:, :, self.recorded_idx]
        loss = (diff ** 2).mean()
        return loss, {'loss': loss.item()}
```

### 3.3 Save / load

```python
model.save_pretrained('./models/16/debug/student_M30')
model = AutoModel.from_pretrained('./models/16/debug/student_M30')
```

## 4. Consistency with the original implementation

- **Dynamics**: Same continuous-time firing-rate equations with Euler discretization; per-neuron gain and bias parameters.
- **Sparse E-I teacher**: Matches the paper's Fig. 1 setup (Gaussian weights with variance $2.4/\sqrt{N}$, 50% sparsity, 70% excitatory neurons, rectified signs).
- **Task-output training**: Students trained only on readout reproduce the output but diverge in single-neuron activity.
- **Recording training**: Connectome-constrained students trained on $M$ recorded neurons predict unrecorded activity once $M$ exceeds the dynamical dimensionality.
- **Linear theory**: Identical construction $A = (I - J)^{\dagger} J$ and SVD-based stiff/sloppy analysis.
- **Optimal selection**: Greedy weighted-SVD selection reproduces the qualitative advantage over random selection.

## 5. Reproduction experiments

- **Notebook (current)**: `notebook/16_connectome_rnn_paradigmB.ipynb` — built on `gain_rnn`,
  faithful to the authors' hyperparameters (N=300, dt=0.05, 60 trials; teacher 1400 /
  students 7000 epochs at `SCALE="full"`).
  - **Section 1–2**: Cycling task (verbatim `create_input` port) + sparse E-I teacher
    (rejection-sampled Dale connectivity, stability-probe h0, Dale hook during training).
  - **Section 3 (Fig. 1)**: Task-output-only student matches the readout but not
    single-neuron activity or gain/bias parameters; per-200-epoch error curves from
    checkpoints; shuffled-identity baselines; gain/bias scatter + linear regression.
  - **Section 4 (Fig. 2)**: Recorded-activity students with M = 30/60/90/180; recorded vs
    unrecorded error curves; the Fig.-1 student as the M=0 baseline.
- **Notebook (archived)**: `notebook/10a_constrained_RNN_paradigmB_executed.ipynb` — the
  earlier reproduction built on the removed `connectome_rnn` class (reduced scale:
  N=150, 1000/800 steps, simplified task). No longer runnable; kept for reference.
- Not yet reproduced: unconstrained-connectivity students (Fig. 2d/f), network-size
  scaling (Fig. 3), model mismatch (Fig. 4), empirical connectomes (Fig. 5), linear
  theory (Fig. 6/8), loss landscape (Fig. 7).

## 6. Key design choices and caveats

- **`GainRNNModel` in `rate` mode keeps the latent state as the current `x`**, not the
  firing rate `r`. Use `get_firing_rates(states)` to obtain `r` for analysis or
  recording objectives (the rate-mode readout already consumes the rates).
- **The shared matrix is copied into `h2h.weight` and frozen** (`freeze_recurrent=True`);
  there is no `fixed_recurrent_weight` config field — save/load goes through safetensors.
- **Dale constraints**: the framework's `dale=True` uses `|W| @ diag(sign)` semantics,
  while the authors' code zeroes sign-violating synapses (`relu(w·s)·s`). Notebook 16
  therefore enforces Dale during teacher training with a `post_step_hook`
  (`w ← relu(w·S)·S ⊙ mask`), matching the paper's "rectified after each epoch".
- **Layer biases**: the reference model has no input/recurrent/readout biases; notebook
  16 zeroes and freezes `input2h.bias`, `h2h.bias` and `readout_layer.bias`.
- **Known minor deviations** (documented in notebook 16 / HANDOFF): gains are not
  ReLU-rectified inside the dynamics (the reference applies `relu(g)`; all gains stay
  positive in practice); the framework emits the readout one transition earlier than
  the reference loop (teacher and student share the convention); student training
  targets are the noiseless teacher readout/activity (the reference uses a single
  noisy realization); evaluation is noise-free.

## 7. Legacy: the removed connectome_rnn class

This paradigm was first ported as a dedicated `models/connectome_rnn` family with a
`fixed_recurrent_weight` config field and `trainable_*` flags. It was generalized by
`models/gain_rnn` (`GainRNNModel`): `nonlinearity_mode="rate"` +
`gain_position="outside"` + `activation="softplus"` reproduced its trajectories exactly
(cross-model parity was unit-tested at `atol=1e-6` before removal), while adding
structural masks, the unified freeze vocabulary, all three nonlinearity modes, and
configurable noise placement. The legacy class has been **removed**; notebook 16 is the
maintained reproduction. See `docs/papers/gain_rnn.md`.
