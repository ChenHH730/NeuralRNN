# E-I RNN: Training Excitatory-Inhibitory Recurrent Neural Networks for Cognitive Tasks

> Song, H.F., Yang, G.R. and Wang, X.J., 2016.
> Training excitatory-inhibitory recurrent neural networks for cognitive tasks:
> a simple and flexible framework.
> PLoS computational biology, 12(2).

---

## 1. Core question and motivation (Why E-I RNN?)

### 1.1 Biological flaws of traditional RNNs

The core idea of using RNNs to model cognitive tasks is to train an RNN on the same task and then analyze how the trained network solves it, thereby inferring the computational mechanisms the brain might use. However, traditional RNNs have several serious biological flaws:

| Flaw | Biological fact | Consequence |
|---|---|---|
| Firing rates can be positive or negative | Real neuronal firing rates ≥ 0 | Network dynamics do not correspond to real circuits |
| No excitatory/inhibitory neuron distinction | Dale's principle: each neuron releases only excitatory or inhibitory neurotransmitter | Cannot produce key properties such as non-normal dynamics |
| All connection patterns are identical | E→E, E→I, I→E, I→I connections differ in sparsity and specificity | Cannot model the real structure of local circuits |
| Readout can come from all units | Long-range projections are mainly excitatory | Inconsistent with cortical architecture |

**The most critical issue is Dale's principle**: in mammalian cortex, neurons are either purely excitatory or purely inhibitory. This constraint has a profound impact on network dynamics—for example, it can produce **non-normal dynamics**, a key mechanism for selective amplification of neural activity patterns (Murphy & Miller, 2009).

### 1.2 Same task, multiple solutions

Another key issue is that **training can produce multiple networks with identical behavior but very different structure and dynamics**. The choice of constraints and regularization determines which solution the training algorithm finds. Therefore, the question is no longer "can an RNN solve the task?" (the answer is almost always "yes"), but rather "what architecture and constraints produce network activity most similar to real neural recordings?".

### 1.3 Contribution of this paper

This paper proposes a **flexible, gradient-descent-based E-I RNN training framework** that can:
- Hard-code biological knowledge such as Dale's principle into the network structure
- Guide training toward biologically plausible solutions by constraining connection patterns (sparsity, E/I ratio, long-range projection rules)
- Train and analyze networks on multiple cognitive tasks with a unified method

---

## 2. Core methods

### 2.1 Network dynamics

Continuous-time equations of the E-I RNN:

$$
\tau \dot{\mathbf{x}} = -\mathbf{x} + W^{\text{rec}} \mathbf{r} + W^{\text{in}} \mathbf{u} + \sqrt{2\tau \sigma_{\text{rec}}^2} \xi
$$

$$
\mathbf{r} = [\mathbf{x}]_+ = \max(\mathbf{x}, 0)
$$

$$
\mathbf{z} = W^{\text{out}} \mathbf{r}
$$

Key points:
- **Threshold-linear (ReLU) activation**: firing rates are strictly non-negative, consistent with biology
- **Continuous time**: implemented by Euler discretization ($\alpha = \Delta t / \tau$)
- **Noise**: shared input noise $\sigma_{\text{in}}$ and private recurrent noise $\sigma_{\text{rec}}$

### 2.2 Implementing Dale's principle

Units are divided into excitatory (E) and inhibitory (I) populations:

- **E units**: all outgoing weights ≥ 0
- **I units**: all outgoing weights ≤ 0
- **E:I ratio**: usually 4:1 (80% E, 20% I)
- **Readout**: only from E units (long-range projections are purely excitatory)

Weight matrix parameterization: $W^{\text{rec}} = W^{\text{rec},+} \cdot D$, where $W^{\text{rec},+}$ is non-negative and $D$ is a diagonal sign matrix (+1 for E, -1 for I).

### 2.3 Connection pattern constraints

In addition to Dale's principle, connection patterns can be constrained:
- **No self-connections**: diagonal is zero
- **Sparse connections**: via hard constraints (mask) or L1 regularization
- **Inter-area connections**: local inhibition, long-range excitation
- **Input specialization**: different inputs project to different E-unit groups

### 2.4 Training method

Modified SGD (Pascanu et al., 2013):
1. **Gradient clipping**: prevents gradient explosion (maximum norm $G$)
2. **Gradient regularization**: prevents vanishing gradients ($\lambda_\Omega$)
3. **Objective**: supervised learning (cross-entropy / MSE) + regularization terms

---

## 3. Analysis methods in detail (Tutorial contents)

### 3.1 Selectivity analysis ($d'$ analysis)

**Motivation**: in experiments, neuroscientists record single neurons under different conditions to determine whether a neuron is "tuned" to a task variable. $d'$ (selectivity index) is the standard metric for measuring such tuning.

**Definition**:

$$
d' = \frac{\mu_1 - \mu_2}{\sqrt{(\sigma_1^2 + \sigma_2^2)/2}}
$$

where $\mu_1, \sigma_1^2$ are the mean and variance of a unit's stimulus-period activity on choice-1 trials, and $\mu_2, \sigma_2^2$ likewise.

**Interpretation**:
- $d' > 0$: unit is more active on choice-1 trials → "selective to choice 1"
- $d' < 0$: unit is more active on choice-2 trials → "selective to choice 2"
- $|d'|$ larger: stronger selectivity

**Why it matters**:
- Reveals which units in the trained network encode task-relevant information
- Provides a sorting basis for subsequent connectivity visualization
- Consistent with single-neuron analysis methods in experiments, facilitating comparison with real data

**Tutorial correspondence**: Section 5 — compute $d'$ for each neuron and plot distributions for E and I units separately.

### 3.2 Connectivity visualization

**Motivation**: the trained network's connection matrix contains structural information about "how the network computes". Sorting units by selectivity can reveal clustering structures that emerge during training.

**Method**:
1. Compute effective recurrent weights $W^{\text{rec}} = |W| \cdot D$ (actual weights under Dale constraint)
2. Sort units by $d'$ (from most selective to choice 1 to most selective to choice 2)
3. Visualize the sorted weight matrix as a heatmap

**Key findings** (paper Fig 3):
- **Without Dale constraint**: units with similar $d'$ have strong excitatory connections; units with different $d'$ have strong inhibitory connections
- **With Dale constraint**: units with different $d'$ interact indirectly through **I units**; E→E and I→I connections naturally differentiate after training
- **With structural constraints**: training further strengthens preset connection patterns

**Why it matters**:
- Directly displays the network's "wiring diagram"
- Reveals how Dale constraints affect the network's computational strategy
- Can be directly compared with connectivity maps drawn by optogenetics or electrophysiology in experiments

**Tutorial correspondence**: Section 6 — plot heatmap of $d'$-sorted weight matrix, mark E/I boundary.

### 3.3 PCA dimensionality reduction

**Motivation**: high-dimensional neural activity (e.g. 50 units) is hard to visualize directly. PCA projects activity onto the low-dimensional plane of maximum variance, revealing the essential structure of population dynamics.

**Method**:
1. Collect neural activity across all trials and time steps
2. Fit PCA and take the first 2-3 principal components
3. Project each trial's trajectory into PC space

**Key findings**:
- In perceptual decision tasks, trajectories of different coherence form a **fan-like expansion** pattern in PC space
- Choice-1 and choice-2 trajectories diverge in different directions
- Zero-coherence trajectories lie in the middle, reflecting random decisions

**Why it matters**:
- Turns unvisualizable high-dimensional dynamics into understandable geometric structure
- Reveals the "neural state-space" trajectories the network uses to make decisions
- Fully consistent with dimensionality-reduction analysis of population neural activity in experiments

**Tutorial correspondence**: Section 7 — fit PCA, plot trajectories colored by coherence and ground truth.

### 3.4 Fixed-point analysis

**Motivation**: fixed points ($F(z^*) = z^*$) are key to understanding dynamical-system behavior. Stable fixed points correspond to attractors—the network state is "attracted" to these points. In decision tasks, fixed points correspond to the network's "choice" states.

**Method**:
1. Under 0-coherence stimulus condition ($[1, 0.5, 0.5]$), start from multiple random initial values
2. Use gradient descent to minimize $\|F(z) - z\|^2$
3. After convergence, check whether speed $\|F(z) - z\|$ is below threshold
4. Project found fixed points into PCA space

**Key findings**:
- Perceptual decision networks contain an **approximately stable fixed-point chain** (line attractor)
- This chain extends along the PC1 direction; different positions correspond to different "evidence accumulation" states
- Network trajectories slide along this chain until converging to endpoints (choice 1 or choice 2)

**Why it matters**:
- Reveals the **dynamical mechanism** by which the network solves the task: decision = sliding along attractor
- Distribution and stability of fixed points explain the network's speed-accuracy tradeoff
- Consistent with dynamical-systems theory analysis of neural data in experiments

**Tutorial correspondence**: Section 8 — numerical fixed-point search + PCA-space visualization.

### 3.5 Jacobian eigenvalue analysis (linearization)

**Motivation**: fixed points only tell us where the network "stops", but Jacobian eigenvalues tell us how the network "behaves near the fixed point"—whether it is attracted (stable) or repelled (unstable), and along which directions.

**Method**:
1. Compute Jacobian $J = \partial F / \partial z$ at fixed point $z^*$
2. Find eigenvalues $\lambda_i$ and eigenvectors
3. Eigenvalues inside unit circle → stable direction; outside unit circle → unstable direction
4. Dominant eigenvectors reveal the main dynamical directions near the fixed point

**Key findings**:
- Points on the fixed-point chain have **one real eigenvalue close to 1** (corresponding to line-attractor direction)
- Other eigenvalues are inside unit circle (stable directions)
- Dominant eigenvector is along PC1, aligned with the fixed-point chain

**Why it matters**:
- Distinguishes "true attractors" from "spurious fixed points"
- Dominant eigenvector reveals the network's **computational direction**—the direction of evidence accumulation
- Eigenvalue spectrum fully characterizes dynamics near the fixed point

**Tutorial correspondence**: Section 9 — distribution of Jacobian eigenvalues in the complex plane + visualization of dominant directions.

---

## 4. Other important analysis methods in the paper

### 4.1 Psychometric function

**Definition**: percentage of choice-1 responses as a function of signed coherence.

**Significance**: directly compared with psychometric curves from monkey experiments to verify that the network's behavior matches the animal. The S-shaped curve reflects the effect of noise on decisions.

### 4.2 Reaction time analysis

**Definition**: time for output to reach threshold as a function of coherence.

**Significance**: reaction time is a core behavioral metric in decision tasks. The network should exhibit a speed-accuracy tradeoff consistent with animals.

### 4.3 Mixed selectivity

**Definition**: single units are simultaneously selective to multiple task variables (e.g. choice, motion, color, context).

**Significance**: mixed selectivity is a hallmark of prefrontal cortex, enabling flexible computation in high-dimensional state space.

### 4.4 State-space regression analysis

**Method**: use linear regression to project population activity onto task-variable axes (choice, motion, color, context).

**Significance**: reveals how different task variables are encoded in population activity and how they interact.

---

## 5. Advantages of E-I RNN over other RNNs

| Dimension | Ordinary RNN | E-I RNN |
|---|---|---|
| Biological plausibility | ❌ Firing rates can be negative, violates Dale's principle | ✅ Firing rates non-negative, satisfies Dale's principle |
| Dynamics type | Symmetric weights → normal dynamics | Asymmetric weights (E/I asymmetry) → non-normal dynamics |
| Selective amplification | No natural mechanism | E/I balance produces selective amplification (Murphy & Miller, 2009) |
| Connection structure | Fully symmetric, no biological meaning | E→E, E→I, I→E, I→I can be constrained separately |
| Comparison with experiments | Requires post-processing to compare with real data | Directly comparable with real data (firing rates, connectivity, selectivity) |
| Long-range projections | All units can be read out | Only E units read out, consistent with cortical architecture |
| Training constraints | No biological constraints | Can incorporate multiple biological knowledge (sparsity, layer structure, inter-area connections) |

**Core advantage**: E-I RNN not only solves the task, but also produces solutions whose **structure and dynamics** are similar to real neural circuits. This makes the trained E-I RNN a source of "computational hypotheses"—analyzing how the network computes can help infer how the brain might compute.

---

## 6. Original code map

| Component | File/Function | Description |
|---|---|---|
| `PosWLinear` | `EI_RNN.ipynb` | Non-negative weight linear layer (Dale constraint) |
| `EIRecLinear` | `EI_RNN.ipynb` | Recurrent layer with Dale mask |
| `EIRNN` | `EI_RNN.ipynb` | E-I RNN cell (state + output) |
| `Net` | `EI_RNN.ipynb` | Full network (EIRNN + E-unit readout) |
| Training loop | `EI_RNN.ipynb` | Adam + CrossEntropy |
| $d'$ computation | `EI_RNN.ipynb` | Selectivity index |
| Connectivity visualization | `EI_RNN.ipynb` | Weight matrix heatmap |

---

## 7. Mapping to NeuralRNN API

### Model

```python
from neuralrnn import AutoConfig, AutoModel

cfg = AutoConfig.for_model('ei_rnn',
                           input_dim=3, latent_dim=50, output_dim=3,
                           dt=20, sigma_rec=0.15, nonlinearity_mode="post_blend")
model = AutoModel.from_config(cfg)
# model.e_size = 40, model.i_size = 10
# model._recurrent_weight() → effective weights under Dale constraint
```

### Analysis

```python
from neuralrnn.analysis import fit_pca, find_fixed_points, linearize, dominant_direction

# Selectivity analysis ($d'$) — computed directly with numpy
d_prime = (mean_0 - mean_1) / np.sqrt((std_0**2 + std_1**2) / 2)

# PCA
pca = fit_pca(activity_all, n_components=2)

# Fixed points
fps = find_fixed_points(model, backend='numeric',
                        task_input=torch.tensor([1, 0.5, 0.5]),
                        n_candidates=128, n_iters=10000)

# Jacobian
lin = linearize(model, fps.points[0].z, task_input=task_input)
d = dominant_direction(lin)
```

---

## 8. Reproduction notebook

`notebook/04_EIRNN_paradigmA.ipynb` reproduces the paper's core analysis pipeline:

| Section | Content | Corresponding paper |
|---|---|---|
| 2 | E-I RNN training | Methods: Training |
| 4 | Unit activity visualization | Fig 2G/H |
| 5 | $d'$ selectivity analysis | Eq 30, Fig 3 |
| 6 | Connectivity visualization | Fig 3 |
| 7 | PCA dimensionality reduction | Fig 4B |
| 8 | Fixed-point analysis | Fig 7 (line attractor) |
| 9 | Jacobian eigenvalues | Stability analysis |

---

## 9. References

1. Original paper: https://doi.org/10.1371/journal.pcbi.1004792
2. Code: https://github.com/xjwanglab/pycog
3. Reference implementation: `reference_project/Neural_network_for_brain_2020/EI_RNN.ipynb`
4. Murphy, B.K. & Miller, K.D. (2009). Balanced amplification: A new mechanism of selective amplification of neural activity patterns. Neuron.
5. Pascanu, R. et al. (2013). On the difficulty of training recurrent neural networks. ICML.
