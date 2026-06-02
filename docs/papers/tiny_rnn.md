# Discovering Cognitive Strategies with Tiny Recurrent Neural Networks

> **Paradigm**: B (Behavioral Fitting)
> **Original repo**: https://github.com/lj-ana/tinyRNN
> **Paper**: https://doi.org/10.1038/s41586-025-09142-4
> **Framework location**: `models/tiny_rnn` + `BehavioralObjective` + `train/cv.py`
> **Notebook**: `notebook/06_tiny_RNN_paradigmB.ipynb`
> **Status**: ✅ Ready

## 1. Problem Statement

Understanding how animals and humans learn from experience to make adaptive decisions is a fundamental goal of neuroscience. Normative frameworks like Bayesian inference and reinforcement learning provide valuable insights but are limited by their simplicity and researcher subjectivity in handcrafting extensions. Larger neural networks offer flexibility but lack interpretability.

**Core idea**: Fit very small RNNs (1-4 GRU units) to individual subjects' behavioral data in reward-learning tasks. These tiny RNNs outperform classical cognitive models in predicting choices while remaining interpretable via dynamical systems analysis.

## 2. Tasks and Datasets

Six reward-learning tasks across 8 datasets:

### Animal Tasks
- **Reversal learning** (Bartolo monkeys, Akam mice): Two actions, one has high reward probability (0.7), contingencies reverse mid-block
- **Two-stage** (Miller rats, Akam mice): Probabilistic transitions between stages
- **Transition-reversal two-stage** (Akam mice): Stochastic reversals in action-state transitions

### Human Tasks
- **3-armed reversal learning** (Suthaharan): Three actions
- **4-armed drifting bandit** (Bahrami): Four actions, continuous rewards
- **Original two-stage** (Gillan): Six actions, three choice states

## 3. Core Method

### 3.1 Architecture

The RNN takes as input the previous trial's observation `[action_{t-1}, stage2_{t-1}, reward_{t-1}]` and outputs logits over actions for the current trial.

**Vanilla GRU** (Equation 1):
```
r_t = σ(W_ir x_t + b_ir + W_hr h_{t-1} + b_hr)
z_t = σ(W_iz x_t + b_iz + W_hz h_{t-1} + b_hz)
n_t = tanh(W_in x_t + b_in + r_t ⊙ (W_hn h_{t-1} + b_hn))
h_t = (1 - z_t) ⊙ n_t + z_t ⊙ h_{t-1}
```

**Switching GRU** (Equation 2, SGRU): Input-dependent recurrent weights — different W_hh for each discrete input combination. Because inputs are discrete (one-hot encoded), the recurrent weights are selected based on the input:
```
W_hh(x) = Σ_k x_k W_hh^(k)
```
where x_k is the k-th component of the one-hot input encoding. For d=1, SGRU consistently outprobes vanilla GRU.

**Readout**: Either fully-connected (`nn.Linear`) or diagonal (`α * h_t`).

### 3.2 Input-Output Structure

Cognitive models and RNNs share the same input-output structure:
- **Inputs**: Previous action a_{t-1}, second-step state s_{t-1}, reward r_{t-1}
- **Internal state**: Dynamical variables summarizing past experience (action values, beliefs, etc.)
- **Output**: Policy — probability of each action

The key difference: classical models use handcrafted update rules, while RNNs learn update rules from data via adjustable weights.

### 3.3 Behavioral Dimensionality

The **dimensionality of behavior** d* is defined as the minimal number of dynamical variables needed to optimize predictability. Results:
- Reversal learning: d* = 1-2
- Two-stage: d* = 1-2
- Transition-reversal: d* = 1-4
- Human tasks: d* = 5-20

## 4. Training

### 4.1 Loss Function

Cross-entropy loss (negative log-likelihood) on action predictions:
```
L = -Σ_t log p_θ(a_t | a_{1:t-1}, s_{1:t-1}, r_{1:t-1})
```

### 4.2 Regularization

- **L1 regularization** on recurrent weights (weight_hh): `L_total = L + λ ||W_hh||_1`
  - Typical λ = 1e-5
  - Only on recurrent weights, not input/output weights
- **Early stopping**: patience = 200 epochs on validation loss

### 4.3 Optimization

- **Optimizer**: Adam, lr = 0.005
- **Gradient clipping**: max norm = 1.0
- **Max epochs**: 2000

### 4.4 Cross-Validation (Original Paper)

The original paper uses **nested cross-validation**:
- **Outer folds**: 5 (train+val vs test)
- **Inner folds**: 4 (train vs val)
- **Seeds**: 2 per fold
- **Total**: 5 × 4 × 2 = 40 models per configuration

This separates training, validation, and evaluation to prevent overfitting given the large parameter count difference between RNNs (40-80 params for 1-2 units) and cognitive models (2-10 params).

**Note**: Our notebook simplifies to a single train/test split (80/20) for demonstration.

### 4.5 Knowledge Distillation (Human Tasks)

For human studies with limited per-participant data (~200 trials):
1. Train a large "teacher" RNN on all participants
2. Train per-participant "student" RNNs to match teacher's probabilistic policy
3. Students learn from teacher's soft predictions (not binary choices)

This enables tiny RNNs to outperform cognitive models with as few as 350 trials per subject.

## 5. Cognitive Models

### MB0 — Model-Based (no decay), d=2

Two state values Q_s[0], Q_s[1] (one per action):
```
Q_s[chosen] += α × (reward - Q_s[chosen])
```
Parameters: α (learning rate), iTemp (inverse temperature)

### MB1 — Model-Based (with decay), d=2

Same as MB0, but unchosen value decays:
```
Q_s[chosen] += α × (reward - Q_s[chosen])
Q_s[unchosen] *= β
```
Parameters: α, β (decay rate), iTemp

### LS0 — Latent State Bayesian, d=1

Posterior probability p1 that action 1 is the high-reward option:
```
p_o_1[s,o] = P(outcome=o | state=s, world in state 1)
p1_new = p_o_1[s,o] × p1 / (p_o_1[s,o] × p1 + p_o_0[s,o] × (1-p1))
p1 = (1-p_r) × p1_new + p_r × (1-p1_new)
```
Parameters: p_r (reversal probability), iTemp

This is a **Bayesian ideal observer** — optimal inference given task structure.

### Q0 — Model-Free Q-Learning (TD(0)), d=4

First-stage values Q_f and second-stage values Q_s:
```
Q_f[chosen] += α × (Q_s[stage2] - Q_f[chosen])
Q_s[stage2] += α × (reward - Q_s[stage2])
```
Parameters: α, iTemp

## 6. Analysis Methods

### 6.1 Phase Portraits (1D models)

For 1D models, plot **logit** vs **logit change**:
- Logit L(t) = log P(action=1) / P(action=0) — current preference
- Logit change ΔL(t) = L(t+1) - L(t) — how preference changes

Each trial is a point colored by trial type (4 types: action × reward). Key features:
- **Fixed points** where trajectories converge (attractors)
- **Curved trajectories** indicating state-dependent learning rates
- **Separation of curves** indicating state-dependent perseveration

### 6.2 Vector Fields (2D models)

For 2D models, evaluate on a grid of initial hidden states:
- Each arrow shows the state change (h_{t+1} - h_t) at that point
- Separate plots for each trial type
- Overlaid readout vector and decision boundary

### 6.3 Dynamical Regression (d>2)

For higher-dimensional models, fit linear regression:
```
Δh(t) = b + Σ_i w_i h_i(t) + ε
```
separately for each trial type. Regression coefficients summarize:
- **Intercept**: asymptotic preference (fixed point)
- **Slopes**: how each state dimension affects dynamics

### 6.4 Novel Cognitive Signatures Discovered

1. **State-dependent learning rates**: Curved phase portraits (not straight lines)
2. **State-dependent perseveration**: Decoupling of unrewarded curves at extreme logits
3. **Reward-dependent bias**: Asymmetric fixed points
4. **Reward-induced indifference**: Rewards after rare transitions lead to indifference
5. **Drift-to-the-other forgetting**: Unrewarded action values drift toward the other action, not zero

## 7. Framework Mapping

| Original | NeuralRNN |
|----------|-----------|
| `agents/network_models.py` RNNnet | `models/tiny_rnn/modeling_tiny_rnn.py` TinyRNNModel |
| `agents/RNNAgent.py` | `NeuralDynamicsModel` (base class) |
| `agents/RNNAgentTrainer.py` | `train/trainer.py` Trainer |
| `datasets/BartoloMonkeyDataset.py` | `data/bartolo_monkey_dataset.py` |
| `Dataset().behav_to('tensor')` | `BartoloMonkeyDataset._build_tensors()` |
| `behavior_cv_training_config_combination` | `train/cv.py` |
| Cross-entropy + L1 loss | `train/objectives/behavioral.py` BehavioralObjective |
| `plot_all_models_value_change` | Notebook inline + `analysis/` |

## 8. Key Results

1. **1-4 unit RNNs outperform all classical models** of equal dimensionality across all datasets
2. **Behavioral dimensionality is low**: d* = 1-2 for simple tasks, d* = 5-20 for human tasks
3. **Knowledge distillation** enables tiny RNNs with limited per-subject data (350 trials)
4. **RNNs form a superset of classical models**: can recover ground-truth model dynamics when trained on simulated data
5. **Novel cognitive signatures** discovered via phase portrait analysis

## 9. References

- Ji-An, L., Benna, M.K. & Mattar, M.G. (2025). Discovering cognitive strategies with tiny recurrent neural networks. *Nature*. https://doi.org/10.1038/s41586-025-09142-4
- Bartolo, R., et al. (2020). Medial prefrontal cortex controls the expression of reward-seeking behaviour via dorsolateral striatum.
- Akam, T., et al. (2015). Dissecting the contributions of distinct dorsal prefrontal cortex subregions to dynamic decision-making.
