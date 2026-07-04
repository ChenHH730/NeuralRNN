# NeuralRNN API Reference

> **Version**: 0.1.0.dev0
>
> This document is the complete API reference for the NeuralRNN framework. It covers every public module, class, and function. For conceptual overviews see [ARCHITECTURE.md](../ARCHITECTURE.md); for step-by-step tutorials see the [notebooks](../../notebook/).

---

## Table of Contents

1. [Top-Level Exports](#1-top-level-exports)
2. [Configuration System](#2-configuration-system)
3. [Model Base Classes](#3-model-base-classes)
4. [Auto Factory](#4-auto-factory)
5. [Built-In Models](#5-built-in-models)
   - [CTRNN Family (Paradigm A)](#ctrnn-family-paradigm-a)
   - [PLRNN Family (Paradigm B)](#plrnn-family-paradigm-b)
   - [Lowrank RNN Model (Paradigm A/B)](#lowrank-rnn-model-paradigm-ab)
   - [Latent Circuit Model (Paradigm A)](#latent-circuit-model-paradigm-a)
   - [Tiny RNN Model (Paradigm B)](#tiny-rnn-model-paradigm-b--behavioral-fitting)
6. [Data Layer](#6-data-layer)
   - [BaseDataset & StandardScaler](#basedataset--standardscaler)
   - [TimeSeriesDataset](#timeseriesdataset)
   - [NeurogymDataset](#neurogymdataset)
   - [CustomDataset](#customdataset)
   - [Registry & load_dataset](#registry--load_dataset)
   - [Download Utilities](#download-utilities)
7. [Training Layer](#7-training-layer)
   - [TrainingArguments](#trainingarguments)
   - [Trainer](#trainer)
   - [Objectives](#objectives)
   - [Cross-Validation](#cross-validation)
8. [Analysis Layer](#8-analysis-layer)
   - [Fixed Points](#fixed-points)
   - [Linearization](#linearization)
   - [Vector Field](#vector-field)
   - [Dimensionality Reduction](#dimensionality-reduction)
   - [Lyapunov Exponent](#lyapunov-exponent)
   - [DSR Metrics (D_stsp, D_H)](#dsr-metrics-d_stsp-d_h)
   - [Manifold Analysis](#manifold-analysis)
9. [Visualization](#9-visualization)

---

## 1. Top-Level Exports

All commonly used symbols are re-exported from `neuralrnn` for convenience:

```python
from neuralrnn import (
    # Core
    NeuralRNNConfig, NeuralDynamicsModel, DynamicsModelOutput,
    # Auto factories
    AutoConfig, AutoModel, register_config, register_model,
    # Data
    BaseDataset, StandardScaler, TimeSeriesDataset,
    DATASET_REGISTRY, DatasetSpec, load_dataset,
    # Training
    Trainer, TrainingArguments,
    Objective, SupervisedObjective, TeacherForcingObjective,
    BehavioralObjective, VariationalObjective,
)
```

---

## 2. Configuration System

### `NeuralRNNConfig`

**Module**: `neuralrnn.configuration_utils`

Base configuration class for all models (analogous to HuggingFace `PretrainedConfig`). Every model family subclasses this and adds its own fields.

```python
class NeuralRNNConfig:
    model_type: str = ""            # Unique registry key, e.g. "shallow_plrnn"

    def __init__(
        self,
        input_dim: int = 0,         # External input dimension K (0 = no input)
        latent_dim: int = 0,        # Latent state dimension M
        output_dim: int = 0,        # Readout dimension
        dt: float | None = None,    # Euler step for continuous-time models
        activation: str = "relu",   # Nonlinearity name
        freeze_input: bool = False,      # Freeze input-layer parameters
        freeze_recurrent: bool = False,  # Freeze recurrent / hidden parameters
        freeze_output: bool = False,     # Freeze output / readout parameters
        freeze_h0: bool = False,         # Freeze initial-state parameters
        **kwargs: Any,              # Forward-compat: unknown keys stored as attrs
    ) -> None: ...
```

The four `freeze_*` flags provide a convenient way to implement echo-state / reservoir-computing training. They are automatically serialized with the config and respected by all built-in models.

**Serialization Methods**:

| Method | Description |
|--------|-------------|
| `to_dict() -> dict` | Serialize all fields to a plain dict |
| `to_json_string() -> str` | JSON string with indentation |
| `to_json_file(save_directory: str) -> str` | Write `config.json` into `save_directory`; returns the path |
| `from_dict(d: dict) -> NeuralRNNConfig` | Class method: reconstruct from dict |
| `from_json_file(json_file: str) -> NeuralRNNConfig` | Class method: read from a `.json` file |
| `from_pretrained(path: str) -> NeuralRNNConfig` | Class method: read from a directory containing `config.json` |

**Usage**:

```python
from neuralrnn import NeuralRNNConfig

# Direct instantiation (rare — prefer AutoConfig)
cfg = NeuralRNNConfig(input_dim=3, latent_dim=64, output_dim=3, dt=100.0)

# Save / load
cfg.to_json_file("my_config/")
cfg2 = NeuralRNNConfig.from_pretrained("my_config/")
```

> **Porting note**: Every model family must define a `<Family>Config(NeuralRNNConfig)` subclass with a unique `model_type`. All constructor hyperparameters from the original paper code must be stored as config fields — never hardcoded in the model `__init__`.

---

## 3. Model Base Classes

### `NeuralDynamicsModel`

**Module**: `neuralrnn.modeling_utils`

The universal base class for all RNN / dynamical systems models. Analogous to HuggingFace `PreTrainedModel`.

**Core abstraction** (the "hard contract"): every subclass must implement two methods that define the discrete dynamical system `z_t = F(z_{t-1}, x_t), y_t = G(z_t)`:

```python
class NeuralDynamicsModel(nn.Module):
    config_class: type[NeuralRNNConfig] = NeuralRNNConfig

    def __init__(self, config: NeuralRNNConfig) -> None: ...

    # ======== Hard contract (must override) ========
    def recurrence(self, x_t: Tensor | None, z_prev: Tensor,
                   *, inputs: Tensor | None = None) -> Tensor:
        """One-step transition F_θ.
        Args:
            x_t:    (B, input_dim) current input, or None for autonomous systems
            z_prev: (B, M) previous latent state
            inputs: full input sequence (unused by most models; for models that need context)
        Returns:
            z_t: (B, M) next latent state
        """
        raise NotImplementedError

    def readout(self, z_t: Tensor) -> Tensor:
        """Readout G_φ: latent state -> observation/output.
        Args:
            z_t: (B, M)
        Returns:
            y_t: (B, output_dim). For DSR with identity observation, return z_t itself.
        """
        raise NotImplementedError

    # ======== Default implementations (override as needed) ========
    def init_state(self, batch_size: int, device: torch.device | str = "cpu") -> Tensor:
        """Initial latent state z_0. Default: zeros. Override for trainable/encoder init."""

    def forward(self, inputs: Tensor | None = None, *,
                initial_state: Tensor | None = None,
                n_steps: int | None = None,
                return_states: bool = True) -> DynamicsModelOutput:
        """Full rollout: loops recurrence + readout over T steps.
        Args:
            inputs: (B, T, input_dim) or None for autonomous rollout
            initial_state: (B, M) or None (uses init_state)
            n_steps: required when inputs is None
            return_states: whether to store latent trajectories
        Returns:
            DynamicsModelOutput
        """

    @torch.no_grad()
    def generate(self, initial_state: Tensor, n_steps: int,
                 inputs: Tensor | None = None) -> Tensor:
        """Free rollout (no teacher forcing). Returns (B, T+1, M) latent trajectory.
        Used for evaluation and analysis."""

    # ======== Analysis support (analytic models override) ========
    @property
    def supports_analytic_fixed_points(self) -> bool:
        """True if the model provides analytic fixed-point solving."""

    def jacobian(self, z: Tensor, *, inputs: Tensor | None = None) -> Tensor:
        """∂F/∂z at state z. Default: autodiff fallback. Analytic models should override.
        Args:
            z: (M,) single state
        Returns:
            J: (M, M) Jacobian matrix
        """

    def analytic_parameters(self) -> dict[str, Tensor]:
        """Expose parameters needed by the analytic fixed-point solver (scy_fi).
        Only required when supports_analytic_fixed_points is True."""

    # ======== Parameter freezing (ESN / reservoir computing) ========
    def freeze_parameters(
        self,
        groups: str | list[str] | None = None,
        patterns: list[str] | None = None,
    ) -> list[str]:
        """Freeze parameters by generic group name(s) and/or regex patterns.
        Common groups: 'input', 'recurrent', 'output', 'h0'."""

    def unfreeze_parameters(
        self,
        groups: str | list[str] | None = None,
        patterns: list[str] | None = None,
    ) -> list[str]:
        """Unfreeze parameters previously frozen via freeze_parameters."""

    def apply_freeze_config(self) -> list[str]:
        """Apply freeze flags stored in self.config. Called automatically
        by built-in models at the end of __init__."""

    # ======== Persistence ========
    def save_pretrained(self, save_directory: str, metadata: dict | None = None) -> None:
        """Write config.json + model.safetensors (+ metadata.json)."""

    @classmethod
    def from_pretrained(cls, path: str, *, map_location: str = "cpu") -> "NeuralDynamicsModel":
        """Restore from a directory. For cross-family loading, use AutoModel.from_pretrained."""

    def num_parameters(self) -> int:
        """Total parameter count."""
```

### `DynamicsModelOutput`

**Module**: `neuralrnn.modeling_utils`

Unified output container returned by `forward()` and `generate()`.

```python
@dataclass
class DynamicsModelOutput:
    outputs: Tensor | None = None    # Readout y_{1:T}, shape (B, T, output_dim)
    states:  Tensor | None = None    # Latent trajectory z_{1:T}, shape (B, T, latent_dim)
    loss:    Tensor | None = None    # Loss computed inside forward (if any)
    extras:  dict[str, Any] | None = None  # Model-specific outputs (e.g. LFADS posteriors)
```

Supports both attribute access (`out.states`) and dict-style access (`out["states"]`).

### Tensor Shape Convention (global)

| Tensor | Shape | Notes |
|--------|-------|-------|
| Input `inputs` | `(B, T, input_dim)` | batch-first; `None` if no input |
| Latent `states` | `(B, T, latent_dim)` | `latent_dim == M` |
| Output `outputs` | `(B, T, output_dim)` | classification: `output_dim = n_classes` |
| Single step `z_t` | `(B, latent_dim)` | recurrence input/output |

> Original paper code often uses time-first. The adapter must `permute` to batch-first at the boundary.

---

## 4. Auto Factory

### `AutoConfig`

**Module**: `neuralrnn.auto.configuration_auto`

```python
class AutoConfig:
    @staticmethod
    def for_model(model_type: str, **kwargs) -> NeuralRNNConfig:
        """Create a config by model_type with optional overrides.
        Example: AutoConfig.for_model("shallow_plrnn", latent_dim=3, hidden_dim=50)
        """

    @staticmethod
    def from_pretrained(path: str) -> NeuralRNNConfig:
        """Load config.json from a directory, dispatching to the correct subclass by model_type."""
```

### `AutoModel`

**Module**: `neuralrnn.auto.modeling_auto`

```python
class AutoModel:
    @staticmethod
    def from_config(config) -> NeuralDynamicsModel:
        """Instantiate a model from a config object."""

    @staticmethod
    def from_pretrained(path: str, *, map_location: str = "cpu") -> NeuralDynamicsModel:
        """Load config.json + model.safetensors from a directory."""
```

### Registration Decorators

```python
from neuralrnn import register_model, register_config

# In your modeling file:
@register_model("my_family")
class MyModel(NeuralDynamicsModel):
    config_class = MyConfig
    ...

# Config is auto-registered when the model is registered.
# Manual config registration (rare):
@register_config("my_family")
class MyConfig(NeuralRNNConfig):
    model_type = "my_family"
    ...
```

### Registries

```python
from neuralrnn import MODEL_REGISTRY, CONFIG_REGISTRY

# MODEL_REGISTRY: dict[str, type[NeuralDynamicsModel]]
# CONFIG_REGISTRY: dict[str, type[NeuralRNNConfig]]
```

Lazy loading: model modules are imported on first access via `_LAZY_MODULES` mapping in `auto/modeling_auto.py`. New models need one line added there.

---

## 5. Built-In Models

### CTRNN Family (Paradigm A)

**Module**: `neuralrnn.models.ctrnn`

Continuous-time RNN with Euler discretization. Transplant of nn-brain.

#### `CTRNNConfig`

```python
class CTRNNConfig(NeuralRNNConfig):
    model_type = "ctrnn"

    def __init__(
        self,
        input_dim: int = 3,
        latent_dim: int = 64,       # Number of hidden units M
        output_dim: int = 3,        # Number of output classes
        dt: float | None = 100.0,   # Euler step (None = discrete vanilla)
        tau: float = 100.0,         # Time constant
        activation: str = "relu",   # "relu" / "tanh" / "softplus"
        dale: bool = False,         # Dale constraint (E/I separation)
        ei_ratio: float = 0.8,      # Excitatory fraction (when dale=True)
        trainable_h0: bool = False, # Trainable initial state
        sigma_rec: float = 0.0,     # Recurrent noise std (0 = off)
        **kwargs,
    ) -> None: ...
```

#### `VanillaRNNConfig`

```python
class VanillaRNNConfig(CTRNNConfig):
    model_type = "vanilla_rnn"
    # Defaults: dt=None (discrete), all other params inherited
```

#### `EIRNNConfig`

```python
class EIRNNConfig(CTRNNConfig):
    model_type = "ei_rnn"

    def __init__(
        self,
        readout_e_only: bool = True,  # Readout from excitatory units only
        init_method: str = "kaiming", # Weight initialization method
        # All CTRNNConfig params inherited, with defaults:
        # dale=True, ei_ratio=0.8
        **kwargs,
    ) -> None: ...
```

- `readout_e_only`: When `True`, readout layer takes only excitatory units (first `e_size` units) as input. This matches the biological constraint that long-range projections are exclusively excitatory.
- `init_method`: Weight initialization method. `"kaiming"` uses Kaiming uniform with E/I balance scaling.

#### `CTRNNModel`

```python
@register_model("ctrnn")
class CTRNNModel(NeuralDynamicsModel):
    config_class = CTRNNConfig

    # Recurrence: h = (1-alpha)*h + alpha * f(W_x*x + W_r*r + b)
    # where alpha = dt/tau
    def recurrence(self, x_t, z_prev, *, inputs=None) -> Tensor: ...
    def readout(self, z_t) -> Tensor: ...
    def init_state(self, batch_size, device) -> Tensor: ...  # Uses trainable h0 if configured
```

`VanillaRNNModel` is an alias that inherits from `CTRNNModel` with `VanillaRNNConfig`.

#### `EIRNNModel`

```python
@register_model("ei_rnn")
class EIRNNModel(CTRNNModel):
    config_class = EIRNNConfig

    # Additional attributes:
    e_size: int  # Number of excitatory units (latent_dim * ei_ratio)
    i_size: int  # Number of inhibitory units (latent_dim - e_size)

    # Key differences from CTRNNModel:
    # 1. Readout from E units only (when readout_e_only=True)
    # 2. EI initialization: E columns scaled by I/E ratio for balance
    def readout(self, z_t) -> Tensor: ...  # z_t[:, :e_size] when readout_e_only
```

**Dale constraint**: When `dale=True`, recurrent weights are decomposed as `|W| @ diag(sign)` where the sign mask is fixed (first 80% excitatory, rest inhibitory).

**EI initialization**: E columns of the recurrent weight matrix are scaled by `(e_size / i_size)` to balance excitatory and inhibitory inputs to each unit.

**Readout**: When `readout_e_only=True` (default), only excitatory units are used for readout, matching the biological constraint that long-range cortical projections are exclusively excitatory.

---

### PLRNN Family (Paradigm B)

**Module**: `neuralrnn.models.plrnn`

Piecewise-linear RNN for dynamical systems reconstruction. Transplant of CNS2023.

#### `ShallowPLRNNConfig`

```python
class ShallowPLRNNConfig(NeuralRNNConfig):
    model_type = "shallow_plrnn"

    def __init__(
        self,
        latent_dim: int = 3,           # Latent dimension M (= observation dim in DSR)
        hidden_dim: int = 50,          # Hidden dimension L
        input_dim: int = 0,            # External input dim K (0 = autonomous)
        output_dim: int | None = None, # Default: latent_dim (identity readout)
        observation: str = "identity", # Observation model
        autonomous: bool | None = None,# Inferred from input_dim==0 if None
        **kwargs,
    ) -> None: ...
```

#### `DendPLRNNConfig`

```python
class DendPLRNNConfig(ShallowPLRNNConfig):
    model_type = "dend_plrnn"
    def __init__(self, n_bases: int = 20, **kwargs) -> None: ...
```

#### `ALRNNConfig`

```python
class ALRNNConfig(ShallowPLRNNConfig):
    model_type = "alrnn"
    def __init__(self, n_linear: int = 1, **kwargs) -> None: ...
```

#### `ShallowPLRNNModel`

```python
@register_model("shallow_plrnn")
class ShallowPLRNNModel(NeuralDynamicsModel):
    config_class = ShallowPLRNNConfig

    # Recurrence: z = A * z + W1 @ ReLU(W2 @ z + h2) + h1 [+ C @ s]
    def recurrence(self, x_t, z_prev, *, inputs=None) -> Tensor: ...
    def readout(self, z_t) -> Tensor: ...  # Identity: returns z_t

    # Analytic support
    supports_analytic_fixed_points = True
    def jacobian(self, z, *, inputs=None) -> Tensor:
        """J(z) = diag(A) + W1 @ diag(1[W2@z + h2 > 0]) @ W2"""
    def analytic_parameters(self) -> dict[str, Tensor]:
        """Returns {"A", "W1", "W2", "h1", "h2"} for the analytic fixed-point solver."""
```

`DendPLRNNModel` and `ALRNNModel` are placeholders that inherit from `ShallowPLRNNModel`.

### Lowrank RNN Model (Paradigm A/B)

**Module**: `neuralrnn.models.lowrank`

Low-rank recurrent neural network where the recurrent connectivity matrix is factorized as $W^{rec} = m \\cdot n^T / N$, constraining dynamics to a low-dimensional subspace spanned by columns of $m$. Ported from Dubreuil et al. (2022) *Nature Neuroscience* and Valente et al. (2022) *NeurIPS*.

The same `LowrankRNNModel` class is used for both **Paradigm A** (task-optimized training, `notebook/07_lowrank_RNN_paradigmA.ipynb`) and **Paradigm B** (LINT inference from full-rank trajectories, `notebook/08_lowrank_RNN_paradigmB.ipynb`). For Paradigm B the full-rank "teacher" network and the LINT training loop are implemented notebook-locally in `08`: the reference `FullRankRNN` keeps separate membrane potentials and firing rates, uses `scale_by_hidden_size=False`, and has per-channel input/output scaling, all of which differ from the existing `CTRNNModel`; LINT fitting also requires a two-phase objective (identity readout trajectory regression followed by readout replacement) that does not map onto the generic `Trainer` + `SupervisedObjective` task pipeline.

A companion notebook, `notebook/08b_lowrank_RNN_paradigmB.ipynb`, demonstrates that LINT *can* be performed with the standard `Trainer` + `SupervisedObjective(regression)` when the teacher is the framework's own `CTRNNModel`: set `output_dim = latent_dim`, freeze the readout to the identity (`wo = N \cdot I`), and pass the teacher firing-rate trajectories as targets. The notebook now reproduces the full downstream analysis from `08_lowrank_RNN_paradigmB.ipynb` (rank scan, population clustering, connectivity overlap, PCA/TDR comparisons, inactivation experiments, $\kappa_1$ trajectories, and gain distributions) using the rank-1 fitted network. No framework modifications are needed.

**Motivation**: Unlike full-rank RNNs where $W^{rec} \\in \\mathbb{R}^{N \\times N}$ has $N^2$ parameters, a low-rank RNN with rank $R \\ll N$ has only $2NR$ recurrent parameters. This:
- Makes the network's computation **transparent**: activity projected onto the $R$ columns of $m$ reveals the low-dimensional dynamics
- Enables **population structure analysis**: neurons cluster naturally by their $(m_i, n_i, w^{in}_i, w^{out}_i)$ vectors
- Supports **resampling**: new networks can be generated from fitted Gaussian mixture distributions while preserving task performance

#### `LowrankRNNConfig`

```python
class LowrankRNNConfig(NeuralRNNConfig):
    model_type = "lowrank_rnn"

    def __init__(
        self,
        input_dim: int = 1,                    # Number of input channels
        latent_dim: int = 500,                 # Number of hidden units N
        output_dim: int = 1,                   # Number of output channels
        rank: int = 1,                         # Rank R of W_rec = m @ n^T / N
        alpha: float = 0.2,                    # dt/tau, Euler discretization step
        noise_std: float = 0.05,               # Recurrent Gaussian noise std
        dt: float | None = None,              # Integration step (alternative to alpha)
        tau: float = 100.0,                   # Membrane time constant
        add_bias: bool = False,               # Whether b is trainable
        scale_by_hidden_size: bool = True,    # Divide rec/output terms by N
        activation: str = "tanh",             # Hidden activation ("tanh"/"relu")
        output_activation: str = "tanh",      # Readout activation ("tanh"/"relu")
        train_wi: bool = True,                # Train input weights
        train_wo: bool = True,                # Train output weights
        train_wrec: bool = True,              # Train m, n low-rank factors
        train_h0: bool = False,               # Train initial state h0
        **kwargs,
    ) -> None: ...
```

**Parameter guide**:
- `rank`: Determines the dimensionality of the recurrent subspace. Rank 1 = line attractor, Rank 2 = plane with rotational dynamics, Rank 3+ = more complex manifolds.
- `scale_by_hidden_size` (default `True`): Matches both original codebases. When True, the recurrent term is divided by `N` and the output term is also divided by `N`.
- `activation` (default `"tanh"`): The original papers use tanh for both hidden and output activation.

#### `LowrankRNNModel`

```python
@register_model("lowrank_rnn")
class LowrankRNNModel(NeuralDynamicsModel):
    config_class = LowrankRNNConfig

    # ── Dynamics ──
    # r_t     = tanh(z_t + b)
    # rec_t   = r_t @ n @ m^T / N
    # z_{t+1} = z_t + sigma*xi_t + alpha*(-z_t + rec_t + x_t @ Wi_full)
    # y_t     = out_act(z_t) @ Wo_full / N
    def recurrence(self, x_t, z_prev, *, inputs=None) -> Tensor: ...
    def readout(self, z_t) -> Tensor: ...

    # ── Forward (override for precision) ──
    # return_dynamics=True → returns (output, trajectories) tuple
    # return_dynamics=False → returns DynamicsModelOutput
    def forward(self, inputs, *, initial_state=None, n_steps=None,
                return_states=True, return_dynamics=False): ...

    # ── Analysis utilities ──
    def svd_reparametrization(self) -> None: ...
    """Orthogonalize m columns via SVD of m @ n^T. Call before vector field analysis."""

    def clone(self) -> "LowrankRNNModel": ...
    """Deep copy for reference train() best-model tracking."""
```

**Key attributes** (for backward compat with reference analysis code):

| Attribute | Shape | Description |
|-----------|-------|-------------|
| `m` | `(N, R)` | Recurrent output directions (columns span the dynamic subspace) |
| `n` | `(N, R)` | Recurrent input directions |
| `wi` | `(input_dim, N)` | Input weight matrix (before scaling) |
| `wo` | `(N, output_dim)` | Output weight matrix (before scaling) |
| `b` | `(N,)` | Bias vector |
| `h0` | `(N,)` | Initial hidden state |
| `wi_full` | `(input_dim, N)` | Effective input weights after `si` scaling |
| `wo_full` | `(N, output_dim)` | Effective output weights after `so` scaling |
| `hidden_size` | `int` | Alias for `latent_dim` |
| `alpha` | `float` | Euler step size |
| `noise_std` | `float` | Noise standard deviation |

**Usage example** (see `notebook/07_lowrank_RNN_paradigmA.ipynb` for the full tutorial):

```python
from neuralrnn import AutoConfig, AutoModel
from neuralrnn.models.lowrank import LowrankRNNConfig, LowrankRNNModel
from neuralrnn.data.tasks import rdm_trials, lr_mante_trials

# ── Create model ──
cfg = LowrankRNNConfig(input_dim=1, latent_dim=256, output_dim=1,
                        rank=1, alpha=0.2, noise_std=0.05)
model = LowrankRNNModel(cfg)

# ── Generate data (neuralrnn built-in task generators) ──
inputs, targets, mask, conditions = rdm_trials(num_trials=800)

# Split train/val
split = int(0.8 * len(inputs))
x_tr, y_tr, m_tr = inputs[:split], targets[:split], mask[:split]
x_v, y_v, m_v = inputs[split:], targets[split:], mask[split:]

# ── Train using notebook-local train() ──
# See 07_lowrank_RNN_paradigmA.ipynb for the local train() definition
train(model, x_tr, y_tr, m_tr, n_epochs=20, lr=5e-3, keep_best=True)

# Or train using NeuralRNN Trainer + SupervisedObjective
from neuralrnn import Trainer, TrainingArguments, SupervisedObjective
from neuralrnn.data import CognitiveTaskDataset
ds = CognitiveTaskDataset(inputs=x_tr, targets=y_tr, mask=m_tr,
                          conditions=[], batch_size=32)
Trainer(model, ds, SupervisedObjective(task_type="regression"),
        TrainingArguments(max_steps=500)).train()

# ── Analysis: project activity onto m-subspace ──
model.svd_reparametrization()
m1 = model.m[:, 0].detach().numpy()
m2 = model.m[:, 1].detach().numpy()  # for rank >= 2

# ── Connectivity inspection ──
wi = model.wi_full[0].detach().numpy()   # effective input weights
wo = model.wo_full[:, 0].detach().numpy()  # effective output weights

# ── Population clustering (locally defined helpers) ──
# See notebook for make_vecs(), gmm_fit(), pop_scatter_linreg() definitions
vecs = make_vecs(model)
z, _ = gmm_fit(vecs, n_components=2)

# ── Save / load (safetensors + json) ──
model.save_pretrained("models/lowrank_rdm/")
model2 = LowrankRNNModel.from_pretrained("models/lowrank_rdm/")
```

**Key notes**:
- All task data generation goes through `neuralrnn.data.tasks` (no reference project dependency).
- Analysis/plotting helpers (`train()`, `loss_mse()`, `accuracy_general()`, `phi_prime()`,
  `gmm_fit()`, `make_vecs()`, `pop_scatter_linreg()`, `overlap_matrix()`,
  `get_lower_tri_heatmap()`, `psychometric_matrices_mante()`) are defined locally
  in the notebook to keep the framework generic.
- `to_support_net()` is now a stub in the notebook; the original `SupportLowRankRNN`
  class has not been ported to neuralrnn, so the notebook contains no reference-project
  imports.
- `forward(return_dynamics=True)` returns a raw `(output_tensor, trajectories_tensor)` tuple.
- `forward(return_dynamics=False)` (default) returns `DynamicsModelOutput`.
- Tensors are made contiguous before saving to avoid safetensors errors.
- `hidden_size`, `input_size`, `output_size`, `rank`, `alpha`, `noise_std` and the
  alias `non_linearity` are exposed for backward compatibility with reference analysis code.
- `print(model)` now shows config fields and parameter shapes.

---

### Latent Circuit Model (Paradigm A)

**Module**: `neuralrnn.models.latent_circuit`

Low-dimensional recurrent circuit embedded in high-dimensional neural space via orthonormal matrix Q (Cayley transform). Ported from Langdon & Engel (2025), Nature Neuroscience.

#### `LatentCircuitConfig`

```python
class LatentCircuitConfig(NeuralRNNConfig):
    model_type = "latent_circuit"

    def __init__(
        self,
        input_dim: int = 6,         # Task input dimension K
        latent_dim: int = 8,        # Number of latent nodes n
        output_dim: int = 2,        # Task output dimension
        embedding_dim: int = 50,    # High-dimensional RNN size N
        dt: float = 40.0,           # Discretization step (ms)
        tau: float = 200.0,         # Time constant (ms), alpha = dt/tau
        sigma_rec: float = 0.15,    # Recurrent noise std
        activation: str = "relu",   # Nonlinearity (only relu supported)
        **kwargs,
    ) -> None: ...
```

#### `LatentCircuitModel`

```python
@register_model("latent_circuit")
class LatentCircuitModel(NeuralDynamicsModel):
    config_class = LatentCircuitConfig

    # Recurrence: x_t = (1-α) x_{t-1} + α ReLU(w_rec x_{t-1} + w_in u_t + noise)
    def recurrence(self, x_t, z_prev, *, inputs=None) -> Tensor: ...
    def readout(self, z_t) -> Tensor: ...

    # Embedding
    @property
    def embedding_matrix(self) -> Tensor:   # Q of shape (n, N), orthonormal
    def embed(self, x: Tensor) -> Tensor:   # x @ Q: latent -> high-dim
    def project(self, y: Tensor) -> Tensor: # y @ Q^T: high-dim -> latent

    # Constraints (call after each gradient step)
    def apply_constraints(self) -> None: ...  # Recompute Q, reapply masks
```

---

### Tiny RNN Model (Paradigm B — Behavioral Fitting)

**Module**: `neuralrnn.models.tiny_rnn`

Small GRU RNN (1-4 units) for behavioral prediction in reward-learning tasks. Ported from Ji-An, Benna & Mattar (2025), Nature.

#### `TinyRNNConfig`

```python
class TinyRNNConfig(NeuralRNNConfig):
    model_type = "tiny_rnn"

    def __init__(
        self,
        input_dim: int = 3,         # [action, stage2, reward]
        latent_dim: int = 2,        # Hidden units (1-4 typically)
        output_dim: int = 2,        # Number of actions
        rnn_type: str = "GRU",      # "GRU" (standard)
        readout_FC: bool = True,    # Fully-connected vs diagonal readout
        trainable_h0: bool = False, # Trainable initial hidden state
        l1_weight: float = 1e-5,    # L1 regularization on recurrent weights
        **kwargs,
    ) -> ...
```

#### `TinyRNNModel`

```python
@register_model("tiny_rnn")
class TinyRNNModel(NeuralDynamicsModel):
    config_class = TinyRNNConfig

    # GRU single-step (manual implementation using gru weights)
    def recurrence(self, x_t, z_prev, *, inputs=None) -> Tensor: ...
    def readout(self, z_t) -> Tensor: ...

    # Efficient full-sequence forward (uses nn.GRU internally)
    def forward(self, inputs, ...) -> DynamicsModelOutput: ...

    # L1 regularization on recurrent weights
    def get_l1_loss(self) -> Tensor: ...
```

**Note**: Input is batch-first `(B, T, input_dim)`. With `output_h0=True` (matching the original tinyRNN implementation), the input at each trial is the **current** trial's `[action, stage2, reward]`, and the target is the **current** trial's action. Because the model prepends a readout of the initial hidden state `h0`, the effective alignment is `readout(h0)` → `action_0` and `readout(h_t)` → `action_t`. The legacy shifted-input convention (`input[t] = previous trial's observation`) is available via `input_format='shifted'` but performs worse with `output_h0=True` because the last observation is never fed into the network.

---

## 6. Data Layer

### `BaseDataset` & `StandardScaler`

**Module**: `neuralrnn.data.base`

```python
class BaseDataset(Dataset):
    """Base class for all datasets. Subclasses implement __len__/__getitem__
    returning standard batch dicts, or sample_batch() for random subsequence sampling."""

    kind: str = ""              # "neurogym" / "timeseries" / "behavioral" / "trajectory"
    input_dim: int = 0
    output_dim: int = 0
    normalizer: Any = None      # StandardScaler or None

    def sample_batch(self) -> dict[str, Tensor]:
        """Sample one batch. Default: __getitem__ + stack; DSR subclasses override."""

    def __iter__(self) -> Iterator[dict]:
        """Infinite random batch iterator (DSR style)."""
```

```python
class StandardScaler:
    """Z-score normalizer with inverse transform for analysis."""

    def fit(self, x: Tensor) -> StandardScaler: ...
    def transform(self, x: Tensor) -> Tensor: ...
    def inverse_transform(self, x: Tensor) -> Tensor: ...
    def fit_transform(self, x: Tensor) -> Tensor: ...
```

### `TimeSeriesDataset`

**Module**: `neuralrnn.data.timeseries_dataset`

For Paradigm B (DSR). Slices a `(T, N)` time series into subsequences of length `sequence_length`. `targets` is `inputs` shifted right by one step.

```python
class TimeSeriesDataset(BaseDataset):
    kind = "timeseries"

    def __init__(
        self,
        data: np.ndarray,                  # (T, N) time series
        external_inputs: np.ndarray = None, # (T, K) optional external inputs
        sequence_length: int = 200,         # Subsequence length
        batch_size: int = 16,               # Batch size for sample_batch
        normalize: bool = False,            # Z-score normalize
        test: np.ndarray = None,            # (T_test, N) held-out test set
    ) -> None: ...

    def sample_batch(self) -> dict:
        """Returns {"inputs": (B,T,N), "targets": (B,T,N), "external_inputs": (B,T,K)|None}"""

    @classmethod
    def from_npy(cls, train_path, test_path=None, **kwargs) -> "TimeSeriesDataset":
        """Registry loader: construct from .npy files (e.g. lorenz63 dataset)."""
```

### `NeurogymDataset`

**Module**: `neuralrnn.data.neurogym_dataset`

For Paradigm A (task-optimized RNN). Wraps `neurogym.Dataset` to produce standard batch dicts. Optional dependency: `pip install 'neuralrnn[neurogym]'`.

```python
class NeurogymDataset(BaseDataset):
    kind = "neurogym"

    def __init__(self, env, dataset, input_dim, output_dim,
                 batch_size=16, seq_len=100) -> None: ...

    @classmethod
    def from_task(cls, task: str, *, batch_size=16, seq_len=100,
                  dt=100, timing=None, **env_kwargs) -> "NeurogymDataset":
        """Construct from a neurogym task name.
        Example: NeurogymDataset.from_task("PerceptualDecisionMaking-v0", dt=100)
        """

    def sample_batch(self) -> dict[str, Tensor]:
        """Returns {"inputs": (B,T,input_dim), "targets": (B,T), "mask": None}"""

    def task_input(self, kind="stimulus") -> Tensor:
        """Return the task-condition input for fixed-point analysis (e.g. 0-coherence mean)."""
```

### `CustomDataset`

**Module**: `neuralrnn.data.custom_dataset`

For importing user-generated data (numpy arrays or torch tensors) into the framework. Supports three use cases:

1. **Supervised (Paradigm A)**: input-output pairs for task optimization
2. **Time-series reconstruction (Paradigm B)**: observed trajectories for dynamical systems reconstruction, with optional internal states for teacher forcing
3. **Free-running generation**: input-only data for model rollout evaluation

```python
class CustomDataset(BaseDataset):
    """User-generated dataset for custom inputs, outputs, and optional internal states.

    Supports two modes:
      - "supervised": inputs + targets for task optimization (Paradigm A)
      - "timeseries": observed time series for DSR (Paradigm B),
        with optional internal_states for teacher forcing

    Batch format:
      Paradigm A (supervised):
        {"inputs": (B,T,input_dim), "targets": (B,T,output_dim), "mask": (B,T)|None}
      Paradigm B (timeseries):
        {"inputs": (B,T,N), "targets": (B,T,N), "external_inputs": (B,T,K)|None}
    """

    def __init__(
        self,
        inputs: np.ndarray | Tensor,            # (T, input_dim) or (T,)
        targets: np.ndarray | Tensor = None,    # (T, output_dim) or (T,); optional
        internal_states: np.ndarray | Tensor = None,  # (T, latent_dim); optional
        external_inputs: np.ndarray | Tensor = None,  # (T, K); optional
        sequence_length: int = 200,              # Subsequence length for slicing
        batch_size: int = 16,                    # Batch size for sample_batch
        mode: str = "auto",                      # "supervised" / "timeseries" / "auto"
        normalize: bool = False,                 # Z-score normalize inputs
        test_fraction: float = 0.0,              # Fraction held out for test set
        seed: int = 0,                           # RNG seed for train/test split
    ) -> None: ...

    @classmethod
    def from_arrays(cls, inputs, targets=None, internal_states=None,
                    external_inputs=None, **kwargs) -> "CustomDataset":
        """Convenience constructor from numpy arrays or torch tensors.
        Accepts 1D (T,), 2D (T,D), or 3D (B,T,D) arrays. 3D arrays are
        reshaped to (B*T, D) for slicing into subsequences.
        """

    @classmethod
    def from_dict(cls, data: dict, **kwargs) -> "CustomDataset":
        """Construct from a dict with keys "inputs", "targets", "internal_states",
        "external_inputs". Useful for loading from .npz or pickle files."""

    @classmethod
    def from_npz(cls, path: str, **kwargs) -> "CustomDataset":
        """Load from a .npz file. Keys: "inputs", "targets" (optional),
        "internal_states" (optional), "external_inputs" (optional)."""

    @classmethod
    def from_mat(cls, path: str, variable_map: dict | None = None, **kwargs) -> "CustomDataset":
        """Load from a MATLAB .mat file.
        variable_map: dict mapping expected keys ("inputs","targets",...) to .mat variable names.
        If None, uses default names "inputs","targets","internal_states","external_inputs".
        """

    def sample_batch(self) -> dict[str, Tensor]:
        """Sample a random batch. Keys depend on mode:
        - supervised: {"inputs", "targets", "mask"}
        - timeseries: {"inputs", "targets", "external_inputs"}
        """

    @property
    def test_set(self) -> "CustomDataset | None":
        """The held-out test fraction (if test_fraction > 0), as a separate CustomDataset."""
```

**Usage Examples**:

```python
import numpy as np
from neuralrnn.data.custom_dataset import CustomDataset

# --- Paradigm A: Supervised task data ---
inputs = np.random.randn(1000, 10, 3)   # 1000 trials, 10 steps, 3 features
targets = np.random.randint(0, 2, (1000, 10))  # binary choice per step
ds = CustomDataset.from_arrays(inputs, targets=targets, mode="supervised", batch_size=32)

# --- Paradigm B: DSR with observed trajectory ---
trajectory = np.load("my_lorenz.npy")   # (50000, 3)
ds = CustomDataset.from_arrays(trajectory, mode="timeseries", sequence_length=200)

# --- Paradigm B with internal states for teacher forcing ---
trajectory = np.load("my_trajectory.npy")       # (50000, 3)
states = np.load("my_internal_states.npy")      # (50000, 10)
ds = CustomDataset.from_arrays(
    trajectory, internal_states=states,
    mode="timeseries", sequence_length=200, normalize=True
)

# --- From .npz file ---
ds = CustomDataset.from_npz("my_data.npz", sequence_length=150, batch_size=8)

# --- Use with Trainer ---
from neuralrnn import Trainer, TrainingArguments, TeacherForcingObjective
Trainer(model, ds, TeacherForcingObjective(alpha=0.1),
        TrainingArguments(max_steps=5000)).train()
```

### `CognitiveTaskDataset`

**Module**: `neuralrnn.data.cognitive_task_dataset`

Wraps cognitive task generators for Paradigm A. Provides a unified interface for training low-rank RNNs and other task-optimized models on cognitive tasks.

```python
class CognitiveTaskDataset(BaseDataset):
    kind = "cognitive_task"

    def __init__(
        self,
        inputs: Tensor,          # (N, T, input_dim)
        targets: Tensor,         # (N, T, output_dim)
        mask: Tensor,            # (N, T, output_dim) boolean
        conditions: list,        # Per-trial metadata dicts
        task_name: str = "",     # Task identifier
        batch_size: int = 128,   # Batch size for sample_batch()
    ) -> None: ...

    @classmethod
    def from_task(cls, task_name: str, batch_size: int = 128, **kwargs) -> "CognitiveTaskDataset":
        """Create from a named task generator in TASK_REGISTRY."""

    def sample_batch(self) -> dict[str, Tensor]:
        """Returns {"inputs": (B,T,input_dim), "targets": (B,T,output_dim), "mask": (B,T,output_dim)}"""
```

**Available task generators** (in `neuralrnn.data.tasks.TASK_REGISTRY`):

| Key | Task | Input | Output | Description |
|-----|------|-------|--------|-------------|
| `rdm` | Random Dot Motion | 1 | 1 | Integrate noisy coherence signal; report sign |
| `romo` | Parametric Working Memory | 1 | 1 | Compare two frequencies across delay |
| `raposo` | Multisensory Decision | 4 | 1 | Attend to visual/auditory/both modalities |
| `dms` | Delayed Match-to-Sample | 2 | 1 | Judge if two sequential stimuli match |
| `lr_mante` | Ctx Decision (low-rank) | 4 | 1 | Context-dependent integration of color/motion |
| `mante` | Ctx Decision (latent circuit) | 6 | 2 | Context-dependent decision with 2 outputs |
| `mante_short` | Ctx Decision (short variant) | 6 | 2 | Shorter version of Mante |
| `siegel_miller` | Siegel-Miller Task | 6 | 2 | Classical context-dependent decision making |
| `two_afc` | Two-Alternative Forced Choice | 1 | 1 | Simple binary choice |
| `delay_match_to_sample` | DMS (latent circuit) | 2 | 1 | Match-to-sample with pre-sliced mask |
| `parametric_wm` | Parametric WM (latent circuit) | 1 | 1 | Working memory with variable delay |

**Usage**:
```python
from neuralrnn.data import CognitiveTaskDataset

# Create from task registry
ds = CognitiveTaskDataset.from_task("rdm", num_trials=800, seed=42)

# Or construct directly from tensors
ds = CognitiveTaskDataset(inputs=x, targets=y, mask=mask, conditions=[],
                          task_name="custom", batch_size=32)
```

### Registry & `load_dataset`

**Module**: `neuralrnn.data.registry`

```python
@dataclass
class DatasetSpec:
    kind: str                    # "neurogym" / "timeseries" / "behavioral" / "trained_rnn"
    loader: str | None = None    # "module:function" loader path
    url: str | None = None       # Download URL
    files: dict[str, str] | None = None  # Logical name -> filename mapping
    filename: str | None = None  # Single file download name
    unpack: str | None = None    # None / "zip" / "tar"
    sha256: str | None = None    # Checksum
    task: str | None = None      # neurogym task name
    extra: dict[str, Any] = field(default_factory=dict)

DATASET_REGISTRY: dict[str, DatasetSpec]  # Global registry

def load_dataset(name: str, **overrides):
    """Unified entry: lookup registry -> download/cache -> instantiate dataset.
    overrides are passed through to the loader (e.g. sequence_length, batch_size)."""
```

**Built-in registered datasets**:

| Name | Kind | Description |
|------|------|-------------|
| `"lorenz63"` | timeseries | Lorenz 63 attractor (CNS2023 benchmark) |
| `"perceptual_decision_making"` | neurogym | Perceptual decision making task |
| `"dms_lowrank_rank2"` | trained_rnn | Rank-2 DMS network (Harvard Dataverse) |
| `"bartolo_monkey"` | behavioral | Bartolo monkey probabilistic reversal learning (Ji-An et al. 2025) |

### Download Utilities

**Module**: `neuralrnn.data.download`

```python
def cache_root() -> Path:
    """Returns the dataset cache directory.
    Priority: NEURALRNN_CACHE env var -> ~/.cache/neuralrnn/datasets"""

def ensure_files(spec: DatasetSpec) -> dict[str, str]:
    """Ensure spec's files are ready (download/cache/verify/extract).
    Returns {logical_name: local_absolute_path}."""
```

---

## 7. Training Layer

### `TrainingArguments`

**Module**: `neuralrnn.train.training_args`

```python
@dataclass
class TrainingArguments:
    # Optimization
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    max_steps: int = 1000
    batch_size: int = 16
    grad_clip_norm: float | None = 1.0   # Gradient clipping; None to disable
    optimizer: str = "adam"               # "adam" / "adamw" / "sgd"

    # Scheduling
    lr_scheduler: str | None = None       # None / "cosine" / "step"
    warmup_steps: int = 0

    # Logging / evaluation / checkpointing
    log_every: int = 50
    eval_every: int | None = None         # None = no eval during training
    save_every: int | None = None
    output_dir: str = "./outputs"
    device: str = "cpu"                   # "cpu" / "cuda" / "cuda:0"
    seed: int = 0

    # Dropout (hidden-state regularization during training)
    # Ported from trainRNNbrain: mask sampled once per forward pass ("dead neuron" strategy).
    # Recommended range: 0.05–0.2. Default 0 = disabled.
    dropout_rate: float = 0.0
    dropout_sampling: str = "uniform"     # "uniform" / "participation" / "output_weights"
    dropout_beta: float = 1.0            # Softmax temperature for non-uniform sampling

    # Forcing annealing (GTF / teacher forcing)
    anneal_forcing: bool = False
    forcing_start: float = 1.0
    forcing_end: float = 0.0

    # Early stopping & best-model saving
    early_stop_loss: float | None = None   # Stop when training loss drops below this
    keep_best: bool = False                # Restore lowest training-loss checkpoint
    # Metric-based early stopping (requires eval_fn + eval_every)
    eval_metric: str | None = None         # Key in eval_fn dict to track
    greater_is_better: bool = False        # Whether eval_metric is maximized
    early_stopping_patience: int | None = None  # Eval checks without improvement before stop

    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict: ...
```

### `Trainer`

**Module**: `neuralrnn.train.trainer`

Universal training loop. Paradigm-agnostic — paradigm differences are entirely determined by the `Objective`.

```python
class Trainer:
    def __init__(
        self,
        model: NeuralDynamicsModel,
        dataset,                            # BaseDataset with sample_batch()
        objective: Objective,               # Determines the training paradigm
        args: TrainingArguments | None = None,
        eval_fn: Callable[[NeuralDynamicsModel], dict] | None = None,
    ) -> None: ...

    def train(self) -> list[dict]:
        """Run the training loop. Returns history of log dicts."""

    def evaluate(self) -> dict:
        """Run eval_fn on the model."""

    def save_checkpoint(self, step: int) -> str:
        """Save model + optimizer state. Returns checkpoint path."""
```

**Training loop internals**:

1. Each step: `batch = dataset.sample_batch()` → move to device
2. Optional forcing annealing (for GTF)
3. `loss, logs = objective.compute_loss(model, batch)`
4. Backward + optional gradient clipping
5. Optimizer step + optional LR scheduler
6. Periodic logging, evaluation, checkpointing

### Objectives

**Module**: `neuralrnn.train.objectives`

All objectives inherit from:

```python
class Objective:
    def compute_loss(self, model: NeuralDynamicsModel,
                     batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        """Returns (scalar_loss, log_dict). Trainer backprops the loss."""

    def set_forcing(self, alpha: float) -> None:
        """Optional: called by Trainer for forcing annealing."""
```

#### `SupervisedObjective` (Paradigm A)

```python
class SupervisedObjective(Objective):
    def __init__(self, task_type: str = "classification"):
        """task_type: "classification" (CrossEntropy) or "regression" (MSE).
        Supports optional mask in batch["mask"]."""
```

#### `TeacherForcingObjective` (Paradigm B — PLRNN)

```python
class TeacherForcingObjective(Objective):
    def __init__(self, alpha: float = 0.1):
        """alpha: forcing strength (1.0 = pure teacher forcing, 0.0 = free running).
        DSR typically uses small alpha (e.g. 0.1) for sparse forcing on chaotic systems.
        Supports alpha annealing via Trainer's forcing schedule."""

def generalized_teacher_forcing(z_pred, z_obs, alpha) -> Tensor:
    """z = alpha * z_obs + (1-alpha) * z_pred"""
```

#### `BehavioralObjective` (Tiny RNN)

```python
class BehavioralObjective(Objective):
    """Next-action NLL for behavioral fitting. Standard batch:
    {"inputs": (B,T,input_dim), "targets": (B,T), "mask": (B,T)|None}

    Supports output_h0=True models: if model.config.output_h0 is True and
    outputs.shape[1] == targets.shape[1] + 1, logits are sliced to logits[:, :-1]
    before computing cross-entropy (matching original tinyRNN scores[:-1]).

    Also adds L1 regularization automatically when model.config.l1_weight > 0
    and the model provides get_l1_loss().
    """
```

#### `VariationalObjective` (LFADS)

```python
class VariationalObjective(Objective):
    def __init__(self, kl_weight: float = 1.0, likelihood: str = "poisson"):
        """ELBO = reconstruction_nll - kl_weight * KL.
        Requires model.forward().extras to contain "rates" and optionally "kl"."""
```

### Loss Functions

**Module**: `neuralrnn.losses`

Reusable loss functions and metrics. All accept either raw `torch.Tensor` or `DynamicsModelOutput` (which delegates arithmetic ops to `.outputs`).

#### `loss_mse`

```python
def loss_mse(output, target, mask) -> Tensor:
    """Masked mean squared error loss.

    Compatible with both raw tensors and DynamicsModelOutput.
    Computes (output - target)^2 elementwise, applies the mask,
    and normalizes per-trial.

    Args:
        output: (B, T, O) tensor or DynamicsModelOutput
        target: (B, T, O) tensor
        mask:   (B, T, 1) or (B, T, O) float/bool tensor

    Returns:
        scalar torch.Tensor
    """
```

#### `accuracy_general`

```python
def accuracy_general(output, targets, mask) -> Tensor:
    """Sign-based accuracy for binary decision tasks.

    Only considers trials where targets are non-zero.
    Decision is the sign of the masked mean output/target over
    valid timesteps.

    Args:
        output:  (B, T, O) tensor or DynamicsModelOutput
        targets: (B, T, O) tensor
        mask:    (B, T, 1) or (B, T, O) float/bool tensor

    Returns:
        scalar torch.Tensor — fraction correct, or NaN if no valid trials
    """
```

**Usage**:

```python
from neuralrnn.losses import loss_mse, accuracy_general

# Compatible with both raw tensors and DynamicsModelOutput
loss = loss_mse(model_output, y_val, mask_val)
acc = accuracy_general(model_output, y_val, mask_val)
```

### Cross-Validation

**Module**: `neuralrnn.train.cv`

For behavioral fitting (Tiny RNN style). Nested CV with configuration grid search.

```python
def config_combination(base_config: dict, config_ranges: dict) -> list[dict]:
    """Cartesian product: base_config x config_ranges.
    Each result gets a "model_name" field from the varied keys."""

@dataclass
class CVResult:
    config: dict
    outer_val_losses: list[float]
    @property
    def mean_val_loss(self) -> float: ...

def behavior_cv_training(base_config: dict, config_ranges: dict,
                         fit_one_fn, n_samples: int) -> list[CVResult]:
    """Nested CV: outer splits for generalization, inner splits for model selection.
    fit_one_fn(config, train_idx, val_idx, seed) -> val_loss"""

def find_best_models_for_exp(results: list[CVResult]) -> CVResult:
    """Select the config with lowest mean validation loss."""
```

---

## 8. Analysis Layer

All analysis tools depend **only** on the `NeuralDynamicsModel` interface (`recurrence`/`readout`/`jacobian`/`generate`). They never import specific model classes. This is the key to making interpretability analysis model-agnostic.

### Fixed Points

**Module**: `neuralrnn.analysis.fixed_points`

```python
@dataclass
class FixedPoint:
    z: np.ndarray                       # Coordinates (M,)
    speed: float                        # ||F(z) - z|| (numeric); 0 (analytic)
    eigenvalues: np.ndarray | None      # Jacobian eigenvalues
    is_stable: bool | None              # Discrete: max|eig| < 1
    order: int = 1                      # 1 = fixed point; k > 1 = k-cycle
    cycle: np.ndarray | None            # k-cycle points (order, M)

@dataclass
class FixedPointSet:
    points: list[FixedPoint]
    def coords(self) -> np.ndarray: ... # Stack all z's
    def __len__(self) -> int: ...
    def __iter__(self): ...
```

#### `NumericFixedPointFinder`

```python
class NumericFixedPointFinder:
    def __init__(
        self,
        n_candidates: int = 64,     # Number of initial candidate states
        n_iters: int = 10000,       # Adam optimization iterations
        lr: float = 1e-3,           # Learning rate
        speed_tol: float = 1e-5,    # Speed threshold for acceptance
        dedup_tol: float = 1e-2,    # Deduplication radius
        init_scale: float = 3.0,    # Random init scale
        init_positive: bool = True, # Init in positive orthant
    ) -> None: ...

    def find(self, model: NeuralDynamicsModel, *,
             task_input: Tensor | None = None,
             init_states: Tensor | None = None) -> FixedPointSet:
        """Minimize ||F(z) - z||^2 with Adam from multiple initial points.
        task_input: (input_dim,) fixed input condition for the search."""
```

#### `AnalyticPLRNNFixedPointFinder`

```python
class AnalyticPLRNNFixedPointFinder:
    def __init__(self, max_order: int = 1, outer_it: int = 300,
                 inner_it: int = 100) -> None: ...

    def find(self, model: NeuralDynamicsModel) -> FixedPointSet:
        """Exact enumeration of fixed points and k-cycles using PLRNN's
        piecewise-linear structure. Requires model.supports_analytic_fixed_points."""
```

#### `ScipyFixedPointFinder`

Ported from trainRNNbrain's `DynamicSystemAnalyzer`. Uses scipy's `fsolve` (exact root-finding) or `minimize(method='Powell')` (approximate). More robust for stiff systems than the gradient-based numeric backend.

```python
class ScipyFixedPointFinder:
    def __init__(
        self,
        n_candidates: int = 100,     # Number of initial candidate points
        mode: str = "exact",         # "exact" (fsolve) or "approx" (Powell)
        fun_tol: float = 1e-12,      # Function value tolerance
        sigma_init_guess: float = 0.01,  # Gaussian noise for init guesses
        diff_cutoff: float = 1e-7,   # Deduplication L2 distance
        seed: int = 42,
    ) -> None: ...

    def find(self, model: NeuralDynamicsModel, *,
             task_input: Tensor | None = None,
             init_states: Tensor | None = None) -> FixedPointSet:
        """Find fixed points using scipy root-finding.
        mode="exact" uses fsolve (Levenberg-Marquardt); mode="approx" uses Powell."""
```

#### Unified Entry

```python
def find_fixed_points(model: NeuralDynamicsModel, *, backend: str = "auto",
                      task_input: Tensor | None = None,
                      max_order: int = 1, **kwargs) -> FixedPointSet:
    """Auto-select backend: analytic (if supported) else numeric.
    backend: "auto" / "numeric" / "analytic" / "scipy"."""
```

### Linearization

**Module**: `neuralrnn.analysis.linearization`

```python
@dataclass
class LinearizationResult:
    jacobian: np.ndarray            # (M, M)
    eigenvalues: np.ndarray         # (M,) complex
    eigenvectors: np.ndarray        # (M, M) complex
    is_stable: bool                 # max|eig| < 1
    n_unstable: int                 # Count of |eig| >= 1

def linearize(model: NeuralDynamicsModel, z, *,
              task_input: Tensor | None = None) -> LinearizationResult:
    """Linearize model at state z. z: (M,) tensor or ndarray."""

def dominant_direction(lin: LinearizationResult) -> np.ndarray:
    """Real part of the eigenvector for the largest eigenvalue."""

def classify_fixed_point(lin: LinearizationResult) -> str:
    """Returns "stable", "unstable", or "saddle(k)" where k is the number of unstable directions."""
```

### Vector Field

**Module**: `neuralrnn.analysis.vector_field`

```python
@dataclass
class VectorField:
    grid_pc: np.ndarray         # Grid points in plane coords (G, 2)
    velocity_pc: np.ndarray     # Velocity vectors projected to plane (G, 2)
    speed: np.ndarray           # ||F(z)-z|| full-dim norm (G,)

@torch.no_grad()
def compute_vector_field(
    model: NeuralDynamicsModel,
    basis: np.ndarray,          # (2, M) — two directions defining the plane
    mean: np.ndarray,           # (M,) — plane origin (data mean)
    *, task_input: Tensor | None = None,
    extent: tuple = (-3.0, 3.0),
    n_grid: int = 20,
) -> VectorField:
    """Sample F(z)-z on a 2D grid in the plane spanned by basis, centered at mean.
    Returns data for quiver plots."""
```

### Dimensionality Reduction

**Module**: `neuralrnn.analysis.dimensionality`

```python
@dataclass
class PCAResult:
    components: np.ndarray              # (n_components, M)
    mean: np.ndarray                    # (M,)
    explained_variance_ratio: np.ndarray

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project (..., M) -> (..., n_components)"""
    def inverse_transform(self, Y: np.ndarray) -> np.ndarray:
        """Back-project (..., n_components) -> (..., M)"""

def fit_pca(X: np.ndarray, n_components: int = 2) -> PCAResult:
    """PCA via SVD (no sklearn dependency). X: (N, M)."""

@torch.no_grad()
def collect_states(model: NeuralDynamicsModel, dataset,
                   n_batches: int = 1) -> np.ndarray:
    """Run model on dataset batches, flatten to (N_points, M) for PCA/analysis."""
```

### Lyapunov Exponent

**Module**: `neuralrnn.analysis.lyapunov`

```python
@torch.no_grad()
def max_lyapunov_exponent(
    model: NeuralDynamicsModel,
    z1: Tensor,                 # (M,) initial condition
    T: int = 10000,             # Steps for exponent computation
    T_trans: int = 1000,        # Transient steps to discard
    ons: int = 1,               # QR re-orthogonalization period
) -> float:
    """Compute the maximal Lyapunov exponent via QR iteration on Jacobians.
    lambda_max > 0 indicates chaos (Lorenz63 ≈ 0.9 with dt=0.01).
    Divide result by dt to get the continuous-time exponent."""
```

### DSR Metrics (D_stsp, D_H)

**Module**: `neuralrnn.analysis.stsp_metrics`

```python
def state_space_divergence(x_gen, x_true, method="auto", **kwargs) -> float:
    """D_stsp: state space distribution divergence.
    Auto-selects: binning (dim <= 4) else GMM Monte Carlo.
    x_*: (T, N) arrays. Lower is better."""

def state_space_divergence_binning(x_gen, x_true, n_bins=30) -> float:
    """D_stsp via histogram + Laplace smoothing + KL. Best for low-dim (≤4)."""

def state_space_divergence_gmm(x_gen, x_true, scaling=1.0,
                                max_used=10000, mc_n=1000) -> float:
    """D_stsp via GMM Monte Carlo KL. More stable for high-dim."""

def power_spectrum_error(X, X_gen, smoothing=20.0) -> float:
    """D_H: per-dimension Hellinger distance of smoothed power spectra.
    Captures temporal structure fidelity. X, X_gen: (T, N)."""

def hellinger_distance(p, q) -> float:
    """Hellinger distance between two distributions (arrays)."""
```

### Line Attractor Analysis

**Module**: `neuralrnn.analysis.line_attractor`

Ported from trainRNNbrain's `DynamicSystemAnalyzerCDDM`. Analyzes line attractors — continuous slow manifolds where the network state drifts very slowly (‖RHS‖ ≈ 0), supporting stable maintenance of continuous variables.

```python
@dataclass
class LineAttractorPoint:
    z: np.ndarray                       # State-space coordinate (M,)
    speed: float                        # ‖F(z) - z‖
    distance: float                     # Cumulative distance along LA
    eigenvalues: np.ndarray | None      # Jacobian eigenvalues
    jacobian: np.ndarray | None         # Jacobian matrix

@dataclass
class LineAttractorResult:
    points: list[LineAttractorPoint]
    endpoints: tuple[np.ndarray, np.ndarray]  # (left, right)
    projection_axes: np.ndarray | None        # (3, M) for 3D viz

def find_line_attractor_endpoints(model, *, context_input, n_steps=1000,
                                   relax_steps=10, initial_state=None):
    """Find left/right endpoints by running with nudged inputs."""

def walk_line_attractor(model, *, context_input, endpoint_left, endpoint_right,
                        n_points=31, max_iter=100):
    """Walk along line attractor, minimizing ‖RHS‖² at each point.
    Uses scipy.optimize.minimize with SLSQP."""

def compute_line_attractor(model, *, context_input, projection_axes=None,
                           n_steps=1000, n_points=31) -> LineAttractorResult:
    """Unified entry: find endpoints → walk → compute analytics."""
```

### Manifold Analysis

**Module**: `neuralrnn.analysis.manifold`

```python
def trajectories_to_pos_vel(traj: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert trajectories (B,T,M) or (T,M) to (position, velocity) pairs.
    velocity[t] = x[t+1] - x[t]. For MARBLE input preparation."""

def marble_embedding(pos, vel, **marble_kwargs):
    """MARBLE unsupervised manifold embedding. Requires: pip install 'neuralrnn[manifold]'
    See PORTING_GUIDE Recipe 6 for integration details."""

def neuralflow_analysis(spike_data, **kwargs):
    """neuralflow continuous-time latent flow analysis.
    See PORTING_GUIDE Recipe 8 for integration details."""
```

### Linear Algebra Utilities

**Module**: `neuralrnn.analysis.linalg_utils`

Model-agnostic linear algebra and trajectory utilities. All functions operate on numpy arrays or torch tensors — no dependency on specific model classes.

| Function | Description |
|----------|-------------|
| `phi_prime(x)` | tanh derivative: `1 - tanh²(x)`, used in gain analysis |
| `gram_schmidt(vecs)` | Classical Gram-Schmidt orthogonalization (numpy) |
| `gram_schmidt_pt(mat)` | Row-wise Gram-Schmidt orthogonalization (torch, in-place) |
| `gram_factorization(G)` | Factorize Gramian/covariance matrix into basis vectors |
| `overlap_matrix(vecs)` | Pairwise inner-product overlap matrix of vectors |
| `corrvecs(v, w)` | Cosine similarity between two vectors |
| `project(v, subspace_vecs)` | Project vector onto a subspace |
| `angle_vectors(v, w)` | Angle between two vectors (radians) |
| `angle_vec_subsp(v, vecs)` | Angle between vector and subspace (radians) |
| `flatten_trajectory(X)` | Reshape `(trials, time, dim)` → `(trials*time, dim)` |
| `unflatten_trajectory(X_flat, n_trials)` | Reverse of `flatten_trajectory` |
| `map_device(tensors, net)` | Move tensor(s) to same device as a network |

### Population Structure Analysis

**Module**: `neuralrnn.analysis.population_structure`

Tools for extracting connectivity vectors and performing Gaussian mixture clustering on neuron feature spaces. Uses duck-typing on model attributes (`m`, `n`, `wi`, `wo`, `rank`, `input_size`, `output_size`).

#### `make_vecs`

```python
def make_vecs(net) -> list[np.ndarray]:
    """Extract connectivity vectors from a low-rank-like network.

    Returns 2R + K + O vectors (each length N):
      - R from columns of m, R from columns of n,
      - K from rows of wi, O from columns of wo.

    Args:
        net: Model with attributes m (N,R), n (N,R), wi (K,N),
             wo (N,O), rank, input_size, output_size.

    Returns:
        list of (N,) numpy arrays
    """
```

#### `gmm_fit`

```python
def gmm_fit(neurons_fs, n_components, algo='bayes', n_init=50,
            random_state=None) -> tuple[np.ndarray, object]:
    """Fit a GMM to neuron feature vectors.

    Args:
        neurons_fs: list of (N,) arrays or (d, N) feature matrix
        n_components: number of clusters
        algo: 'em' (GaussianMixture) or 'bayes' (BayesianGaussianMixture)
        n_init: number of EM initializations
        random_state: seed

    Returns:
        labels: (N,) int array of cluster assignments
        model: fitted sklearn mixture model
    """
```

#### `compute_population_means` / `compute_population_covariances`

```python
def compute_population_means(X, labels) -> np.ndarray:
    """(n_pops, d) mean feature vector per population."""

def compute_population_covariances(X, labels) -> list[np.ndarray]:
    """List of (d, d) covariance matrices per population."""
```

---

## 9. Visualization

**Module**: `neuralrnn.visualization`

Comprehensive plotting utilities for dynamical systems analysis. All functions accept **data** (numpy arrays, dataclasses), not models — visualization is decoupled from analysis. All functions accept an optional `ax` parameter for composition and return `(fig, ax)`.

### Trajectory Plots

```python
def plot_trajectories_2d(trajectories, pca_result=None, *,
                         colors=None, labels=None, ax=None,
                         alpha=0.5, linewidth=0.5, show_legend=True):
    """Plot multiple trajectories projected to 2D PCA plane.
    trajectories: list of (T_i, M) arrays. Each gets its own color/label."""

def plot_trajectories_3d(trajectories, pca_result=None, *,
                         colors=None, labels=None, ax=None,
                         alpha=0.5, linewidth=0.5, elev=25, azim=45):
    """Plot trajectories in 3D PCA space.
    elev/azim: camera angles for 3D rotation control."""
```

### Fixed Point Plots

```python
def plot_fixed_points(fixed_points, pca_result=None, *,
                      ax=None, colors=None, markers=None, size=80):
    """Plot fixed points on 2D PCA plane with stability coloring.
    stable=blue circle, unstable=red X, saddle=orange triangle."""

def plot_fixed_points_3d(fixed_points, pca_result=None, *,
                         ax=None, colors=None, elev=25, azim=45):
    """Plot fixed points in 3D PCA space."""
```

### Vector Field

```python
def plot_vector_field(vector_field, *, ax=None, color='gray',
                      scale=None, width=0.003, alpha=0.7,
                      speed_colormap='YlOrRd', show_speed=False):
    """Quiver plot of vector field on 2D plane.
    show_speed=True: color-encodes speed magnitude."""
```

### Combined Phase Portrait

```python
def plot_phase_portrait(trajectories=None, fixed_points=None,
                        vector_field=None, pca_result=None, *,
                        colors=None, labels=None, title=None, figsize=(8, 6)):
    """Combined: trajectories + fixed points + vector field on single 2D plot."""
```

### Weight Matrices

```python
def plot_weight_matrix(W, *, title=None, ax=None, cmap='RdBu_r',
                       center_zero=True, colorbar=True):
    """Heatmap of a weight matrix (W_inp, W_rec, W_out)."""

def plot_connectivity(W_inp, W_rec, W_out, *, dale_mask=None,
                      sort=True, figsize=(15, 4)):
    """Side-by-side heatmaps of input/recurrent/output weight matrices.
    If dale_mask given, sorts neurons by E/I identity."""
```

### Line Attractor Plots

```python
def plot_line_attractor(la_result, pca_result=None, *,
                        ax=None, show_rhs=True, color='#2196F3'):
    """Plot line attractor in 2D PCA space with optional |RHS|² color coding."""

def plot_line_attractor_3d(la_result, projection_axes, *,
                           ax=None, trajectories=None, elev=25, azim=45):
    """Plot line attractor in 3D subspace (e.g. choice/context/sensory).
    projection_axes: (3, M) defining the 3D subspace."""
```

### Animation

```python
def animate_trajectories_3d(trajectories, pca_result=None, *,
                            colors=None, labels=None, fps=30,
                            duration=10.0, elev=25, step=2):
    """Create rotating 3D animation of trajectories (camera orbits z-axis).
    Returns matplotlib.animation.FuncAnimation."""

def animate_trajectory_progression(trajectories, pca_result=None, *,
                                   colors=None, labels=None, projection="3d",
                                   fps=30, n_frames=None, elev=25, azim=45,
                                   linewidth=1.0, alpha=0.8, trail_alpha=0.15):
    """Progressive trajectory drawing animation — reveals trajectories timestep
    by timestep, showing how RNN states evolve over time.
    projection: "2d" or "3d". Returns matplotlib.animation.FuncAnimation."""
```

### Other Plots

```python
def plot_averaged_responses(responses, dale_mask=None, *, ax=None,
                            cmap='RdBu_r', labels=None):
    """Heatmap of per-cluster averaged firing rate trajectories."""

def plot_psychometric_curves(curves, *, ax=None, colors=None):
    """Plot psychometric curves (coherence vs P(right))."""

def plot_trial_predictions(predictions, targets, *, mask=None,
                           n_trials=6, figsize=(12, 8)):
    """Plot multiple trials: predicted vs target output traces.
    Handles dimension mismatches: integer class labels (B, T) are auto-expanded
    to one-hot when predictions is (B, T, O) with O > 1."""
```

---

## Quick Reference: End-to-End Patterns

### Paradigm A: Task-Optimized RNN

```python
from neuralrnn import AutoConfig, AutoModel, Trainer, TrainingArguments, load_dataset
from neuralrnn import SupervisedObjective
from neuralrnn.analysis import find_fixed_points, fit_pca, collect_states

# Data
ds = load_dataset("perceptual_decision_making", dt=100, seq_len=200, batch_size=16)

# Model
cfg = AutoConfig.for_model("ctrnn", input_dim=ds.input_dim, latent_dim=64,
                           output_dim=ds.output_dim, dt=ds.dt)
model = AutoModel.from_config(cfg)

# Train
Trainer(model, ds, SupervisedObjective("classification"),
        TrainingArguments(max_steps=1000, learning_rate=0.01)).train()

# Analyze
states = collect_states(model, ds, n_batches=10)
pca = fit_pca(states, n_components=2)
fps = find_fixed_points(model, task_input=ds.task_input())

# Save / load
model.save_pretrained("runs/ctrnn_pdm")
model = AutoModel.from_pretrained("runs/ctrnn_pdm")
```

### Paradigm B: Dynamical Systems Reconstruction

```python
from neuralrnn import AutoConfig, AutoModel, Trainer, TrainingArguments, load_dataset
from neuralrnn import TeacherForcingObjective
from neuralrnn.analysis import find_fixed_points, state_space_divergence, max_lyapunov_exponent

# Data (auto-downloads)
ds = load_dataset("lorenz63", sequence_length=75, batch_size=16, normalize=True)

# Model
cfg = AutoConfig.for_model("shallow_plrnn", latent_dim=ds.dim, hidden_dim=50)
model = AutoModel.from_config(cfg)

# Train
Trainer(model, ds, TeacherForcingObjective(alpha=0.1),
        TrainingArguments(max_steps=5000, learning_rate=1e-3)).train()

# Evaluate
orbit = model.generate(ds.test[0], n_steps=len(ds.test)).numpy()
print("D_stsp:", state_space_divergence(orbit, ds.test.numpy()))
print("D_H:   ", power_spectrum_error(orbit, ds.test.numpy()))

# Analyze (PLRNN -> automatic analytic backend)
fps = find_fixed_points(model)
```

### Custom Data Import

```python
from neuralrnn.data.custom_dataset import CustomDataset
from neuralrnn import Trainer, TrainingArguments, TeacherForcingObjective

# Load your own data
my_data = CustomDataset.from_npz("experiment_data.npz",
                                  sequence_length=200, batch_size=16, normalize=True)

# Or from arrays
import numpy as np
traj = np.load("neural_trajectory.npy")    # (T, N)
my_data = CustomDataset.from_arrays(traj, mode="timeseries", sequence_length=200)

# Train
Trainer(model, my_data, TeacherForcingObjective(alpha=0.1),
        TrainingArguments(max_steps=5000)).train()
```
