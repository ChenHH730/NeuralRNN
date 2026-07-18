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
   - [Constrained RNN Model (Paradigm A)](#constrained-rnn-model-paradigm-a)
   - [Multi-Area RNN Model (Paradigm A)](#multi-area-rnn-model-paradigm-a)
   - [Gain RNN Family (gain_rnn / stp_rnn)](#gain-rnn-family-gain_rnn--stp_rnn)
   - [Short-Term Plasticity RNN (Paradigm A)](#short-term-plasticity-rnn-paradigm-a)
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
   - [Fixed-Point Hyperparameter Guide](fixed_point.md)
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
    Objective, SupervisedObjective, RegularizedSupervisedObjective,
    TeacherForcingObjective, BehavioralObjective, VariationalObjective,
    LatentCircuitObjective, ConstrainedSupervisedObjective,
    build_objective, register_objective, OBJECTIVE_REGISTRY,
    masked_mse, masked_cross_entropy, masked_nll, loss_mse,
    activity_l2, weight_l2, weight_l1,
    orthogonality_penalty, model_orthogonality_penalty,
    accuracy_classification, accuracy_general,
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
        activation: str = "relu",   # Nonlinearity name (see neuralrnn.activations.SUPPORTED_ACTIVATIONS)
        freeze_input: bool = False,      # Freeze input-layer parameters
        freeze_recurrent: bool = False,  # Freeze recurrent / hidden parameters
        freeze_output: bool = False,     # Freeze output / readout parameters
        freeze_h0: bool = False,         # Freeze initial-state parameters
        **kwargs: Any,              # Forward-compat: unknown keys stored as attrs
    ) -> None: ...
```

The four `freeze_*` flags provide a convenient way to implement echo-state / reservoir-computing training. They are automatically serialized with the config and respected by all built-in models.

**Freezing has two layers.** Layer 1 is the unified vocabulary: the four `freeze_*` flags plus
`model.freeze_parameters(groups=...)/unfreeze_parameters(...)` (per-model `_freeze_groups` map
`input`/`recurrent`/`output`/`h0` — and for the gain_rnn family also
`gains`/`biases`/`stp` — to parameter regexes; gain_rnn-family configs add matching
`freeze_gain`/`freeze_bias`/`freeze_stp` flags). Layer 2 is family-specific flags that only
exist where Layer 1 cannot express the semantics — e.g. CTRNN-family `trainable_h0`
(structural: `False` makes h0 a buffer, not a parameter at all) and low-rank
`train_wi`/`train_wo` (selectors choosing between the weight and its per-channel scaling, not
freeze flags). Rule: **`freeze_*` always wins** — models apply family flags first and
`apply_freeze_config()` last.

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

### Euler Discretization: `alpha` / `dt` / `tau`

All continuous-time model configs expose the same three knobs for the Euler update
`z_t = (1 - alpha) * z_{t-1} + alpha * f(pre)`, resolved by
`neuralrnn.resolve_euler_alpha(dt, tau, alpha, *, default_dt, model_type)` with a single
deterministic priority:

1. **`alpha` given explicitly** → used directly (highest priority). If `dt` is also given and
   `alpha != dt/tau`, a `UserWarning` is emitted and `alpha` wins.
2. **`alpha=None`, `dt` given** → `alpha = dt / tau`.
3. **both None** → family default: `dt = default_dt` and `alpha = default_dt / tau`
   (with `default_dt=None` the model is fully discrete, `alpha = 1.0`).

Validation: `tau <= 0` and `alpha <= 0` raise `ValueError`; `alpha > 1` warns (Euler may be unstable).
The resolved `alpha` and effective `dt` are stored on the config and serialized to `config.json`;
**models always read `config.alpha`** and never recompute it. Family defaults:

| Family | `default_dt` | `tau` | resolved default `alpha` |
|---|---|---|---|
| `ctrnn` / `ei_rnn` / `constrained_rnn` (and `se_rnn`, `sparse_rnn`, `modular_rnn`, `multiarea_rnn`) | 100.0 | 100.0 | 1.0 |
| `gain_rnn` | 100.0 | 100.0 | 1.0 |
| `stp_rnn` | 10.0 | 100.0 | 0.1 |
| `latent_circuit` | 40.0 | 200.0 | 0.2 |
| `lowrank_rnn` | 20.0 | 100.0 | 0.2 |

Old checkpoints without an `alpha` field re-resolve from their stored `dt`/`tau` and are fully
compatible. Discrete-map models (`shallow_plrnn`, `dend_plrnn`, `alrnn`) and gated models
(`tiny_rnn`) have no Euler step and do not expose `alpha`.

### Nonlinearity Placement: `nonlinearity_mode`

CTRNN-lineage configs select **where the nonlinearity `f` sits in the Euler step** via the
unified `nonlinearity_mode` field (`neuralrnn.SUPPORTED_NONLINEARITY_MODES`). With
`pre = W@state + B@x + b` (recurrent noise added on `pre`):

| Mode | Update rule | Meaning |
|---|---|---|
| `"pre_activation"` | `z' = (1-α)z + α·f(pre)` | standard Euler: state relaxes toward `f(pre)` |
| `"post_blend"` | `z' = f((1-α)z + α·pre)` | nn-brain / Masse-style: leak blended **inside** `f` |
| `"rate"` | `r = f(z)`; `z' = (1-α)z + α·(W@r + B@x + b)` | classic firing-rate form: state is a current, `f` maps it to the rate that drives the recurrence |

Family defaults (defaults never change existing behavior):

| Family | default `nonlinearity_mode` |
|---|---|
| `ctrnn` / `ei_rnn` / `constrained_rnn` / `se_rnn` / `sparse_rnn` / `modular_rnn` / `multiarea_rnn` / `latent_circuit` | `"pre_activation"` |
| `gain_rnn` | `"rate"` |
| `stp_rnn` | `"post_blend"` |
| `lowrank_rnn` | `"rate"` |

Key distinctions to be aware of:

- **`post_blend` vs `rate`** share the same skeleton (one `f` per step, `W` always reads the
  rate) and differ only in the **leak**: decay acts on the rate (`post_blend`) or on the
  current (`rate`). They coincide exactly at `alpha = 1`; with `relu` they differ in
  subthreshold memory — `rate` keeps negative currents (units can sit below threshold and
  take time to re-activate), `post_blend` rectifies them away every step.
- **Noise rectification**: in `pre_activation` and `post_blend` the noise on `pre` passes
  through `f` (rectified for relu); in `rate` mode noise enters the blend unrectified.
- **`rate` mode details (CTRNN family)**: `h2h.bias` stays in `pre` (not inside `f`), and the
  readout still reads the state `z` (not `f(z)`). `lowrank_rnn` keeps its own family traits in
  all modes: bias `b` inside `f` in `"rate"` mode (`r = f(z + b)`, moving into the drive in the
  other modes), noise added **after** the Euler update, and `output_activation` on readout.
  `latent_circuit` always scales noise as `sqrt(2α)·σ` on `pre`.
- **Fixed points / Jacobians differ across modes** (e.g. `pre_activation` FP: `z = f(Wz+Bx+b)`;
  `rate` FP: `z = W·f(z)+Bx+b`). All analysis modules go through `recurrence`, so they
  automatically reflect the configured mode.

Validation: unknown mode strings raise `ValueError` at config construction. The mode is stored
on the config and serialized to `config.json`; **models always read `config.nonlinearity_mode`**.

Scope: discrete-map models (`shallow_plrnn`, `dend_plrnn`, `alrnn`) and gated models
(`tiny_rnn`) have no Euler step and do not expose this field. The connectome-constrained
paradigm (Beiran & Litwin-Kumar 2025) is structurally `"rate"` (firing rates
`gain·f(z + bias)` computed before the recurrent mix); the `gain_rnn` family (below)
implements it by composing the three modes with per-neuron gain/bias placement.

---

## Activation Functions

All built-in models select their nonlinearities through `neuralrnn.activations.get_activation(name, **kwargs)`:

```python
from neuralrnn.activations import get_activation, SUPPORTED_ACTIVATIONS

fn = get_activation("leaky_relu", negative_slope=0.1)
print(SUPPORTED_ACTIVATIONS)
```

Supported names: `relu`, `tanh`, `sigmoid`, `softplus` (with `beta`), `leaky_relu`/`leakyrelu` (with `negative_slope`), `elu` (with `alpha`), `selu`, `gelu`, `silu`/`swish`, `piecewise_tanh` (with `r0`, `rmax`: piecewise-saturating tanh with unit slope at the origin — the Stroud et al. 2018 gain function with the gain factored out; used with `gain_position="inside"`).

Each model keeps its original default activation, so existing configs and saved checkpoints are unaffected.

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

    def analytic_parameters(self, task_input: Tensor | None = None) -> dict[str, Tensor]:
        """Expose parameters needed by the analytic fixed-point solver (scy_fi).
        Only required when supports_analytic_fixed_points is True.
        Optional task_input allows folding constant external inputs into the effective bias."""

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
        dt: float | None = None,    # Euler step (None + alpha=None -> 100.0, i.e. alpha=1.0)
        tau: float = 100.0,         # Time constant
        alpha: float | None = None, # Euler update fraction; overrides dt/tau when given
        activation: str = "relu",   # Nonlinearity: relu, tanh, sigmoid, softplus,
                                    # leaky_relu/leakyrelu, elu, selu, gelu, silu/swish
        dale: bool = False,         # Dale constraint (E/I separation)
        ei_ratio: float = 0.8,      # Excitatory fraction (when dale=True)
        dale_signs: list[float] | None = None,  # Optional per-unit sign vector (+1 E / -1 I),
                                    # length latent_dim; implies dale=True and overrides
                                    # the global ei_ratio split (e.g. per-area 80/20 splits)
        trainable_h0: bool = False, # Trainable initial state
        sigma_rec: float = 0.0,     # Recurrent noise std (0 = off)
        noise_alpha_scaling: bool = False,  # True: noise std sqrt(2*alpha*sigma^2)
        nonlinearity_mode: str = "pre_activation",  # pre_activation | post_blend | rate
        **kwargs,
    ) -> None: ...
```

See *Nonlinearity Placement: `nonlinearity_mode`* (§2) for the three update rules.

#### `VanillaRNNConfig`

```python
class VanillaRNNConfig(CTRNNConfig):
    model_type = "vanilla_rnn"
    # Defaults: dt=None (discrete, resolves to alpha=1.0), all other params inherited
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

    # Recurrence (default "pre_activation" mode):
    #   h = (1-alpha)*h + alpha * f(W_x*x + W_r*h + b)
    # "post_blend": h = f((1-alpha)*h + alpha*pre); "rate": r = f(h), h = (1-alpha)*h + alpha*(W_r*r + W_x*x + b)
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

#### `DendPLRNNConfig`

```python
class DendPLRNNConfig(ShallowPLRNNConfig):
    model_type = "dend_plrnn"
    def __init__(
        self,
        n_bases: int = 20,          # Number of spline bases B per latent unit
        use_clipping: bool = False, # Hard-clip basis expansion for bounded orbits
        clip_range: float | None = None,
        **kwargs,
    ) -> None: ...
```

#### `DendPLRNNModel`

```python
@register_model("dend_plrnn")
class DendPLRNNModel(NeuralDynamicsModel):
    config_class = DendPLRNNConfig

    # Recurrence:
    # z = A*z + W @ sum_b alpha_b ReLU(z - H_b) + h [+ C @ s]
    def recurrence(self, x_t, z_prev, *, inputs=None) -> Tensor: ...
    def readout(self, z_t) -> Tensor: ...  # Identity

    # Analytic support
    supports_analytic_fixed_points = True
    def jacobian(self, z, *, inputs=None) -> Tensor:
        """J(z) = diag(A) + W @ diag(sum_b alpha_b 1[z > H_b])"""
    def analytic_parameters(self) -> dict[str, Tensor]:
        """Returns effective {"A", "W1", "W2", "h1", "h2"} for the analytic solver."""
```

#### `ALRNNConfig`

```python
class ALRNNConfig(ShallowPLRNNConfig):
    model_type = "alrnn"
    def __init__(
        self,
        n_linear: int = 1,          # Number of linear units; P = latent_dim - n_linear
        use_clipping: bool = False,
        clip_range: float | None = None,
        **kwargs,
    ) -> None: ...
```

#### `ALRNNModel`

```python
@register_model("alrnn")
class ALRNNModel(NeuralDynamicsModel):
    config_class = ALRNNConfig

    # Recurrence:
    # z = A*z + W @ Phi*(z) + h [+ C @ s]
    # Phi*(z) = [z_1, ..., z_{M-P}, ReLU(z_{M-P+1}), ..., ReLU(z_M)]
    def recurrence(self, x_t, z_prev, *, inputs=None) -> Tensor: ...
    def readout(self, z_t) -> Tensor: ...  # Identity

    # Analytic support
    supports_analytic_fixed_points = True
    def jacobian(self, z, *, inputs=None) -> Tensor:
        """J(z) = diag(A) + W @ diag([1_{M-P}; 1[z_{-P:}>0]])"""
    def analytic_parameters(self) -> dict[str, Tensor]:
        """Returns effective {"A", "W1", "W2", "h1", "h2"} for the analytic solver."""
```

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
        alpha: float | None = None,            # Euler update fraction; overrides dt/tau when given
        sigma_rec: float = 0.05,               # Recurrent Gaussian noise std
        dt: float | None = None,              # Integration step (None + alpha=None -> 20.0)
        tau: float = 100.0,                   # Membrane time constant
        add_bias: bool = False,               # Whether b is trainable
        scale_by_hidden_size: bool = True,    # Divide rec/output terms by N
        activation: str = "tanh",             # Hidden activation: relu, tanh, sigmoid,
                                              # softplus, leaky_relu/leakyrelu, elu, selu,
                                              # gelu, silu/swish
        output_activation: str = "tanh",      # Readout activation: same supported names
        nonlinearity_mode: str = "rate",      # rate (native) | pre_activation | post_blend
        train_wi: bool = True,                # Selector: train wi (si frozen) or train si (wi frozen)
        train_wo: bool = True,                # Selector: train wo (so frozen) or train so (wo frozen)
        **kwargs,
    ) -> None: ...
```

**Freezing** uses the framework-wide `freeze_*` flags: `freeze_recurrent` freezes the low-rank
`m, n` factors, `freeze_h0` freezes the initial state. Note the family default **`freeze_h0=True`**
(the original code never trained h0) — pass `freeze_h0=False` to train it. The removed flags
`train_wrec`/`train_h0` are still accepted as deprecated aliases (mapped to `freeze_recurrent`/
`freeze_h0` with inverted logic, `DeprecationWarning`), so old checkpoints keep working.

**Parameter guide**:
- `rank`: Determines the dimensionality of the recurrent subspace. Rank 1 = line attractor, Rank 2 = plane with rotational dynamics, Rank 3+ = more complex manifolds.
- `scale_by_hidden_size` (default `True`): Matches both original codebases. When True, the recurrent term is divided by `N` and the output term is also divided by `N`.
- `activation` (default `"tanh"`): The original papers use tanh for both hidden and output activation.
- `nonlinearity_mode` (default `"rate"`): the family's native form `r = f(z + b)` with the bias
  inside `f`; in `"pre_activation"`/`"post_blend"` the bias moves into the drive term. Noise is
  always added after the Euler update, and the reference forward's step-0 no-bias rate
  asymmetry exists only in `"rate"` mode. See *Nonlinearity Placement* (§2).

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
| `sigma_rec` | `float` | Noise standard deviation |

**Usage example** (see `notebook/07_lowrank_RNN_paradigmA.ipynb` for the full tutorial):

```python
from neuralrnn import AutoConfig, AutoModel
from neuralrnn.models.lowrank import LowrankRNNConfig, LowrankRNNModel
from neuralrnn.data.tasks import rdm_trials, lr_mante_trials

# ── Create model ──
cfg = LowrankRNNConfig(input_dim=1, latent_dim=256, output_dim=1,
                        rank=1, alpha=0.2, sigma_rec=0.05)
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
- `hidden_size`, `input_size`, `output_size`, `rank`, `alpha`, `sigma_rec` and the
  alias `non_linearity` are exposed for backward compatibility with reference analysis code.
- `print(model)` now shows config fields and parameter shapes.

---

### Constrained RNN Model (Paradigm A)

**Module**: `neuralrnn.models.constrained_rnn`

CTRNN with hard structural masks on input/recurrent/output weights, plus three common constrained variants:
spatially-embedded RNN (`se_rnn`), sparse RNN (`sparse_rnn`), and modular RNN (`modular_rnn`).

#### `ConstrainedRNNConfig`

```python
class ConstrainedRNNConfig(CTRNNConfig):
    model_type = "constrained_rnn"

    def __init__(
        self,
        input_dim: int = 3,
        latent_dim: int = 64,
        output_dim: int = 3,
        dt: float | None = None,
        tau: float = 100.0,
        alpha: float | None = None,   # overrides dt/tau when given
        activation: str = "relu",
        rec_mask: list | np.ndarray | None = None,   # (M, M) recurrent mask
        in_mask: list | np.ndarray | None = None,    # (input_dim, M) input mask
        out_mask: list | np.ndarray | None = None,   # (M, output_dim) output mask
        nonlinearity_mode: str = "pre_activation",   # pre_activation | post_blend | rate
        **kwargs,
    ) -> None: ...
```

#### `SERNNConfig`

```python
class SERNNConfig(ConstrainedRNNConfig):
    model_type = "se_rnn"

    def __init__(
        self,
        grid_shape: tuple | list | None = None,      # e.g. (5, 5, 4) for 100 units in 3D
        embedding_dim: int | None = 3,               # 2 or 3
        distance_power: float = 1.0,
        se1_weight: float = 0.5,                     # spatial L1 coefficient
        comms_factor: float = 1.0,                   # communicability exponent (0 to disable)
        distance_metric: str = "euclidean",
        orthogonal_init: bool = True,
        **kwargs,
    ) -> None: ...
```

#### `SparseRNNConfig`

```python
class SparseRNNConfig(ConstrainedRNNConfig):
    model_type = "sparse_rnn"

    def __init__(
        self,
        sparsity: float = 0.1,                       # fraction of recurrent connections kept
        allow_self_connections: bool = False,
        seed: int | None = 42,
        **kwargs,
    ) -> None: ...
```

#### `ModularRNNConfig`

```python
class ModularRNNConfig(ConstrainedRNNConfig):
    model_type = "modular_rnn"

    def __init__(
        self,
        n_modules: int = 4,
        p_inter: float = 0.05,                       # inter-module connection probability
        intra_density: float = 1.0,                  # intra-module connection density
        allow_self_connections: bool = False,
        seed: int | None = 42,
        **kwargs,
    ) -> None: ...
```

#### `ConstrainedRNNModel`

```python
@register_model("constrained_rnn")
class ConstrainedRNNModel(CTRNNModel):
    config_class = ConstrainedRNNConfig

    # Recurrence applies rec_mask/in_mask; readout applies out_mask
    def recurrence(self, x_t, z_prev, *, inputs=None) -> Tensor: ...
    def readout(self, z_t) -> Tensor: ...

    def constraint_loss(self) -> Tensor: ...        # base returns 0; se_rnn overrides
```

#### `SERNNModel` / `SparseRNNModel` / `ModularRNNModel`

```python
@register_model("se_rnn")
class SERNNModel(ConstrainedRNNModel):
    config_class = SERNNConfig
    # constraint_loss() returns distance-weighted L1 + optional communicability
    def get_neuron_positions(self) -> np.ndarray: ...

@register_model("sparse_rnn")
class SparseRNNModel(ConstrainedRNNModel): ...

@register_model("modular_rnn")
class ModularRNNModel(ConstrainedRNNModel): ...
```

### Multi-Area RNN Model (Paradigm A)

Cascaded multi-area CTRNN (Kleinman et al. 2025, eLife; see
`docs/papers/multi-area_rnn.md`). A thin Contract-A adapter over
`ConstrainedRNNModel`: all dynamics are inherited; the family only generates
block-structured masks (dense intra-area, sparse excitatory-source inter-area
split into feedforward/feedback) and a per-area Dale sign vector. Weight
convention: `rec_mask[target, source]`; inter-area blocks connect only
adjacent areas and originate exclusively from the source area's E units.

#### `MultiAreaRNNConfig`

```python
class MultiAreaRNNConfig(ConstrainedRNNConfig):
    model_type = "multiarea_rnn"

    def __init__(
        self,
        area_sizes: tuple | list = (100, 100, 100),  # latent_dim = sum(area_sizes)
        ei_ratio: float = 0.8,          # E fraction *within each area*
        intra_density: float = 1.0,     # intra-area recurrent density
        ff_ee_density: float = 0.10,    # feedforward E->E density (adjacent areas)
        ff_ei_density: float = 0.02,    # feedforward E->I density
        fb_density: float = 0.05,       # feedback E->E density (0 = pure cascade)
        fb_ei_density: float = 0.0,     # feedback E->I density
        input_areas: tuple | list = (0,),  # areas receiving external input
        input_e_only: bool = False,
        output_area: int = -1,          # readout area (negative = from the end)
        output_e_only: bool = True,     # readout only from the output area's E units
        allow_self_connections: bool = True,
        rec_spectral_radius: float | None = 1.0,  # rescale effective |W|@dale
                                    # spectral radius at init (None = disable);
                                    # default framework init explodes under Dale
        spectral_norm_iters: int = 100,
        mask_seed: int = 42,
        **kwargs,
    ) -> None: ...
```

#### `MultiAreaRNNModel` and mask utilities

```python
@register_model("multiarea_rnn")
class MultiAreaRNNModel(ConstrainedRNNModel):
    config_class = MultiAreaRNNConfig
    area_slices: list[slice]            # per-area unit-index slices

    def area_states(self, states, area: int) -> Tensor: ...  # slice (..., M) to one area

# Pure mask builder (also used for the manual constrained_rnn construction path):
def build_multiarea_masks(area_sizes, input_dim, output_dim, ...) \
        -> (rec_mask, in_mask, out_mask, dale_signs): ...
def area_slices(area_sizes) -> list[slice]: ...
def area_ei_indices(area_sizes, ei_ratio) -> (dale_signs, e_indices, i_indices): ...
def rescale_effective_spectral_radius(model, target, iters=100): ...
    # Works on any ConstrainedRNNModel (manual-mask path uses it explicitly).
```

Masks are regenerated deterministically from `mask_seed` on load (same pattern
as `sparse_rnn` / `modular_rnn`); `area_sizes` etc. are serialized in
`config.json`.
```

#### `ConstrainedSupervisedObjective`

**Module**: `neuralrnn.train.objectives.constrained`

```python
class ConstrainedSupervisedObjective(SupervisedObjective):
    def __init__(self, task_type: str = "classification", constraint_weight: float = 0.0): ...
```

Adds `constraint_weight * model.constraint_loss()` to the supervised task loss. Use with `se_rnn`;
for hard-masked sparse/modular RNNs the standard `SupervisedObjective` is sufficient.

---

### Gain RNN Family (gain_rnn / stp_rnn)

**Module**: `neuralrnn.models.gain_rnn`

Unified "fixed weights + neuronal gain modulation" paradigm: per-neuron gains/biases
parameterize the firing-rate map while connectivity can be hard-masked and/or frozen.
The family implements the connectome-constrained teacher–student paradigm
(Beiran & Litwin-Kumar 2025, gain outside the nonlinearity — see
`notebook/16_connectome_rnn_paradigmB.ipynb`), covers the gain-modulation model of
Stroud et al. 2018 (gain inside), and hosts `stp_rnn`, where short-term plasticity
acts as a *dynamic* gain parameterization (Masse et al. 2019 / Zhou & Buonomano 2024).
See `docs/papers/gain_rnn.md` for the theory. (The legacy `connectome_rnn` model class
was removed in favor of this family.)

Inheritance: `StpRNNModel → GainRNNModel → ConstrainedRNNModel → CTRNNModel`, so all
CTRNN-lineage features (masks, Dale, `nonlinearity_mode`, Euler resolution) apply.

#### Rate map and gain placement

Every nonlinearity application goes through `rate_map(u)` (default gain=1, bias=0 ⇒ plain
activation, i.e. exact ConstrainedRNN behavior):

| `gain_position` | `rate_map(u)` | Semantics |
|---|---|---|
| `"outside"` (default) | `gain * act(u + bias)` | output gain ≡ presynaptic column scaling of the recurrent matrix (amplitude modulation; Beiran & Litwin-Kumar) |
| `"inside"` | `act(gain * u + bias)` | input gain ≡ slope modulation, saturation unchanged (Stroud et al.; use `activation="piecewise_tanh"`) |

The bias always sits inside `act`. Note: for ReLU with positive gains the two placements
are exactly equivalent (scale degeneracy — regularize if both placements are trained);
negative gains under Dale constraints flip a neuron's E/I identity.

Composition with `nonlinearity_mode`: `"rate"` (family default — state is a current, the
recurrence and the **readout** consume `rate_map(z)`), `"post_blend"`, `"pre_activation"`.

**Family deviations from the CTRNN base (documented and tested):**
- In `"rate"` mode the readout reads `rate_map(z)` instead of the raw state (connectome
  behavior; CTRNN's rate mode reads the state).
- `recurrence` accepts `x_t=None` for autonomous rollout (input term skipped entirely).
- `noise_position="post"` adds recurrent noise to the leaked blend before the final
  nonlinearity (state-level noise) instead of to the pre-activation; the std still follows
  `noise_alpha_scaling` (`sqrt(2α)·σ` when True).

#### `GainRNNConfig`

```python
class GainRNNConfig(ConstrainedRNNConfig):
    model_type = "gain_rnn"
    # Family defaults: nonlinearity_mode="rate", activation="softplus"
    def __init__(
        self,
        gain_position: str = "outside",        # outside | inside
        gain_init: float | array = 1.0,        # scalar or (M,)
        bias_init: float | array = 0.0,
        freeze_gain: bool = False,             # Layer-1 freeze vocabulary extensions
        freeze_bias: bool = False,
        noise_position: str = "pre",           # pre (CTRNN behavior) | post (state-level)
        positive_input_weights: bool = False,  # ReLU on input weights in forward
        positive_output_weights: bool = False, # ReLU on readout weights in forward
        activation_params: dict | None = None, # kwargs for get_activation (e.g. {"beta": 1.0})
        h0_init: float | array = 0.0,          # initial h0 value at construction
        **kwargs,                              # all ConstrainedRNN/CTRNN fields
    ) -> None: ...
```

#### `GainRNNModel`

```python
@register_model("gain_rnn")
class GainRNNModel(ConstrainedRNNModel):
    config_class = GainRNNConfig

    def rate_map(self, u) -> Tensor: ...          # parameterized firing-rate map
    def firing_rate(self, z) -> Tensor: ...       # alias (firing-rate API naming)
    def get_firing_rates(self, states) -> Tensor: ...
    def recurrence(self, x_t, z_prev, *, inputs=None) -> Tensor: ...
    def readout(self, z_t) -> Tensor: ...         # rate mode: from rate_map(z)
```

Freeze groups add `gains` (`^gain$`) and `biases` (`^bias$`) to the base four. Reproduction
mappings: connectome student = `rate` + `outside` + `softplus` + `freeze_input/recurrent/
output` + copying the shared J into `h2h.weight` (per-neuron teacher input via an identity
`input2h`); Stroud 2018 = `rate` + `inside` + `activation="piecewise_tanh"`.

#### `StpRNNConfig`

Short-term plasticity as a dynamic gain: the effective presynaptic gain is
`syn_x * syn_u` (Tsodyks–Markram rate form, cell-specific per presynaptic neuron):

```
syn_x' = syn_x + (dt/tau_x)(1 - syn_x) - dt_sec * syn_u * syn_x * r
syn_u' = syn_u + (dt/tau_u)(U_eff - syn_u) + dt_sec * U_eff * (1 - syn_u) * r
U_eff  = clamp(stp_alpha * U, 0, 1);      rec_in = r * syn_x' * syn_u'
```

`dt` in ms, `dt_sec = dt/1000` (release terms assume rates in spikes/s). STP dynamics use
the physical `dt` and ignore an explicit `alpha` override, so `dt` is required (the model
raises when `config.dt is None`). Family defaults reproduce the notebook-11 model
(Masse 2019 style): `post_blend` + `relu` + `noise_position="post"` +
`noise_alpha_scaling=True` + positive input/output weights + `h0_init=0.1` + frozen static
gain/bias + `dt=10, tau=100`.

```python
class StpRNNConfig(GainRNNConfig):
    model_type = "stp_rnn"
    def __init__(
        self,
        stp_tau_x: float | array = 200.0,      # depression tau (ms); scalar or (M,)
        stp_tau_u: float | array = 1500.0,     # facilitation tau (ms)
        stp_U: float | array = 0.2,            # baseline release probability
        stp_init: str = "constant",            # constant | alternating | random
        tau_x_fac=1500.0, tau_u_fac=200.0, U_fac=0.15,      # "alternating" (nb11)
        tau_x_dep=200.0, tau_u_dep=1500.0, U_dep=0.45,
        stp_U_mean=0.5, stp_U_std=0.17, stp_U_min=0.001, stp_U_max=0.99,   # "random" (Zhou 2024)
        stp_tau_mean=1000.0, stp_tau_std=330.0, stp_tau_min=100.0, stp_tau_max=3000.0,
        stp_seed: int | None = None,
        stp_alpha: float | array = 1.0,        # neuromodulator cue (runtime buffer)
        freeze_stp: bool = True,               # both reference papers never train STP
        init_method: str = "default",          # "gamma" = notebook-11 gamma init
        gamma_shape_exc=0.1, gamma_shape_inh=0.2, gamma_scale=1.0,
        init_seed: int | None = None,
        **kwargs,
    ) -> None: ...
```

Explicit arrays for `stp_tau_x/stp_tau_u/stp_U` take precedence over `stp_init`.
`"alternating"` = notebook 11 (even indices facilitating, odd depressing); `"random"` =
Zhou & Buonomano 2024 truncated normals (tau_x/tau_u sampled independently).

#### `StpRNNModel`

```python
@register_model("stp_rnn")
class StpRNNModel(GainRNNModel):
    config_class = StpRNNConfig

    # Latent state is [h, syn_x, syn_u] (3M). forward returns states = h only
    # (B,T,M) and extras = {"syn_x", "syn_u"}.
    def recurrence(self, x_t, z_prev, *, inputs=None):
        """(B,3M) -> (B,3M); analysis fallback (B,M) treats syn_x=syn_u=1 -> (B,M)."""
    def init_state(self, batch_size, device) -> Tensor: ...   # [h0, 1, U_eff] steady state
    def set_stp_alpha(self, value) -> None: ...   # trial-level cue; scalar or (M,)
    def get_stp_alpha(self) -> Tensor: ...
    @staticmethod
    def synaptic_efficacy(extras) -> Tensor: ...  # syn_x * syn_u
    def forward_with_dropout(self, inputs, ...): ...  # h-only dropout override
```

Additional freeze group `stp` (`stp_tau_x`/`stp_tau_u`/`stp_U`). Notebook-11 parity
details: `input2h.bias` is zeroed and frozen (the reference model has no input bias);
`init_method="gamma"` reproduces the reference weight init. Helper
`make_stp_masks(latent_dim, output_dim, ei_ratio, no_self_connections, readout_e_only)`
builds the rec/out masks (zero diagonal, E-only readout) in the framework's conventions.

**Analysis caveat**: fixed-point / linearization tools call `recurrence` with M-dim
states, so they analyze the frozen-efficacy (syn_x = syn_u = 1) M-dim map, not the full
3M system — interpret fixed points accordingly. `forward(initial_state=...)` requires the
full (B, 3M) state and raises otherwise. `stp_alpha` is a runtime buffer: the config field
is only its initial value, and `save/load` restores the buffer from the checkpoint.

Reproduction mappings: notebook 11 = defaults + `stp_init="alternating"` +
`make_stp_masks` + `dale=True` + `init_method="gamma"`; Zhou & Buonomano 2024 =
`nonlinearity_mode="rate"` + `stp_init="random"` + `positive_output_weights=False` +
per-trial `model.set_stp_alpha(alpha)`.

---

### Short-Term Plasticity RNN (Paradigm A)

**Status**: Migrated — the notebook-11 inline model is now part of the gain_rnn family as
`model_type="stp_rnn"` (`neuralrnn.models.gain_rnn.StpRNNModel`, defaults reproduce the
notebook-11 math). See *Gain RNN Family* above and `docs/papers/stp_rnn.md`.

STP-RNN from Masse et al. (2019). Continuous-time ReLU RNN with Dale constraints and
per-neuron short-term synaptic plasticity (facilitating / depressing). Used to study
activity-silent maintenance (DMS) vs. manipulation-driven persistent activity (DRMS,
ABBA). See `notebook/11_STP_RNN_paradigmA.ipynb` for the original reproduction (its
inline class still shadows the registry inside that notebook; its old checkpoints are
incompatible with the new src class and were already marked for deletion).

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
        dt: float | None = None,  # Discretization step in ms (None + alpha=None -> 40 ms)
        tau: float = 200.0,         # Time constant (ms), alpha = dt/tau
        alpha: float | None = None, # Euler update fraction; overrides dt/tau when given
        sigma_rec: float = 0.15,    # Recurrent noise std
        activation: str = "relu",   # Recurrence nonlinearity: relu, tanh, sigmoid,
                                    # softplus, leaky_relu/leakyrelu, elu, selu, gelu,
                                    # silu/swish. Connectivity masks in apply_constraints()
                                    # remain hard ReLU constraints regardless.
        nonlinearity_mode: str = "pre_activation",  # pre_activation | post_blend | rate
        **kwargs,
    ) -> None: ...
```

#### `LatentCircuitModel`

```python
@register_model("latent_circuit")
class LatentCircuitModel(NeuralDynamicsModel):
    config_class = LatentCircuitConfig

    # Recurrence (default "pre_activation" mode; noise always sqrt(2α)·σ on pre):
    #   x_t = (1-α) x_{t-1} + α f(w_rec x_{t-1} + w_in u_t + noise)
    # "post_blend": x_t = f((1-α)x + α·pre); "rate": r = f(x), x_t = (1-α)x + α·(w_rec r + w_in u)
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

### `TrialTimeseriesDataset`

**Module**: `neuralrnn.data.trial_dataset`

For Paradigm B task-state reconstruction where the data are naturally organized as independent trials. Preserves the `(n_trials, trial_length, n_variable)` structure and performs trial-wise train/test splitting, avoiding cross-trial leakage.

```python
class TrialTimeseriesDataset(BaseDataset):
    kind = "trial_timeseries"

    def __init__(
        self,
        inputs: np.ndarray | Tensor,                  # (n_trials, T, N)
        targets: np.ndarray | Tensor | None = None,   # (n_trials, T, N)
        external_inputs: np.ndarray | Tensor | None = None,  # (n_trials, T, K)
        batch_size: int = 16,
        normalize: bool = False,
        normalize_externals: bool = False,
        test_fraction: float = 0.0,
        seed: int = 0,
    ) -> None: ...

    def sample_batch(self) -> dict[str, Tensor]:
        """Returns {"inputs": (B,T,N), "targets": (B,T,N), "external_inputs": (B,T,K)|None}"""

    @property
    def test_set(self) -> "TrialTimeseriesDataset | None": ...

    @classmethod
    def from_arrays(cls, inputs, targets=None, external_inputs=None, **kwargs) -> "TrialTimeseriesDataset": ...
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

**Available task generators** (in `neuralrnn.data.tasks.TASK_REGISTRY`). For a full catalog including parameter defaults, timing, overlap analysis, and notebook usage, see [`src/neuralrnn/data/tasks/tasks.md`](../../src/neuralrnn/data/tasks/tasks.md).

| Key | Task | Input | Output | Description |
|-----|------|-------|--------|-------------|
| `rdm` | Random Dot Motion | 1 | 1 | Integrate noisy coherence signal; report sign |
| `two_afc` | Two-Alternative Forced Choice | 1 | 1 | Backward-compatible alias for `rdm` |
| `raposo` | Multisensory Decision | 4 | 1 | Attend to visual/auditory/both modalities |
| `dms` | Delayed Match-to-Sample | 2 | 1 | Judge if two sequential A/B symbols match |
| `dms_continuous` | Delayed Match-to-Sample (continuous) | 4 | 2 | Match-to-sample with continuous coherences |
| `wm_angle` | Parametric WM (angle) | 2 | 2 | Remember and reproduce a circular angle |
| `parametric_wm` | Parametric WM (angle) | 2 | 2 | Backward-compatible alias for `wm_angle` |
| `wm_frequency` | Parametric WM (frequency) | 1 | 1 | Compare two frequencies across delay |
| `romo` | Parametric WM (frequency) | 1 | 1 | Backward-compatible alias for `wm_frequency` |
| `lr_mante` | Ctx Decision (low-rank) | 4 | 1 | Context-dependent integration of color/motion (low-rank format) |
| `mante` | Ctx Decision (Mante 2013) | 6 | 2 | Context-dependent decision with 2 readout channels |
| `siegel_miller` | Ctx Decision (Siegel 2015) | 6 | 2 | Backward-compatible alias for `mante` |
| `multitask_yang` | Yang 20-task set | 85 | 33 | 20 simultaneous cognitive tasks with one-hot rule input |
| `multitask_flexible` | Driscoll 15-task set | 20 | 3 | 15 flexible tasks with 2D circular stimuli/responses |
| `checkerboard` | Checkerboard decision (Kleinman 2025) | 4 | 2 | Report majority color by reaching to the matching target; color/direction decisions independent; 10% catch trials |

**Multitask dataset wrappers**:
```python
from neuralrnn.data.tasks.multitask_yang_dataset import MultitaskYangDataset
from neuralrnn.data.tasks.multitask_flexible_dataset import MultitaskFlexibleDataset

# Yang et al. (2019) — contextdm1/contextdm2 oversampled 5x by default
yang_ds = MultitaskYangDataset(batch_size=64, rule_prob_map={"contextdm1": 5.0, "contextdm2": 5.0})

# Driscoll et al. (2024) — contextdelaydm1/contextdelaydm2 oversampled 5x by default
flex_ds = MultitaskFlexibleDataset(batch_size=64)
```

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

Objectives can be registered by name and built through the factory:

```python
from neuralrnn import build_objective, OBJECTIVE_REGISTRY, register_objective

obj = build_objective("supervised", task_type="classification")
```

The same factory is also exposed as ``AutoObjective.from_name`` for symmetry with ``AutoConfig`` / ``AutoModel``:

```python
from neuralrnn import AutoObjective

obj = AutoObjective.from_name("teacher_forcing", alpha=0.1)
```

#### `SupervisedObjective` (Paradigm A)

```python
class SupervisedObjective(Objective):
    def __init__(self, task_type: str = "classification"):
        """task_type: "classification" (CrossEntropy) or "regression" (MSE).

        Batch shapes:
          - inputs:  (B, T, K)
          - targets: (B, T)     for classification, integer class indices
          - targets: (B, T, O)  for regression, continuous values
          - mask:    (B, T) or (B, T, O), optional

        Decision tasks can use either form: integer labels with argmax accuracy,
        or signed continuous targets (e.g. +1 / -1) with accuracy_general().
        """
```

#### `RegularizedSupervisedObjective` (Paradigm A with regularizers)

```python
class RegularizedSupervisedObjective(SupervisedObjective):
    def __init__(
        self,
        task_type: str = "classification",
        activity_weight: float = 0.0,      # coefficient for E[h^2]
        weight_weight: float = 0.0,        # coefficient for L2 weight penalty
        weight_patterns: list[str] | None = None,
        weight_reduce: str = "mean",       # "mean" or "sum" over matched parameters
        ortho_weight: float = 0.0,         # coefficient for input/output orthogonality
        ortho_input_name: str = "input2h",
        ortho_output_name: str = "readout_layer",
        mse_reduce: str = "per_trial",     # "per_trial" (default) or "global"
        activity_reduce: str = "per_trial", # "per_trial" (default) or "global"
    )
```

Combines the supervised task loss with optional L2 activity, L2 weight, and input/output orthogonality penalties. Safe to use with models that do not expose the requested weight attributes (the orthogonality term returns 0).

The `*_reduce` options let the objective match reference-notebook conventions:
- `mse_reduce="global"` computes a single masked MSE over the whole batch (used in the latent-circuit and flexible-multitask notebooks).
- `activity_reduce="global"` ignores the loss mask and regularizes global mean firing rate (used in the same notebooks).
- `weight_reduce="sum"` applies the coefficient to the raw sum of squared weights (used in the flexible-multitask notebook).

#### `TeacherForcingObjective` (Paradigm B — PLRNN)

```python
class TeacherForcingObjective(Objective):
    def __init__(self, alpha: float = 0.1, forcing_interval: int | None = None):
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

#### `LatentCircuitObjective` (Paradigm A/B hybrid)

```python
class LatentCircuitObjective(Objective):
    def __init__(self, l_y: float = 1.0):
        """Fit low-dimensional circuit to high-dim RNN responses.
        Loss = MSE(output) + l_y * NMSE(x @ Q, y_rnn)."""
```

#### `ConstrainedSupervisedObjective` (Paradigm A with structural regularizer)

```python
class ConstrainedSupervisedObjective(SupervisedObjective):
    def __init__(self, task_type: str = "classification", constraint_weight: float = 0.0):
        """Adds constraint_weight * model.constraint_loss() to the supervised loss."""
```

### Loss Functions and Metrics

**Primary module**: `neuralrnn.train.losses`

Reusable loss functions, regularizers, and metrics. All loss functions accept either raw `torch.Tensor` or `DynamicsModelOutput` (which delegates arithmetic ops to `.outputs`).

#### Loss terms

```python
def masked_mse(output, target, mask=None, reduction="per_trial") -> Tensor:
    """Masked mean squared error.
    reduction: "per_trial" (mean MSE across trials) or "global" (total SE / total mask).
    """

def masked_cross_entropy(logits, targets, mask=None) -> Tensor:
    """Masked cross-entropy for (B,T,C) logits and (B,T) targets."""

def masked_nll(logits, targets, mask=None) -> Tensor:
    """Alias for masked_cross_entropy."""
```

`loss_mse` is retained as an alias for `masked_mse`.

#### Regularizers

```python
def activity_l2(states, mask=None, reduction="per_trial") -> Tensor:
    """Mean squared activity E[h^2].
    reduction: "per_trial" (masked mean per trial, then mean over trials)
               or "global" (ignore mask, mean over all elements).
    """

def weight_l2(model, patterns=None, reduction="mean") -> Tensor:
    """Squared L2 over matched parameters.
    reduction: "mean" (mean squared value) or "sum" (raw sum of squares).
    """

def weight_l1(model, patterns=None) -> Tensor:
    """Mean L1 over matched parameters."""

def orthogonality_penalty(input_weight, output_weight, normalize_columns=True) -> Tensor:
    """||B^T B - diag(B^T B)||_2 with B = normalize([W_in | W_out^T])."""

def model_orthogonality_penalty(model, input_name="input2h", output_name="readout_layer") -> Tensor:
    """Convenience wrapper; returns 0 if attributes are missing."""
```

#### Metrics

```python
def accuracy_classification(logits, targets, mask=None) -> Tensor:
    """Standard argmax accuracy, optionally masked."""

def accuracy_general(output, targets, mask) -> Tensor:
    """Sign-based accuracy for binary decision tasks."""
```

**Usage**:

```python
from neuralrnn.train.losses import masked_mse, activity_l2, accuracy_classification

loss = masked_mse(model_output, y_val, mask_val)
reg = activity_l2(states, mask_val)
acc = accuracy_classification(model_output, targets_val, mask_val)
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

#### `OriginalStyleFixedPointFinder`

PyTorch reimplementation of the TensorFlow `fixed-point-finder` algorithm used by Golub & Sussillo (2018) and in the `multitask/flexible_multitask` reference code. Useful as a drop-in alternative to `NumericFixedPointFinder` when Adam optimization fails to spread candidates along a ring attractor.

```python
class OriginalStyleFixedPointFinder:
    def __init__(
        self,
        n_candidates: int = 1000,    # Number of initial candidate states
        n_iters: int = 5000,         # Gradient-descent iterations
        initial_rate: float = 1.0,   # Initial adaptive learning rate
        decrease_factor: float = 0.95,  # LR multiplier when objective increases
        tol_q: float = 1e-9,         # Convergence threshold on q = 0.5*||F(z)-z||^2
        noise_scale: float = 0.05,   # Noise added to trajectory-based init states
        dedup_tol: float = 1e-2,     # Deduplication radius
        seed: int | None = None,
    ) -> None: ...

    def find(self, model: NeuralDynamicsModel, *,
             task_input: Tensor | None = None,
             init_states: Tensor | np.ndarray | None = None) -> FixedPointSet:
        """Adaptive-learning-rate gradient descent on 0.5*||F(z)-z||^2.
        If init_states are provided, n_candidates are sampled from them and
        Gaussian noise with std=noise_scale is added (mirroring FixedPointFinder.sample_states)."""
```

For a detailed guide on how each parameter affects fixed-point recovery (number, position, stability), see [Fixed-Point Hyperparameter Guide](fixed_point.md).

#### `AnalyticPLRNNFixedPointFinder`

```python
class AnalyticPLRNNFixedPointFinder:
    def __init__(self, max_order: int = 1, outer_it: int = 300,
                 inner_it: int = 100) -> None: ...

    def find(self, model: NeuralDynamicsModel, *,
             task_input: Tensor | None = None) -> FixedPointSet:
        """Exact enumeration of fixed points and k-cycles using PLRNN's
        piecewise-linear structure. Requires model.supports_analytic_fixed_points.
        Constant task_input is folded into the effective bias before solving."""
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
    backend: "auto" / "numeric" / "analytic" / "scipy".
    task_input: constant external input condition. For the analytic backend it is
                folded into the effective bias (e.g. h1 + C*s); for numeric/scipy
                it is held fixed during the search.
    max_order: highest cycle order for the analytic backend."""
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

def effective_dimensionality(
    X: np.ndarray,
    variance_threshold: float = 0.95,
) -> int:
    """Number of PCs needed to explain `variance_threshold` fraction of variance."""

@torch.no_grad()
def collect_states(model: NeuralDynamicsModel, dataset,
                   n_batches: int = 1) -> np.ndarray:
    """Run model on dataset batches, flatten to (N_points, M) for PCA/analysis."""
```

### Sequentiality Analysis

**Module**: `neuralrnn.analysis.sequentiality`

Tools for quantifying and visualizing neural sequences in task-trained RNNs. The Sequentiality Index and peak-time sorting follow Orhan & Ma (2019); the same helpers are reused for the Zhou et al. (2023) T+WM ramp-to-sequence analysis in `notebook/15_neural_sequence_paradigmA.ipynb`. `split_ei_weight_submatrices` is provided as a general E/I submatrix utility.

```python
def compute_sequentiality_index(
    states: np.ndarray,
    threshold: float = 0.1,
    window: int = 5,
    n_bins: int = 20,
) -> float:
    """Orhan & Ma Sequentiality Index.
    states: (N_trials, T, M) or (N_points, M).
    Returns mean SI across trials."""

def sort_neurons_by_peak_time(states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (peak_times, sort_idx) for each unit."""

def weight_profile_by_peak_order(
    weight: np.ndarray,
    peak_times: np.ndarray,
    max_lag: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean/sd recurrent weights as a function of peak-time order difference."""

def split_ei_weight_submatrices(
    weight: np.ndarray,
    ei_mask: Sequence[int] | np.ndarray,
) -> dict[str, np.ndarray]:
    """Split recurrent weight matrix into EE/EI/IE/II submatrices."""
```

### Cue-Time Decoding

Cue-time decoding is implemented **inline in the Zhou et al. (2023) section of `notebook/15_neural_sequence_paradigmA.ipynb`** rather than as a reusable public API. The notebook uses `scikit-learn` linear SVMs to decode elapsed time from delay-period population activity.

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
    dt: float | None = None,    # Sampling interval: divide result by dt for continuous-time exponent
) -> float:
    """Compute the maximal Lyapunov exponent via QR iteration on Jacobians.
    lambda_max > 0 indicates chaos. For discrete-time maps trained on data sampled
    at interval dt (e.g. Lorenz-63 with dt=0.01), pass dt to obtain the continuous-time
    exponent comparable to the ODE literature (Lorenz63 ≈ 0.906)."""
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

### Demixed PCA and Axis Alignment

**Module**: `neuralrnn.analysis.demixed`

Marginalization-based demixed PCA (Kobak et al. 2016, eLife 5:e10989;
official code: https://github.com/wielandbrendel/dPCA, reference copy at
`reference_project/analysis/dPCA`) for trial-aligned RNN
states, plus the axis-alignment utilities used by Kleinman et al. (2025) to
relate representational axes to inter-area weight structure. Pure numpy;
consumes arrays and the per-trial condition dicts from the data layer.
Centering is per neuron across trials and time (as in the official package);
axes come from the SVD of each marginalized condition-mean matrix (a
simplification of the official regularized encoder/decoder optimization —
equivalent when trial noise is ignored, as for RNN trajectories).

```python
def fit_dpca(states, conditions, variables=("direction", "color"), n_axes=1) -> DPCAResult:
    """states: (n_trials, T, M). conditions: per-trial dicts containing every
    key in `variables` (filter out catch trials beforehand). Returns unit axes
    and trial-count-weighted variance fractions per marginalization
    ("condition_independent", each variable, all interactions)."""

@dataclass
class DPCAResult:
    axes: dict[str, np.ndarray]         # marginalization -> (n_axes, M)
    variance_ratio: dict[str, float]    # marginalization -> variance fraction
    def transform(self, states, marginalization: str) -> np.ndarray: ...

def axis_overlap_matrix(axes_a, axes_b) -> np.ndarray:
    """Dot products between unit axis sets (partial orthogonalization, Fig. 4b)."""

def axis_svd_alignment(axis, W, n_random=100, seed=0) -> dict:
    """|cos| of an axis with each right singular vector of W (e.g. W21) plus a
    random-vector baseline (Fig. 4f)."""

def potent_null_projection(axis, W, rank: int) -> dict:
    """Squared-norm fractions of an axis in the top-`rank` potent space vs the
    null space of W (Fig. 4c)."""
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

### PLRNN Invariant Manifolds

**Module**: `neuralrnn.analysis.manifolds`

DetectingManifolds-style tracing of stable/unstable manifolds for PLRNN-family models (shallow, dend, ALRNN). Works on the effective shallowPLRNN parameters returned by `model.analytic_parameters(task_input)`.

```python
@dataclass
class ManifoldSegment:
    points: np.ndarray           # (n_points, M) support points
    center: np.ndarray           # (M,) anchor point
    basis: np.ndarray            # (manifold_dim, M) orthonormal basis
    eigenvalues: np.ndarray      # (manifold_dim,) local eigenvalues
    region_id: int | None = None
    is_stable: bool = True

@dataclass
class ManifoldTrace:
    segments: list[ManifoldSegment]
    fixed_point: np.ndarray | None = None
    is_stable: bool = True

class PLRNNManifoldTracer:
    def __init__(self, max_iter=10, n_samples=500, factor=0.1,
                 propagation_steps=100, variance_threshold=0.95, seed=None): ...

    def trace(self, A, W1, W2, h1, h2, z0, stable=True) -> ManifoldTrace: ...

def compute_manifold(model, fixed_point: np.ndarray,
                     task_input: Tensor | None = None,
                     stable: bool = True, **tracer_kwargs) -> ManifoldTrace:
    """Trace stable/unstable manifold of a PLRNN fixed point under constant task input."""
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
