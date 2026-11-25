"""Model base class (≈ transformers.PreTrainedModel).

Core abstraction (ARCHITECTURE §2.1): every model is a "discrete dynamical system with readout".
The only hard contract is that subclasses must implement two methods:
    recurrence(x_t, z_prev, *, inputs=None) -> z_t   # single-step transition F_θ
    readout(z_t) -> y_t                              # readout G_φ
Implementing these two methods is enough to plug into the unified trainer (train/) and analysis (analysis/).

Tensor shape convention (unified across the framework, batch-first):
    inputs : (batch, T, input_dim)   single-step x_t : (batch, input_dim)
    states : (batch, T, latent_dim)  single-step z_t : (batch, latent_dim)
    outputs: (batch, T, output_dim)
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from .configuration_utils import NeuralRNNConfig

WEIGHTS_FILE_NAME = "model.safetensors"
METADATA_FILE_NAME = "metadata.json"


@dataclass
class DynamicsModelOutput:
    """Unified model output container (≈ transformers.ModelOutput). Supports attribute access and dict unpacking.

    Supports arithmetic ops that delegate to the .outputs tensor for backward
    compatibility with reference code that expects raw tensors from forward().
    """

    outputs: torch.Tensor | None = None   # readout y_{1:T}     (B, T, output_dim)
    states: torch.Tensor | None = None    # latent trajectory z_{1:T}    (B, T, latent_dim)
    loss: torch.Tensor | None = None      # loss computed inside forward, if any
    extras: dict[str, Any] | None = None  # model-specific outputs (e.g., complete-state trajectories)

    def __getitem__(self, k):  # out["states"] or tensor indexing out[0, :, :]
        if isinstance(k, str):
            return getattr(self, k)
        # Delegate to .outputs for tensor-like indexing
        if self.outputs is not None:
            return self.outputs[k]
        raise KeyError(f"Cannot index {type(self).__name__} with {k}: outputs is None")

    # ---- Arithmetic ops delegate to .outputs for backward compat ----
    def __sub__(self, other):
        if isinstance(other, DynamicsModelOutput):
            return self.outputs - other.outputs
        return self.outputs - other

    def __rsub__(self, other):
        return other - self.outputs

    def __add__(self, other):
        if isinstance(other, DynamicsModelOutput):
            return self.outputs + other.outputs
        return self.outputs + other

    def __radd__(self, other):
        return other + self.outputs

    def __mul__(self, other):
        if isinstance(other, DynamicsModelOutput):
            return self.outputs * other.outputs
        return self.outputs * other

    def __rmul__(self, other):
        return other * self.outputs

    def __truediv__(self, other):
        if isinstance(other, DynamicsModelOutput):
            return self.outputs / other.outputs
        return self.outputs / other

    def __rtruediv__(self, other):
        return other / self.outputs

    def sign(self):
        """Element-wise sign of .outputs (tensor delegation)."""
        return self.outputs.sign()

    def squeeze(self, dim=None):
        """Squeeze .outputs (tensor delegation)."""
        return self.outputs.squeeze(dim) if dim is not None else self.outputs.squeeze()

    def mean(self, *args, **kwargs):
        """Mean of .outputs (tensor delegation)."""
        return self.outputs.mean(*args, **kwargs)

    def sum(self, *args, **kwargs):
        """Sum of .outputs (tensor delegation)."""
        return self.outputs.sum(*args, **kwargs)

    def pow(self, n):
        """Element-wise power of .outputs (tensor delegation)."""
        return self.outputs.pow(n)

    def detach(self):
        """Return a new DynamicsModelOutput with detached outputs and states."""
        return DynamicsModelOutput(
            outputs=self.outputs.detach() if self.outputs is not None else None,
            states=self.states.detach() if self.states is not None else None,
            loss=self.loss,
            extras=self.extras,
        )

    def detach_(self):
        """Detach outputs and states in-place."""
        if self.outputs is not None:
            self.outputs = self.outputs.detach()
        if self.states is not None:
            self.states = self.states.detach()
        return self

    @property
    def device(self):
        """Device of .outputs (None when outputs is None)."""
        return self.outputs.device if self.outputs is not None else None


class NeuralDynamicsModel(nn.Module):
    """Base class for all RNN / dynamical-system models.

    Subclass conventions:
        - Set `config_class = <Family>Config`
        - Decorate with `@register_model("<family>")` (see auto/modeling_auto.py)
        - In __init__(self, config) only read parameters from config to build submodules
        - Implement recurrence / readout (hard contract)
        - Analytic models can implement jacobian and set supports_analytic_fixed_points=True
    """

    config_class: type[NeuralRNNConfig] = NeuralRNNConfig

    def __init__(self, config: NeuralRNNConfig) -> None:
        super().__init__()
        self.config = config
        self._noise_override = None
        self._noise_generator = None
        self._strict_state = False

    @property
    def state_dim(self) -> int:
        """Width of the complete Markov state, which may exceed latent_dim."""
        p = next(self.parameters(), None)
        return self.init_state(1, p.device if p is not None else "cpu").shape[-1]

    def validate_state(self, state: torch.Tensor) -> None:
        """Reject activity-only vectors at complete-state analysis boundaries."""
        if state.ndim < 1 or state.shape[-1] != self.state_dim:
            raise ValueError(f"Expected complete state of width {self.state_dim}; "
                             f"got {tuple(state.shape)}. Use init_state or full_state_trajectory.")

    @property
    def recurrent_noise_enabled(self) -> bool:
        """Training uses noise; evaluation opts in through evaluate_noise."""
        return (self.training or self.evaluate_noise) if self._noise_override is None else self._noise_override

    @property
    def sigma_rec(self) -> float:
        """Noise magnitude, kept in config for serialization and runtime edits."""
        return self.config.sigma_rec

    @sigma_rec.setter
    def sigma_rec(self, value: float) -> None:
        import math
        if not math.isfinite(value) or value < 0:
            raise ValueError("sigma_rec must be finite and nonnegative")
        self.config.sigma_rec = float(value)

    @property
    def evaluate_noise(self) -> bool:
        """Whether configured recurrent noise is enabled in eval mode."""
        return self.config.evaluate_noise

    @evaluate_noise.setter
    def evaluate_noise(self, enabled: bool) -> None:
        if not isinstance(enabled, bool):
            raise ValueError("evaluate_noise must be bool")
        for module in self.modules():
            if isinstance(module, NeuralDynamicsModel):
                module.config.evaluate_noise = enabled
                # Keep serialized nested core configs consistent with runtime flags.
                if isinstance(getattr(module.config, 'core_config', None), dict):
                    module.config.core_config['evaluate_noise'] = enabled

    def _noise_std(self) -> float:
        """Standard deviation at the injection site, before subsequent transforms."""
        sigma = self.config.sigma_rec
        return sigma * (2 * getattr(self, 'alpha', 1.0)) ** 0.5 if self.config.noise_alpha_scaling else sigma

    def _add_recurrent_noise(self, value, position, *, noise=None):
        """Apply noise at exactly the configured location; zero noise is a no-op."""
        if (self.config.noise_position != position or self.config.sigma_rec == 0
                or not self.recurrent_noise_enabled):
            return value
        return value + self._noise_std() * (self._randn_like(value) if noise is None else noise)

    def _euler_update(self, pre, state, activation):
        """Shared Euler/nonlinearity/noise placement for CTRNN-lineage models."""
        pre = self._add_recurrent_noise(pre, 'pre')
        mode = self.config.nonlinearity_mode
        if mode == 'pre_activation':
            result = (1 - self.alpha) * state + self.alpha * activation(pre)
            result = self._add_recurrent_noise(result, 'post')
        else:
            result = (1 - self.alpha) * state + self.alpha * pre
            result = self._add_recurrent_noise(result, 'post')
            if mode == 'post_blend':
                result = activation(result)
        return self._add_recurrent_noise(result, 'state')

    def _randn_like(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.randn(tensor.shape, device=tensor.device, dtype=tensor.dtype,
                           generator=self._noise_generator)

    @contextmanager
    def noise_context(self, enabled: bool | None = None, *, generator=None):
        """Temporarily control process noise, including nested dynamics cores.

        None inherits the surrounding policy. Supply a device-matched
        torch.Generator to isolate process noise from the global RNG.
        """
        if enabled is not None and not isinstance(enabled, bool):
            raise ValueError("enabled must be True, False, or None")
        modules = [m for m in self.modules() if isinstance(m, NeuralDynamicsModel)]
        saved = [(m, m._noise_override, m._noise_generator) for m in modules]
        try:
            for m in modules:
                if enabled is not None:
                    m._noise_override = enabled
                if generator is not None:
                    m._noise_generator = generator
            yield self
        finally:
            for m, policy, rng in saved:
                m._noise_override, m._noise_generator = policy, rng

    @contextmanager
    def evaluation_context(self, *, noise=None, generator=None, strict_state=False):
        """Restore every module mode and parameter flag, even after an exception."""
        modes = [(m, m.training) for m in self.modules()]
        flags = [(p, p.requires_grad) for p in self.parameters()]
        states = [(m, m._strict_state) for m in self.modules()
                  if isinstance(m, NeuralDynamicsModel)]
        try:
            self.eval()
            for m, _ in states:
                m._strict_state = strict_state or m._strict_state
            with self.noise_context(noise, generator=generator):
                yield self
        finally:
            for m, mode in modes:
                m.training = mode
            for p, flag in flags:
                p.requires_grad_(flag)
            for m, strict in states:
                m._strict_state = strict

    def full_state_trajectory(self, inputs=None, *, initial_state=None, n_steps=None):
        """Differentiable packed trajectory (B,T,state_dim), excluding z0.

        Uses recurrence with the current noise policy. Existing forward().states
        retains its activity semantics. For deterministic collection use
        evaluation_context(noise=False).
        """
        if inputs is not None:
            if inputs.ndim != 3:
                raise ValueError("inputs must be (B,T,K)")
            batch, steps, _ = inputs.shape
            z = self.init_state(batch, inputs.device) if initial_state is None else initial_state
        else:
            if initial_state is None or n_steps is None:
                raise ValueError("Autonomous rollout requires initial_state and n_steps")
            z, steps = initial_state, n_steps
        self.validate_state(z)
        trajectory = []
        for t in range(steps):
            z = self.recurrence(None if inputs is None else inputs[:, t], z, inputs=inputs)
            self.validate_state(z)
            trajectory.append(z)
        return torch.stack(trajectory, dim=1) if trajectory else z[:, None, :][:, :0]

    # ====================== Parameter freezing support (ESN / reservoir computing) ======================
    def freeze_parameters(
        self,
        groups: str | list[str] | None = None,
        patterns: list[str] | None = None,
    ) -> list[str]:
        """Freeze parameters by generic group name(s) and/or regex patterns.

        Args:
            groups: Generic layer group(s) supported by this model, e.g.
                ``"input"``, ``"recurrent"``, ``"output"``, ``"h0"``.
            patterns: Optional list of regex patterns matched against full
                parameter names.

        Returns:
            Sorted list of parameter names whose ``requires_grad`` was set to False.
        """
        names = self._match_parameters(groups, patterns)
        self._set_requires_grad(names, False)
        return sorted(names)

    def unfreeze_parameters(
        self,
        groups: str | list[str] | None = None,
        patterns: list[str] | None = None,
    ) -> list[str]:
        """Unfreeze parameters previously frozen via ``freeze_parameters``."""
        names = self._match_parameters(groups, patterns)
        self._set_requires_grad(names, True)
        return sorted(names)

    def apply_freeze_config(self) -> list[str]:
        """Apply freeze flags stored in ``self.config``.

        Subclasses should call this at the end of ``__init__`` if they want
        config-level freezing to take effect automatically.
        """
        groups = []
        if getattr(self.config, "freeze_input", False):
            groups.append("input")
        if getattr(self.config, "freeze_recurrent", False):
            groups.append("recurrent")
        if getattr(self.config, "freeze_output", False):
            groups.append("output")
        if getattr(self.config, "freeze_h0", False):
            groups.append("h0")
        if not groups:
            return []
        return self.freeze_parameters(groups=groups)

    def _freeze_groups(self) -> dict[str, list[str]]:
        """Mapping from generic group names to regex patterns for this model.

        Override per model family. The default empty mapping means that only
        explicit ``patterns`` can be used.
        """
        return {}

    def _match_parameters(
        self,
        groups: str | list[str] | None,
        patterns: list[str] | None,
    ) -> set[str]:
        import re

        group_map = self._freeze_groups()
        if isinstance(groups, str):
            groups = [groups]
        matched: set[str] = set()
        for g in groups or []:
            if g not in group_map:
                available = list(group_map.keys())
                raise ValueError(
                    f"Unknown freeze group '{g}' for {type(self).__name__}. "
                    f"Available groups: {available}"
                )
            for pat in group_map[g]:
                matched.update({n for n, _ in self.named_parameters() if re.search(pat, n)})
        for pat in patterns or []:
            matched.update({n for n, _ in self.named_parameters() if re.search(pat, n)})
        return matched

    def _set_requires_grad(self, names: set[str], value: bool) -> None:
        for n, p in self.named_parameters():
            if n in names:
                p.requires_grad = value

    # ====================== Hard contract (subclasses must implement) ======================
    def recurrence(self, x_t: torch.Tensor | None, z_prev: torch.Tensor,
                   *, inputs: torch.Tensor | None = None) -> torch.Tensor:
        """Single-step transition F_θ. z_prev:(B,M), x_t:(B,input_dim) or None -> z_t:(B,M)."""
        raise NotImplementedError(f"{type(self).__name__} must implement recurrence()")

    def readout(self, z_t: torch.Tensor) -> torch.Tensor:
        """Readout G_φ. z_t:(B,M) -> y_t:(B,output_dim). Returns z_t when DSR directly observes latent states."""
        raise NotImplementedError(f"{type(self).__name__} must implement readout()")

    # ====================== Base default implementations (override allowed) ======================
    def init_state(self, batch_size: int, device: torch.device | str = "cpu") -> torch.Tensor:
        """Initial value z_0. Default zero vector; trainable initial value / encoder initial value can be overridden in subclasses."""
        return torch.zeros(batch_size, self.config.latent_dim, device=device,
                           dtype=next(self.parameters(), torch.empty(0)).dtype)

    def init_state_from_obs(self, x0: torch.Tensor) -> torch.Tensor:
        """Initial latent state from the first observation x0: (B, N) -> (B, M).

        Default (DSR identity observation): zero state with the first N dims set to
        x0. Models with a learned observation->latent lift (e.g. ALRNN's B matrix)
        override this.
        """
        if x0.shape[-1] > self.config.latent_dim:
            raise ValueError("Observation width exceeds the activity dimension")
        z = self.init_state(x0.shape[0], x0.device).clone()
        z[..., :x0.shape[-1]] = x0
        return z

    def forward(self, inputs: torch.Tensor | None = None, *,
                initial_state: torch.Tensor | None = None,
                n_steps: int | None = None,
                return_states: bool = True) -> DynamicsModelOutput:
        """Full rollout: loop recurrence + readout.

        - If inputs is given (B,T,input_dim), rollout over its time length with x_t = inputs[:,t].
        - If inputs is None, provide n_steps for autonomous rollout (x_t=None).
        """
        if inputs is not None:
            assert inputs.dim() == 3, "inputs must have shape (batch, T, input_dim)"
            batch_size, T = inputs.shape[0], inputs.shape[1]
            device = inputs.device
        else:
            assert n_steps is not None, "n_steps must be provided for autonomous rollout"
            assert initial_state is not None, "initial_state must be provided for autonomous rollout"
            batch_size, T, device = initial_state.shape[0], n_steps, initial_state.device

        z = initial_state if initial_state is not None else self.init_state(batch_size, device)

        states, outputs = [], []
        for t in range(T):
            x_t = inputs[:, t] if inputs is not None else None
            z = self.recurrence(x_t, z, inputs=inputs)
            states.append(z)
            outputs.append(self.readout(z))

        states_t = torch.stack(states, dim=1) if return_states else None   # (B,T,M)
        outputs_t = torch.stack(outputs, dim=1)                            # (B,T,output_dim)
        return DynamicsModelOutput(outputs=outputs_t, states=states_t)

    @torch.no_grad()
    def generate(self, initial_state: torch.Tensor, n_steps: int,
                 inputs: torch.Tensor | None = None) -> torch.Tensor:
        """Free rollout (no teacher forcing), returning the latent trajectory
        (B, n_steps+1, M) including the given initial state at index 0.
        For analysis / evaluation."""
        with self.evaluation_context():
            self.validate_state(initial_state)
            z = initial_state
            traj = [z]
            for t in range(n_steps):
                x_t = inputs[:, t] if inputs is not None else None
                z = self.recurrence(x_t, z, inputs=inputs)
                traj.append(z)
            return torch.stack(traj, dim=1)

    # ====================== Dropout support (training only) ======================
    def forward_with_dropout(
        self,
        inputs: torch.Tensor,
        *,
        dropout_rate: float = 0.0,
        dropout_sampling: str = "uniform",
        dropout_beta: float = 1.0,
        participation: torch.Tensor | None = None,
        initial_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Rollout with dropout (training only).

        The dropout mask is sampled once before rollout and applied to the hidden state z_t
        at every time step (multiply by mask and scale by 1/(1-p) to preserve expectation),
        consistent with the "dead neuron" strategy in trainRNNbrain.

        Subclasses can override this for model-specific dropout behavior (e.g., dropout on W_rec).

        Returns:
            states_clean:   (B, T, M)  hidden states without dropout
            outputs_clean:  (B, T, O)  outputs without dropout
            states_dropped: (B, T, M)  hidden states with dropout
            outputs_dropped:(B, T, O)  outputs with dropout
        """
        if dropout_rate <= 0:
            out = self.forward(inputs, initial_state=initial_state, return_states=True)
            s = out.states
            return s, out.outputs, s, out.outputs

        assert inputs.dim() == 3, "inputs must have shape (batch, T, input_dim)"
        batch_size, T = inputs.shape[0], inputs.shape[1]
        device = inputs.device
        M = self.config.latent_dim

        z0 = initial_state if initial_state is not None else self.init_state(batch_size, device)

        # ---- Sample dropout mask (M,) once and reuse across the whole rollout ----
        mask = self._sample_dropout_mask(M, dropout_rate, dropout_sampling,
                                         dropout_beta, participation, device)
        scale = 1.0 / (1.0 - dropout_rate)  # inverted dropout scaling

        # ---- Clean rollout ----
        z = z0.clone()
        states_clean, outputs_clean = [], []
        for t in range(T):
            z = self.recurrence(inputs[:, t], z, inputs=inputs)
            states_clean.append(z[..., :M])
            outputs_clean.append(self.readout(z))

        # ---- Dropout rollout ----
        z = z0.clone()
        states_dropped, outputs_dropped = [], []
        for t in range(T):
            z = self.recurrence(inputs[:, t], z, inputs=inputs)
            # Drop activity coordinates, preserving LSTM/STP memory variables.
            z = torch.cat([z[..., :M] * mask * scale, z[..., M:]], dim=-1)
            states_dropped.append(z[..., :M])
            outputs_dropped.append(self.readout(z))

        sc = torch.stack(states_clean, dim=1)
        oc = torch.stack(outputs_clean, dim=1)
        sd = torch.stack(states_dropped, dim=1)
        od = torch.stack(outputs_dropped, dim=1)
        return sc, oc, sd, od

    @staticmethod
    def _sample_dropout_mask(
        M: int, rate: float, sampling: str, beta: float,
        participation: torch.Tensor | None, device: torch.device,
    ) -> torch.Tensor:
        """Sample a dropout mask (M,). Three strategies: uniform / participation / output_weights."""
        if sampling == "uniform":
            probs = torch.ones(M, device=device)
        elif sampling == "participation":
            if participation is None:
                raise ValueError("sampling='participation' requires participation tensor")
            probs = torch.softmax(beta * participation.to(device).float(), dim=0)
        elif sampling == "output_weights":
            # Do not access W_out here (model-agnostic); fall back to uniform, subclasses may override
            probs = torch.ones(M, device=device)
        else:
            raise ValueError(f"Unknown dropout_sampling: {sampling}")

        p_drop = torch.clamp(rate * M * probs, 0.0, 0.999)
        keep_prob = 1.0 - p_drop
        mask = torch.bernoulli(keep_prob)
        # Ensure at least one neuron remains active
        if mask.sum() == 0:
            mask[torch.randint(0, M, (1,))] = 1.0
        return mask

    # ---------- Analysis support (analytic models may override for speed) ----------
    @property
    def supports_analytic_fixed_points(self) -> bool:
        """Whether the model exposes a closed-form Jacobian / fixed-point solver
        (override to True and implement analytic_parameters / jacobian)."""
        return False

    def jacobian(self, z: torch.Tensor, *, inputs: torch.Tensor | None = None) -> torch.Tensor:
        """∂F/∂z evaluated at state z. Defaults to autograd; analytic models should override.
        z:(M,) -> J:(M,M)."""
        self.validate_state(z)
        z = z.detach().requires_grad_(True)
        x_t = inputs[:1] if inputs is not None else None

        def f(zz):
            """Single-step map F with fixed input, unbatched: (M,) -> (M,)."""
            return self.recurrence(x_t, zz.unsqueeze(0)).squeeze(0)

        with self.evaluation_context(noise=False, strict_state=True), torch.enable_grad():
            return torch.autograd.functional.jacobian(f, z)

    def analytic_parameters(self, task_input: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Expose parameters needed by analytic fixed-point / cycle solvers.

        Only required when ``supports_analytic_fixed_points`` is True.
        The optional ``task_input`` lets a model fold a constant external input into an effective bias,
        e.g. ``h1_eff = h1 + C @ task_input``, so the autonomous solver can be reused.

        Args:
            task_input: (input_dim,) constant external input, or None (autonomous system).

        Returns:
            dict whose keys follow the convention of the specific analytic algorithm (e.g., shallowPLRNN's
            SCYFI uses {"A", "W1", "W2", "h1", "h2"}).
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement analytic_parameters(); "
            "use the numeric/scipy backend, or expose analytic parameters for this model."
        )

    # ====================== Unified save/load (safetensors + json) ======================
    def save_pretrained(self, save_directory: str, metadata: dict | None = None) -> None:
        """Write config.json + model.safetensors (+ metadata.json)."""
        os.makedirs(save_directory, exist_ok=True)
        self.config.to_json_file(save_directory)
        try:
            from safetensors.torch import save_file
            save_file(self.state_dict(), os.path.join(save_directory, WEIGHTS_FILE_NAME))
        except ImportError:
            torch.save(self.state_dict(), os.path.join(save_directory, "model.pt"))
        if metadata is not None:
            import json
            with open(os.path.join(save_directory, METADATA_FILE_NAME), "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

    @classmethod
    def from_pretrained(cls, path: str, *, map_location: str = "cpu") -> "NeuralDynamicsModel":
        """Restore model from a directory. Call on a concrete subclass; for cross-family loading use AutoModel.from_pretrained."""
        config = cls.config_class.from_pretrained(path)
        model = cls(config)
        st_path = os.path.join(path, WEIGHTS_FILE_NAME)
        if os.path.exists(st_path):
            from safetensors.torch import load_file
            state = load_file(st_path, device=map_location)
        else:
            state = torch.load(os.path.join(path, "model.pt"), map_location=map_location)
        model.load_state_dict(state)
        return model

    def num_parameters(self) -> int:
        """Total number of model parameters (trainable + frozen)."""
        return sum(p.numel() for p in self.parameters())
