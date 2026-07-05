"""模型基类（≈ transformers.PreTrainedModel）。

核心抽象（ARCHITECTURE §2.1）：所有模型都是"带读出的离散动力系统"。
唯一硬契约 —— 子类必须实现两个方法：
    recurrence(x_t, z_prev, *, inputs=None) -> z_t   # 单步转移 F_θ
    readout(z_t) -> y_t                              # 读出 G_φ
实现这两个方法即可接入统一的训练器(train/)与分析器(analysis/)。

张量形状约定（全框架统一，batch-first）：
    inputs : (batch, T, input_dim)   单步 x_t : (batch, input_dim)
    states : (batch, T, latent_dim)  单步 z_t : (batch, latent_dim)
    outputs: (batch, T, output_dim)
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from .configuration_utils import NeuralRNNConfig

WEIGHTS_FILE_NAME = "model.safetensors"
METADATA_FILE_NAME = "metadata.json"


@dataclass
class DynamicsModelOutput:
    """统一模型输出容器（≈ transformers.ModelOutput）。可属性访问，也可 dict 解包。

    Supports arithmetic ops that delegate to the .outputs tensor for backward
    compatibility with reference code that expects raw tensors from forward().
    """

    outputs: torch.Tensor | None = None   # 读出 y_{1:T}     (B, T, output_dim)
    states: torch.Tensor | None = None    # 潜轨迹 z_{1:T}    (B, T, latent_dim)
    loss: torch.Tensor | None = None      # 若 forward 内算了损失
    extras: dict[str, Any] | None = None  # 模型特异输出（如 LFADS 后验）

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
        return self.outputs.sign()

    def squeeze(self, dim=None):
        return self.outputs.squeeze(dim) if dim is not None else self.outputs.squeeze()

    def mean(self, *args, **kwargs):
        return self.outputs.mean(*args, **kwargs)

    def sum(self, *args, **kwargs):
        return self.outputs.sum(*args, **kwargs)

    def pow(self, n):
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
        return self.outputs.device if self.outputs is not None else None


class NeuralDynamicsModel(nn.Module):
    """所有 RNN / 动力系统模型的基类。

    子类约定：
        - 设 `config_class = <Family>Config`
        - 用 `@register_model("<family>")` 装饰（见 auto/modeling_auto.py）
        - 在 __init__(self, config) 中只从 config 读参数构建子模块
        - 实现 recurrence / readout（硬契约）
        - 解析模型可实现 jacobian 并令 supports_analytic_fixed_points=True
    """

    config_class: type[NeuralRNNConfig] = NeuralRNNConfig

    def __init__(self, config: NeuralRNNConfig) -> None:
        super().__init__()
        self.config = config

    # ====================== 参数冻结支持（ESN / reservoir computing）======================
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

    # ====================== 硬契约（子类必须实现）======================
    def recurrence(self, x_t: torch.Tensor | None, z_prev: torch.Tensor,
                   *, inputs: torch.Tensor | None = None) -> torch.Tensor:
        """单步转移 F_θ。z_prev:(B,M) , x_t:(B,input_dim) 或 None -> z_t:(B,M)。"""
        raise NotImplementedError(f"{type(self).__name__} 必须实现 recurrence()")

    def readout(self, z_t: torch.Tensor) -> torch.Tensor:
        """读出 G_φ。z_t:(B,M) -> y_t:(B,output_dim)。DSR 直接观测潜状态时返回 z_t。"""
        raise NotImplementedError(f"{type(self).__name__} 必须实现 readout()")

    # ====================== 基类默认实现（可覆盖）======================
    def init_state(self, batch_size: int, device: torch.device | str = "cpu") -> torch.Tensor:
        """初值 z_0。默认零向量；可训练初值/编码器初值在子类覆盖。"""
        return torch.zeros(batch_size, self.config.latent_dim, device=device)

    def forward(self, inputs: torch.Tensor | None = None, *,
                initial_state: torch.Tensor | None = None,
                n_steps: int | None = None,
                return_states: bool = True) -> DynamicsModelOutput:
        """整段 rollout：循环 recurrence + readout。

        - 若 inputs 给定 (B,T,input_dim)，按其时间长度 rollout，x_t = inputs[:,t]。
        - 若 inputs 为 None，需给 n_steps 做自治 rollout（x_t=None）。
        """
        if inputs is not None:
            assert inputs.dim() == 3, "inputs 形状应为 (batch, T, input_dim)"
            batch_size, T = inputs.shape[0], inputs.shape[1]
            device = inputs.device
        else:
            assert n_steps is not None, "自治 rollout 需提供 n_steps"
            assert initial_state is not None, "自治 rollout 需提供 initial_state"
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
        """自由 rollout（无 teacher forcing），返回潜轨迹 (B,T,M)。分析/评估用。"""
        self.eval()
        z = initial_state
        traj = [z]
        for t in range(n_steps):
            x_t = inputs[:, t] if inputs is not None else None
            z = self.recurrence(x_t, z, inputs=inputs)
            traj.append(z)
        return torch.stack(traj, dim=1)

    # ====================== Dropout 支持（训练用）======================
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
        """带 dropout 的 rollout（训练用）。

        Dropout mask 在 rollout 前采样一次，对每个时间步的隐藏状态 z_t 应用
        （乘以 mask，再缩放 1/(1-p) 保持期望值不变），与 trainRNNbrain 的
        "dead neuron" 策略一致。

        子类可覆盖此方法以实现模型特定的 dropout 行为（如对 W_rec 做 dropout）。

        Returns:
            states_clean:   (B, T, M)  无 dropout 的隐藏状态
            outputs_clean:  (B, T, O)  无 dropout 的输出
            states_dropped: (B, T, M)  有 dropout 的隐藏状态
            outputs_dropped:(B, T, O)  有 dropout 的输出
        """
        if dropout_rate <= 0:
            out = self.forward(inputs, initial_state=initial_state, return_states=True)
            s = out.states
            return s, out.outputs, s, out.outputs

        assert inputs.dim() == 3, "inputs 形状应为 (batch, T, input_dim)"
        batch_size, T = inputs.shape[0], inputs.shape[1]
        device = inputs.device
        M = self.config.latent_dim

        z0 = initial_state if initial_state is not None else self.init_state(batch_size, device)

        # ---- 采样 dropout mask (M,) —— 一次采样，整个 rollout 复用 ----
        mask = self._sample_dropout_mask(M, dropout_rate, dropout_sampling,
                                         dropout_beta, participation, device)
        scale = 1.0 / (1.0 - dropout_rate)  # inverted dropout scaling

        # ---- Clean rollout ----
        z = z0.clone()
        states_clean, outputs_clean = [], []
        for t in range(T):
            z = self.recurrence(inputs[:, t], z, inputs=inputs)
            states_clean.append(z)
            outputs_clean.append(self.readout(z))

        # ---- Dropout rollout ----
        z = z0.clone()
        states_dropped, outputs_dropped = [], []
        for t in range(T):
            z = self.recurrence(inputs[:, t], z, inputs=inputs)
            z = z * mask * scale                    # apply dropout
            states_dropped.append(z)
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
        """采样 dropout mask (M,)。三种策略：uniform / participation / output_weights。"""
        if sampling == "uniform":
            probs = torch.ones(M, device=device)
        elif sampling == "participation":
            if participation is None:
                raise ValueError("sampling='participation' requires participation tensor")
            probs = torch.softmax(beta * participation.to(device).float(), dim=0)
        elif sampling == "output_weights":
            # 不在此处访问 W_out（模型无关）；用均匀兜底，子类可覆盖
            probs = torch.ones(M, device=device)
        else:
            raise ValueError(f"Unknown dropout_sampling: {sampling}")

        p_drop = torch.clamp(rate * M * probs, 0.0, 0.999)
        keep_prob = 1.0 - p_drop
        mask = torch.bernoulli(keep_prob)
        # 保证至少保留一个神经元
        if mask.sum() == 0:
            mask[torch.randint(0, M, (1,))] = 1.0
        return mask

    # ---------- 分析支持（解析模型可覆盖以加速）----------
    @property
    def supports_analytic_fixed_points(self) -> bool:
        return False

    def jacobian(self, z: torch.Tensor, *, inputs: torch.Tensor | None = None) -> torch.Tensor:
        """∂F/∂z 在状态 z 处。默认用自动微分兜底；解析模型应覆盖此方法。
        z:(M,) -> J:(M,M)。"""
        z = z.detach().requires_grad_(True)
        x_t = inputs[:1] if inputs is not None else None

        def f(zz):
            return self.recurrence(x_t, zz.unsqueeze(0)).squeeze(0)

        return torch.autograd.functional.jacobian(f, z)

    def analytic_parameters(self, task_input: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """暴露解析不动点/环算法所需的参数。

        仅当 ``supports_analytic_fixed_points`` 为 True 时需要实现。
        可选的 ``task_input`` 允许模型将常值外部输入折叠到有效偏置中，例如
        ``h1_eff = h1 + C @ task_input``，从而复用自治系统的解析求解器。

        Args:
            task_input: (input_dim,) 常值外部输入，或 None（自治系统）。

        Returns:
            dict，键与具体解析算法约定一致（如 shallowPLRNN 的 SCYFI 使用
            {"A", "W1", "W2", "h1", "h2"}）。
        """
        raise NotImplementedError(
            f"{type(self).__name__} 未实现 analytic_parameters(); "
            "请用 numeric/scipy 后端，或为此模型实现解析参数暴露。"
        )

    # ====================== 统一存读（safetensors + json）======================
    def save_pretrained(self, save_directory: str, metadata: dict | None = None) -> None:
        """写 config.json + model.safetensors (+ metadata.json)。"""
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
        """从目录恢复模型。在具体子类上调用；跨家族请用 AutoModel.from_pretrained。"""
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
        return sum(p.numel() for p in self.parameters())
