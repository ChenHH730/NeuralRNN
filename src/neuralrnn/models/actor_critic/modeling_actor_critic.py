"""Actor-critic agent model (RL agent layer).

Wraps any registered NeuralRNN core (built from ``config.core_config`` via AutoModel)
with a policy head (Categorical for discrete actions, diagonal Gaussian for
continuous actions) and a scalar value head. Implements the standard
NeuralDynamicsModel hard contract (recurrence / readout delegate to the core; readout
returns scaled policy logits), so all analysis modules (fixed points, PCA, vector
fields) work on the agent's hidden dynamics unchanged.

RL interface (CleanRL-style, batch-first single steps, full-sequence replay for BPTT):

    step(x_t, z_prev, done)               -> action, log_prob, entropy, value, z_t
    get_value(x_t, z_prev, done)          -> value
    evaluate_sequence(inputs, z0, dones, actions)
                                          -> log_probs, entropies, values, states, z_T
    project_constraints()                 -> post-optimizer-step |W| projections

``done`` convention (CleanRL): ``done[t]`` marks that the transition INTO step t ended
an episode, so the hidden state is masked BEFORE processing x_t. Rollout buffers store
``dones[step] = done_before_action`` accordingly.
"""
from __future__ import annotations

import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal

from ...modeling_utils import NeuralDynamicsModel
from ...auto.modeling_auto import register_model, AutoModel
from ...auto.configuration_auto import AutoConfig
from .configuration_actor_critic import ActorCriticConfig

_HEAD_ACTIVATIONS = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU,
                     "identity": nn.Identity}


def _build_head(in_dim: int, out_dim: int, hidden_dims: tuple,
                activation: str, bias: bool = True) -> nn.Module:
    """Policy/value head: a plain Linear when ``hidden_dims`` is empty (original
    behavior, keeps old checkpoint keys), else Linear->act->...->Linear."""
    if not hidden_dims:
        return nn.Linear(in_dim, out_dim, bias=bias)
    act_cls = _HEAD_ACTIVATIONS[activation]
    layers: list[nn.Module] = []
    dims = (in_dim, *hidden_dims)
    for d_in, d_out in zip(dims[:-1], dims[1:]):
        layers += [nn.Linear(d_in, d_out), act_cls()]
    layers.append(nn.Linear(dims[-1], out_dim, bias=bias))
    return nn.Sequential(*layers)


def _iter_linears(head: nn.Module):
    """Yield all Linear layers of a head (works for both Linear and Sequential)."""
    if isinstance(head, nn.Linear):
        yield head
    else:
        yield from (m for m in head.modules() if isinstance(m, nn.Linear))


@register_model("actor_critic")
class ActorCriticModel(NeuralDynamicsModel):
    """Actor-critic agent = RNN core + policy head + value head.

    Core-agnostic: any registered RNN family works (ctrnn, ei_rnn, gated_rnn,
    ...). Critic-only value agents (no policy) use ``action_type="none"``.
    """

    config_class = ActorCriticConfig

    def __init__(self, config: ActorCriticConfig) -> None:
        super().__init__(config)
        core_dict = dict(config.core_config)
        core_type = core_dict.pop("model_type", None)
        if core_type is None:
            raise ValueError("ActorCriticConfig.core_config must contain 'model_type'")
        core_cfg = AutoConfig.for_model(core_type, **core_dict)
        self.core = AutoModel.from_config(core_cfg)

        # Feature slice for the heads: E units only for E-I cores when requested.
        self.e_only = (bool(config.readout_e_only)
                       and getattr(self.core, "e_size", None) is not None
                       and getattr(self.core.config, "readout_e_only", True))
        self.feat_dim = self.core.e_size if self.e_only else config.latent_dim

        if config.action_type in ("discrete", "continuous"):
            self.actor = _build_head(self.feat_dim, config.action_dim,
                                     config.head_hidden_dims,
                                     config.head_activation)
        else:
            self.actor = None
        if config.action_type == "continuous":
            self.log_std = nn.Parameter(
                torch.full((config.action_dim,), float(config.log_std_init)))
        else:
            self.log_std = None
        self.critic = _build_head(self.feat_dim, 1, config.head_hidden_dims,
                                  config.head_activation, bias=config.value_bias)
        # Optional auxiliary prediction head (e.g. a learned world model,
        # Jensen-2024): input is [features] or [features, one-hot(action)].
        if config.aux_head_out_dim > 0:
            aux_in = self.feat_dim + (config.action_dim
                                      if config.aux_input == "state_action" else 0)
            self.aux = _build_head(aux_in, config.aux_head_out_dim,
                                   config.aux_head_hidden_dims,
                                   config.aux_head_activation)
        else:
            self.aux = None
        self._init_heads()
        self.apply_freeze_config()

    # ---------------- construction helpers ----------------
    def _init_heads(self) -> None:
        """Head weight init.

        "uniform" (default): |U(0,1)| * head_init_scale / sqrt(feat_dim), zero bias.
        "orthogonal": orthogonal init, gain sqrt(2) on hidden layers, gain 0.01 on
        the actor output layer and 1.0 on the critic output layer, zero biases
        (the Singh-2023 / ikostrikov recipe).
        """
        cfg = self.config
        with torch.no_grad():
            if cfg.head_init == "orthogonal":
                for head, out_gain in ((self.actor, 0.01), (self.critic, 1.0),
                                       (self.aux, 1.0)):
                    if head is None:
                        continue
                    linears = list(_iter_linears(head))
                    for lin in linears[:-1]:
                        nn.init.orthogonal_(lin.weight, gain=2 ** 0.5)
                        if lin.bias is not None:
                            lin.bias.zero_()
                    nn.init.orthogonal_(linears[-1].weight, gain=out_gain)
                    if linears[-1].bias is not None:
                        linears[-1].bias.zero_()
            else:
                scale = cfg.head_init_scale / max(1, self.feat_dim) ** 0.5
                for head in (self.actor, self.critic, self.aux):
                    if head is None:
                        continue
                    for lin in _iter_linears(head):
                        lin.weight.data.copy_(torch.rand_like(lin.weight) * scale)
                        if lin.bias is not None:
                            lin.bias.data.zero_()

    def _freeze_groups(self) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = {}
        for g, pats in self.core._freeze_groups().items():
            groups[g] = ["^core\\." + re.sub(r"^\^", "", p) for p in pats]
        groups.setdefault("output", [])
        groups["output"] = groups["output"] + [r"^actor\.", r"^critic\.",
                                               r"^aux\.", r"^log_std$"]
        return groups

    # ---------------- hard contract (delegates to the core) ----------------
    def recurrence(self, x_t, z_prev, *, inputs=None):
        """Single-step transition of the core. x_t: (B, input_dim), z_prev: (B, M)."""
        return self.core.recurrence(x_t, z_prev, inputs=inputs)

    def readout(self, z_t):
        """Scaled policy logits (B, action_dim) for discrete agents, action mean
        (B, action_dim) for continuous agents, or value (B, 1) for critic-only agents."""
        if self.actor is not None:
            if self.config.action_type == "continuous":
                return self.actor(self._features(z_t))
            return self._logits(z_t)
        return self.critic(self._features(z_t))

    def init_state(self, batch_size, device="cpu"):
        """Initial hidden state z_0 (delegates to the core)."""
        return self.core.init_state(batch_size, device)

    # ---------------- heads ----------------
    def _features(self, z: torch.Tensor) -> torch.Tensor:
        """Head input features: E-unit slice for E-I cores, else the full state."""
        return z[..., :self.core.e_size] if self.e_only else z

    def _logits(self, z: torch.Tensor) -> torch.Tensor:
        return self.actor(self._features(z)) * self.config.policy_logit_scale

    def _value(self, z: torch.Tensor) -> torch.Tensor:
        return self.critic(self._features(z)).squeeze(-1)

    def value(self, z: torch.Tensor) -> torch.Tensor:
        """State value V(z) from a hidden state (B, M) -> (B,)."""
        return self._value(z)

    def aux_logits(self, z: torch.Tensor,
                   actions: torch.Tensor | None = None) -> torch.Tensor:
        """Auxiliary head logits from a hidden state.

        Args:
            z: (B, M) hidden state.
            actions: (B,) long actions, required when ``aux_input='state_action'``
                (the aux head consumes a one-hot of the taken action, e.g. the
                world-model head of Jensen-2024).

        Returns:
            (B, aux_head_out_dim) logits; interpret slices via
            ``config.aux_slices`` (e.g. next-state / reward-location logits).
        """
        if self.aux is None:
            raise RuntimeError("This agent has no auxiliary head "
                               "(config.aux_head_out_dim == 0)")
        feats = self._features(z)
        if self.config.aux_input == "state_action":
            if actions is None:
                raise ValueError("actions are required when aux_input='state_action'")
            ahot = F.one_hot(actions.long(), self.config.action_dim).to(feats.dtype)
            feats = torch.cat([feats, ahot], dim=-1)
        return self.aux(feats)

    def _mask_reset(self, z: torch.Tensor, done: torch.Tensor | None,
                    z0: torch.Tensor | None = None) -> torch.Tensor:
        """CleanRL done-masking that resets to the core's initial state (h0)
        instead of zeros. Numerically identical to plain zero-masking whenever
        h0 == 0 (the default for all cores), but lets a learnable h0
        (``trainable_h0=True``) receive gradients through episode boundaries,
        matching Flux's ``state0`` semantics."""
        if done is None:
            return z
        d = done.float().unsqueeze(-1)
        if not bool(d.any()):
            return z
        if z0 is None:
            z0 = self.init_state(z.shape[0], z.device)
        return z * (1.0 - d) + z0.to(dtype=z.dtype) * d

    def _policy_dist(self, z: torch.Tensor):
        """Action distribution at hidden state z.

        Discrete: Categorical over scaled logits. Continuous: diagonal Gaussian
        with the actor-head mean and a learned state-independent std (Singh-2023
        style; any action squashing happens in the ENVIRONMENT, so log-probs are
        always taken on the raw Gaussian sample and no Jacobian correction is
        needed).
        """
        if self.config.action_type == "continuous":
            mean = self.actor(self._features(z))
            return Normal(mean, self.log_std.exp())
        return Categorical(logits=self._logits(z))

    def _log_prob(self, dist, action: torch.Tensor) -> torch.Tensor:
        """Per-sample log prob (B,); summed over action dims for continuous."""
        lp = dist.log_prob(action)
        return lp.sum(-1) if self.config.action_type == "continuous" else lp

    def _entropy(self, dist) -> torch.Tensor:
        """Per-sample entropy (B,); summed over action dims for continuous
        (matches the ikostrikov-style reference PPO implementations)."""
        ent = dist.entropy()
        return ent.sum(-1) if self.config.action_type == "continuous" else ent

    def policy_mode(self, z: torch.Tensor) -> torch.Tensor:
        """Deterministic action: argmax logits (discrete) or mean (continuous)."""
        if self.config.action_type == "continuous":
            return self.actor(self._features(z))
        return self._logits(z).argmax(-1)

    # ---------------- RL interface ----------------
    def step(self, x_t: torch.Tensor, z_prev: torch.Tensor,
             done: torch.Tensor | None = None):
        """Single environment step.

        Args:
            x_t: (B, input_dim) observation.
            z_prev: (B, M) hidden state before this step.
            done: (B,) float/bool mask; 1 marks that the previous transition ended an
                episode, so the hidden state is reset before processing x_t.

        Returns:
            action (B,) long or (B, action_dim) float, log_prob (B,),
            entropy (B,), value (B,), z_t (B, M).
        """
        z_prev = self._mask_reset(z_prev, done)
        z = self.core.recurrence(x_t, z_prev)
        dist = self._policy_dist(z)
        action = dist.sample()
        hook = getattr(self, "planner_hook", None)
        if hook is not None:
            hook(z.detach())
        return action, self._log_prob(dist, action), self._entropy(dist), \
            self._value(z), z

    def get_value(self, x_t: torch.Tensor, z_prev: torch.Tensor,
                  done: torch.Tensor | None = None) -> torch.Tensor:
        """Value at the current observation (used for bootstrapping). (B,) -> (B,)."""
        z_prev = self._mask_reset(z_prev, done)
        z = self.core.recurrence(x_t, z_prev)
        return self._value(z)

    def evaluate_sequence(self, inputs: torch.Tensor, z0: torch.Tensor,
                          dones: torch.Tensor, actions: torch.Tensor,
                          *, return_aux: bool = False):
        """Replay a rollout for the PPO update (BPTT, episode-boundary masked).

        Args:
            inputs: (B, T, input_dim) observations.
            z0: (B, M) hidden state at the start of the chunk.
            dones: (B, T) float masks with the CleanRL convention (see module docstring).
            actions: (B, T) long (discrete) or (B, T, action_dim) float
                (continuous), actions taken during collection.
            return_aux: If True (requires an aux head), also return the aux-head
                logits computed per step from the replayed z_t and the
                teacher-forced actions (world-model semantics of Jensen-2024).

        Returns:
            log_probs (B, T), entropies (B, T), values (B, T),
            states (B, T, M), z_T (B, M); with ``return_aux=True`` a sixth
            element ``aux`` (B, T, aux_head_out_dim) is appended.
        """
        assert inputs.dim() == 3, "inputs must be (B, T, input_dim)"
        B, T = inputs.shape[0], inputs.shape[1]
        z = z0
        # Initial state for episode-boundary resets (created only when needed).
        init = self.init_state(B, inputs.device) if bool(dones.any()) else None
        logps, ents, vals, states, auxs = [], [], [], [], []
        for t in range(T):
            z = self._mask_reset(z, dones[:, t], z0=init)
            z = self.core.recurrence(inputs[:, t], z)
            dist = self._policy_dist(z)
            logps.append(self._log_prob(dist, actions[:, t]))
            ents.append(self._entropy(dist))
            vals.append(self._value(z))
            states.append(z)
            if return_aux:
                auxs.append(self.aux_logits(
                    z, actions[:, t] if self.config.aux_input == "state_action"
                    else None))
        out = (torch.stack(logps, 1), torch.stack(ents, 1), torch.stack(vals, 1),
               torch.stack(states, 1), z)
        if return_aux:
            out = out + (torch.stack(auxs, 1),)
        return out

    # ---------------- constraint projection ----------------
    @torch.no_grad()
    def project_constraints(self) -> None:
        """Post-optimizer-step projections requested by the config (|W| on the
        heads / core input weights).

        The core's Dale signs (if any) are enforced on the fly in
        ``core.recurrence`` (|W| @ diag(sign)), so no recurrent projection is
        needed here.
        """
        cfg = self.config
        if self.actor is not None and cfg.actor_positive:
            for lin in _iter_linears(self.actor):
                lin.weight.data.abs_()
        if cfg.critic_positive:
            for lin in _iter_linears(self.critic):
                lin.weight.data.abs_()
        if cfg.core_input_positive and hasattr(self.core, "input2h"):
            self.core.input2h.weight.data.abs_()
