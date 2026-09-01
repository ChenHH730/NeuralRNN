"""Actor-critic agent configuration (RL agent layer).

An actor-critic agent wraps ANY registered NeuralRNN core (ctrnn, ei_rnn, gated_rnn,
lowrank_rnn, ...) and adds lightweight policy / value readout heads, turning a
task-optimized-RNN core into an RL agent (see docs/RL_PLAN.md).

The core is specified as a serialized config dict (``SomeCoreConfig(...).to_dict()``),
so the agent round-trips through ``save_pretrained`` / ``AutoModel.from_pretrained``
like any other family.
"""
from __future__ import annotations

from ...configuration_utils import NeuralRNNConfig

SUPPORTED_ACTION_TYPES = ("discrete", "none", "continuous")


class ActorCriticConfig(NeuralRNNConfig):
    """Actor-critic agent = RNN core + policy head + value head.

    Args:
        core_config: Serialized config dict of any registered RNN family
            (e.g. ``EIRNNConfig(...).to_dict()``); must contain ``model_type``.
        action_dim: Number of discrete actions. Use 0 (with action_type="none")
            for critic-only agents (e.g. value_rnn-style TD paradigms).
        action_type: "discrete" (Categorical policy), "continuous" (diagonal
            Gaussian policy; the actor head outputs the mean and a state-independent
            ``log_std`` parameter is learned), or "none" (critic-only).
        policy_logit_scale: Temperature multiplier applied to actor logits
            (e.g. Battista-2026 uses 4/7.5 to match monkey choice steepness).
            Discrete policies only.
        log_std_init: Initial value of the state-independent log-std parameter
            (continuous policies only; Singh-2023 uses 0.0).
        head_hidden_dims: Hidden layer sizes for BOTH the actor and critic heads
            (e.g. ``(64, 64)`` for the Singh-2023 heads). Empty = single Linear
            (the original behavior).
        head_activation: Activation between hidden head layers ("tanh", "relu", ...).
        head_init: "uniform" = |U(0,1)| * head_init_scale / sqrt(feat_dim), zero bias
            (the original behavior); "orthogonal" = orthogonal init with gain
            sqrt(2) on hidden layers, gain 0.01 on the actor output layer and 1.0
            on the critic output layer (the Singh-2023 / ikostrikov recipe).
        readout_e_only: If True and the core exposes ``e_size`` (E-I cores), the
            heads read only from excitatory units (long-range projections are
            excitatory). Ignored for cores without an E/I split.
        actor_positive: Project actor weights to |W| after each optimizer step
            (e.g. Battista-2026 readouts are non-negative).
        critic_positive: Same for the critic weights.
        core_input_positive: Project the core input weights (``core.input2h``)
            to |W| after each optimizer step (e.g. for cores whose inputs
            should act excitatory).
        head_init_scale: Heads are initialized |U(0,1)| * head_init_scale / sqrt(feat_dim)
            with zero bias.
        value_bias: Whether the critic head has a bias.
        aux_head_out_dim: Output size of an optional auxiliary prediction head
            (e.g. a learned world model; Jensen-2024 uses 33). 0 = no aux head
            (default; no behavior change).
        aux_head_hidden_dims: Hidden layer sizes of the aux head
            (e.g. ``(33,)`` for Jensen-2024). Empty = single Linear.
        aux_head_activation: Activation between aux-head hidden layers.
        aux_input: What feeds the aux head: "state_action" = [hidden state,
            one-hot of the taken action] (Jensen-2024), or "state" = hidden
            state only. "state_action" requires a discrete policy.
        aux_slices: Named ``{name: (start, stop)}`` logit slices of the aux
            output, used by aux-aware RL losses (e.g. ``a2c_pred``) and by
            planners. Must lie within ``aux_head_out_dim``.

    The generic ``input_dim`` / ``latent_dim`` / ``output_dim`` fields are derived from
    ``core_config`` (output_dim = action_dim, or 1 for critic-only) unless overridden,
    so the standard config plumbing (freeze flags, serialization) keeps working.
    """

    model_type = "actor_critic"

    def __init__(
        self,
        core_config: dict | None = None,
        action_dim: int = 0,
        action_type: str = "discrete",
        policy_logit_scale: float = 1.0,
        log_std_init: float = 0.0,
        head_hidden_dims: tuple = (),
        head_activation: str = "tanh",
        head_init: str = "uniform",
        readout_e_only: bool = True,
        actor_positive: bool = False,
        critic_positive: bool = False,
        core_input_positive: bool = False,
        head_init_scale: float = 0.4,
        value_bias: bool = True,
        aux_head_out_dim: int = 0,
        aux_head_hidden_dims: tuple = (),
        aux_head_activation: str = "relu",
        aux_input: str = "state_action",
        aux_slices: dict | None = None,
        **kwargs,
    ) -> None:
        core_config = dict(core_config) if core_config is not None else {}
        if action_type not in SUPPORTED_ACTION_TYPES:
            raise ValueError(
                f"action_type must be one of {SUPPORTED_ACTION_TYPES}, got {action_type!r}")
        if action_type == "discrete" and action_dim < 1:
            raise ValueError("action_type='discrete' requires action_dim >= 1")
        if action_type == "continuous" and action_dim < 1:
            raise ValueError("action_type='continuous' requires action_dim >= 1")
        if head_init not in ("uniform", "orthogonal"):
            raise ValueError(f"head_init must be 'uniform' or 'orthogonal', got {head_init!r}")
        if aux_input not in ("state_action", "state"):
            raise ValueError(
                f"aux_input must be 'state_action' or 'state', got {aux_input!r}")
        if aux_head_out_dim > 0 and aux_input == "state_action" \
                and action_type != "discrete":
            raise ValueError(
                "aux_input='state_action' requires action_type='discrete' "
                "(the aux head consumes a one-hot of the taken action)")
        aux_slices = dict(aux_slices) if aux_slices is not None else {}
        for name, sl in aux_slices.items():
            start, stop = int(sl[0]), int(sl[1])
            if not (0 <= start < stop <= aux_head_out_dim):
                raise ValueError(
                    f"aux_slices[{name!r}] = {(start, stop)} is outside "
                    f"[0, {aux_head_out_dim})")
        input_dim = kwargs.pop("input_dim", core_config.get("input_dim", 0))
        latent_dim = kwargs.pop("latent_dim", core_config.get("latent_dim", 0))
        output_dim = kwargs.pop("output_dim", action_dim if action_dim > 0 else 1)
        super().__init__(input_dim=input_dim, latent_dim=latent_dim,
                         output_dim=output_dim, **kwargs)
        self.core_config = core_config
        self.action_dim = action_dim
        self.action_type = action_type
        self.policy_logit_scale = policy_logit_scale
        self.log_std_init = log_std_init
        self.head_hidden_dims = tuple(head_hidden_dims)
        self.head_activation = head_activation
        self.head_init = head_init
        self.readout_e_only = readout_e_only
        self.actor_positive = actor_positive
        self.critic_positive = critic_positive
        self.core_input_positive = core_input_positive
        self.head_init_scale = head_init_scale
        self.value_bias = value_bias
        self.aux_head_out_dim = aux_head_out_dim
        self.aux_head_hidden_dims = tuple(aux_head_hidden_dims)
        self.aux_head_activation = aux_head_activation
        self.aux_input = aux_input
        self.aux_slices = {k: (int(v[0]), int(v[1])) for k, v in aux_slices.items()}
