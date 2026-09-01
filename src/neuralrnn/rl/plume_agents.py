"""Agent factory for the Singh-2023 plume-tracking experiments.

Ports the RNN network constructions of the reference project
(``reference_project/reinforcement_learning/Singh-2023-plumetracknets``,
``code/ppo/models.py``) onto the NeuralRNN actor_critic family:

* ``"vrnn"``: vanilla tanh RNN core = ``CTRNNConfig(activation="tanh",
  dt=None)`` (alpha = 1, pre-activation nonlinearity; exact ``nn.RNN(3, 64,
  tanh)`` equivalent), 3-D observation input. This is the paper's main model.
  (The paper's GRU/MLP appendix controls are not ported.)

All agents share the reference head recipe: separate 2-layer tanh MLPs
(64, 64) for actor and critic, diagonal Gaussian policy with a learned
state-independent log_std (init 0.0), orthogonal head init (hidden gain
sqrt(2), actor-output gain 0.01, critic-output gain 1.0).

The reference initializes the recurrent core with ``N(0, 1/sqrt(64))``
weights and zero biases; since that is not the default init of the CTRNN /
gated_rnn families, it is applied by :func:`init_plume_agent_` after
construction (in-place).
"""
from __future__ import annotations

import math

import torch

from ..models.actor_critic import ActorCriticConfig, ActorCriticModel

PLUME_ARCHS = ("vrnn",)


def build_plume_agent(arch: str = "vrnn", seed: int = 0,
                      init: bool = True) -> ActorCriticModel:
    """Build a plume-tracking actor-critic agent.

    Args:
        arch: one of ``PLUME_ARCHS`` (currently only "vrnn", the paper's
            main model; the GRU/MLP controls are not ported).
        seed: torch seed for weight initialization.
        init: apply the reference core initialization (``init_plume_agent_``).

    Returns:
        ActorCriticModel with a continuous 2-D [move, turn] action head.
        Note: action squashing (tanh -> [0, 1]) happens in the ENVIRONMENT
        (``PlumeEnv(squash_action=True)``), not in the agent.
    """
    if arch not in PLUME_ARCHS:
        raise ValueError(f"unknown arch {arch!r}; choose from {PLUME_ARCHS}")
    torch.manual_seed(seed)

    # dt=None -> alpha=1 -> exact vanilla tanh RNN: z = tanh(Wx + Uh + b)
    from ..models.ctrnn import CTRNNConfig
    core_config = CTRNNConfig(input_dim=3, latent_dim=64,
                              activation="tanh", dt=None).to_dict()

    config = ActorCriticConfig(
        core_config=core_config, action_dim=2,
        action_type="continuous", log_std_init=0.0,
        head_hidden_dims=(64, 64), head_activation="tanh",
        head_init="orthogonal")
    agent = ActorCriticModel(config)
    if init:
        init_plume_agent_(agent)
    return agent


def init_plume_agent_(agent: ActorCriticModel) -> ActorCriticModel:
    """Reference core initialization, in-place: all recurrent-core weights
    ``N(0, 1/sqrt(64))``, biases zero. Heads keep the ActorCriticModel
    orthogonal init (same as the reference)."""
    std = 1.0 / math.sqrt(64)
    core = agent.core
    with torch.no_grad():
        for name, p in core.named_parameters():
            if "weight" in name:
                p.normal_(0.0, std)
            elif "bias" in name:
                p.zero_()
    return agent
