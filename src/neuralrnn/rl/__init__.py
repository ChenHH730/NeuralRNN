"""Reinforcement learning layer for NeuralRNN.

Generic, task-agnostic API (see docs/RL_PLAN.md):
    envs:     env registry + adapters (neurogym, gym) and SyncVectorEnv
    buffers:  RecurrentRolloutBuffer (chunk-initial hidden state, env-slice minibatches)
    returns:  return/advantage estimators (n-step, GAE)
    losses:   PPOLoss / ReinforceLoss / A2CPredLoss (+ RL_LOSS_REGISTRY for
              pluggable objectives)
    trainer:  RLTrainer + RLTrainingArguments (on-policy rollout -> BPTT loop)
    rollout:  collect_episodes (evaluation / neural-behavioral analysis data)
    planning: WorldModelPlanner (model-based rollouts fed back as observations —
              the planning meta-action protocol of Jensen-2024)
    vecnormalize: RunningMeanStd / VecNormalize / OnlineObsNorm

Agents live in the model layer: ``neuralrnn.models.actor_critic.ActorCriticModel``.

Task-specific code (used by the reproduction notebooks, not part of the
generic API) is importable from its own modules:
    neuralrnn.rl.envs.echoice   EchoiceEnv (Battista-2026 economic choice)
    neuralrnn.rl.envs.plume     PlumeEnv (Singh-2023 plume tracking)
    neuralrnn.rl.envs.maze      MazeEnv (Jensen-2024 toroidal maze + think action)
    neuralrnn.rl.plume_agents   build_plume_agent / init_plume_agent_
    neuralrnn.rl.plume_eval     fixed-grid evaluation assay + sparsity sweep
    neuralrnn.rl.probe          probe_value_model + trial alignment helpers
                                (value-RNN / Qian-Burrell-2024 analyses)
"""
from .buffers import RecurrentRolloutBuffer
from .envs import SyncVectorEnv, make_env, register_env
from .losses import (A2CPredLoss, PPOLoss, ReinforceLoss, RL_LOSS_REGISTRY,
                     build_rl_loss, register_rl_loss)
from .planning import RolloutResult, WorldModelPlanner, bind_planners
from .returns import GAE, NStepReturns, build_return_estimator
from .rollout import collect_episodes
from .trainer import RLTrainer, RLTrainingArguments
from .vecnormalize import OnlineObsNorm, RunningMeanStd, VecNormalize

__all__ = [
    "RecurrentRolloutBuffer", "SyncVectorEnv", "make_env", "register_env",
    "PPOLoss", "ReinforceLoss", "A2CPredLoss", "RL_LOSS_REGISTRY",
    "build_rl_loss", "register_rl_loss", "GAE", "NStepReturns",
    "build_return_estimator", "collect_episodes", "RLTrainer",
    "RLTrainingArguments", "VecNormalize", "OnlineObsNorm", "RunningMeanStd",
    "WorldModelPlanner", "RolloutResult", "bind_planners",
]
