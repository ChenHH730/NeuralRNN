# `neuralrnn.rl` — Reinforcement Learning Layer

Generic, task-agnostic RL machinery for recurrent agents. An RL model in
NeuralRNN is always:

> **agent = any registered RNN core + light actor/critic heads**
> (`neuralrnn.models.actor_critic.ActorCriticModel`)

trained on-policy by `RLTrainer` against gym-style vectorized environments,
with a pluggable loss (algorithm registry) and pluggable return estimator.

## Layer map

| file | contents | scope |
|---|---|---|
| `envs/base.py` | `SyncVectorEnv`, old-gym/gymnasium API shims | generic |
| `envs/__init__.py` | `ENV_REGISTRY`, `register_env`, `make_env` (incl. `"neurogym:<Task>"` and `"gym:<id>"` passthroughs) | generic |
| `envs/neurogym_env.py` | `NeurogymEnvAdapter` — gym view of any neurogym task | generic |
| `buffers.py` | `RecurrentRolloutBuffer` (time-first, chunk-initial hidden state, env-slice minibatches) | generic |
| `returns.py` | `NStepReturns`, `GAE`, `build_return_estimator` | generic |
| `losses.py` | `PPOLoss`, `ReinforceLoss`, `RL_LOSS_REGISTRY` / `register_rl_loss` / `build_rl_loss` | generic |
| `trainer.py` | `RLTrainer` + `RLTrainingArguments` (rollout → returns → BPTT replay update) | generic |
| `rollout.py` | `collect_episodes` (evaluation / analysis episode collector) | generic |
| `vecnormalize.py` | `RunningMeanStd`, `VecNormalize`, `OnlineObsNorm` | generic |
| `planning.py` | `WorldModelPlanner` + `RolloutResult` + `bind_planners` — imagined rollouts via the agent's aux head, fed back as observation channels | generic (needs an env-supplied input builder) |
| `envs/echoice.py` | `EchoiceEnv` (Battista-2026 economic choice) | task-specific |
| `envs/plume.py`, `envs/plume_sim.py` | `PlumeEnv` + puff/wind simulator (Singh-2023) | task-specific |
| `envs/maze.py` | `MazeEnv` + `generate_maze` (Jensen-2024 toroidal maze with a `think` action) | task-specific |
| `plume_agents.py`, `plume_eval.py` | plume agent factory / fixed-grid eval assay (Singh-2023) | task-specific |
| `probe.py` | `probe_value_model` + trial alignment helpers (value-RNN analyses) | task-flavored (generic functions, Qian-Burrell-2024 defaults) |

Task-specific modules are **not** re-exported at the `neuralrnn.rl` package
level; import them from their module path, e.g.
`from neuralrnn.rl.plume_eval import evaluate_agent`.

Critic-only (value-only) training on offline episode datasets uses the
supervised stack instead: `neuralrnn.train.objectives.td.TDObjective` with
`ActorCriticConfig(action_type="none")` (see notebook 19).

## Using the framework for a new RL-RNN study

### 1. Environment

Any object with the gymnasium contract (`observation_space`, `action_space`,
`reset() -> (obs, info)`, `step(a) -> (obs, reward, terminated, truncated,
info)`) works. Register it once:

```python
from neuralrnn import register_env, make_env, SyncVectorEnv

register_env("mytask", lambda **kw: MyEnv(**kw))
envs = SyncVectorEnv([make_env("mytask", seed=i) for i in range(32)])
```

Neurogym tasks need no code: `make_env("neurogym:PerceptualDecisionMaking-v0")`.

Envs should put per-episode scalar statistics in `info["episode"]` (e.g.
`{"r": return, "p_correct": ...}`) — `RLTrainer` averages and logs them
generically.

### 2. Agent

```python
from neuralrnn import ActorCriticConfig, ActorCriticModel
from neuralrnn.models.gated_rnn import GatedRNNConfig

agent = ActorCriticModel(ActorCriticConfig(
    core_config=GatedRNNConfig(input_dim=3, latent_dim=64).to_dict(),
    action_dim=3, action_type="discrete",   # or "continuous" / "none" (critic-only)
))
```

`core_config` accepts the serialized config of **any** registered RNN family
(ctrnn, gated_rnn, lowrank, plrnn, ...). The agent implements the full
`NeuralDynamicsModel` contract (`recurrence` / `readout` / `init_state`), so
every `neuralrnn.analysis` tool works on it unchanged.

### 3. Loss (algorithm)

```python
from neuralrnn import build_rl_loss, RLTrainingArguments, RLTrainer

loss = build_rl_loss("ppo", clip_coef=0.1)        # or "reinforce"
args = RLTrainingArguments(update_epochs=4, num_minibatches=4,   # PPO
                           estimator="gae", gamma=0.99)
# REINFORCE is strictly on-policy: update_epochs=1, num_minibatches=1
trainer = RLTrainer(agent, envs, args, loss=loss, eval_fn=my_eval_fn)
history = trainer.train()
```

New algorithms plug in without touching the trainer:

```python
from neuralrnn import register_rl_loss

@register_rl_loss("my_algo")
class MyAlgoLoss:
    def __call__(self, new_logp, old_logp, advantages, new_values,
                 old_values, returns, entropies, states=None):
        ...
        return loss, {"pg_loss": ..., "v_loss": ..., "entropy": ...}
```

## Reproduction notebooks using this layer

| notebook | task | env | loss |
|---|---|---|---|
| `18_rl_rnn_paradigmA.ipynb` | Battista-2026 economic choice | `envs/echoice.py` | PPO (discrete) |
| `19_value_rnn_paradigmA.ipynb` | Qian-Burrell-2024 value RNN | offline dataset + `TDObjective` | semi-gradient TD |
| `s1_plumetracknets.ipynb` | Singh-2023 plume tracking | `envs/plume.py` | PPO (continuous) |
| `s2_metalearning.ipynb` | Jensen-2024 maze meta-learning / replay | `envs/maze.py` + `WorldModelPlanner` | A2C + world-model aux (`A2CPredLoss`) |
| `reinforcement_learning.ipynb` | Song-2017 RDM (tutorial) | neurogym PDM | REINFORCE |
