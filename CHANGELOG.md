# Changelog

All notable changes to NeuralRNN are documented in this file.

## [0.5.0] - Unreleased

### Licensing

- Changed the project license from MIT to GPL-3.0-only so that the distributed
  work is compatible with the GPL-licensed implementations adapted by the
  framework.
- Added a file-level source and license ledger for the installable package,
  retained the applicable upstream MIT and Apache notices, and included all
  legal files in wheel and source archives.
- Clarified which components adapt licensed source code and which independently
  reimplement published methods, task protocols or interfaces.

### Packaging

- Replaced the legacy Gym dependency with Gymnasium. A standard
  `pip install neuralrnn` installs the complete supported task, RL and notebook
  stack without requiring extras suffixes.

### Fixed

- Kept the nn-brain tutorial as the E-I default recipe; the pycog constraints,
  initialization, spectral scaling, trainable initial current, and bias policy
  remain available through explicit configuration.
- Replaced obsolete and stochastic test skips with deterministic reconstruction,
  fixed-point, and manifold checks. The Python 3.11 suite now completes with
  857 passed, 0 skipped, and 0 failed tests (75% statement coverage).

## [0.4.0] - 2026-09-02

Reinforcement learning layer: a generic RL-RNN framework (environment / agent / loss / trainer) plus reproductions of published RL-RNN studies.

### Added

- **RL layer (`neuralrnn.rl`)**
  - Environments: `make_env` / `register_env` registry; `SyncVectorEnv` with auto-reset and episode statistics; `reset_env` / `step_env` unifying old-gym and gymnasium APIs; `NeurogymEnvAdapter` (`"neurogym:<Task>"`, one trial = one episode) and `"gym:<id>"` passthrough.
  - Built-in environments: `EchoiceEnv` (multi-task economic choice, Battista-2026), `PlumeEnv` + `plume_sim` simulator with on-disk caching (odor plume tracking, Singh-2023), `MazeEnv` (toroidal maze with a "think" meta-action, Jensen-2024).
  - `RecurrentRolloutBuffer`: time-first rollout storage with chunk-initial hidden states for BPTT replay, env-slice minibatches, optional auxiliary targets.
  - Return estimators: `NStepReturns` and `GAE` (CleanRL conventions), pluggable via `build_return_estimator`.
  - Losses: `PPOLoss` (clipped surrogate), `ReinforceLoss` (REINFORCE with value baseline, Song-Yang-Wang-2017 recipe), `A2CPredLoss` (policy gradient + auxiliary world-model cross-entropy, Jensen-2024 recipe); pluggable via `RL_LOSS_REGISTRY` / `register_rl_loss`.
  - `RLTrainer` + `RLTrainingArguments`: on-policy recurrent PPO/A2C loop with episode-boundary hidden-state masking, advantage estimators, constraint
    projection hooks, eval callbacks with early stopping, checkpointing via `save_pretrained`, training curves and history logging.
  - `collect_episodes`: full-episode recording (observations, hidden states, actions, rewards, values) for neural/behavioral analysis.
  - `VecNormalize` / `OnlineObsNorm` / `RunningMeanStd`: observation and reward normalization with SB3 semantics.
  - `WorldModelPlanner` / `bind_planners`: model-based internal rollouts fed back as observations (planning as a meta-action, Jensen-2024).
  - Task-specific utilities for the reproductions: `plume_agents`, `plume_eval`, `probe` (value-RNN analyses).
- **Model layer**: new `actor_critic` family. `ActorCriticModel` wraps any registered RNN core (ctrnn, gated_rnn, ei_rnn, ...) with a policy head
  (Categorical for discrete actions, diagonal Gaussian for continuous) and a scalar value head; critic-only mode (`action_type="none"`); optional
  auxiliary prediction head (world model); non-negative readout constraint projection; full `save_pretrained` / `AutoModel.from_pretrained` roundtrip.
  Implements the standard `NeuralDynamicsModel` contract, so all analysis modules (fixed points, PCA, vector fields) work on agent dynamics unchanged.
- **Train layer**: `TDObjective` (semi-gradient TD(0) value learning for critic-only paradigms, Qian & Burrell 2024).
- **Notebooks**
  - `reinforcement_learning.ipynb`: tutorial for RL modeling with NeuralRNN (Song, Yang & Wang 2017).
  - `18_rl_rnn_paradigmA.ipynb`: Battista-2026 economic choice (PPO).
  - `19_value_rnn_paradigmA.ipynb`: Qian & Burrell 2024 value RNN (TD).
  - `s1_plumetracknets.ipynb`: Singh-2023 plume tracking (continuous-action PPO, full figure reproduction).
  - `s2_metalearning.ipynb`: Jensen-2024 maze meta-learning with planning.
- **Tests**: `test/test_rl.py` (68 tests covering return estimators, rollout buffer, env APIs, actor-critic agents, losses, VecNormalize, TD objective,
  planning, and an end-to-end trainer smoke run). RL layer coverage 9.7% -> 70.9%; overall test coverage 59% -> 73% (635 tests).
