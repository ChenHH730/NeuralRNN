"""VecNormalize for the NeuralRNN RL layer (SB3 semantics; Singh-2023 port).

The reference plumetracknets PPO training wraps its vector env in SB3's
``VecNormalize`` (``make_vec_envs`` with ``gamma=0.99``): observations are
normalized by a running mean/std (clipped to ±10) and rewards are scaled by
the running std of the discounted return (clipped to ±10). This is crucial
for training stability on the plume task, whose raw rewards span ±100.

Two pieces:

- ``VecNormalize``: duck-typed wrapper around ``SyncVectorEnv`` (training).
  Fresh stats per training stage, matching the reference, which rebuilds the
  vec env (and thus the normalizer) at every curriculum stage.
- ``OnlineObsNorm``: single-env gym wrapper used at EVALUATION time. The
  reference eval (`evalCli.py`) wraps the eval env in a fresh VecNormalize in
  training mode, so observations are normalized by stats accumulated over the
  eval stream itself; this wrapper reproduces exactly that (rewards are left
  raw — the reference logs raw rewards via the Monitor inside the wrapper).
"""
from __future__ import annotations

import numpy as np


class RunningMeanStd:
    """Running mean/var (Welford), SB3-compatible."""

    def __init__(self, shape=(), epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = float(epsilon)

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0] if x.ndim > len(self.mean.shape) else 1
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count) -> None:
        delta = batch_mean - self.mean
        tot = self.count + batch_count
        self.mean = self.mean + delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot
        self.var = m_2 / tot
        self.count = tot


class VecNormalize:
    """Duck-typed drop-in around SyncVectorEnv (obs + reward normalization).

    SB3 semantics: obs -> clip((obs - mean) / sqrt(var + eps), ±clip_obs);
    returns r <- r * gamma each step, ret_rms updated BEFORE reward scaling,
    reward -> clip(r / sqrt(ret_var + eps), ±clip_reward); per-env returns
    reset at done. Stats update only when ``training=True``.
    """

    def __init__(self, venv, training: bool = True, gamma: float = 0.99,
                 clip_obs: float = 10.0, clip_reward: float = 10.0,
                 epsilon: float = 1e-8):
        self.venv = venv
        self.training = training
        self.gamma = gamma
        self.clip_obs = clip_obs
        self.clip_reward = clip_reward
        self.epsilon = epsilon
        obs_dim = venv.single_observation_space.shape
        self.obs_rms = RunningMeanStd(shape=obs_dim)
        self.ret_rms = RunningMeanStd(shape=())
        self.returns = np.zeros(venv.num_envs, dtype=np.float64)

    # ---- duck-typed SyncVectorEnv interface ----
    @property
    def num_envs(self) -> int:
        return self.venv.num_envs

    @property
    def single_observation_space(self):
        return self.venv.single_observation_space

    @property
    def single_action_space(self):
        return self.venv.single_action_space

    def _norm_obs(self, obs: np.ndarray) -> np.ndarray:
        if self.training:
            self.obs_rms.update(obs)
        return np.clip((obs - self.obs_rms.mean) /
                       np.sqrt(self.obs_rms.var + self.epsilon),
                       -self.clip_obs, self.clip_obs).astype(np.float32)

    def _norm_reward(self, rews: np.ndarray) -> np.ndarray:
        self.returns = self.returns * self.gamma + rews
        if self.training:
            self.ret_rms.update(self.returns.reshape(-1, 1))
        out = rews / np.sqrt(self.ret_rms.var + self.epsilon)
        return np.clip(out, -self.clip_reward, self.clip_reward).astype(
            np.float32)

    def reset(self, **kwargs):
        try:
            out = self.venv.reset(**kwargs)
        except TypeError:
            out = self.venv.reset()
        if isinstance(out, tuple):  # SyncVectorEnv: (obs, infos)
            return self._norm_obs(out[0]), out[1]
        return self._norm_obs(out)

    def step(self, actions):
        obs, rews, terms, truncs, infos = self.venv.step(actions)
        obs = self._norm_obs(obs)
        rews = self._norm_reward(np.asarray(rews, dtype=np.float64))
        dones = np.asarray(terms, dtype=bool) | np.asarray(truncs, dtype=bool)
        self.returns[dones] = 0.0
        return obs, rews, terms, truncs, infos

    def close(self) -> None:
        self.venv.close()

    def state_dict(self) -> dict:
        return {"obs_mean": self.obs_rms.mean, "obs_var": self.obs_rms.var,
                "obs_count": self.obs_rms.count,
                "ret_mean": self.ret_rms.mean, "ret_var": self.ret_rms.var,
                "ret_count": self.ret_rms.count}


class OnlineObsNorm:
    """Single-env gym wrapper: online (eval-stream) observation normalization.

    Reproduces the reference eval setup (a fresh VecNormalize in training
    mode around the eval env): stats start from scratch and accumulate over
    the evaluation episode stream. Rewards/infos pass through unchanged.
    """

    def __init__(self, env, clip_obs: float = 10.0, epsilon: float = 1e-8):
        object.__setattr__(self, "env", env)
        object.__setattr__(self, "obs_rms",
                           RunningMeanStd(shape=env.observation_space.shape))
        object.__setattr__(self, "clip_obs", clip_obs)
        object.__setattr__(self, "epsilon", epsilon)
        # transparent attribute access (PlumeEnv fields used by eval/plots)
        object.__setattr__(self, "action_space", env.action_space)
        object.__setattr__(self, "observation_space", env.observation_space)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "env"), name)

    def __setattr__(self, name, value):
        # forward config writes (e.g. fixed_x for the eval grid) to the env
        if name in ("env", "obs_rms", "clip_obs", "epsilon",
                    "action_space", "observation_space"):
            object.__setattr__(self, name, value)
        else:
            setattr(self.env, name, value)

    def _norm(self, obs: np.ndarray) -> np.ndarray:
        self.obs_rms.update(np.asarray(obs, dtype=np.float64).reshape(1, -1))
        return np.clip((obs - self.obs_rms.mean) /
                       np.sqrt(self.obs_rms.var + self.epsilon),
                       -self.clip_obs, self.clip_obs).astype(np.float32)

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        obs, info = out if isinstance(out, tuple) else (out, None)
        obs = self._norm(obs)
        return (obs, info) if info is not None else obs

    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(action)
        return self._norm(obs), rew, term, trunc, info
