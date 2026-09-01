"""Environment base utilities for the RL layer.

Unified **gymnasium-style** API across old-gym (4-tuple ``step``, ``reset() -> obs``)
and gymnasium (5-tuple ``step``, ``reset() -> (obs, info)``) environments:

    reset_env(env)              -> (obs, info)
    step_env(env, action)       -> (obs, reward, terminated, truncated, info)

``SyncVectorEnv`` is a minimal synchronous vector wrapper (no dependency on
``gymnasium.vector`` semantics): on ``terminated | truncated`` the env is auto-reset,
the final observation is exposed in ``info["final_observation"]``, and per-episode
statistics are expected in ``info["episode"]`` (dict) when an episode ends.
"""
from __future__ import annotations

import numpy as np


def reset_env(env, *, seed=None, options=None):
    """Reset an env regardless of old-gym / gymnasium API. Returns (obs, info)."""
    try:
        out = env.reset(seed=seed, options=options)
    except TypeError:
        out = env.reset()
        if seed is not None and hasattr(env, "seed"):
            env.seed(seed)
    if isinstance(out, tuple) and len(out) == 2:
        return out
    return out, {}


def step_env(env, action):
    """Step an env regardless of old-gym / gymnasium API.

    Returns (obs, reward, terminated, truncated, info).
    """
    out = env.step(action)
    if len(out) == 5:
        return out
    obs, reward, done, info = out
    return obs, reward, bool(done), False, info


class SyncVectorEnv:
    """Minimal synchronous vector environment with auto-reset.

    Args:
        env_fns: list of callables, each returning a fresh single env.
    """

    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)
        env0 = self.envs[0]
        self.single_observation_space = env0.observation_space
        self.single_action_space = env0.action_space

    def reset(self, *, seed=None):
        obs, infos = [], []
        for i, env in enumerate(self.envs):
            o, info = reset_env(env, seed=None if seed is None else seed + i)
            obs.append(o)
            infos.append(info)
        return np.stack(obs), infos

    def step(self, actions):
        obs, rewards, terms, truncs, infos = [], [], [], [], []
        for env, a in zip(self.envs, actions):
            o, r, term, trunc, info = step_env(env, a)
            if term or trunc:
                info = dict(info)
                info["final_observation"] = o
                o, _ = reset_env(env)
            obs.append(o)
            rewards.append(r)
            terms.append(term)
            truncs.append(trunc)
            infos.append(info)
        return (np.stack(obs), np.asarray(rewards, dtype=np.float32),
                np.asarray(terms, dtype=np.float32), np.asarray(truncs, dtype=np.float32),
                infos)

    def close(self):
        for env in self.envs:
            close = getattr(env, "close", None)
            if callable(close):
                close()
