"""Recurrent rollout buffer for on-policy RL with RNN agents.

Storage layout follows CleanRL ``ppo_atari_lstm.py``: time-first tensors
``(T, N, ...)``; only the **chunk-initial hidden state** is stored (sequences are
replayed through the RNN during the update for BPTT). Minibatches are contiguous
**env-axis slices** so temporal structure is never broken.
"""
from __future__ import annotations

import numpy as np
import torch


class RecurrentRolloutBuffer:
    """Preallocated rollout storage for one policy iteration.

    Args:
        num_steps: rollout length T per env.
        num_envs: number of parallel envs N.
        obs_shape: observation shape (e.g. (obs_dim,)).
        device: torch device for storage.
        action_shape: per-step action shape; () for discrete actions,
            (action_dim,) for continuous actions.
        action_dtype: torch.long for discrete, torch.float32 for continuous.
        aux_dim: if > 0, also store per-step auxiliary targets
            (T, N, aux_dim) — e.g. world-model prediction targets for
            aux-head losses like ``a2c_pred``.
    """

    def __init__(self, num_steps: int, num_envs: int, obs_shape: tuple, device,
                 action_shape: tuple = (), action_dtype=torch.long,
                 aux_dim: int = 0):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = device
        self.obs = torch.zeros((num_steps, num_envs) + tuple(obs_shape), device=device)
        self.actions = torch.zeros((num_steps, num_envs) + tuple(action_shape),
                                   dtype=action_dtype, device=device)
        self.logprobs = torch.zeros((num_steps, num_envs), device=device)
        self.rewards = torch.zeros((num_steps, num_envs), device=device)
        self.dones = torch.zeros((num_steps, num_envs), device=device)
        self.values = torch.zeros((num_steps, num_envs), device=device)
        self.aux_dim = int(aux_dim)
        self.aux_targets = (torch.zeros((num_steps, num_envs, self.aux_dim),
                                        device=device)
                            if self.aux_dim > 0 else None)
        self.z_init: torch.Tensor | None = None  # (N, M) hidden state at chunk start
        self.returns: torch.Tensor | None = None
        self.advantages: torch.Tensor | None = None

    def add(self, step: int, obs, done, action, logprob, value, reward,
            aux=None) -> None:
        """Write one time slice. All inputs are (N, ...) tensors on ``device``."""
        self.obs[step] = obs
        self.dones[step] = done
        self.actions[step] = action
        self.logprobs[step] = logprob
        self.values[step] = value
        self.rewards[step] = reward
        if self.aux_targets is not None:
            if aux is None:
                raise ValueError("buffer created with aux_dim > 0 but add() "
                                 "received aux=None")
            self.aux_targets[step] = aux

    def compute_returns(self, estimator, next_value, next_done, gamma: float) -> None:
        """Fill ``self.returns`` / ``self.advantages`` via a return estimator."""
        self.returns, self.advantages = estimator.compute(
            self.rewards, self.values, self.dones, next_value, next_done, gamma)

    def iterate_minibatches(self, num_minibatches: int, rng: np.random.Generator):
        """Yield batch-first minibatches as contiguous env-axis slices.

        Each yield is a dict with:
            obs (nB, T, K), dones (nB, T), actions (nB, T), logprobs (nB, T),
            advantages (nB, T), returns (nB, T), values (nB, T), z_init (nB, M);
            plus ``aux_targets`` (nB, T, aux_dim) when aux_dim > 0.
        """
        assert self.returns is not None, "call compute_returns first"
        assert self.num_envs % num_minibatches == 0
        envs_per_batch = self.num_envs // num_minibatches
        envinds = np.arange(self.num_envs)
        rng.shuffle(envinds)
        for start in range(0, self.num_envs, envs_per_batch):
            mb = envinds[start:start + envs_per_batch]
            mb_t = torch.as_tensor(mb, device=self.device)
            batch = {
                "obs": self.obs[:, mb].transpose(0, 1).contiguous(),
                "dones": self.dones[:, mb].transpose(0, 1).contiguous(),
                "actions": self.actions[:, mb].transpose(0, 1).contiguous(),
                "logprobs": self.logprobs[:, mb].transpose(0, 1).contiguous(),
                "advantages": self.advantages[:, mb].transpose(0, 1).contiguous(),
                "returns": self.returns[:, mb].transpose(0, 1).contiguous(),
                "values": self.values[:, mb].transpose(0, 1).contiguous(),
                "z_init": self.z_init[mb_t].contiguous(),
            }
            if self.aux_targets is not None:
                batch["aux_targets"] = self.aux_targets[:, mb].transpose(0, 1).contiguous()
            yield batch
