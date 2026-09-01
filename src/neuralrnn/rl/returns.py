"""Return / advantage estimators for on-policy RL (pluggable, CleanRL conventions).

Both estimators operate on time-first rollout tensors:
    rewards, values, dones: (T, N)     next_value, next_done: (N,)
with the CleanRL ``done`` convention: dones[t] marks that the transition INTO
step t ended an episode (so rewards[t] is the reward OF step t, and
dones[t+1] tells whether step t's transition was terminal).
"""
from __future__ import annotations

import torch


class NStepReturns:
    """Bootstrapped n-step (full-rollout) returns (PPO with gae=False).

        R_t = r_t + gamma * (1 - d_{t+1}) * R_{t+1},   R_T = V(s_T)
        A_t = R_t - V_t

    This is the estimator used e.g. by Battista-2026; GAE with
    ``gae_lambda`` recovers the bias-variance trade-off of CleanRL defaults.
    """

    def compute(self, rewards, values, dones, next_value, next_done, gamma: float):
        T = rewards.shape[0]
        returns = torch.zeros_like(rewards)
        for t in reversed(range(T)):
            if t == T - 1:
                nextnonterminal = 1.0 - next_done
                next_return = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                next_return = returns[t + 1]
            returns[t] = rewards[t] + gamma * nextnonterminal * next_return
        advantages = returns - values
        return returns, advantages


class GAE:
    """Generalized Advantage Estimation (CleanRL default; used e.g. by Singh-2023).

        delta_t = r_t + gamma * V_{t+1} * (1 - d_{t+1}) - V_t
        A_t = delta_t + gamma * lambda * (1 - d_{t+1}) * A_{t+1}
        R_t = A_t + V_t
    """

    def __init__(self, gae_lambda: float = 0.95):
        self.gae_lambda = gae_lambda

    def compute(self, rewards, values, dones, next_value, next_done, gamma: float):
        T = rewards.shape[0]
        advantages = torch.zeros_like(rewards)
        lastgaelam = 0
        for t in reversed(range(T)):
            if t == T - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
            delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = (
                delta + gamma * self.gae_lambda * nextnonterminal * lastgaelam)
        returns = advantages + values
        return returns, advantages


RETURN_ESTIMATORS = {"nstep": NStepReturns, "gae": GAE}


def build_return_estimator(name: str, gae_lambda: float = 0.95):
    """Factory: ``build_return_estimator("nstep")`` / ``("gae", gae_lambda=0.95)``."""
    if name not in RETURN_ESTIMATORS:
        raise KeyError(f"Unknown return estimator '{name}'. "
                       f"Available: {sorted(RETURN_ESTIMATORS)}")
    if name == "gae":
        return GAE(gae_lambda)
    return RETURN_ESTIMATORS[name]()
