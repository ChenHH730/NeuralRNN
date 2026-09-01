"""Step-based RL adapter for neurogym tasks.

Gives neurogym ``TrialEnv``s a gymnasium-style step/reset interface for the RL layer
(one env step = one dt; episode = one trial). This complements — does not replace —
the dataset-oriented ``NeurogymDataset`` used by the supervised trainer.
"""
from __future__ import annotations

import numpy as np

from .base import reset_env, step_env


class NeurogymEnvAdapter:
    """Gymnasium-style wrapper around a neurogym task.

    Neurogym ``TrialEnv``s run trials back-to-back and never return
    ``terminated``; trial boundaries are signaled by ``info["new_trial"]``.
    This adapter turns each trial into an episode: when ``new_trial`` fires,
    it reports ``terminated=True`` and attaches ``info["episode"]`` with the
    finished trial's statistics (``"r"`` return, ``"l"`` length, plus
    ``"performance"`` when the task provides it), so ``SyncVectorEnv`` /
    ``RLTrainer`` reset the agent's hidden state at trial boundaries and log
    per-trial returns — no task-specific code needed.

    Args:
        task: neurogym task id (e.g. "PerceptualDecisionMaking-v0" or a short name;
            "-v0" is appended when missing).
        seed: env seed.
        **env_kwargs: forwarded to ``neurogym.make`` (e.g. dt, timing).
    """

    def __init__(self, task: str, seed: int | None = None, **env_kwargs):
        import neurogym as ngym

        task_id = task if "-v" in task else f"{task}-v0"
        self.env = ngym.make(task_id, **env_kwargs).unwrapped
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self._seed = seed
        if seed is not None and hasattr(self.env, "seed"):
            self.env.seed(seed)
        self._ep_r = 0.0
        self._ep_l = 0

    def reset(self, *, seed: int | None = None, options=None):
        """Reset the underlying trial env. Returns (obs, info)."""
        self._ep_r = 0.0
        self._ep_l = 0
        return reset_env(self.env, seed=seed)

    def step(self, action):
        """Step one dt. Returns (obs, reward, terminated, truncated, info)."""
        obs, r, term, trunc, info = step_env(self.env, action)
        self._ep_r += r
        self._ep_l += 1
        if info.get("new_trial", False):
            term = True
            info = dict(info)
            ep = {"r": self._ep_r, "l": self._ep_l}
            perf = info.get("performance")
            if perf is not None:
                try:
                    ep["performance"] = float(np.asarray(perf, dtype=float).ravel()[0])
                except (TypeError, ValueError, IndexError):
                    pass
            info["episode"] = ep
            self._ep_r = 0.0
            self._ep_l = 0
        return obs, r, term, trunc, info
