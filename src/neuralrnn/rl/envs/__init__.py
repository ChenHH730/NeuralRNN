"""Environment registry for the RL layer.

``make_env(name, **kwargs)`` returns a thunk (callable -> fresh env), ready for
``SyncVectorEnv``. Built-in names:

    "echochoice"        -> EchoiceEnv (Battista-2026 economic choice tasks)
    "plume"             -> PlumeEnv (Singh-2023 odor plume tracking)
    "maze"              -> MazeEnv (Jensen-2024 toroidal maze + think action)
    "neurogym:<Task>"   -> NeurogymEnvAdapter (any installed neurogym task)
    "gym:<id>"          -> gymnasium.make("<id>") passthrough

Custom envs can be registered with ``register_env(name, factory)`` where the
factory takes **kwargs and returns an env with observation_space/action_space.
"""
from __future__ import annotations

from typing import Callable

ENV_REGISTRY: dict[str, Callable] = {}


def register_env(name: str, factory: Callable) -> None:
    """Register an env factory ``factory(**kwargs) -> env`` under ``name``."""
    ENV_REGISTRY[name] = factory


def _echochoice_factory(**kwargs):
    from .echoice import EchoiceEnv
    return EchoiceEnv(**kwargs)


def _plume_factory(**kwargs):
    from .plume import PlumeEnv
    return PlumeEnv(**kwargs)


def _maze_factory(**kwargs):
    from .maze import MazeEnv
    return MazeEnv(**kwargs)


def _neurogym_factory(task: str, **kwargs):
    from .neurogym_env import NeurogymEnvAdapter
    return NeurogymEnvAdapter(task, **kwargs)


def _gym_factory(env_id: str, **kwargs):
    try:
        import gymnasium as gym
    except ImportError:  # pragma: no cover
        import gym
    env = gym.make(env_id, **kwargs)
    # adds info["episode"] = {"r", "l", "t"} at episode end (trainer metric source)
    return gym.wrappers.RecordEpisodeStatistics(env)


register_env("echochoice", _echochoice_factory)
register_env("plume", _plume_factory)
register_env("maze", _maze_factory)


def make_env(name: str, **kwargs) -> Callable:
    """Return a thunk that builds a fresh env. ``name`` as described above."""
    if name.startswith("neurogym:"):
        task = name.split(":", 1)[1]
        return lambda: _neurogym_factory(task, **kwargs)
    if name.startswith("gym:"):
        env_id = name.split(":", 1)[1]
        return lambda: _gym_factory(env_id, **kwargs)
    if name not in ENV_REGISTRY:
        raise KeyError(
            f"Unknown env '{name}'. Registered: {sorted(ENV_REGISTRY)}; "
            "or use 'neurogym:<Task>' / 'gym:<id>'.")
    return lambda: ENV_REGISTRY[name](**kwargs)


from .base import SyncVectorEnv, reset_env, step_env  # noqa: E402
from .echoice import EchoiceEnv  # noqa: E402

__all__ = ["make_env", "register_env", "ENV_REGISTRY", "SyncVectorEnv",
           "reset_env", "step_env", "EchoiceEnv"]
