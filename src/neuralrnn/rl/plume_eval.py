"""Fixed-grid evaluation for plume-tracking agents (Singh et al. 2023).

Port of the evaluation assay in the reference ``code/ppo/evalCli.py``:
a fixed 240-episode grid (3 x-locations x 2 time offsets x 5 y-locations
x 8 headings) per dataset, run with the deterministic policy (Gaussian mean).

Eval environments use the reference eval configuration: no flipping, no odor
scaling, dense plume (birthx = birthx_max = 1.0), qvar = 0, no observation or
action noise, squashed actions; ``switch*`` datasets start at t_val_min=58 s
with reset_offset_tmax=3 s so episodes straddle the wind switch.

Known deviation: the reference evalCli adds +100 to any reward > 9 (a debug
hack per the source comment); that is NOT ported here.
"""
from __future__ import annotations

import os
import pickle

import numpy as np
import torch

from .envs.plume import PlumeEnv

EVAL_DATASETS = ("constantx5b5", "switch45x5b5", "noisy3x5b5")
SPARSITY_LEVELS = (0.8, 0.6, 0.4, 0.2)
_GRID_XS = (4.0, 6.0, 8.0)
_GRID_TIMES = (0.0, 1.0)
_GRID_ANGLES = np.linspace(0, 2, 9)[:8] * np.pi
_GRID_Y_STRAY = 0.5


def make_eval_env(dataset: str, data_dir: str, *,
                  birthx_max: float = 1.0, squash_action: bool = True,
                  seed: int = 0, online_obs_norm: bool = True,
                  **overrides) -> PlumeEnv:
    """Evaluation PlumeEnv with the reference eval configuration.

    ``online_obs_norm=True`` wraps the env in ``OnlineObsNorm``, reproducing
    the reference evalCli setup (a fresh VecNormalize in training mode around
    the eval env, so obs are normalized by stats accumulated over the eval
    stream itself). Agents trained with ``VecNormalize`` must be evaluated
    with it (default); pass False for untrained-agent sanity checks."""
    kwargs = dict(dataset=dataset, data_dir=data_dir,
                  flipping=False, odor_scaling=False,
                  birthx=1.0, birthx_max=birthx_max, qvar=0.0,
                  obs_noise=0.0, act_noise=0.0, squash_action=squash_action,
                  loc_algo="fixed", time_algo="fixed", angle_algo="fixed",
                  seed=seed)
    if "switch" in dataset:
        kwargs.update(t_val_min=58.0, reset_offset_tmax=3.0)
    kwargs.update(overrides)
    env = PlumeEnv(**kwargs)
    if online_obs_norm:
        from .vecnormalize import OnlineObsNorm
        env = OnlineObsNorm(env)
    return env


def fixed_eval_grid(env: PlumeEnv) -> np.ndarray:
    """Build the fixed evaluation grid; (240, 4) rows of
    [loc_y, angle, loc_x, time_offset]. Port of evalCli's meshgrid block."""
    grids = []
    for fixed_x in _GRID_XS:
        for time_offset in _GRID_TIMES:
            env.fixed_x, env.fixed_time_offset = fixed_x, time_offset
            env.reset()
            px, py = env.puff_centers()
            band = py[np.abs(px - fixed_x) <= 0.5]
            y_min, y_max = (np.quantile(band, [0.0, 1.0]) if len(band)
                            else np.array([-1.0, 1.0]))
            if ("switch" in env.dataset) or ("noisy" in env.dataset):
                ys = np.linspace(y_min, y_max, 5)  # on-plume starts only
            else:  # constant: include two off-plume starts
                ys = np.concatenate([[y_min - _GRID_Y_STRAY],
                                     np.linspace(y_min, y_max, 3),
                                     [y_max + _GRID_Y_STRAY]])
            gy, ga = np.meshgrid(ys, _GRID_ANGLES)
            grid = np.stack([gy.ravel(), ga.ravel(),
                             np.full(gy.size, fixed_x),
                             np.full(gy.size, time_offset)], axis=1)
            grids.append(grid)
    return np.concatenate(grids, axis=0)


@torch.no_grad()
def evaluate_agent(agent, env: PlumeEnv, grid: np.ndarray | None = None, *,
                   deterministic: bool = True, device: str = "cpu",
                   progress: bool = False) -> list[dict]:
    """Run the agent over the fixed grid (or the env's stochastic resets if
    ``grid=None``), recording full behavioral + neural traces per episode.

    Returns a list of per-episode dicts:
        obs (T, K), states (T, M), actions_raw (T, A) policy mean (pre-squash),
        actions (T, A) squashed to [0, 1] (pre-flip), rewards (T,),
        values (T,), loc_x/loc_y (T+1,) incl. initial position,
        agent_angle (T,), wind_x/wind_y (T,), stray_distance (T,), t_idx (T,),
        plus scalars: outcome ("HOME"/"OOB"/"OOT"), reward_total, length,
        flipx, init_x/init_y/init_angle/time_offset (grid row), dataset.
    """
    was_training = agent.training
    agent.eval()
    agent = agent.to(device)
    if grid is None:
        grid = fixed_eval_grid(env)

    episodes = []
    iterator = enumerate(grid)
    if progress:
        from tqdm import tqdm
        iterator = tqdm(iterator, total=len(grid), desc="eval", unit="ep")
    for i_ep, (loc_y, angle, loc_x, time_offset) in iterator:
        env.fixed_y = float(loc_y)
        env.fixed_angle = float(angle)
        env.fixed_x = float(loc_x)
        env.fixed_time_offset = float(time_offset)
        obs, _ = env.reset()

        z = agent.init_state(1, device)
        done = torch.ones(1, device=device)  # reset hidden state at t=0
        rec = {k: [] for k in ("obs", "states", "actions_raw", "actions",
                               "rewards", "values", "agent_angle",
                               "wind_x", "wind_y", "stray_distance", "t_idx")}
        locs = [env.agent_location_init.copy()]
        outcome, ep_reward = None, 0.0
        while True:
            x_t = torch.as_tensor(obs, dtype=torch.float32,
                                  device=device).view(1, -1)
            z_in = z * (1.0 - done).unsqueeze(-1)
            z = agent.core.recurrence(x_t, z_in)
            mean = agent.policy_mode(z)  # continuous: actor mean
            value = agent.value(z)
            if deterministic:
                action_raw = mean
            else:
                action_raw = mean + agent.log_std.exp() * torch.randn_like(mean)
            a_np = action_raw.squeeze(0).cpu().numpy()
            obs, reward, terminated, truncated, info = env.step(a_np)
            done = torch.tensor([float(terminated or truncated)],
                                device=device)

            a_env = np.clip((np.tanh(a_np) + 1.0) / 2.0, 0.0, 1.0) \
                if env.squash_action else np.clip(a_np, 0.0, 1.0)
            rec["obs"].append(obs.copy())
            rec["states"].append(z.squeeze(0).cpu().numpy())
            rec["actions_raw"].append(a_np.copy())
            rec["actions"].append(a_env)
            rec["rewards"].append(reward)
            rec["values"].append(float(value.squeeze(0).cpu()))
            rec["agent_angle"].append(
                float(np.angle(info["angle"][0] + 1j * info["angle"][1])))
            rec["wind_x"].append(float(info["wind_ground"][0]))
            rec["wind_y"].append(float(info["wind_ground"][1]))
            rec["stray_distance"].append(float(info["stray_distance"]))
            rec["t_idx"].append(int(info["t_idx"]))
            locs.append(info["location"].copy())
            ep_reward += reward
            if terminated or truncated:
                outcome = info["episode"]["outcome"]
                break

        locs = np.asarray(locs)
        episodes.append({
            "obs": np.asarray(rec["obs"], dtype=np.float32),
            "states": np.asarray(rec["states"], dtype=np.float32),
            "actions_raw": np.asarray(rec["actions_raw"], dtype=np.float32),
            "actions": np.asarray(rec["actions"], dtype=np.float32),
            "rewards": np.asarray(rec["rewards"], dtype=np.float32),
            "values": np.asarray(rec["values"], dtype=np.float32),
            "loc_x": locs[:, 0], "loc_y": locs[:, 1],
            "agent_angle": np.asarray(rec["agent_angle"], dtype=np.float32),
            "wind_x": np.asarray(rec["wind_x"], dtype=np.float32),
            "wind_y": np.asarray(rec["wind_y"], dtype=np.float32),
            "stray_distance": np.asarray(rec["stray_distance"],
                                         dtype=np.float32),
            "t_idx": np.asarray(rec["t_idx"], dtype=np.int64),
            "outcome": outcome, "reward_total": float(ep_reward),
            "length": len(rec["rewards"]), "flipx": float(env.flipx),
            "init_x": float(locs[0, 0]), "init_y": float(locs[0, 1]),
            "init_angle": float(angle), "time_offset": float(time_offset),
            "dataset": env.dataset,
        })
    if was_training:
        agent.train()
    return episodes


def eval_all_datasets(agent, datasets=EVAL_DATASETS, *, data_dir: str,
                      deterministic: bool = True,
                      seed: int = 0, progress: bool = False) -> dict:
    """Evaluate one agent on each dataset; returns {dataset: episodes}."""
    results = {}
    for dataset in datasets:
        env = make_eval_env(dataset, data_dir, seed=seed)
        results[dataset] = evaluate_agent(
            agent, env, deterministic=deterministic, progress=progress)
        env.close()
    return results


def sparsity_sweep(agent, dataset: str = "constantx5b5", *,
                   birthx_values=SPARSITY_LEVELS, data_dir: str,
                   deterministic: bool = True,
                   seed: int = 0, progress: bool = False) -> dict:
    """Sparsity generalization: evaluate on plumes sparsified at load time
    (birthx_max); returns {birthx_max: episodes}."""
    results = {}
    for bx in birthx_values:
        env = make_eval_env(dataset, data_dir,
                            birthx_max=bx, seed=seed)
        results[bx] = evaluate_agent(agent, env, deterministic=deterministic,
                                     progress=progress)
        env.close()
    return results


def save_eval(path: str, episodes) -> None:
    """Pickle an episode list (or {key: episodes} dict) to ``path``."""
    os.makedirs(os.path.dirname(os.fspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(episodes, f)


def load_eval(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)
