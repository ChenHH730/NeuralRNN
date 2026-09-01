"""Plume-tracking environment (faithful port of Singh et al. 2023).

Port of ``PlumeEnvironment`` from
``reference_project/reinforcement_learning/Singh-2023-plumetracknets/code/plume_env.py``
onto the plume data layer of ``plume_sim.py``, with a gymnasium-style API
(``reset() -> (obs, info)``, ``step() -> (obs, reward, terminated, truncated, info)``).

Behavioral contract (all ported line-by-line from the reference):
  * observation (3,): egocentric relative wind [wx, wy] (ground wind minus the
    agent's last self-motion velocity, rotated into the heading frame, divided
    by ``wind_obsx``) and odor concentration at the agent location (sum of
    ``(0.01/radius)**3`` over puffs whose bounding box contains the point,
    thresholded at 1e-4, clipped to [0, 1]);
  * action (2,) in [0, 1]: [move, turn]. With ``squash_action=True`` the raw
    (Gaussian) action is squashed by ``(tanh(a)+1)/2`` INSIDE the env, so the
    agent's log-probs are always taken on the pre-squash sample;
  * dynamics: turn by ``turn_capacity * turnx * (turn - 0.5) * dt``, move
    forward up to ``move_capacity * movex * move * dt``, then advected by wind;
  * episode: ``sim_steps_max`` (300) steps at env_dt = 0.04 s; start time
    uniform in a ``reset_offset_tmax`` window after ``t_val_min``; start
    location from the "quantile" curriculum (sample a quantile
    q ~ U(diff_min, diff_max) of the puff x-distribution at the start tidx);
  * termination: HOME (within ``homed_radius`` of the source), OOB
    (``stray_distance`` > ``stray_max`` from the nearest puff center), or OOT;
  * reward: +101 HOME, else tick -10/sim_steps_max (an extra 5x tick when the
    CURRENT odor observation is at/below threshold); ``r_shaping=('step','oob')``
    adds radial-distance-decrease shaping ``5*(|loc_last|-|loc|)`` (zeroed
    when positive while off-plume) and the OOB penalty
    ``-(5*|loc| + stray)`` (doubled when x < 0);
  * per-episode generalization noise (training): plume x-axis flip
    (``flipping``; also inverts the turn action and the sign of obs[1]),
    odor gain U(0.5, 1.5) (``odor_scaling``), per-episode puff sparsity
    U(birthx, 1) (``birthx`` < 1), per-episode diffusion multiplier
    U(diffusion_min, diffusion_max).

Deviations from the reference (behavior-preserving):
  * per-env ``np.random.Generator`` instead of the global numpy RandomState;
  * gymnasium 5-tuple API; ``info['episode'] = {'r', 'l', 'outcome'}`` at
    termination (SyncVectorEnv / RLTrainer consume this);
  * ``loc_algo='fixed'`` / ``angle_algo='fixed'`` / ``time_algo='fixed'`` use
    the fixed_x/fixed_y/fixed_angle/fixed_time_offset kwargs (fixed eval grid).
"""
from __future__ import annotations

import numpy as np

try:
    from gymnasium import spaces
except ImportError:  # pragma: no cover
    from gym import spaces

from .plume_sim import load_or_generate_plume, MIN_RADIUS, ENV_DT

ODOR_THRESHOLD = 1e-4


class PlumeEnv:
    """Odor plume tracking environment (Singh et al. 2023). See module docstring."""

    def __init__(self, *, dataset: str = "constantx5b5", data_dir: str,
                 env_dt: float = ENV_DT, sim_steps_max: int = 300,
                 t_val_min: float = 60.0, reset_offset_tmax: float = 30.0,
                 move_capacity: float = 2.0, turn_capacity: float = 6.25 * np.pi,
                 wind_obsx: float = 1.0, movex: float = 1.0, turnx: float = 1.0,
                 loc_algo: str = "quantile", time_algo: str = "uniform",
                 angle_algo: str = "uniform", qvar: float = 1.0,
                 diff_min: float = 0.4, diff_max: float = 0.8,
                 birthx: float = 1.0, birthx_max: float = 1.0,
                 diffusion_min: float = 1.0, diffusion_max: float = 1.0,
                 homed_radius: float = 0.2, stray_max: float = 2.0,
                 wind_rel: bool = True, r_shaping=("step", "oob"),
                 squash_action: bool = False, flipping: bool = False,
                 odor_scaling: bool = False, obs_noise: float = 0.0,
                 act_noise: float = 0.0, radiusx: float = 1.0,
                 odor_threshold: float = ODOR_THRESHOLD,
                 fixed_x: float = 7.0, fixed_y: float = 0.0,
                 fixed_angle: float = 0.0, fixed_time_offset: float = 0.0,
                 seed: int | None = None):
        assert loc_algo in ("quantile", "uniform", "fixed")
        assert time_algo in ("uniform", "fixed")
        assert angle_algo in ("uniform", "fixed")
        self.dataset = dataset
        self.data_dir = data_dir
        self.dt = env_dt
        self.fps = int(round(1.0 / env_dt))
        self.episode_steps_max = sim_steps_max
        self.t_val_min = t_val_min
        self.reset_offset_tmax = reset_offset_tmax
        self.move_capacity = move_capacity
        self.turn_capacity = turn_capacity
        self.wind_obsx = wind_obsx
        self.movex = movex
        self.turnx = turnx
        self.loc_algo = loc_algo
        self.time_algo = time_algo
        self.angle_algo = angle_algo
        self.qvar = qvar
        self.diff_min = diff_min
        self.diff_max = diff_max
        self.birthx = birthx
        self.birthx_max = birthx_max
        self.diffusion_min = diffusion_min
        self.diffusion_max = diffusion_max
        self.homed_radius = homed_radius
        self.stray_max = stray_max
        self.wind_rel = wind_rel
        self.r_shaping = tuple(r_shaping)
        self.squash_action = squash_action
        self.flipping = flipping
        self.odor_scaling = odor_scaling
        self.obs_noise = obs_noise
        self.act_noise = act_noise
        self.radiusx = radiusx
        self.odor_threshold = odor_threshold
        self.fixed_x = fixed_x
        self.fixed_y = fixed_y
        self.fixed_angle = fixed_angle
        self.fixed_time_offset = fixed_time_offset
        self.rng = np.random.default_rng(seed)

        self.rewards = {"tick": -10.0 / self.episode_steps_max, "homed": 101.0}

        t_val_max = (self.t_val_min + self.reset_offset_tmax
                     + 1.0 * self.episode_steps_max / self.fps + 1.0)
        self.plume = load_or_generate_plume(
            dataset, data_dir, t_val_min=self.t_val_min, t_val_max=t_val_max,
            env_dt=env_dt)
        if self.plume.n_steps < self.episode_steps_max:
            self.episode_steps_max = self.plume.n_steps

        # Base (birthx_max / radiusx / diffusion_max-adjusted) per-step puff
        # arrays. birthx_max < 1 sparsifies once at load (reference behavior).
        self._base = self._build_base_view()

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,),
                                            dtype=np.float32)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2,),
                                       dtype=np.float32)

        # Episode state (set properly in reset()).
        self.episode_step = 0
        self.step_offset = 0
        self.agent_location = np.zeros(2)
        self.agent_location_last = np.zeros(2)
        self.agent_location_init = np.zeros(2)
        self.agent_angle = np.array([1.0, 0.0])
        self.agent_velocity_last = np.zeros(2)
        self.wind_ground = np.zeros(2)
        self.stray_distance = 0.0
        self.stray_distance_last = 0.0
        self.flipx = 1.0
        self.odorx = 1.0
        self.episode_reward = 0.0
        self.found_plume = False
        self._ep_view = self._base  # per-episode (sparsified/diffused) view

    # ------------------------------------------------------------------ views
    def _build_base_view(self) -> list[tuple]:
        """Per-step (x, y, radius, concentration, pid) after load-time
        adjustments (birthx_max sparsity, radiusx, diffusion_max)."""
        plume = self.plume
        view = []
        keep_pids = None
        if self.birthx_max < 0.99:
            all_pids = np.unique(plume.puff_pid)
            n_drop = int(round((1.0 - np.clip(self.birthx_max, 0.01, 1.0))
                               * len(all_pids)))
            drop = self.rng.choice(all_pids, size=n_drop, replace=False)
            keep_pids = np.setdiff1d(all_pids, drop)
        for k in range(plume.n_steps):
            x, y, r, pid = plume.puffs_at(k)
            if keep_pids is not None:
                m = np.isin(pid, keep_pids)
                x, y, r, pid = x[m], y[m], r[m], pid[m]
            r = r * self.radiusx
            if self.diffusion_max != 1.0:
                r = (r - MIN_RADIUS) * self.diffusion_max + MIN_RADIUS
            c = (MIN_RADIUS / np.maximum(r, 1e-9)) ** 3
            view.append((x, y, r, c, pid))
        return view

    def _build_episode_view(self, k_lo: int, k_hi: int) -> list[tuple]:
        """Per-episode view over global steps [k_lo, k_hi): per-episode puff
        sparsity (birthx) and diffusion multiplier (diffusion_min/max)."""
        birthx = 1.0
        drop_pids = None
        if self.birthx < 0.99:
            birthx = float(np.clip(self.rng.uniform(self.birthx, 1.0), 0.0, 1.0))
            window_pids = np.unique(np.concatenate(
                [self._base[k][4] for k in range(k_lo, min(k_hi, self.plume.n_steps))
                 if len(self._base[k][4])])) if k_lo < self.plume.n_steps else []
            if len(window_pids):
                n_drop = int(round((1.0 - birthx) * len(window_pids)))
                drop_pids = set(self.rng.choice(
                    window_pids, size=n_drop, replace=False).tolist())
        diffx = 1.0
        if self.diffusion_min < self.diffusion_max - 0.01:
            diffx = float(self.rng.uniform(self.diffusion_min, self.diffusion_max))
        view = []
        for k in range(k_lo, min(k_hi, self.plume.n_steps)):
            x, y, r, c, pid = self._base[k]
            if drop_pids and len(pid):
                keep = ~np.isin(pid, list(drop_pids))
                x, y, r, c, pid = x[keep], y[keep], r[keep], c[keep], pid[keep]
            if diffx != 1.0:
                r = (r - MIN_RADIUS) * (diffx / self.diffusion_max) + MIN_RADIUS
                c = (MIN_RADIUS / np.maximum(r, 1e-9)) ** 3
                x, y = x.copy(), y.copy()
            view.append((x, y, r, c))
        return view

    # ---------------------------------------------------------------- queries
    def _concentration_at(self, k: int, x: float, y: float) -> float:
        """Box-intersection odor query (reference get_concentration_at_tidx)."""
        if not (0 <= k < len(self._ep_view)):
            return 0.0
        px, py, pr, pc = self._ep_view[k]
        if len(px) == 0:
            return 0.0
        hit = (np.abs(px - x) < pr) & (np.abs(py - y) < pr)
        return float(pc[hit].sum())

    def _stray_distance(self, k: int, max_samples: int = 300) -> float:
        """Min Euclidean distance to (up to ``max_samples``) puff centers."""
        if not (0 <= k < len(self._ep_view)):
            return 0.0
        px, py = self._ep_view[k][0], self._ep_view[k][1]
        if len(px) == 0:
            return 0.0  # reference: exception path returns 0
        if len(px) > max_samples:
            idx = self.rng.choice(len(px), size=max_samples, replace=False)
            px, py = px[idx], py[idx]
        d = np.hypot(px - self.agent_location[0], py - self.agent_location[1])
        return float(d.min())

    def puff_centers(self, k: int | None = None, max_samples: int = 300):
        """Public (x, y) puff-center arrays at episode step ``k`` (default:
        current step). Port of the reference ``get_abunchofpuffs``; used to
        build the fixed evaluation grid's plume y-extent."""
        k = self._k() if k is None else int(k)
        px, py = self._ep_view[k][0], self._ep_view[k][1]
        if len(px) > max_samples:
            idx = self.rng.choice(len(px), size=max_samples, replace=False)
            px, py = px[idx], py[idx]
        return px, py

    # ---------------------------------------------------------------- sensing
    def _sense(self) -> np.ndarray:
        wind_absolute = (self.wind_ground - self.agent_velocity_last
                         if self.wind_rel else self.wind_ground.copy())
        agent_angle = np.angle(self.agent_angle[0] + 1j * self.agent_angle[1])
        wind_angle = np.angle(wind_absolute[0] + 1j * wind_absolute[1])
        rel = wind_angle - agent_angle
        mag = np.linalg.norm(wind_absolute) / self.wind_obsx
        wind_obs = np.array([np.cos(rel) * mag, np.sin(rel) * mag])
        if self.obs_noise > 0:
            wind_obs = wind_obs * (1.0 + self.rng.uniform(-self.obs_noise,
                                                          self.obs_noise, 2))
        odor = self._concentration_at(self._k(), *self.agent_location)
        if self.odor_scaling:
            odor *= self.odorx
        if self.obs_noise > 0:
            odor *= 1.0 + self.rng.uniform(-self.obs_noise, self.obs_noise)
        odor = 0.0 if odor < self.odor_threshold else odor
        odor = float(np.clip(odor, 0.0, 1.0))
        return np.array([wind_obs[0], wind_obs[1], odor], dtype=np.float32)

    def _k(self) -> int:
        """Current LOCAL step index into the per-episode view."""
        return min(self.episode_step, len(self._ep_view) - 1)

    def _k_global(self) -> int:
        """Current GLOBAL step index into the plume (wind) arrays."""
        return min(self.episode_step + self.step_offset, self.plume.n_steps - 1)

    # ------------------------------------------------------------------- gym
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.episode_reward = 0.0
        self.episode_step = 0

        if self.time_algo == "uniform":
            self.step_offset = int(self.fps * self.rng.uniform(
                0.0, self.reset_offset_tmax))
        else:
            self.step_offset = int(self.fps * self.fixed_time_offset)
        max_offset = self.plume.n_steps - self.episode_steps_max - 1
        self.step_offset = int(np.clip(self.step_offset, 0, max(0, max_offset)))

        k0 = self.step_offset
        k1 = k0 + self.episode_steps_max + self.fps  # reference: tidx_max slack
        self._ep_view = self._build_episode_view(k0, k1)
        self._ep_k0 = k0

        if self.flipping:
            self.flipx = -1.0 if self.rng.uniform() > 0.5 else 1.0
        else:
            self.flipx = 1.0
        if self.odor_scaling:
            self.odorx = float(self.rng.uniform(0.5, 1.5))
        else:
            self.odorx = 1.0

        self.agent_location = self._initial_location()
        self.agent_location_last = self.agent_location.copy()
        self.agent_location_init = self.agent_location.copy()
        self.stray_distance = self._stray_distance(0)
        self.stray_distance_last = self.stray_distance
        if self.angle_algo == "uniform":
            a = self.rng.uniform(0, 2 * np.pi)
        else:
            a = self.fixed_angle
        self.agent_angle = np.array([np.cos(a), np.sin(a)])
        self.agent_velocity_last = np.zeros(2)
        self.wind_ground = np.array(self.plume.wind_at(k0))

        obs = self._sense()
        self.found_plume = bool(obs[-1] > 0.0)
        if self.flipx < 0:
            obs[1] *= -1.0
        return obs, {}

    def _initial_location(self) -> np.ndarray:
        if self.loc_algo == "fixed":
            return np.array([self.fixed_x, self.fixed_y], dtype=float)
        if self.loc_algo == "uniform":
            return np.array([2 + self.rng.uniform(-1, 1),
                             self.rng.uniform(-0.5, 0.5)])
        # quantile curriculum: q ~ U(diff_min, diff_max) of the puff x-dist
        px, py = self._ep_view[0][0], self._ep_view[0][1]
        if len(px) == 0:
            return np.array([2 + self.rng.uniform(-1, 1),
                             self.rng.uniform(-0.5, 0.5)])
        if len(px) > 300:
            idx = self.rng.choice(len(px), size=300, replace=False)
            px, py = px[idx], py[idx]
        q = self.rng.uniform(self.diff_min, self.diff_max)
        x_lo, x_hi = np.quantile(px, [max(q - 0.1, 0.0), q])
        x_mean, x_var = x_hi, x_hi - x_lo
        ys = py[np.abs(px - x_mean) <= x_var]
        if len(ys) == 0:
            ys = py
        y_lo, y_hi = np.quantile(ys, [0.05, 0.5])
        y_mean, y_var = y_hi, min(1.0, y_hi - y_lo)
        return np.array([x_mean + self.qvar * x_var * self.rng.standard_normal(),
                         y_mean + self.qvar * y_var * self.rng.standard_normal()])

    def step(self, action):
        self.episode_step += 1
        self.agent_location_last = self.agent_location.copy()

        k = self._k()
        self.stray_distance_last = self.stray_distance
        self.stray_distance = self._stray_distance(k)  # pre-move (reference order)
        self.wind_ground = np.array(self.plume.wind_at(self._k_global()))

        action = np.asarray(action, dtype=np.float64)
        if self.squash_action:
            action = (np.tanh(action) + 1.0) / 2.0
        action = np.clip(action, 0.0, 1.0)
        move_action, turn_action = float(action[0]), float(action[1])
        if self.act_noise > 0:
            move_action *= 1.0 + self.rng.uniform(-self.act_noise, self.act_noise)
            turn_action *= 1.0 + self.rng.uniform(-self.act_noise, self.act_noise)
        if self.flipping and self.flipx < 0:
            turn_action = 1.0 - turn_action

        # Turn, then move, then wind advection.
        old_angle = np.angle(self.agent_angle[0] + 1j * self.agent_angle[1])
        new_angle = old_angle + self.turn_capacity * self.turnx \
            * (turn_action - 0.5) * self.dt
        self.agent_angle = np.array([np.cos(new_angle), np.sin(new_angle)])
        move_x = self.agent_angle[0] * self.move_capacity * self.movex \
            * move_action * self.dt
        move_y = self.agent_angle[1] * self.move_capacity * self.movex \
            * move_action * self.dt
        drift = self.wind_ground * self.dt
        self.agent_location = np.array([
            self.agent_location[0] + move_x + drift[0],
            self.agent_location[1] + move_y + drift[1]])
        self.agent_velocity_last = np.array([move_x, move_y]) / self.dt

        # Termination.
        loc_norm = float(np.linalg.norm(self.agent_location))
        is_home = loc_norm <= self.homed_radius
        is_outoftime = self.episode_step >= self.episode_steps_max - 1
        is_oob = self.stray_distance > self.stray_max
        done = bool(is_home or is_oob or is_outoftime)

        obs = self._sense()

        # Reward.
        reward = self.rewards["homed"] if is_home else self.rewards["tick"]
        if obs[2] <= self.odor_threshold:  # off-plume: extra tick penalty
            reward += 5 * self.rewards["tick"]
        if is_oob and "oob" in self.r_shaping:
            oob_penalty = 5 * loc_norm + self.stray_distance
            oob_penalty *= 2.0 if self.agent_location[0] < 0 else 1.0
            reward -= oob_penalty
        if "step" in self.r_shaping:
            r_step = 5 * (float(np.linalg.norm(self.agent_location_last))
                          - loc_norm)
            if obs[2] <= self.odor_threshold:
                r_step = min(0.0, r_step)
            if "overshoot" in self.r_shaping and self.agent_location[0] < 0:
                r_step *= 2.0
            reward += r_step
        if "turn" in self.r_shaping:
            reward -= 0.05 * abs(2 * (turn_action - 0.5))
        if "move" in self.r_shaping:
            reward -= 0.05 * abs(move_action)

        outcome = "HOME" if is_home else "OOB" if is_oob else \
            "OOT" if is_outoftime else None
        if self.flipx < 0:
            obs[1] = obs[1] * -1.0
        self.episode_reward += reward

        info = {
            "t_idx": int(self._k_global()),
            "location": self.agent_location.copy(),
            "location_last": self.agent_location_last.copy(),
            "location_initial": self.agent_location_init.copy(),
            "angle": self.agent_angle.copy(),
            "wind_ground": self.wind_ground.copy(),
            "stray_distance": self.stray_distance,
            "flipx": self.flipx,
            "done": outcome,
        }
        if done:
            info["episode"] = {"r": float(self.episode_reward),
                               "l": float(self.episode_step),
                               "outcome": outcome}
        # gymnasium: OOT is a truncation, HOME/OOB are terminations.
        terminated = bool(is_home or is_oob)
        truncated = bool(is_outoftime and not terminated)
        return obs, float(reward), terminated, truncated, info

    def close(self):
        pass


