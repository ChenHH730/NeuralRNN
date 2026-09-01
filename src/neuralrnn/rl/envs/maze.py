"""Toroidal maze environment with a hidden goal and a "think" meta-action.

Port of the dynamic maze task of Jensen, Hennequin & Mattar (2024, Nature
Neuroscience, "A recurrent network model of planning explains hippocampal
replay and human behavior"; Julia reference: ``src/walls*.jl``, ``src/maze.jl``,
``src/initializations.jl`` of the ToPlanOrNotToPlan repository).

Task: a new random maze (recursive backtracker on a torus, plus a few extra
walls removed to create multiple routes) and a new hidden reward location are
sampled every episode. The agent starts at a random non-goal cell and must
discover the goal, then exploit it: reaching the goal yields +1 and teleports
the agent to a random non-goal cell (repeat foraging until the time budget
runs out). Walls and goal stay fixed within an episode, so all within-episode
adaptation must happen through recurrent dynamics (meta-RL / RL^2 protocol).

Actions (Discrete(5)): 0-3 = move +/-x/+/-y (toroidal), 4 = "think" — triggers
an internal rollout through the agent's world model (see
``neuralrnn.rl.planning.WorldModelPlanner``) whose result is appended to the
next observation. Physical actions cost 1.0 time unit, thinking costs
``planning_time`` (0.3); the episode ends when time reaches the budget
(50 time units, so <= 50 actions and <= 167 steps).

Observation (88-dim, matching the reference layout):
    [0:5]   one-hot of the previous (attempted) action
    [5]     previous reward
    [6]     elapsed time / episode_budget
    [7:23]  one-hot of the current agent location (16 states, index = L*x + y)
    [23:55] wall configuration: all +x walls (16), then all +y walls (16)
    [55:88] planning input: flattened one-hot imagined action sequence
            (4 * plan_depth) + 1 goal-reached flag

Timing conventions replicated from the Julia reference: the clock starts at
1.0; the prediction target for the world-model head is the post-move,
PRE-teleport state (returned in ``info["aux_targets"]``); an agent standing
on the goal is teleported on the FOLLOWING step (teleport overrides its move);
thinking while on the goal produces no rollout and costs a full time unit;
bumping into a wall means no movement and no penalty.
"""
from __future__ import annotations

import numpy as np
from gymnasium import spaces

# Direction convention (Julia actions 1-4, here 0-3): +x, -x, +y, -y.
DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))
OPPOSITE = {0: 1, 1: 0, 2: 3, 3: 2}
THINK = 4


def generate_maze(arena_size: int, n_holes: int | None = None,
                  rng=None) -> np.ndarray:
    """Recursive-backtracker maze on a torus (port of ``maze.jl``).

    Starts from all walls present, carves a spanning tree via a randomized
    depth-first walk with toroidal wrapping, then removes ``n_holes``
    additional walls (default ``3 * (L - 3)`` = 3 for L = 4) to create
    multiple routes between cells.

    Returns:
        walls (L*L, 4) float32: ``walls[L*x + y, a] == 1`` iff a wall blocks
        direction ``a`` from cell ``(x, y)``. Symmetric by construction
        (``walls[s, a] == walls[neighbor(s, a), OPPOSITE[a]]``).
    """
    rng = np.random.default_rng(rng)
    L = int(arena_size)
    maz = np.ones((L, L, 4), dtype=np.float32)  # maz[x, y, a]
    visited = set()

    def walk(cell):
        visited.add(cell)
        for a in rng.permutation(4):
            dx, dy = DIRS[a]
            nb = ((cell[0] + dx) % L, (cell[1] + dy) % L)
            if nb not in visited:
                maz[cell[0], cell[1], a] = 0.0
                maz[nb[0], nb[1], OPPOSITE[a]] = 0.0
                walk(nb)

    walk((int(rng.integers(L)), int(rng.integers(L))))

    if n_holes is None:
        n_holes = 3 * (L - 3)
    for _ in range(int(n_holes)):
        walled = np.argwhere(maz == 1.0)
        x, y, a = walled[rng.integers(len(walled))]
        dx, dy = DIRS[a]
        maz[x, y, a] = 0.0
        maz[(x + dx) % L, (y + dy) % L, OPPOSITE[a]] = 0.0

    # state index s = L*x + y (0-based equivalent of the Julia
    # permute-and-reshape: wall_loc[L*(x-1)+y, a] = maz[x, y, a])
    return maz.reshape(L * L, 4)


class MazeEnv:
    """Jensen-2024 toroidal maze (gymnasium-style API, numpy only).

    Args:
        arena_size: side length L of the (toroidal) arena; Nstates = L^2.
        episode_budget: time units per episode (T = 50 in the paper; a
            physical action costs 1.0, planning costs ``planning_time``).
        plan_depth: maximum rollout length of the planner (Lplan = 8).
        planning_time: time cost of one think action (0.3 in the paper,
            i.e. 120 ms vs 400 ms per action).
        planning_cost: reward cost of planning (0 in the paper).
        n_holes: extra walls removed after maze generation
            (default 3*(L-3), matching ``maze.jl``).
        planner: optional WorldModelPlanner, or set later via
            ``bind_planners``. Without a planner, think actions produce a
            zero planning input (and still cost ``planning_time``).
        seed: RNG seed.
    """

    def __init__(self, *, arena_size: int = 4, episode_budget: float = 50.0,
                 plan_depth: int = 8, planning_time: float = 0.3,
                 planning_cost: float = 0.0, n_holes: int | None = None,
                 planner=None, seed: int | None = None):
        self.L = int(arena_size)
        self.n_states = self.L ** 2
        self.budget = float(episode_budget)
        self.plan_depth = int(plan_depth)
        self.planning_time = float(planning_time)
        self.planning_cost = float(planning_cost)
        self.n_holes = n_holes
        self.planner = planner
        self.planner_env_index = 0  # row within a vector env (bind_planners)
        self._seed = seed
        self._rng = np.random.default_rng(seed)

        self.n_actions = 5
        self.n_plan_in = 4 * self.plan_depth + 1
        # 5 (prev action) + 1 (reward) + 1 (time) + N (location) + 2N (walls)
        self.obs_dim = 7 + 3 * self.n_states + self.n_plan_in
        self.observation_space = spaces.Box(-np.inf, np.inf,
                                            shape=(self.obs_dim,),
                                            dtype=np.float32)
        self.action_space = spaces.Discrete(self.n_actions)

        # set by reset()
        self.walls: np.ndarray | None = None
        self.goal: int = -1
        self.agent_state: int = -1
        self.time: float = 1.0
        self._n_steps = 0
        self._n_plan = 0
        self._total_reward = 0.0
        self._first_rew_step = np.nan

    # ------------------------------------------------------------- helpers
    def _obs(self, prev_action: int, prev_rew: float,
             plan_input: np.ndarray) -> np.ndarray:
        """Assemble the 88-dim observation (layout of ``gen_input``)."""
        x = np.zeros(self.obs_dim, dtype=np.float32)
        if 0 <= prev_action < self.n_actions:
            x[prev_action] = 1.0
        x[5] = prev_rew
        x[6] = self.time / self.budget
        x[7 + self.agent_state] = 1.0
        w0 = 7 + self.n_states
        x[w0:w0 + self.n_states] = self.walls[:, 0]      # all +x walls
        x[w0 + self.n_states:w0 + 2 * self.n_states] = self.walls[:, 2]  # +y
        x[-self.n_plan_in:] = plan_input
        return x

    def _make_input_builder(self):
        """Closure building the simulated RNN input for imagined rollout
        steps (mirrors ``gen_input`` on an imagined WorldState: one-hot of the
        previous imagined action, zero reward, time + k, one-hot of the
        imagined state, real walls, zero planning input)."""
        walls, t0, budget = self.walls, self.time, self.budget
        ns, obs_dim = self.n_states, self.obs_dim

        def builder(s_prev: int, a_prev: int, k: int) -> np.ndarray:
            x = np.zeros(obs_dim, dtype=np.float32)
            x[a_prev] = 1.0
            x[6] = (t0 + k) / budget
            x[7 + s_prev] = 1.0
            x[7 + ns:7 + 2 * ns] = walls[:, 0]
            x[7 + 2 * ns:7 + 3 * ns] = walls[:, 2]
            return x

        return builder

    # ----------------------------------------------------------------- API
    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        rng = self._rng
        self.walls = generate_maze(self.L, self.n_holes, rng)
        self.goal = int(rng.integers(self.n_states))
        # start at a uniformly random non-goal cell
        self.agent_state = int(rng.integers(self.n_states - 1))
        if self.agent_state >= self.goal:
            self.agent_state += 1
        self.time = 1.0
        self._n_steps = 0
        self._n_plan = 0
        self._total_reward = 0.0
        self._first_rew_step = np.nan
        self._start_loc = self.agent_state
        obs = self._obs(-1, 0.0, np.zeros(self.n_plan_in, dtype=np.float32))
        info = {"goal_loc": self.goal, "start_loc": self.agent_state,
                "walls": self.walls.copy()}
        return obs, info

    def step(self, action: int):
        a = int(action)
        s_old = self.agent_state
        at_rew = (s_old == self.goal)

        # 1. wall-checked toroidal move (think = stay in place)
        s_new = s_old
        moved = False
        if a < 4 and self.walls[s_old, a] < 0.5:
            x, y = divmod(s_old, self.L)
            dx, dy = DIRS[a]
            s_new = self.L * ((x + dx) % self.L) + ((y + dy) % self.L)
            moved = True

        # 2. world-model targets: post-move PRE-teleport state + goal location
        aux_targets = np.array([s_new, self.goal], dtype=np.float32)

        # 3. reward for moving onto the goal
        rew = 1.0 if (moved and s_new == self.goal) else 0.0

        # 4. agents standing on the goal (from the previous step) teleport to
        #    a uniform non-goal cell; the teleport overrides the move
        if at_rew:
            s_new = int(self._rng.integers(self.n_states - 1))
            if s_new >= self.goal:
                s_new += 1

        # 5. planning on the think action (never while standing on the goal)
        plan_input = np.zeros(self.n_plan_in, dtype=np.float32)
        rollout = None
        believed_goal = -1
        planned = bool(a == THINK and not at_rew and self.planner is not None)
        if planned:
            i = self.planner_env_index
            z = self.planner.last_z[i:i + 1]
            rollout = self.planner.rollout(z, a, self._make_input_builder())
            plan_input = rollout.plan_input
            believed_goal = rollout.goal
            rew += self.planning_cost

        # 6. time accounting: planning is cheaper than acting
        self.time += self.planning_time if planned else 1.0
        self._n_steps += 1
        self._n_plan += int(a == THINK)
        self._total_reward += rew
        if rew > 0.5 and not np.isfinite(self._first_rew_step):
            self._first_rew_step = float(self._n_steps)

        # 7. termination (Julia: while any(time < T + 1 - 1e-2))
        terminated = not (self.time < self.budget + 1.0 - 1e-2)

        self.agent_state = s_new
        obs = self._obs(a, rew, plan_input)

        info = {
            "aux_targets": aux_targets,
            "agent_loc": s_new,
            "goal_loc": self.goal,
            "at_rew": bool(at_rew),
            "planned": planned,
            "rollout": rollout,
            "believed_goal": believed_goal,
            "time": self.time,
        }
        if terminated:
            info["episode"] = {
                "r": self._total_reward,
                "l": float(self._n_steps),
                "n_plan": float(self._n_plan),
                "plan_frac": self._n_plan / max(1, self._n_steps),
                "first_rew_step": self._first_rew_step,
                # non-scalar analysis metadata (ignored by the trainer's
                # scalar logging, picked up by collect_episodes)
                "goal_loc": self.goal,
                "start_loc": self._start_loc,
                "walls": self.walls.copy(),
            }
        return obs, rew, terminated, False, info
