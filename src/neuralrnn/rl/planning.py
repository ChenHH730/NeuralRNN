"""Model-based planning via internal rollouts (planning as a meta-action).

Generic machinery for agents that can "think": instead of (only) acting in the
environment, the agent simulates short trajectories with its OWN learned world
model (the aux prediction head of ``ActorCriticModel``) and receives the
simulated action sequence back as an observation. The rollout changes the
policy purely through recurrent dynamics (the feedback input updates the
hidden state), not through weight updates.

This is the planning protocol of Jensen, Hennequin & Mattar (2024, Nature
Neuroscience; ported from ``model_planner.jl`` / ``planning.jl``):

1. The believed goal is the argmax of the reward-location slice of the aux
   head, computed from the current hidden state and the think action.
2. Up to ``plan_depth`` imagined steps: sample an imagined move from the
   policy logits (restricted to the first ``n_move_actions`` physical
   actions), predict the next imagined state as the argmax of the state
   slice of the aux head, and roll the RNN hidden state forward with a
   simulated input (built by the environment via ``input_builder``).
3. Stop early when the imagined state equals the believed goal.
4. Return the flattened one-hot imagined action sequence plus a
   goal-reached flag — the environment appends this to the next observation.

Everything runs under ``no_grad`` on detached hidden states: planner outputs
are data, matching the reference's ``Zygote.ignore`` blocks.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class RolloutResult:
    """Outcome of a single internal rollout.

    Attributes:
        actions: (plan_depth,) int64 imagined move actions, -1-padded after
            an early stop.
        states: (plan_depth,) int64 imagined next states, -1-padded.
        goal: believed goal state (argmax of the reward slice).
        found_goal: whether the rollout reached ``goal``.
        plan_input: (n_move_actions * plan_depth + 1,) float32 observation
            feedback: flattened one-hot action sequence (step-major, action
            index fastest) followed by the goal-reached flag.
    """
    actions: np.ndarray
    states: np.ndarray
    goal: int
    found_goal: bool
    plan_input: np.ndarray


class WorldModelPlanner:
    """Open-loop rollouts through an agent's own world-model (aux) head.

    The planner works with any ``ActorCriticModel`` that has a discrete
    policy and an aux head (``config.aux_head_out_dim > 0``). Environment
    specifics (how to build the simulated input for an imagined state) are
    injected via the ``input_builder`` callback passed to ``rollout``.

    Hidden-state handoff: environments never see the agent's hidden state in
    the gym contract, so the planner caches it — ``bind(agent)`` registers
    ``planner.update_hidden`` as ``agent.planner_hook``, which
    ``ActorCriticModel.step`` (and ``collect_episodes``) call after every
    step. Vector-env row ``i`` then reads ``planner.last_z[i]``.

    Args:
        plan_depth: maximum number of imagined steps (Jensen-2024: Lplan = 8).
        n_move_actions: number of physical move actions; imagined actions are
            sampled from the policy logits restricted to this prefix
            (Jensen-2024: 4 moves; action 4 is "think").
        state_slice: (start, stop) slice of the aux logits holding the
            next-state distribution.
        reward_slice: (start, stop) slice of the aux logits holding the
            reward-location distribution.
        sample_actions: if True, sample imagined actions from the (renormalized)
            move-action policy; if False, take the argmax.
        seed: seed for the planner's private RNG (independent of env RNGs).
    """

    def __init__(self, *, plan_depth: int = 8, n_move_actions: int = 4,
                 state_slice: tuple = (0, 16), reward_slice: tuple = (17, 33),
                 sample_actions: bool = True, seed: int | None = None):
        self.plan_depth = int(plan_depth)
        self.n_move_actions = int(n_move_actions)
        self.state_slice = tuple(state_slice)
        self.reward_slice = tuple(reward_slice)
        self.sample_actions = bool(sample_actions)
        self.agent = None
        self.last_z: torch.Tensor | None = None  # (B, M) detached
        self._rng = torch.Generator(device="cpu")
        if seed is not None:
            self._rng.manual_seed(seed)

    # ------------------------------------------------------------- wiring
    def bind(self, agent):
        """Attach to an agent: registers ``update_hidden`` as the agent's
        ``planner_hook`` (called with the detached hidden state after every
        ``agent.step``). Returns self for chaining."""
        self.agent = agent
        agent.planner_hook = self.update_hidden
        return self

    def update_hidden(self, z: torch.Tensor) -> None:
        """Cache the latest hidden state (B, M). Called by the agent."""
        self.last_z = z.detach()

    # ------------------------------------------------------------ rollout
    @torch.no_grad()
    def rollout(self, z: torch.Tensor, current_action: int,
                input_builder) -> RolloutResult:
        """Run one internal rollout from hidden state ``z``.

        Args:
            z: (1, M) hidden state of the current step (a single env row).
                Never modified in place and never written back to the agent.
            current_action: the action just taken (the think meta-action);
                its one-hot enters the aux head for the goal belief.
            input_builder: callable ``(imagined_state, imagined_action, k) ->
                (input_dim,) np.ndarray`` building the simulated RNN input for
                imagined step ``k`` (0-based; only called for k >= 1), where
                ``imagined_state`` / ``imagined_action`` are the results of
                the previous imagined step.

        Returns:
            RolloutResult.
        """
        if self.agent is None:
            raise RuntimeError("call planner.bind(agent) first")
        agent = self.agent
        device = z.device

        # believed goal from the reward slice, with the taken (think) action
        a_taken = torch.tensor([current_action], dtype=torch.long, device=device)
        aux = agent.aux_logits(z, a_taken)
        r0, r1 = self.reward_slice
        goal = int(aux[0, r0:r1].argmax())

        s0, s1 = self.state_slice
        actions = np.full(self.plan_depth, -1, dtype=np.int64)
        states = np.full(self.plan_depth, -1, dtype=np.int64)
        found = False
        h = z
        s_prev, a_prev = -1, -1
        for k in range(self.plan_depth):
            if k > 0:
                # roll the hidden state forward with the simulated input
                x = input_builder(s_prev, a_prev, k)
                xt = torch.as_tensor(np.asarray(x), dtype=torch.float32,
                                     device=device).unsqueeze(0)
                h = agent.recurrence(xt, h)
            # imagined move from the policy restricted to physical actions
            logits = agent.readout(h)[0, :self.n_move_actions].float()
            probs = torch.softmax(logits.cpu(), dim=-1)
            if self.sample_actions:
                a = int(torch.multinomial(probs, 1, generator=self._rng))
            else:
                a = int(probs.argmax())
            # predicted next imagined state from the world-model head
            a_t = torch.tensor([a], dtype=torch.long, device=device)
            aux = agent.aux_logits(h, a_t)
            s_new = int(aux[0, s0:s1].argmax())
            actions[k] = a
            states[k] = s_new
            if s_new == goal:
                found = True
                break
            s_prev, a_prev = s_new, a

        plan_input = np.zeros(self.n_move_actions * self.plan_depth + 1,
                              dtype=np.float32)
        for k, a in enumerate(actions):
            if a < 0:
                break
            plan_input[k * self.n_move_actions + a] = 1.0
        plan_input[-1] = float(found)
        return RolloutResult(actions=actions, states=states, goal=goal,
                             found_goal=found, plan_input=plan_input)


def bind_planners(envs, planner: WorldModelPlanner) -> int:
    """Attach a shared planner to every env of a SyncVectorEnv that accepts
    one (i.e. exposes a ``planner`` attribute, e.g. MazeEnv), and tell each
    env its row index so it can read the right hidden state from
    ``planner.last_z``. Returns the number of envs bound."""
    bound = 0
    for i, env in enumerate(getattr(envs, "envs", [])):
        if hasattr(env, "planner"):
            env.planner = planner
            env.planner_env_index = i
            bound += 1
    if bound == 0:
        raise ValueError("no env with a `planner` attribute found in envs")
    return bound
