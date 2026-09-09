"""Tests for the RL layer (neuralrnn.rl + actor_critic agent + TD objective).

Covers return estimators (n-step / GAE) against hand-computed values, the
recurrent rollout buffer (storage, minibatch slicing, aux targets), env base
utilities (old-gym / gymnasium API unification, SyncVectorEnv auto-reset),
the built-in environments (echochoice / maze) API sanity, the ActorCriticModel
agent (discrete / continuous / critic-only, done-masking, aux head, constraint
projection, save/load roundtrip), RL losses (PPO / REINFORCE / A2C-pred math
and the loss registry), VecNormalize, the TD value objective, episode
collection, and an end-to-end RLTrainer smoke run on a toy bandit env.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from neuralrnn import AutoConfig, AutoModel
from neuralrnn.rl import (
    A2CPredLoss,
    GAE,
    NStepReturns,
    PPOLoss,
    ReinforceLoss,
    RecurrentRolloutBuffer,
    RLTrainer,
    RLTrainingArguments,
    RunningMeanStd,
    SyncVectorEnv,
    VecNormalize,
    build_return_estimator,
    build_rl_loss,
    collect_episodes,
    make_env,
    register_env,
)
from neuralrnn.rl.envs import reset_env, step_env
from neuralrnn.rl.envs.base import reset_env as _reset_env  # explicit re-export check
from neuralrnn.rl.planning import RolloutResult, WorldModelPlanner, bind_planners
from neuralrnn.train.objectives.td import TDObjective

try:  # spaces: prefer gymnasium, fall back to gym (same as the env modules)
    from gymnasium import spaces
except ImportError:  # pragma: no cover
    from gym import spaces


# ============================ toy envs ============================

class _OldGymEnv:
    """Old-gym API: reset() -> obs, step() -> 4-tuple."""

    def __init__(self):
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(2,))
        self.action_space = spaces.Discrete(2)
        self.t = 0

    def reset(self):
        self.t = 0
        return np.zeros(2, dtype=np.float32)

    def step(self, action):
        self.t += 1
        done = self.t >= 3
        return np.ones(2, dtype=np.float32) * self.t, 1.0, done, {}


class _GymnasiumEnv:
    """Gymnasium API: reset() -> (obs, info), step() -> 5-tuple."""

    def __init__(self):
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(2,))
        self.action_space = spaces.Discrete(2)
        self.t = 0

    def reset(self, *, seed=None, options=None):
        self.t = 0
        return np.zeros(2, dtype=np.float32), {"seed": seed}

    def step(self, action):
        self.t += 1
        terminated = self.t >= 3
        obs = np.ones(2, dtype=np.float32) * self.t
        info = {}
        if terminated:
            info["episode"] = {"r": float(self.t), "l": self.t}
        return obs, 1.0, terminated, False, info


class _ToyBandit:
    """2-step bandit: obs is a one-hot target, reward 1 for the matching action."""

    def __init__(self, seed=None):
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(2,))
        self.action_space = spaces.Discrete(2)
        self._rng = np.random.default_rng(seed)
        self.t = 0
        self.target = 0
        self._ret = 0.0

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.t = 0
        self._ret = 0.0
        self.target = int(self._rng.integers(0, 2))
        obs = np.zeros(2, dtype=np.float32)
        obs[self.target] = 1.0
        return obs, {}

    def step(self, action):
        self.t += 1
        r = 1.0 if int(action) == self.target else 0.0
        self._ret += r
        terminated = self.t >= 3
        obs = np.zeros(2, dtype=np.float32)
        obs[self.target] = 1.0
        info = {}
        if terminated:
            info["episode"] = {"r": self._ret, "l": float(self.t)}
        return obs, r, terminated, False, info


def _make_agent(action_type="discrete", action_dim=2, latent_dim=8, **overrides):
    kw = dict(
        core_config={"model_type": "ctrnn", "input_dim": 2, "latent_dim": latent_dim,
                     "output_dim": max(action_dim, 1)},
        action_dim=action_dim,
        action_type=action_type,
    )
    kw.update(overrides)
    return AutoModel.from_config(AutoConfig.for_model("actor_critic", **kw))


# ============================ return estimators ============================

class TestReturnEstimators:

    def test_nstep_hand_computed(self):
        # T=3, no episode ends; bootstrap from next_value=2.
        r = torch.ones(3, 1)
        v = torch.full((3, 1), 0.5)
        d = torch.zeros(3, 1)
        returns, adv = NStepReturns().compute(
            r, v, d, next_value=torch.tensor([2.0]),
            next_done=torch.tensor([0.0]), gamma=0.9)
        expected = torch.tensor([[1 + 0.9 * 3.52], [1 + 0.9 * 2.8], [1 + 0.9 * 2.0]])
        assert torch.allclose(returns, expected, atol=1e-6)
        assert torch.allclose(adv, returns - v, atol=1e-6)

    def test_nstep_respects_done(self):
        # dones[2]=1: the transition INTO step 2 ended an episode, so step 1
        # must NOT bootstrap into step 2.
        r = torch.ones(3, 1)
        v = torch.zeros(3, 1)
        d = torch.tensor([[0.0], [0.0], [1.0]])
        returns, _ = NStepReturns().compute(
            r, v, d, next_value=torch.tensor([5.0]),
            next_done=torch.tensor([0.0]), gamma=0.9)
        assert returns[1].item() == pytest.approx(1.0)
        # step 2 bootstraps from next_value despite the done before it
        assert returns[2].item() == pytest.approx(1 + 0.9 * 5.0)

    def test_gae_lambda_zero_is_td_residual(self):
        torch.manual_seed(0)
        r = torch.randn(4, 2)
        v = torch.randn(4, 2)
        d = torch.zeros(4, 2)
        nv, nd = torch.randn(2), torch.zeros(2)
        _, adv = GAE(gae_lambda=0.0).compute(r, v, d, nv, nd, gamma=0.9)
        for t in range(4):
            next_v = nv if t == 3 else v[t + 1]
            next_d = nd if t == 3 else d[t + 1]
            delta = r[t] + 0.9 * next_v * (1 - next_d) - v[t]
            assert torch.allclose(adv[t], delta, atol=1e-6)

    def test_gae_lambda_one_matches_nstep(self):
        torch.manual_seed(0)
        r = torch.randn(5, 2)
        v = torch.randn(5, 2)
        d = (torch.rand(5, 2) > 0.8).float()
        nv, nd = torch.randn(2), torch.tensor([0.0, 1.0])
        ret_n, adv_n = NStepReturns().compute(r, v, d, nv, nd, gamma=0.95)
        ret_g, adv_g = GAE(gae_lambda=1.0).compute(r, v, d, nv, nd, gamma=0.95)
        assert torch.allclose(ret_n, ret_g, atol=1e-5)
        assert torch.allclose(adv_n, adv_g, atol=1e-5)

    def test_factory_and_unknown(self):
        assert isinstance(build_return_estimator("nstep"), NStepReturns)
        assert isinstance(build_return_estimator("gae", gae_lambda=0.9), GAE)
        with pytest.raises(KeyError, match="Unknown return estimator"):
            build_return_estimator("monte-carlo")


# ============================ rollout buffer ============================

class TestRolloutBuffer:

    def _filled_buffer(self, aux_dim=0):
        buf = RecurrentRolloutBuffer(num_steps=4, num_envs=4, obs_shape=(3,),
                                     device="cpu", aux_dim=aux_dim)
        for t in range(4):
            obs = torch.full((4, 3), float(t))
            done = torch.zeros(4)
            action = torch.arange(4)
            logprob = torch.zeros(4)
            value = torch.ones(4)
            reward = torch.ones(4) * t
            aux = torch.ones(4, aux_dim) if aux_dim else None
            buf.add(t, obs, done, action, logprob, value, reward, aux=aux)
        buf.z_init = torch.arange(4 * 5, dtype=torch.float32).reshape(4, 5)
        return buf

    def test_storage_shapes(self):
        buf = self._filled_buffer()
        assert buf.obs.shape == (4, 4, 3)
        assert buf.actions.shape == (4, 4) and buf.actions.dtype == torch.long
        assert buf.rewards.shape == (4, 4)
        buf.compute_returns(NStepReturns(), torch.zeros(4), torch.zeros(4), 0.99)
        assert buf.returns.shape == (4, 4)
        # rewards[t]=t with zero values/bootstrap: returns[t] = sum gamma^k r
        assert buf.returns[3, 0].item() == pytest.approx(3.0)
        assert buf.returns[2, 0].item() == pytest.approx(2 + 0.99 * 3.0)

    def test_continuous_action_storage(self):
        buf = RecurrentRolloutBuffer(num_steps=2, num_envs=2, obs_shape=(3,),
                                     device="cpu", action_shape=(2,),
                                     action_dtype=torch.float32)
        buf.add(0, torch.zeros(2, 3), torch.zeros(2), torch.ones(2, 2),
                torch.zeros(2), torch.zeros(2), torch.zeros(2))
        assert buf.actions.shape == (2, 2, 2)
        assert buf.actions.dtype == torch.float32

    def test_minibatches_are_batch_first_env_slices(self):
        buf = self._filled_buffer()
        buf.compute_returns(NStepReturns(), torch.zeros(4), torch.zeros(4), 0.99)
        rng = np.random.default_rng(0)
        mbs = list(buf.iterate_minibatches(2, rng))
        assert len(mbs) == 2
        seen = []
        for mb in mbs:
            assert mb["obs"].shape == (2, 4, 3)
            assert mb["z_init"].shape == (2, 5)
            # env identity is traceable through z_init (row i starts at 5*i)
            seen.extend((mb["z_init"][:, 0] / 5).long().tolist())
        assert sorted(seen) == [0, 1, 2, 3]

    def test_aux_targets(self):
        buf = self._filled_buffer(aux_dim=3)
        buf.compute_returns(NStepReturns(), torch.zeros(4), torch.zeros(4), 0.99)
        mb = next(buf.iterate_minibatches(4, np.random.default_rng(0)))
        assert mb["aux_targets"].shape == (1, 4, 3)

    def test_aux_dim_requires_aux_on_add(self):
        buf = RecurrentRolloutBuffer(num_steps=1, num_envs=1, obs_shape=(2,),
                                     device="cpu", aux_dim=2)
        with pytest.raises(ValueError, match="aux"):
            buf.add(0, torch.zeros(1, 2), torch.zeros(1), torch.zeros(1).long(),
                    torch.zeros(1), torch.zeros(1), torch.zeros(1), aux=None)

    def test_minibatches_must_divide_envs(self):
        buf = self._filled_buffer()
        buf.compute_returns(NStepReturns(), torch.zeros(4), torch.zeros(4), 0.99)
        with pytest.raises(AssertionError):
            list(buf.iterate_minibatches(3, np.random.default_rng(0)))

    def test_returns_before_compute_raises(self):
        buf = self._filled_buffer()
        with pytest.raises(AssertionError, match="compute_returns"):
            list(buf.iterate_minibatches(2, np.random.default_rng(0)))


# ============================ env base ============================

class TestEnvBase:

    def test_old_gym_api_unified(self):
        env = _OldGymEnv()
        obs, info = reset_env(env)
        assert obs.shape == (2,) and info == {}
        obs, r, term, trunc, info = step_env(env, 0)
        assert r == 1.0 and term is False and trunc is False
        step_env(env, 0)
        _, _, term, _, _ = step_env(env, 0)
        assert term is True

    def test_gymnasium_api_unified(self):
        env = _GymnasiumEnv()
        obs, info = reset_env(env, seed=7)
        assert info == {"seed": 7}
        obs, r, term, trunc, info = step_env(env, 0)
        assert trunc is False

    def test_sync_vector_env_auto_reset(self):
        venv = SyncVectorEnv([_GymnasiumEnv, _GymnasiumEnv])
        obs, infos = venv.reset(seed=3)
        assert obs.shape == (2, 2)
        for _ in range(3):  # both envs terminate at t=3
            obs, rews, terms, truncs, infos = venv.step(np.zeros(2, dtype=int))
        assert terms.all()
        for info in infos:
            assert "final_observation" in info
            assert info["episode"]["r"] == 3.0
        # auto-reset: the returned obs after termination is the fresh obs
        assert np.allclose(obs, 0.0)
        venv.close()

    def test_env_registry(self):
        from neuralrnn.rl.envs.maze import MazeEnv
        env = make_env("maze", arena_size=3, episode_budget=5.0)()
        assert isinstance(env, MazeEnv)
        register_env("toy_bandit", lambda **kw: _ToyBandit(**kw))
        assert isinstance(make_env("toy_bandit")(), _ToyBandit)
        with pytest.raises(KeyError, match="Unknown env"):
            make_env("no_such_env")


# ============================ built-in envs ============================

class TestEchoiceEnv:

    def test_reset_and_step_api(self):
        env = make_env("echochoice", seed=0)()
        obs, info = env.reset()
        assert obs.shape == (16,)
        assert np.allclose(obs, 0.0)  # reference behavior: zeros at trial start
        assert info["rule"] in (0, 1, 2, 3, 4)
        obs, r, term, trunc, info = env.step(0)
        assert obs.shape == (16,)
        assert isinstance(term, bool) or term in (0, 1)
        assert trunc is False

    def test_breaking_fixation_aborts_trial(self):
        env = make_env("echochoice", seed=1)()
        env.reset()
        # answering while fixation must be held -> reward -1 and trial abort
        _, r, term, _, info = env.step(1)
        assert r == -1.0
        assert term

    def test_never_deciding_terminates_with_penalty(self):
        env = make_env("echochoice", seed=2)()
        env.reset()
        term, last_r = False, 0.0
        for _ in range(2000):
            _, last_r, term, _, _ = env.step(0)  # hold forever
            if term:
                break
        assert term, "trial should terminate even without a decision"
        assert last_r == -1.0

    def test_rule_subset_validation(self):
        from neuralrnn.rl.envs.echoice import EchoiceEnv
        with pytest.raises(ValueError, match="rules"):
            EchoiceEnv(rules=())
        with pytest.raises(ValueError, match="rules"):
            EchoiceEnv(rules=(0, 99))
        env = EchoiceEnv(rules=(1,), seed=0)
        _, info = env.reset()
        assert info["rule"] == 1

    def test_seeded_reproducibility(self):
        from neuralrnn.rl.envs.echoice import EchoiceEnv
        a, b = EchoiceEnv(seed=42), EchoiceEnv(seed=42)
        oa, ia = a.reset()
        ob, ib = b.reset()
        assert ia["rule"] == ib["rule"]


class TestMazeEnv:

    def test_reset_and_step_api(self):
        env = make_env("maze", arena_size=4, episode_budget=6.0, seed=0)()
        obs, info = env.reset()
        assert obs.shape == (env.obs_dim,)
        assert np.all(np.isfinite(obs))
        terminated = False
        for _ in range(100):
            obs, r, terminated, truncated, info = env.step(
                int(np.random.randint(0, env.n_actions)))
            assert np.all(np.isfinite(obs))
            if terminated or truncated:
                break
        assert terminated or truncated, "episode should end within the budget"


# ============================ actor-critic agent ============================

class TestActorCriticAgent:

    def test_step_shapes_discrete(self):
        agent = _make_agent()
        x = torch.randn(4, 2)
        z = agent.init_state(4)
        action, logp, ent, value, z_new = agent.step(x, z, torch.zeros(4))
        assert action.shape == (4,) and action.dtype == torch.long
        assert 0 <= action.min() and action.max() < 2
        assert logp.shape == (4,) and (logp <= 0).all()
        assert ent.shape == (4,) and (ent >= 0).all()
        assert value.shape == (4,)
        assert z_new.shape == (4, 8)

    def test_step_shapes_continuous(self):
        agent = _make_agent(action_type="continuous", action_dim=3)
        x = torch.randn(4, 2)
        z = agent.init_state(4)
        action, logp, ent, value, z_new = agent.step(x, z, torch.zeros(4))
        assert action.shape == (4, 3)
        assert logp.shape == (4,) and ent.shape == (4,)
        # deterministic mode is the Gaussian mean
        mode = agent.policy_mode(z_new)
        assert mode.shape == (4, 3)

    def test_critic_only_agent(self):
        agent = _make_agent(action_type="none", action_dim=0)
        assert agent.actor is None
        z = torch.randn(4, 8)
        out = agent.readout(z)
        assert out.shape == (4, 1)  # scalar value readout
        assert agent.value(z).shape == (4,)

    def test_get_value_and_policy_mode(self):
        agent = _make_agent()
        x = torch.randn(3, 2)
        z = agent.init_state(3)
        v = agent.get_value(x, z, torch.zeros(3))
        assert v.shape == (3,)
        z2 = agent.core.recurrence(x, z)
        assert agent.policy_mode(z2).shape == (3,)

    def test_evaluate_sequence_shapes(self):
        agent = _make_agent()
        B, T, K = 3, 5, 2
        inputs = torch.randn(B, T, K)
        z0 = agent.init_state(B)
        dones = torch.zeros(B, T)
        actions = torch.randint(0, 2, (B, T))
        logp, ent, val, states, zT = agent.evaluate_sequence(inputs, z0, dones, actions)
        assert logp.shape == (B, T)
        assert ent.shape == (B, T)
        assert val.shape == (B, T)
        assert states.shape == (B, T, 8)
        assert zT.shape == (B, 8)

    def test_done_masking_resets_hidden_state(self):
        # With done=1 at step t, the hidden state is reset to the core's
        # initial state BEFORE processing x_t.
        agent = _make_agent()
        B, T, K = 1, 4, 2
        torch.manual_seed(0)
        inputs = torch.randn(B, T, K)
        z0 = agent.init_state(B)
        dones = torch.zeros(B, T)
        dones[0, 2] = 1.0
        actions = torch.zeros(B, T, dtype=torch.long)
        _, _, _, states, _ = agent.evaluate_sequence(inputs, z0, dones, actions)
        z_fresh = agent.init_state(B)
        expected = agent.core.recurrence(inputs[:, 2], z_fresh)
        assert torch.allclose(states[:, 2], expected, atol=1e-6)

    def test_step_done_masking(self):
        agent = _make_agent()
        x = torch.randn(2, 2)
        z = torch.randn(2, 8)  # non-zero prior state
        done = torch.tensor([1.0, 0.0])
        z_in = agent._mask_reset(z, done)
        assert torch.allclose(z_in[0], agent.init_state(2)[0])
        assert torch.allclose(z_in[1], z[1])

    def test_policy_logit_scale(self):
        agent = _make_agent(policy_logit_scale=2.5)
        z = torch.randn(3, 8)
        logits = agent.readout(z)
        raw = agent.actor(agent._features(z))
        assert torch.allclose(logits, raw * 2.5)

    def test_aux_head_state_action(self):
        agent = _make_agent(aux_head_out_dim=6,
                            aux_slices={"a": (0, 4), "b": (4, 6)})
        z = torch.randn(3, 8)
        a = torch.randint(0, 2, (3,))
        out = agent.aux_logits(z, a)
        assert out.shape == (3, 6)
        with pytest.raises(ValueError, match="actions are required"):
            agent.aux_logits(z, None)
        # evaluate_sequence can replay aux logits per step
        logp, ent, val, states, zT, aux = agent.evaluate_sequence(
            torch.randn(3, 4, 2), agent.init_state(3), torch.zeros(3, 4),
            torch.zeros(3, 4, dtype=torch.long), return_aux=True)
        assert aux.shape == (3, 4, 6)

    def test_aux_head_missing_raises(self):
        agent = _make_agent()
        with pytest.raises(RuntimeError, match="no auxiliary head"):
            agent.aux_logits(torch.randn(2, 8))

    def test_config_validation(self):
        with pytest.raises(ValueError, match="action_dim >= 1"):
            _make_agent(action_type="discrete", action_dim=0)
        with pytest.raises(ValueError, match="action_type"):
            _make_agent(action_type="semi")
        with pytest.raises(ValueError, match="head_init"):
            _make_agent(head_init="zeros")
        with pytest.raises(ValueError, match="aux_slices"):
            _make_agent(aux_head_out_dim=4, aux_slices={"bad": (2, 9)})
        with pytest.raises(ValueError, match="state_action"):
            _make_agent(action_type="continuous", action_dim=2,
                        aux_head_out_dim=4, aux_input="state_action")

    def test_head_init_orthogonal(self):
        agent = _make_agent(head_init="orthogonal", head_hidden_dims=(16,))
        w = list(agent.critic.modules())[-1].weight
        assert w.shape == (1, 16)

    def test_project_constraints_actor_positive(self):
        agent = _make_agent(actor_positive=True, critic_positive=True)
        with torch.no_grad():
            agent.actor.weight[0, 0] = -5.0
            agent.critic.weight[0, 0] = -5.0
        agent.project_constraints()
        assert (agent.actor.weight >= 0).all()
        assert (agent.critic.weight >= 0).all()

    def test_save_load_roundtrip(self, tmp_path):
        agent = _make_agent(policy_logit_scale=1.7)
        agent.save_pretrained(str(tmp_path))
        loaded = AutoModel.from_pretrained(str(tmp_path))
        assert type(loaded) is type(agent)
        assert loaded.config.policy_logit_scale == pytest.approx(1.7)
        for p1, p2 in zip(agent.parameters(), loaded.parameters()):
            assert torch.allclose(p1, p2)

    def test_readout_scales_logits_for_analysis(self):
        # The analysis hard-contract: readout returns scaled policy logits.
        agent = _make_agent()
        out = agent(torch.randn(2, 6, 2))
        assert out.outputs.shape == (2, 6, 2)
        assert out.states.shape == (2, 6, 8)


# ============================ RL losses ============================

class TestRLLosses:

    def _inputs(self, B=2, T=3):
        torch.manual_seed(0)
        old_logp = torch.randn(B, T)
        return dict(
            new_logp=old_logp.clone().requires_grad_(True),
            old_logp=old_logp,
            advantages=torch.randn(B, T),
            new_values=torch.randn(B, T).requires_grad_(True),
            old_values=torch.randn(B, T),
            returns=torch.randn(B, T),
            entropies=torch.rand(B, T),
        )

    def test_ppo_math_ratio_one(self):
        kw = self._inputs()
        loss_fn = PPOLoss(clip_coef=0.1, vf_coef=0.5, ent_coef=0.01)
        loss, logs = loss_fn(**kw)
        # ratio == 1 -> pg_loss = -advantages.mean()
        pg = -kw["advantages"].mean()
        v = 0.5 * ((kw["new_values"] - kw["returns"]) ** 2).mean()
        ent = kw["entropies"].mean()
        expected = pg - 0.01 * ent + 0.5 * v
        assert loss.item() == pytest.approx(expected.item(), abs=1e-6)
        assert logs["approx_kl"] == pytest.approx(0.0, abs=1e-6)
        assert set(logs) >= {"pg_loss", "v_loss", "entropy", "clipfrac"}

    def test_ppo_gradient_flows(self):
        kw = self._inputs()
        loss, _ = PPOLoss()(**kw)
        loss.backward()
        assert kw["new_logp"].grad is not None
        assert torch.isfinite(kw["new_logp"].grad).all()
        assert kw["new_values"].grad is not None

    def test_ppo_clipping(self):
        kw = self._inputs()
        kw["new_logp"] = kw["old_logp"] + 10.0  # ratio = e^10, far outside the clip
        loss_fn = PPOLoss(clip_coef=0.1)
        _, logs = loss_fn(**kw)
        assert logs["clipfrac"] == pytest.approx(1.0)

    def test_ppo_activity_l2_requires_states(self):
        kw = self._inputs()
        with pytest.raises(AssertionError, match="states"):
            PPOLoss(activity_l2=1e-4)(**kw)  # states=None
        loss, logs = PPOLoss(activity_l2=1e-4)(**kw, states=torch.randn(2, 3, 4))
        assert logs["activity_l2"] > 0

    def test_reinforce_math(self):
        kw = self._inputs()
        loss_fn = ReinforceLoss(vf_coef=0.5, ent_coef=0.01)
        loss, logs = loss_fn(**kw)
        pg = -(kw["advantages"] * kw["new_logp"]).mean()
        v = 0.5 * ((kw["new_values"] - kw["returns"]) ** 2).mean()
        ent = kw["entropies"].mean()
        expected = pg - 0.01 * ent + 0.5 * v
        assert loss.item() == pytest.approx(expected.item(), abs=1e-6)
        assert logs["approx_kl"] == 0.0 and logs["clipfrac"] == 0.0

    def test_a2c_pred_loss(self):
        kw = self._inputs(B=2, T=4)
        D = 6
        aux_logits = torch.randn(2, 4, D, requires_grad=True)
        aux_targets = torch.stack([torch.randint(0, 4, (2, 4)),
                                   torch.randint(0, 2, (2, 4))], dim=-1)
        loss_fn = A2CPredLoss(aux_slices={"a": (0, 4), "b": (4, 6)})
        loss, logs = loss_fn(**kw, extras={"aux_logits": aux_logits,
                                           "aux_targets": aux_targets})
        assert torch.isfinite(loss)
        assert "pred_loss" in logs and "pred_loss_a" in logs
        loss.backward()
        assert aux_logits.grad is not None

    def test_a2c_pred_requires_extras(self):
        kw = self._inputs()
        with pytest.raises(ValueError, match="extras"):
            A2CPredLoss()(**kw)

    def test_loss_registry(self):
        assert isinstance(build_rl_loss("ppo"), PPOLoss)
        assert isinstance(build_rl_loss("reinforce"), ReinforceLoss)
        assert isinstance(build_rl_loss("a2c_pred"), A2CPredLoss)
        with pytest.raises(KeyError, match="Unknown RL loss"):
            build_rl_loss("dqn")


# ============================ vecnormalize ============================

class TestVecNormalize:

    def test_running_mean_std_matches_numpy(self):
        rms = RunningMeanStd(shape=(3,))
        rng = np.random.default_rng(0)
        x1 = rng.normal(2.0, 3.0, size=(50, 3))
        x2 = rng.normal(-1.0, 0.5, size=(70, 3))
        rms.update(x1)
        rms.update(x2)
        full = np.concatenate([x1, x2])
        assert np.allclose(rms.mean, full.mean(axis=0), atol=1e-10)
        assert np.allclose(rms.var, full.var(axis=0), atol=1e-10)

    def test_obs_normalization_and_training_flag(self):
        venv = SyncVectorEnv([_GymnasiumEnv, _GymnasiumEnv])
        nenv = VecNormalize(venv, training=True)
        obs, _ = nenv.reset()
        assert obs.shape == (2, 2)
        count_after_reset = nenv.obs_rms.count
        assert count_after_reset > 1e-4  # stats updated
        obs, rews, terms, truncs, infos = nenv.step(np.zeros(2, dtype=int))
        assert np.abs(obs).max() <= 10.0
        assert np.abs(rews).max() <= 10.0

        nenv_eval = VecNormalize(venv, training=False)
        nenv_eval.reset()
        c = nenv_eval.obs_rms.count
        nenv_eval.step(np.zeros(2, dtype=int))
        assert nenv_eval.obs_rms.count == c  # frozen stats in eval mode

    def test_state_dict(self):
        venv = SyncVectorEnv([_GymnasiumEnv])
        nenv = VecNormalize(venv)
        nenv.reset()
        sd = nenv.state_dict()
        assert set(sd) >= {"obs_mean", "obs_var", "obs_count"}


# ============================ TD objective ============================

class _ConstantValueModel(torch.nn.Module):
    """Fake scalar-value model: outputs w * ones(B, T, 1) regardless of input."""

    def __init__(self, w=0.3):
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor(float(w)))

    def forward(self, x):
        from types import SimpleNamespace
        B, T = x.shape[0], x.shape[1]
        return SimpleNamespace(outputs=self.w * torch.ones(B, T, 1))


class TestTDObjective:

    def test_hand_computed_td0(self):
        model = _ConstantValueModel(w=0.5)
        obj = TDObjective(gamma=0.9)
        r = torch.zeros(2, 5, 1)
        r[0, 2, 0] = 1.0  # one reward pulse
        batch = {"inputs": torch.zeros(2, 5, 3), "targets": r, "mask": None}
        loss, logs = model_loss = obj.compute_loss(model, batch)
        # V = 0.5 everywhere; target_t = r_{t+1} + 0.9 * 0.5
        # delta_t = 0.5 - r_{t+1} - 0.45; only t=1 of row 0 has r_{t+1}=1
        deltas = torch.full((2, 4), 0.5 - 0.45)
        deltas[0, 1] = 0.5 - 1.0 - 0.45
        expected = (deltas ** 2).mean()
        assert loss.item() == pytest.approx(expected.item(), abs=1e-6)
        assert set(logs) >= {"loss", "mean_value"}

    def test_gradient_only_flows_through_current_value(self):
        model = _ConstantValueModel(w=0.5)
        obj = TDObjective(gamma=0.9)
        batch = {"inputs": torch.zeros(1, 4, 2),
                 "targets": torch.zeros(1, 4, 1), "mask": None}
        loss, _ = obj.compute_loss(model, batch)
        loss.backward()
        # semi-gradient: target = gamma * w.detach() is a constant 0.45, so
        # d/dw mean((w - 0.45)^2) = 2 (w - 0.45) = 0.1  (NOT 2 (1-gamma)^2 w,
        # which would be the gradient if the bootstrap value were not detached)
        expected = 2 * (0.5 - 0.9 * 0.5)
        assert model.w.grad.item() == pytest.approx(expected, abs=1e-6)

    def test_mask_excludes_boundaries(self):
        model = _ConstantValueModel(w=0.5)
        obj = TDObjective(gamma=0.9)
        r = torch.zeros(1, 6, 1)
        mask = torch.ones(1, 6)
        mask[0, 4:] = 0  # last two steps padded
        loss_masked, _ = obj.compute_loss(
            model, {"inputs": torch.zeros(1, 6, 2), "targets": r, "mask": mask})
        r2 = r.clone()
        r2[0, 5, 0] = 100.0  # garbage reward in the padded tail
        loss_masked2, _ = obj.compute_loss(
            model, {"inputs": torch.zeros(1, 6, 2), "targets": r2, "mask": mask})
        assert loss_masked.item() == pytest.approx(loss_masked2.item(), abs=1e-7)

    def test_scalar_readout_required(self):
        model = _make_agent()  # discrete actor: outputs (B, T, 2)
        obj = TDObjective()
        with pytest.raises(ValueError, match="scalar readout"):
            obj.compute_loss(model, {"inputs": torch.randn(1, 3, 2),
                                     "targets": torch.zeros(1, 3, 1),
                                     "mask": None})


# ============================ episode collection ============================

class TestCollectEpisodes:

    def test_collect_episodes_structure(self):
        agent = _make_agent()
        envs = SyncVectorEnv([_ToyBandit, _ToyBandit])
        eps = collect_episodes(agent, envs, n_episodes=3, device="cpu")
        assert len(eps) == 3
        for ep in eps:
            T = ep["rewards"].shape[0]
            assert ep["obs"].shape == (T, 2)
            assert ep["states"].shape == (T, 8)
            assert ep["actions"].shape == (T,)
            assert ep["values"].shape == (T,)
            assert ep["info"]["r"] == pytest.approx(ep["rewards"].sum())

    def test_collect_episodes_deterministic(self):
        agent = _make_agent()
        envs = SyncVectorEnv([_ToyBandit])
        eps1 = collect_episodes(agent, envs, 2, deterministic=True)
        eps2 = collect_episodes(agent, envs, 2, deterministic=True)
        # deterministic policy + env RNG continuing is fine; just check it runs
        # and actions are valid
        for ep in eps1 + eps2:
            assert set(np.unique(ep["actions"])) <= {0, 1}


# ============================ trainer smoke ============================

class TestRLTrainer:

    def _smoke_args(self, tmp_path, **overrides):
        kw = dict(total_timesteps=64, num_envs=4, num_steps=8,
                  num_minibatches=2, update_epochs=1, gamma=0.9,
                  device="cpu", progress_bar=False, seed=0,
                  output_dir=str(tmp_path))
        kw.update(overrides)
        return RLTrainingArguments(**kw)

    def test_end_to_end_smoke(self, tmp_path):
        agent = _make_agent()
        envs = SyncVectorEnv([lambda: _ToyBandit(seed=i) for i in range(4)])
        args = self._smoke_args(tmp_path)
        trainer = RLTrainer(agent, envs, args, loss=PPOLoss())
        history = trainer.train()
        scalars = history["scalars"]
        assert len(scalars) == 2  # 64 timesteps / (4 envs * 8 steps)
        for s in scalars:
            assert np.isfinite(s["pg_loss"])
            assert np.isfinite(s["v_loss"])
            assert s["global_step"] > 0
        # episode statistics from info["episode"] are averaged into the log
        assert np.isfinite(scalars[-1]["episodic_return"])
        # checkpoints + history files
        import os
        assert os.path.isdir(tmp_path / "final")
        assert os.path.isfile(tmp_path / "history.json")
        assert os.path.isfile(tmp_path / "rl_training_args.json")

    def test_checkpoint_is_loadable(self, tmp_path):
        agent = _make_agent()
        envs = SyncVectorEnv([lambda: _ToyBandit(seed=0) for _ in range(4)])
        trainer = RLTrainer(agent, envs, self._smoke_args(tmp_path))
        trainer.train()
        loaded = AutoModel.from_pretrained(str(tmp_path / "final"))
        assert isinstance(loaded, type(agent))

    def test_reinforce_loss_smoke(self, tmp_path):
        agent = _make_agent()
        envs = SyncVectorEnv([lambda: _ToyBandit(seed=i) for i in range(4)])
        args = self._smoke_args(tmp_path, update_epochs=1, num_minibatches=1,
                                estimator="gae")
        trainer = RLTrainer(agent, envs, args, loss=ReinforceLoss())
        history = trainer.train()
        assert np.isfinite(history["scalars"][-1]["pg_loss"])

    def test_aux_loss_requires_aux_target_dim(self, tmp_path):
        agent = _make_agent()
        envs = SyncVectorEnv([lambda: _ToyBandit(seed=0) for _ in range(4)])
        args = self._smoke_args(tmp_path, aux_target_dim=0)
        trainer = RLTrainer(agent, envs, args, loss=A2CPredLoss())
        with pytest.raises(ValueError, match="requires_aux"):
            trainer.train()

    def test_reset_at_update_start(self, tmp_path):
        agent = _make_agent()
        envs = SyncVectorEnv([lambda: _ToyBandit(seed=i) for i in range(4)])
        args = self._smoke_args(tmp_path, reset_at_update_start=True)
        history = RLTrainer(agent, envs, args).train()
        assert len(history["scalars"]) == 2


# ============================ plume env (Singh-2023) ============================

class TestPlumeEnv:
    """PlumeEnv tests generate a tiny 30 s plume simulation into a tmp dir
    (the default 120 s datasets are cached on first use in real runs)."""

    @pytest.fixture()
    def plume_env(self, tmp_path, monkeypatch):
        import neuralrnn.rl.envs.plume_sim as plume_sim
        monkeypatch.setattr(plume_sim, "DURATION", 30.0)
        from neuralrnn.rl.envs.plume import PlumeEnv
        return PlumeEnv(dataset="constantx5b5", data_dir=str(tmp_path),
                        t_val_min=0.0, seed=0)

    def test_reset_and_step_api(self, plume_env):
        obs, info = plume_env.reset()
        assert obs.shape == plume_env.observation_space.shape
        assert np.all(np.isfinite(obs))
        # continuous 2-dim action (move, turn)
        action = plume_env.action_space.sample()
        obs, r, term, trunc, info = plume_env.step(action)
        assert obs.shape == plume_env.observation_space.shape
        assert np.isfinite(r) and trunc is False

    def test_episode_terminates(self, plume_env):
        plume_env.reset()
        term = False
        for _ in range(400):
            _, _, term, _, _ = plume_env.step(plume_env.action_space.sample())
            if term:
                break
        assert term, "episode should end within sim_steps_max"

    def test_plume_sim_cache_roundtrip(self, plume_env, tmp_path):
        # second construction hits the cache written by the fixture
        from neuralrnn.rl.envs.plume import PlumeEnv
        env2 = PlumeEnv(dataset="constantx5b5", data_dir=str(tmp_path),
                        t_val_min=0.0, seed=1)
        obs, _ = env2.reset()
        assert np.all(np.isfinite(obs))


# ============================ neurogym adapter ============================

class TestNeurogymEnvAdapter:

    def test_trial_boundaries_become_episodes(self):
        pytest.importorskip("neurogym")
        from neuralrnn.rl.envs.neurogym_env import NeurogymEnvAdapter
        env = NeurogymEnvAdapter(
            "PerceptualDecisionMaking", dt=100,
            timing={"fixation": 100, "stimulus": 200, "decision": 200})
        obs, info = env.reset()
        assert obs.shape == env.observation_space.shape
        terminated = False
        for _ in range(200):
            obs, r, terminated, truncated, info = env.step(
                env.action_space.sample())
            if terminated:
                break
        assert terminated, "a neurogym trial should report terminated=True"
        ep = info["episode"]
        assert "r" in ep and "l" in ep

    def test_registry_prefix(self):
        pytest.importorskip("neurogym")
        env = make_env("neurogym:PerceptualDecisionMaking", dt=100,
                       timing={"fixation": 100, "stimulus": 200,
                               "decision": 200})()
        obs, _ = env.reset()
        assert obs.shape == env.observation_space.shape


# ============================ world-model planning (Jensen-2024) ============================

class TestWorldModelPlanner:

    def _maze_agent(self, obs_dim):
        return AutoModel.from_config(AutoConfig.for_model(
            "actor_critic",
            core_config={"model_type": "ctrnn", "input_dim": obs_dim,
                         "latent_dim": 16, "output_dim": 5},
            action_dim=5, action_type="discrete",
            aux_head_out_dim=33,
            aux_slices={"next_state": (0, 16), "reward": (17, 33)}))

    def test_rollout_result_structure(self):
        agent = self._maze_agent(obs_dim=88)
        planner = WorldModelPlanner(plan_depth=4, seed=0).bind(agent)
        z = agent.init_state(1)
        builder = lambda s_prev, a_prev, k: np.zeros(88, dtype=np.float32)
        ro = planner.rollout(z, current_action=4, input_builder=builder)
        assert isinstance(ro, RolloutResult)
        assert ro.actions.shape == (4,) and ro.states.shape == (4,)
        assert 0 <= ro.goal < 16
        assert ro.plan_input.shape == (4 * 4 + 1,)
        assert ro.plan_input[-1] in (0.0, 1.0)
        # one-hot encoding of the imagined action sequence
        k = int((ro.actions >= 0).sum())
        assert ro.plan_input[:-1].sum() == pytest.approx(float(k))

    def test_unbound_planner_raises(self):
        planner = WorldModelPlanner(plan_depth=2)
        with pytest.raises(RuntimeError, match="bind"):
            planner.rollout(torch.zeros(1, 16), 4, lambda s, a, k: np.zeros(4))

    def test_maze_think_action_uses_planner(self):
        env_fn = make_env("maze", arena_size=4, episode_budget=10.0,
                          plan_depth=4, seed=0)
        env0 = env_fn()
        agent = self._maze_agent(obs_dim=env0.obs_dim)
        venv = SyncVectorEnv([env_fn])
        planner = WorldModelPlanner(plan_depth=4, seed=0)
        assert bind_planners(venv, planner) == 1
        planner.bind(agent)
        obs, _ = venv.reset()
        z = agent.init_state(1)
        # one physical step to populate planner.last_z via the agent hook
        agent.step(torch.as_tensor(obs, dtype=torch.float32), z, torch.ones(1))
        obs, r, term, trunc, info = venv.step(np.array([4]))  # think
        assert info[0]["planned"] is True
        assert isinstance(info[0]["rollout"], RolloutResult)
        assert 0 <= info[0]["believed_goal"] < 16
        assert np.isfinite(obs).all()

    def test_think_without_planner_is_noop(self):
        env = make_env("maze", arena_size=4, episode_budget=10.0,
                       plan_depth=4, seed=0)()
        env.reset()
        obs, r, term, trunc, info = env.step(4)
        assert info["planned"] is False
        assert np.allclose(obs[-env.n_plan_in:], 0.0)  # zero planning input
