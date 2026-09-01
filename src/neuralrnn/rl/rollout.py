"""Episode collection utility for evaluation and neural/behavioral analysis.

``collect_episodes`` runs an agent in a vector env and records full per-episode
data (observations, hidden states, actions, rewards, values, episode info
dicts), kept in memory for analysis notebooks.
"""
from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def collect_episodes(agent, envs, n_episodes: int, *, deterministic: bool = False,
                     device: str = "cpu", progress: bool = False,
                     keep_step_infos: bool = False,
                     obs_transform=None) -> list[dict]:
    """Collect complete episodes with hidden-state recordings.

    Args:
        agent: ActorCriticModel (step interface).
        envs: SyncVectorEnv (auto-resetting).
        n_episodes: number of complete episodes to collect.
        deterministic: if True, take the policy mode (argmax logits / Gaussian
            mean) instead of sampling.
        device: device the agent runs on.
        progress: show a tqdm counter.
        keep_step_infos: if True, also record the per-step env info dicts
            (``step_infos``: list of T dicts). Default off (memory).
        obs_transform: optional callable applied to the observation tensor
            (B, K) BEFORE the agent sees it (e.g. zeroing a slice for
            ablations). The recorded ``obs`` is the original, untransformed
            observation.

    Returns:
        List of per-episode dicts:
            obs (T, K) np.float32, states (T, M) np.float32,
            actions (T,) np.int64 (discrete) or (T, A) np.float32 (continuous),
            rewards (T,) np.float32, values (T,) np.float32,
            info: the env's info["episode"] dict (env-specific episode variables;
                ECHOICE provides trial variables, n_dec, n_corr, etc.).
    """
    agent = agent.to(device)
    continuous = getattr(agent.config, "action_type", "discrete") == "continuous"
    episodes: list[dict] = []
    B = envs.num_envs

    obs_np, _ = envs.reset()
    obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
    done = torch.ones(B, device=device)  # force hidden-state reset at t=0
    z = agent.init_state(B, device)

    live = [dict(obs=[], states=[], actions=[], rewards=[], values=[],
                 step_infos=[]) for _ in range(B)]
    pbar = None
    if progress:
        from tqdm import tqdm
        pbar = tqdm(total=n_episodes, desc="collecting episodes", unit="ep")

    while len(episodes) < n_episodes:
        agent_obs = obs if obs_transform is None else obs_transform(obs.clone())
        if deterministic:
            # policy mode (argmax logits / Gaussian mean) without sampling
            z_in = agent._mask_reset(z, done)
            z = agent.recurrence(agent_obs, z_in)
            if continuous:
                action = agent.readout(z)
            else:
                action = agent.readout(z).argmax(dim=-1)
            value = agent._value(z)
            logp = torch.zeros(B, device=device)
            hook = getattr(agent, "planner_hook", None)
            if hook is not None:
                hook(z.detach())
        else:
            action, logp, _, value, z = agent.step(agent_obs, z, done)
        action_np = (action.cpu().numpy().astype(np.float32) if continuous
                     else action.cpu().numpy())
        next_obs_np, reward, term, trunc, infos = envs.step(action_np)

        for i in range(B):
            live[i]["obs"].append(obs_np[i])
            live[i]["states"].append(z[i].cpu().numpy())
            live[i]["actions"].append(action_np[i])
            live[i]["rewards"].append(reward[i])
            live[i]["values"].append(value[i].item())
            if keep_step_infos:
                live[i]["step_infos"].append(
                    infos[i] if isinstance(infos[i], dict) else {})
            ep = infos[i].get("episode") if isinstance(infos[i], dict) else None
            if ep is not None and len(episodes) < n_episodes:
                rec = live[i]
                episodes.append({
                    "obs": np.asarray(rec["obs"], dtype=np.float32),
                    "states": np.asarray(rec["states"], dtype=np.float32),
                    "actions": np.asarray(rec["actions"], dtype=(
                        np.float32 if continuous else np.int64)),
                    "rewards": np.asarray(rec["rewards"], dtype=np.float32),
                    "values": np.asarray(rec["values"], dtype=np.float32),
                    "info": ep,
                    **({"step_infos": rec["step_infos"]} if keep_step_infos else {}),
                })
                if pbar is not None:
                    pbar.update(1)
                live[i] = dict(obs=[], states=[], actions=[], rewards=[], values=[],
                               step_infos=[])

        obs_np = next_obs_np
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
        done = torch.as_tensor(np.maximum(term, trunc), dtype=torch.float32, device=device)

    if pbar is not None:
        pbar.close()
    return episodes
