"""On-policy RL trainer (CleanRL recurrent-PPO skeleton).

Loop per update (see docs/RL_PLAN.md):
    1. collect ``num_steps`` x ``num_envs`` rollout with ``agent.step`` (done-masked
       hidden-state reset; only the chunk-initial hidden state is stored);
    2. bootstrap the value and compute returns/advantages (n-step or GAE);
    3. ``update_epochs`` passes over env-slice minibatches, replaying full sequences
       through the RNN (``agent.evaluate_sequence``) for BPTT;
    4. grad clipping, optimizer step, ``post_step_hook`` (default:
       ``agent.project_constraints`` — Dale / |W| projections);
    5. logging (tqdm + history dict + training_curves.png), periodic ``eval_fn``
       validation with optional early stop, checkpoints via ``save_pretrained``.

Note: the agent stays in ``train()`` mode during rollouts so cores with recurrent
noise (``sigma_rec > 0``) keep their stochasticity, matching the reference
implementations. ``eval_fn`` receives the agent and may manage modes itself.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Callable

import numpy as np
import torch
from torch import optim

from .buffers import RecurrentRolloutBuffer
from .losses import PPOLoss
from .returns import build_return_estimator


@dataclass
class RLTrainingArguments:
    """Hyperparameters for RLTrainer (defaults: reduced tutorial config).

    Reference values from published work (e.g. Battista-2026's
    ``network_training.py``) are noted per field.
    """
    total_timesteps: int = 1_000_000   # reference: effectively unbounded (early stop)
    num_envs: int = 32                 # reference (Battista-2026): 128
    num_steps: int = 128               # reference (Battista-2026): 512
    num_minibatches: int = 4
    update_epochs: int = 4
    gamma: float = 0.99
    estimator: str = "nstep"           # "nstep" | "gae"
    gae_lambda: float = 0.95
    lr: float = 2.5e-4
    adam_eps: float = 1e-5
    weight_decay: float = 1e-6
    max_grad_norm: float = 1.0
    anneal_lr: bool = False
    adv_norm: str = "none"             # "none" | "rollout" (normalize advantages
                                       # over the whole rollout before the epoch
                                       # loop; reference PPO convention, e.g.
                                       # Singh-2023's ikostrikov-style PPO)
    target_kl: float | None = None
    aux_target_dim: int = 0            # >0: collect per-step info["aux_targets"]
                                       # (aux_dim,) from the env into the buffer
                                       # (required by losses with requires_aux=True)
    reset_at_update_start: bool = False  # reset all envs at each update start so
                                       # windows begin at episode boundaries
                                       # (meta-RL / episodic protocols, e.g.
                                       # Jensen-2024's 40-fresh-episodes batches)
    seed: int = 1
    device: str = "cuda"
    output_dir: str | None = None      # checkpoints + training_curves.png
    save_every: int = 0                # updates between checkpoints; 0 = final only
    eval_every: int = 0                # updates between eval_fn calls; 0 = off
    early_stop_metric: str | None = None   # key in eval_fn metrics, e.g. "pc_correct"
    early_stop_threshold: float | None = None
    progress_bar: bool = True

    @property
    def batch_size(self) -> int:
        return self.num_envs * self.num_steps


class RLTrainer:
    """On-policy trainer for ActorCriticModel agents.

    Args:
        agent: ActorCriticModel (or compatible: step / get_value / evaluate_sequence /
            init_state / project_constraints).
        envs: SyncVectorEnv with num_envs == args.num_envs.
        args: RLTrainingArguments.
        loss: RL loss module (default PPOLoss()).
        eval_fn: callable(agent) -> dict[str, float]; called every args.eval_every
            updates. Used for held-out validation (e.g. task-performance criteria).
        post_step_hook: callable(agent) after each optimizer step
            (default: agent.project_constraints).
    """

    def __init__(self, agent, envs, args: RLTrainingArguments,
                 loss=None, eval_fn: Callable | None = None,
                 post_step_hook: Callable | None = None):
        assert envs.num_envs == args.num_envs
        self.agent = agent
        self.envs = envs
        self.args = args
        self.loss = loss if loss is not None else PPOLoss()
        self.eval_fn = eval_fn
        # post_step_hook(agent); default None -> agent.project_constraints() is called
        self.post_step_hook = post_step_hook
        self.history: dict[str, list] = {"scalars": [], "eval": []}

    # ------------------------------------------------------------------ train
    def train(self) -> dict[str, list]:
        args = self.args
        device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available())
                              else "cpu")
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        rng = np.random.default_rng(args.seed)

        agent = self.agent.to(device)
        agent.train()
        optimizer = optim.Adam(agent.parameters(), lr=args.lr,
                               eps=args.adam_eps, weight_decay=args.weight_decay)
        estimator = build_return_estimator(args.estimator, args.gae_lambda)

        obs_shape = tuple(self.envs.single_observation_space.shape)
        action_space = self.envs.single_action_space
        if hasattr(action_space, "n"):  # Discrete
            action_shape, action_dtype = (), torch.long
        else:  # Box (continuous)
            action_shape = tuple(action_space.shape)
            action_dtype = torch.float32
        buffer = RecurrentRolloutBuffer(args.num_steps, args.num_envs, obs_shape,
                                        device, action_shape=action_shape,
                                        action_dtype=action_dtype,
                                        aux_dim=args.aux_target_dim)
        want_aux = bool(getattr(self.loss, "requires_aux", False))
        if want_aux and args.aux_target_dim < 1:
            raise ValueError(
                f"{type(self.loss).__name__} declares requires_aux=True but "
                "args.aux_target_dim is 0 — set aux_target_dim so the buffer "
                "stores per-step aux targets from info['aux_targets'].")

        num_updates = max(1, args.total_timesteps // args.batch_size)
        global_step = 0
        start_time = time.time()
        best_metric = -np.inf

        next_obs, _ = self.envs.reset(seed=args.seed)
        next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
        next_done = torch.zeros(args.num_envs, device=device)
        next_z = agent.init_state(args.num_envs, device)

        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            with open(os.path.join(args.output_dir, "rl_training_args.json"), "w") as f:
                json.dump(asdict(args), f, indent=2)

        pbar = range(1, num_updates + 1)
        if args.progress_bar:
            from tqdm import tqdm
            pbar = tqdm(pbar, desc="RL training", unit="update")

        for update in pbar:
            if args.anneal_lr:
                frac = 1.0 - (update - 1.0) / num_updates
                optimizer.param_groups[0]["lr"] = frac * args.lr

            # Optional: start every update at fresh episode boundaries (meta-RL).
            if args.reset_at_update_start and update > 1:
                next_obs_np, _ = self.envs.reset()
                next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32,
                                           device=device)
                next_done = torch.ones(args.num_envs, device=device)
                next_z = agent.init_state(args.num_envs, device)

            # ---------------- rollout collection ----------------
            buffer.z_init = next_z.clone()
            ep_infos: dict[str, list[float]] = {}
            for step in range(args.num_steps):
                global_step += args.num_envs
                with torch.no_grad():
                    action, logprob, _, value, new_z = agent.step(next_obs, next_z, next_done)
                obs_t, done_t = next_obs, next_done
                next_obs_np, reward, term, trunc, infos = self.envs.step(action.cpu().numpy())
                aux_t = None
                if args.aux_target_dim > 0:
                    aux_rows = []
                    for info in infos:
                        a = info.get("aux_targets") if isinstance(info, dict) else None
                        if a is None:
                            a = np.zeros(args.aux_target_dim, dtype=np.float32)
                        aux_rows.append(np.asarray(a, dtype=np.float32)
                                        .ravel()[:args.aux_target_dim])
                    aux_t = torch.as_tensor(np.stack(aux_rows), device=device)
                buffer.add(step, obs_t, done_t, action, logprob, value,
                           torch.as_tensor(reward, dtype=torch.float32, device=device),
                           aux=aux_t)
                next_z = new_z
                next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)
                next_done = torch.as_tensor(np.maximum(term, trunc),
                                            dtype=torch.float32, device=device)
                # Generic episode statistics: average every scalar field of
                # the envs' info["episode"] dicts (gym RecordEpisodeStatistics
                # provides "r"/"l"/"t", possibly as 1-element arrays; richer
                # envs may add task variables).
                for info in infos:
                    ep = info.get("episode") if isinstance(info, dict) else None
                    if ep:
                        for k, v in ep.items():
                            try:
                                val = float(np.asarray(v, dtype=float).ravel()[0])
                            except (TypeError, ValueError, IndexError):
                                continue  # non-scalar entry (arrays, str, ...)
                            if np.isfinite(val):
                                ep_infos.setdefault(k, []).append(val)

            # ---------------- returns / advantages ----------------
            with torch.no_grad():
                next_value = agent.get_value(next_obs, next_z, next_done)
            buffer.compute_returns(estimator, next_value, next_done, args.gamma)
            if args.adv_norm == "rollout":
                adv = buffer.advantages
                buffer.advantages = (adv - adv.mean()) / (adv.std() + 1e-5)
            elif args.adv_norm != "none":
                raise ValueError(f"unknown adv_norm '{args.adv_norm}'")

            # ---------------- policy update ----------------
            last_logs = {}
            stop_update = False
            for _epoch in range(args.update_epochs):
                for mb in buffer.iterate_minibatches(args.num_minibatches, rng):
                    out = agent.evaluate_sequence(
                        mb["obs"], mb["z_init"], mb["dones"], mb["actions"],
                        return_aux=want_aux)
                    extras = None
                    if want_aux:
                        newlogp, entropy, newvalue, states, _, aux_logits = out
                        extras = {"aux_logits": aux_logits,
                                  "aux_targets": mb.get("aux_targets")}
                    else:
                        newlogp, entropy, newvalue, states, _ = out
                    loss, last_logs = self.loss(
                        newlogp, mb["logprobs"], mb["advantages"], newvalue,
                        mb["values"], mb["returns"], entropy, states,
                        extras=extras)
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()
                    if self.post_step_hook is not None:
                        self.post_step_hook(agent)
                    elif hasattr(agent, "project_constraints"):
                        agent.project_constraints()
                if args.target_kl is not None and last_logs.get("approx_kl", 0) > args.target_kl:
                    stop_update = True
                    break

            # ---------------- logging ----------------
            sps = int(global_step / (time.time() - start_time))
            ep_means = {k: float(np.mean(v)) for k, v in ep_infos.items()}
            scalars = {
                "update": update, "global_step": global_step, "sps": sps,
                **{k: (float(v) if np.isscalar(v) or np.ndim(v) == 0 else np.nan)
                   for k, v in last_logs.items()},
                "episodic_return": ep_means.get("r", np.nan),
                **{f"episodic_{k}": v for k, v in ep_means.items() if k != "r"},
            }
            self.history["scalars"].append(scalars)
            if args.progress_bar:
                postfix = {
                    "ret": f"{scalars['episodic_return']:.3f}",
                    "vloss": f"{scalars['v_loss']:.3f}",
                }
                if args.early_stop_metric is not None:
                    m = args.early_stop_metric
                    if m in ep_means:
                        postfix[m] = f"{ep_means[m]:.2f}"
                    elif self.history["eval"] and m in self.history["eval"][-1]:
                        postfix[m] = f"{self.history['eval'][-1][m]:.2f}"
                pbar.set_postfix(postfix)

            # ---------------- eval / checkpoints ----------------
            if self.eval_fn is not None and args.eval_every and update % args.eval_every == 0:
                metrics = dict(self.eval_fn(agent))
                metrics["update"] = update
                self.history["eval"].append(metrics)
                if args.output_dir:
                    with open(os.path.join(args.output_dir, "eval_history.json"), "w") as f:
                        json.dump(self.history["eval"], f, indent=2)
                m = args.early_stop_metric
                if m is not None and m in metrics:
                    if metrics[m] > best_metric:
                        best_metric = metrics[m]
                        self.save_checkpoint("best", update, global_step)
                    if (args.early_stop_threshold is not None
                            and metrics[m] >= args.early_stop_threshold):
                        self.save_checkpoint("final", update, global_step)
                        break

            if args.output_dir and args.save_every and update % args.save_every == 0:
                self.save_checkpoint(f"checkpoint-update{update}", update, global_step)

        self.save_checkpoint("final", update, global_step)
        if args.output_dir:
            self._save_history_and_curves()
        return self.history

    # ------------------------------------------------------------- checkpoint
    def save_checkpoint(self, name: str, update: int, global_step: int) -> None:
        """Save agent via save_pretrained into output_dir/<name>."""
        if not self.args.output_dir:
            return
        path = os.path.join(self.args.output_dir, name)
        self.agent.save_pretrained(path, metadata={
            "update": update, "global_step": global_step,
            "rl_training_args": asdict(self.args)})

    def _save_history_and_curves(self) -> None:
        out = self.args.output_dir
        with open(os.path.join(out, "history.json"), "w") as f:
            json.dump(self.history, f, indent=2)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            scalars = self.history["scalars"]
            if not scalars:
                return
            keys = [k for k in ("episodic_return", "pg_loss", "v_loss", "entropy")
                    if any(np.isfinite(s.get(k, np.nan)) for s in scalars)]
            keys += sorted({k for s in scalars for k in s
                            if k.startswith("episodic_") and k != "episodic_return"
                            and np.isfinite(s[k])})
            if not keys:
                return
            fig, axes = plt.subplots(len(keys), 1, figsize=(7, 2.2 * len(keys)),
                                     sharex=True)
            axes = np.atleast_1d(axes)
            xs = [s["global_step"] for s in scalars]
            for ax, k in zip(axes, keys):
                ax.plot(xs, [s.get(k, np.nan) for s in scalars])
                ax.set_ylabel(k)
                ax.grid(alpha=0.3)
            axes[-1].set_xlabel("global step")
            fig.tight_layout()
            fig.savefig(os.path.join(out, "training_curves.png"), dpi=150)
            plt.close(fig)
        except Exception:
            pass  # curves are best-effort; history.json is the source of truth
