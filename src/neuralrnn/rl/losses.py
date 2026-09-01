"""RL loss modules (algorithm layer).

``PPOLoss`` implements the CleanRL clipped-surrogate objective (defaults follow
CleanRL's recurrent PPO recipe: clip 0.1, vf 0.5, ent 0.01, unclipped value MSE,
no advantage normalization, optional activity L2 on hidden states).

``ReinforceLoss`` implements the classic REINFORCE-with-baseline (one-step
actor-critic) objective: unclipped policy gradient on the advantage plus a
value regression towards the observed returns (the reward-only recipe of
Song, Yang & Wang 2017). Use it with ``RLTrainingArguments(update_epochs=1,
num_minibatches=1)`` so each batch is exactly on-policy.

Losses are registered in ``RL_LOSS_REGISTRY`` so alternative objectives
(critic-only TD, a torchrl bridge, future algorithms) plug into ``RLTrainer``
without changes. For offline/critic-only TD value learning on fixed episode
datasets see ``neuralrnn.train.objectives.td.TDObjective``.
"""
from __future__ import annotations

import torch

RL_LOSS_REGISTRY: dict[str, type] = {}


def register_rl_loss(name: str):
    """Decorator: register a loss class under ``name`` in RL_LOSS_REGISTRY."""
    def deco(cls):
        RL_LOSS_REGISTRY[name] = cls
        return cls
    return deco


def build_rl_loss(name: str, **kwargs):
    """Factory: ``build_rl_loss("ppo", clip_coef=0.1, ...)``."""
    if name not in RL_LOSS_REGISTRY:
        raise KeyError(f"Unknown RL loss '{name}'. Registered: {sorted(RL_LOSS_REGISTRY)}")
    return RL_LOSS_REGISTRY[name](**kwargs)


@register_rl_loss("ppo")
class PPOLoss:
    """Clipped PPO objective (CleanRL ppo_atari_lstm.py math).

    Args:
        clip_coef: surrogate clipping epsilon.
        vf_coef: value-loss coefficient.
        ent_coef: entropy bonus coefficient.
        clip_vloss: clip the value loss around old values.
        norm_adv: normalize advantages per minibatch.
        activity_l2: coefficient of the mean squared hidden-state penalty
            (applied to ALL states of the replayed sequence; e.g.
            Battista-2026 uses 1e-6).
    """

    def __init__(self, clip_coef: float = 0.1, vf_coef: float = 0.5,
                 ent_coef: float = 0.01, clip_vloss: bool = False,
                 norm_adv: bool = False, activity_l2: float = 0.0):
        self.clip_coef = clip_coef
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.clip_vloss = clip_vloss
        self.norm_adv = norm_adv
        self.activity_l2 = activity_l2

    def __call__(self, new_logp, old_logp, advantages, new_values, old_values,
                 returns, entropies, states=None, extras=None):
        """Compute the PPO loss.

        All tensor args are (B, T). ``states`` (B, T, M) is only needed when
        ``activity_l2 > 0``. ``extras`` is accepted for interface uniformity
        (aux-aware losses) and ignored.

        Returns:
            (loss, logs) with logs: pg_loss, v_loss, entropy, approx_kl,
            old_approx_kl, clipfrac, activity_l2.
        """
        logratio = new_logp - old_logp
        ratio = logratio.exp()

        with torch.no_grad():
            old_approx_kl = (-logratio).mean()
            approx_kl = ((ratio - 1.0) - logratio).mean()
            clipfrac = ((ratio - 1.0).abs() > self.clip_coef).float().mean()

        mb_advantages = advantages
        if self.norm_adv:
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                mb_advantages.std() + 1e-8)

        # Policy loss (clipped surrogate)
        pg_loss1 = -mb_advantages * ratio
        pg_loss2 = -mb_advantages * torch.clamp(
            ratio, 1 - self.clip_coef, 1 + self.clip_coef)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        # Value loss
        if self.clip_vloss:
            v_loss_unclipped = (new_values - returns) ** 2
            v_clipped = old_values + torch.clamp(
                new_values - old_values, -self.clip_coef, self.clip_coef)
            v_loss_clipped = (v_clipped - returns) ** 2
            v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
        else:
            v_loss = 0.5 * ((new_values - returns) ** 2).mean()

        entropy_loss = entropies.mean()

        loss = pg_loss - self.ent_coef * entropy_loss + self.vf_coef * v_loss
        act_l2 = torch.zeros((), device=loss.device)
        if self.activity_l2 > 0:
            assert states is not None, "states required when activity_l2 > 0"
            act_l2 = (states ** 2).mean()
            loss = loss + self.activity_l2 * act_l2

        logs = {
            "pg_loss": pg_loss.item(), "v_loss": v_loss.item(),
            "entropy": entropy_loss.item(), "approx_kl": approx_kl.item(),
            "old_approx_kl": old_approx_kl.item(), "clipfrac": clipfrac.item(),
            "activity_l2": act_l2.item(),
        }
        return loss, logs


@register_rl_loss("reinforce")
class ReinforceLoss:
    """REINFORCE with a learned value baseline (one-step actor-critic).

    Policy gradient ``-log pi(a|s) * A`` with ``A = return - V(s)`` (computed
    by the trainer's return estimator) plus a value regression towards the
    observed returns and an optional entropy bonus. No importance ratio and
    no clipping: the batch must be strictly on-policy, so train with
    ``RLTrainingArguments(update_epochs=1, num_minibatches=1)``.

    This is the reward-only recipe of Song, Yang & Wang (2017): BPTT through
    the whole rollout, sparse end-of-episode rewards, Adam.

    Args:
        vf_coef: value-loss coefficient.
        ent_coef: entropy bonus coefficient.
        norm_adv: normalize advantages per batch.
        activity_l2: coefficient of the mean squared hidden-state penalty
            (applied to ALL states of the replayed sequence).
    """

    def __init__(self, vf_coef: float = 0.5, ent_coef: float = 0.0,
                 norm_adv: bool = False, activity_l2: float = 0.0):
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.norm_adv = norm_adv
        self.activity_l2 = activity_l2

    def __call__(self, new_logp, old_logp, advantages, new_values, old_values,
                 returns, entropies, states=None, extras=None):
        """Compute the REINFORCE loss.

        All tensor args are (B, T). ``old_logp``/``old_values`` are ignored
        (on-policy, ratio == 1). ``states`` (B, T, M) is only needed when
        ``activity_l2 > 0``. ``extras`` is accepted for interface uniformity
        and ignored.

        Returns:
            (loss, logs) with the same keys as ``PPOLoss`` (approx_kl /
            old_approx_kl / clipfrac are 0 — there is no ratio).
        """
        mb_advantages = advantages
        if self.norm_adv:
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                mb_advantages.std() + 1e-8)

        pg_loss = -(mb_advantages * new_logp).mean()
        v_loss = 0.5 * ((new_values - returns) ** 2).mean()
        entropy_loss = entropies.mean()

        loss = pg_loss - self.ent_coef * entropy_loss + self.vf_coef * v_loss
        act_l2 = torch.zeros((), device=loss.device)
        if self.activity_l2 > 0:
            assert states is not None, "states required when activity_l2 > 0"
            act_l2 = (states ** 2).mean()
            loss = loss + self.activity_l2 * act_l2

        zero = torch.zeros((), device=loss.device)
        logs = {
            "pg_loss": pg_loss.item(), "v_loss": v_loss.item(),
            "entropy": entropy_loss.item(), "approx_kl": zero.item(),
            "old_approx_kl": zero.item(), "clipfrac": zero.item(),
            "activity_l2": act_l2.item(),
        }
        return loss, logs


@register_rl_loss("a2c_pred")
class A2CPredLoss:
    """REINFORCE-with-baseline + auxiliary world-model cross-entropy.

    The Jensen, Hennequin & Mattar (2024) recipe: on-policy policy gradient
    with undiscounted Monte-Carlo advantages (use with
    ``RLTrainingArguments(update_epochs=1, num_minibatches=1, gamma=1.0,
    estimator="nstep", aux_target_dim=<K>)``), a value baseline regressed to
    the observed returns, an entropy bonus (gradient-equivalent to the paper's
    uniform-prior KL regularizer), and a cross-entropy term on the agent's
    auxiliary prediction head::

        loss = pg_coef * (-mean(A * logp)) + vf_coef * 0.5*mean((V - R)^2)
               - ent_coef * mean(H[pi])
               + pred_coef * mean_s CE(aux_logits[:, :, slice_s], aux_targets[:, :, s])

    ``extras`` must provide ``aux_logits`` (B, T, D) — from
    ``ActorCriticModel.evaluate_sequence(..., return_aux=True)`` — and
    ``aux_targets`` (B, T, K) — per-step class indices collected by the trainer
    from ``info["aux_targets"]``; column order matches the insertion order of
    ``aux_slices``. The trainer wires both automatically when this loss is
    passed (it declares ``requires_aux = True``).

    Args:
        pg_coef: policy-gradient coefficient (paper: beta_r = 1.0).
        vf_coef: value-loss coefficient (paper: beta_v = 0.05).
        ent_coef: entropy bonus coefficient (paper: beta_e = 0.05).
        pred_coef: aux prediction loss coefficient (paper: beta_p = 0.5).
        aux_slices: ``{name: (start, stop)}`` logit slices, one per aux-target
            column; default matches Jensen-2024's 33-dim world-model head
            (16 next-state logits, 1 unused slot, 16 reward-location logits).
        norm_adv: normalize advantages per batch.
        activity_l2: coefficient of the mean squared hidden-state penalty.
    """

    requires_aux = True

    def __init__(self, pg_coef: float = 1.0, vf_coef: float = 0.05,
                 ent_coef: float = 0.05, pred_coef: float = 0.5,
                 aux_slices: dict | None = None,
                 norm_adv: bool = False, activity_l2: float = 0.0):
        self.pg_coef = pg_coef
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.pred_coef = pred_coef
        self.aux_slices = (dict(aux_slices) if aux_slices is not None else
                           {"next_state": (0, 16), "reward": (17, 33)})
        self.norm_adv = norm_adv
        self.activity_l2 = activity_l2

    def __call__(self, new_logp, old_logp, advantages, new_values, old_values,
                 returns, entropies, states=None, extras=None):
        """Compute the loss. All tensor args are (B, T); see class docstring."""
        if extras is None or "aux_logits" not in extras or "aux_targets" not in extras:
            raise ValueError("A2CPredLoss requires extras={'aux_logits', 'aux_targets'}")
        aux_logits = extras["aux_logits"]      # (B, T, D)
        aux_targets = extras["aux_targets"]    # (B, T, K) class indices

        mb_advantages = advantages
        if self.norm_adv:
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                mb_advantages.std() + 1e-8)

        pg_loss = -(mb_advantages * new_logp).mean()
        v_loss = 0.5 * ((new_values - returns) ** 2).mean()
        entropy_loss = entropies.mean()

        pred_terms = {}
        for s, (name, (start, stop)) in enumerate(self.aux_slices.items()):
            logits_s = aux_logits[:, :, start:stop].reshape(-1, stop - start)
            target_s = aux_targets[:, :, s].reshape(-1).long()
            pred_terms[name] = torch.nn.functional.cross_entropy(logits_s, target_s)
        pred_loss = torch.stack(list(pred_terms.values())).mean()

        loss = (self.pg_coef * pg_loss - self.ent_coef * entropy_loss
                + self.vf_coef * v_loss + self.pred_coef * pred_loss)
        act_l2 = torch.zeros((), device=loss.device)
        if self.activity_l2 > 0:
            assert states is not None, "states required when activity_l2 > 0"
            act_l2 = (states ** 2).mean()
            loss = loss + self.activity_l2 * act_l2

        zero = torch.zeros((), device=loss.device)
        logs = {
            "pg_loss": pg_loss.item(), "v_loss": v_loss.item(),
            "entropy": entropy_loss.item(), "approx_kl": zero.item(),
            "old_approx_kl": zero.item(), "clipfrac": zero.item(),
            "activity_l2": act_l2.item(),
            "pred_loss": pred_loss.item(),
            **{f"pred_loss_{k}": v.item() for k, v in pred_terms.items()},
        }
        return loss, logs
