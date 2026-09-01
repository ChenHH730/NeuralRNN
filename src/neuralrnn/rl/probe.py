"""Probing utilities for value models (critic-only RNNs) on episodic tasks.

Runs a trained value model over dataset episodes and packages the trial-level
records used by value-RNN analyses (value traces, reward prediction errors,
aligned averaging). The alignment/labeling conventions port
``r_analysis_code/Fig6_functions.R`` (``new_label_data`` / ``new_align_time``
/ ``shift_iti``) of Qian & Burrell (2024):

* trial types are labeled from the *input* sums (so trials whose cue is
  hidden fall into the unpredicted-reward / blank classes);
* ``aligned_time = t - first_event_time`` (first nonzero input step);
  trials whose first event is the reward itself (unpredicted background
  rewards) are shifted by ``+ isi + 1`` so their reward lands one step after
  the cued-reward RPE index;
* RPE is indexed by the *transition start*: ``rpe[t] = y[t+1] + gamma *
  V[t+1] - V[t]``, so the cue-triggered RPE sits at aligned_time ``-1`` and
  the cued-reward RPE at ``isi - 1``;
* ``shift_iti`` folds ITI steps earlier than ``-pre_cue_steps`` back onto the
  end of the previous trial, leaving exactly ``pre_cue_steps`` pre-cue steps
  per trial.

The labeling functions are generic over the number and names of cues via the
``cue_names`` / ``hidden_rewarded`` / ``hidden_blank`` arguments; the module
constant ``TRIAL_TYPES`` is the Qian & Burrell (2024) configuration
(3 cues A/B/C), provided as the example used by notebook 19.
"""
from __future__ import annotations

import numpy as np
import torch

# Qian & Burrell (2024) trial-type configuration (3 cues: A, B, C).
TRIAL_TYPES = (
    "Cue A Rewarded", "Cue A Unrewarded", "Cue B",
    "Cue C Rewarded", "Cue C Unrewarded", "Degradation", "Blank",
)


@torch.no_grad()
def probe_value_model(model, dataset, n_episodes: int | None = None,
                      gamma: float = 0.99, device: str | None = None,
                      seed: int | None = None) -> list[dict]:
    """Run a value model over dataset episodes and collect per-episode records.

    Args:
        model: trained value model; ``model(inputs)`` must return an output
            with ``.states`` (1, T, H) and ``.outputs`` (1, T, 1) or (1, T).
        dataset: ContingencyDataset-like (provides ``episodes`` and
            ``episode_arrays``).
        n_episodes: how many episodes to probe (default: all).
        gamma: discount factor used for the RPE (paper value-RNNs: 0.83).
        device: torch device (default: model's device).
        seed: if given, sample episodes randomly instead of taking the first.

    Returns:
        List of episode records with keys:
        ``X`` (T, D), ``y`` (T,), ``Z`` (T, H), ``V`` (T,), ``rpe`` (T,;
        last step NaN), ``trials`` (list of per-trial metadata dicts with
        added ``start`` / ``stop`` boundaries), ``episode`` (index).
    """
    was_training = model.training
    model.eval()
    device = device or next(model.parameters()).device
    n = len(dataset.episodes) if n_episodes is None else min(n_episodes, len(dataset.episodes))
    if seed is None:
        idx = np.arange(n)
    else:
        idx = np.random.RandomState(seed).choice(len(dataset.episodes), n, replace=False)

    records = []
    for ep in idx:
        X, y = dataset.episode_arrays(int(ep))
        inputs = torch.from_numpy(X)[None].to(device)
        out = model(inputs)
        Z = out.states[0].detach().cpu().numpy()
        V = out.outputs[0].detach().cpu().numpy().reshape(-1)
        y = y.reshape(-1)
        rpe = np.full(X.shape[0], np.nan)
        rpe[:-1] = y[1:] + gamma * V[1:] - V[:-1]

        trials = []
        t0 = 0
        for trial in dataset.episodes[int(ep)]:
            t1 = t0 + trial["length"]
            trials.append({k: v for k, v in trial.items() if k not in ("X", "y")}
                           | {"start": t0, "stop": t1})
            t0 = t1
        records.append({"episode": int(ep), "X": X, "y": y, "Z": Z, "V": V,
                        "rpe": rpe, "trials": trials})
    if was_training:
        model.train()
    return records


def label_trial_type(X_trial: np.ndarray,
                     cue_names: tuple = ("A", "B", "C"),
                     reward_free_cues: tuple = (1,),
                     hidden_rewarded: str = "Degradation",
                     hidden_blank: str = "Blank") -> str:
    """Label a trial from its input sums (reference ``new_label_data``).

    Args:
        X_trial: (T, n_cues + 1) inputs; last channel is reward.
        cue_names: name of each cue channel; cue i is labeled
            ``"Cue <name> Rewarded/Unrewarded"``.
        reward_free_cues: indices of cues that never predict reward; they are
            labeled just ``"Cue <name>"`` (the reference labels cue B
            reward-free).
        hidden_rewarded: label for trials with reward but no cue.
        hidden_blank: label for trials with neither cue nor reward.
    """
    cues = X_trial[:, :-1].sum(axis=0)
    reward = X_trial[:, -1].sum()
    for i, name in enumerate(cue_names[:len(cues)]):
        if cues[i] > 0:
            if i in reward_free_cues:
                return f"Cue {name}"
            return f"Cue {name} Rewarded" if reward > 0 else f"Cue {name} Unrewarded"
    return hidden_rewarded if reward > 0 else hidden_blank


def align_time(X_trial: np.ndarray, isi_shift: int = 9) -> np.ndarray:
    """Cue-aligned time base for one trial (reference ``new_align_time``).

    ``aligned_time = t - first_event_time``; if the first event is the
    reward itself (hidden-cue / blank trials), shift by ``isi_shift``
    (reference: 9 = ISI 8 + 1) so the reward RPE lands one step after the
    cued-reward RPE.
    """
    active = np.where(X_trial.sum(axis=1) > 0)[0]
    if len(active) == 0:                      # nothing at all: align to trial start
        return np.arange(X_trial.shape[0])
    first = active[0]
    aligned = np.arange(X_trial.shape[0]) - first
    if X_trial[first, -1] > 0:                # first event is reward
        aligned = aligned + isi_shift
    return aligned


def split_episode_trials(record: dict, isi_shift: int = 9,
                         **label_kwargs) -> list[dict]:
    """Split one episode record into aligned per-trial records.

    Each trial dict has: ``trial_type``, ``aligned_time`` (L,), ``V`` (L,),
    ``rpe`` (L,), ``Z`` (L, H), plus the dataset trial metadata.
    ``label_kwargs`` are forwarded to ``label_trial_type``.
    """
    X, V, rpe, Z = record["X"], record["V"], record["rpe"], record["Z"]
    trials = []
    for meta in record["trials"]:
        s, e = meta["start"], meta["stop"]
        X_t = X[s:e]
        trials.append({
            **{k: v for k, v in meta.items() if k not in ("start", "stop")},
            "trial_type": label_trial_type(X_t, **label_kwargs),
            "aligned_time": align_time(X_t, isi_shift),
            "V": V[s:e], "rpe": rpe[s:e], "Z": Z[s:e],
        })
    return trials


def shift_iti(aligned: np.ndarray, pre_cue_steps: int = 4) -> np.ndarray:
    """Fold ITI steps before ``-pre_cue_steps`` onto the previous trial's end.

    Reference ``shift_iti``: each trial keeps exactly ``pre_cue_steps``
    pre-cue steps; earlier ITI steps are appended after the previous trial's
    last aligned step (their relative order is preserved).
    """
    aligned = np.asarray(aligned, dtype=int).copy()
    overflow = aligned < -pre_cue_steps
    if overflow.any():
        # Previous trial's post-cue part spans 0..(L - pre_cue - first_shift);
        # the folded steps continue right after its last *kept* index.
        n_overflow = int(overflow.sum())
        last_kept = np.max(aligned[~overflow]) if (~overflow).any() else -1
        aligned[overflow] = last_kept + np.arange(1, n_overflow + 1)
    return aligned


def average_by_aligned_time(trials: list[dict], key: str = "rpe",
                            pre_cue_steps: int = 4,
                            trial_types: tuple = TRIAL_TYPES):
    """Average a per-trial signal over aligned time, per trial type.

    Applies ``shift_iti`` to each trial's time base, then pools steps at each
    aligned time across trials of the same type.

    Returns:
        Dict ``trial_type -> (times (K,), mean (K,), sem (K,), n_trials)``;
        types with no trials are omitted.
    """
    out = {}
    for tt in trial_types:
        bucket: dict[int, list[float]] = {}
        n_trials = 0
        for trial in trials:
            if trial["trial_type"] != tt:
                continue
            n_trials += 1
            aligned = shift_iti(trial["aligned_time"], pre_cue_steps)
            for a, v in zip(aligned, trial[key]):
                if np.isfinite(v):
                    bucket.setdefault(int(a), []).append(float(v))
        if not bucket:
            continue
        times = np.array(sorted(bucket))
        mean = np.array([np.mean(bucket[t]) for t in times])
        sem = np.array([np.std(bucket[t], ddof=1) / np.sqrt(len(bucket[t]))
                        if len(bucket[t]) > 1 else 0.0 for t in times])
        out[tt] = (times, mean, sem, n_trials)
    return out


def mean_rpe_at_events(trials: list[dict], isi: int = 8,
                       pre_cue_steps: int = 4,
                       unpredicted_types: tuple = ("Degradation", "Blank")
                       ) -> dict[str, dict[str, float]]:
    """Fig-6c-style summary: mean RPE at the cue and reward events.

    Cue-triggered RPE is read at aligned_time ``-1``; reward RPE at
    ``isi - 1`` for cued trials and at ``isi`` for ``unpredicted_types``
    (their time base is shifted by ``isi + 1``).

    Returns:
        ``{trial_type: {"cue": float|nan, "reward": float|nan, "n": int}}``.
    """
    out = {}
    for trial in trials:
        tt = trial["trial_type"]
        aligned = shift_iti(trial["aligned_time"], pre_cue_steps)
        rpe = trial["rpe"]
        entry = out.setdefault(tt, {"cue": [], "reward": []})
        rew_t = isi if tt in unpredicted_types else isi - 1
        for a, v in zip(aligned, rpe):
            if not np.isfinite(v):
                continue
            if a == -1:
                entry["cue"].append(float(v))
            elif a == rew_t:
                entry["reward"].append(float(v))
    return {tt: {"cue": float(np.mean(e["cue"])) if e["cue"] else np.nan,
                 "reward": float(np.mean(e["reward"])) if e["reward"] else np.nan,
                 "n": len(e["reward"]) or len(e["cue"])}
            for tt, e in out.items()}
