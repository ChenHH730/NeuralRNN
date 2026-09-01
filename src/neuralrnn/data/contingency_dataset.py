"""Contingency degradation experiment dataset (value-RNN task).

Simulated Pavlovian contingency experiments used to train value-RNNs
(critic-only RNNs trained with TD learning) and belief-state reference models.
Ported from Qian & Burrell (2024/2025, Nat. Neurosci.) ``value_rnn`` package
(``valuernn/tasks/contingency.py`` and ``valuernn/tasks/trial.py``).

Trial structure (time step ``dt``, default 0.5 s in the reference):

    [ ITI (nothing) ] cue onset [ ISI ] reward (or omission)

* cues: A (index 0), B / "nogo" (index 1), C (index 2, shown only in
  ``"cue-c"`` mode; in the other modes it is a *hidden* cue whose trials
  deliver "background" / unexpected rewards);
* inputs are one-hot cue channels plus a reward channel
  (``input_dim = n_cues + 1 = 4``);
* the target is the reward channel (used by the TD objective as ``r_t``);
* the ITI is geometric (p = 1/12, truncated to [8, 20]) + ``iti_min`` steps;
  the reward timing can be jittered by up to ``jitter`` steps to simulate
  internal timing uncertainty.

Modes (reward probabilities / cue visibility):

    "conditioning": rew_probs=[0.75, 0, 0],    cue_shown=[T, T, F]
    "degradation":  rew_probs=[0.75, 0, 0.75], cue_shown=[T, T, F]
    "cue-c":        rew_probs=[0.75, 0, 0.75], cue_shown=[T, T, T]

Batches are **episodes**: ``ntrials_per_episode`` consecutive trials
concatenated into one stream (the RNN hidden state is carried across trials
within an episode, as in the reference). ``sample_batch()`` returns
``{"inputs": (B, T, 4), "targets": (B, T, 1), "mask": (B, T, 1)}`` with
zero-padding to the longest episode in the batch. Trial-level metadata is
kept in ``self.episodes`` (list of episodes, each a list of trial dicts) for
analysis (see ``neuralrnn.rl.probe``).
"""
from __future__ import annotations

import numpy as np
import torch

from .base import BaseDataset, Trials

MODES = ("conditioning", "degradation", "cue-c")

# Per-mode (rew_probs, cue_shown); cue order is [A, B("nogo"), C].
_MODE_PARAMS = {
    "conditioning": ([0.75, 0.0, 0.0], [True, True, False]),
    "degradation": ([0.75, 0.0, 0.75], [True, True, False]),
    "cue-c": ([0.75, 0.0, 0.75], [True, True, True]),
}


class ContingencyDataset(BaseDataset):
    """Simulated contingency experiment (see module docstring).

    Args:
        mode: "conditioning" | "degradation" | "cue-c" (sets rew_probs and
            cue_shown; pass both explicitly for custom experiments).
        ntrials: total number of trials in the simulated session.
        ntrials_per_episode: trials concatenated per episode (batch element).
        rew_times: base ISI (cue -> reward delay) per cue, in steps.
        rew_sizes: reward magnitude per cue.
        rew_probs / cue_shown: override the mode defaults (both or neither).
        cue_probs: trial-type frequencies [A, B, C].
        jitter: reward timing is jittered uniformly in [-jitter, +jitter]
            steps (0 for display/test datasets).
        iti_min: minimum ITI in steps (geometric part is drawn on top).
        iti_p: geometric parameter of the ITI distribution.
        iti_trunc: (lo, hi) truncation range of the geometric draw
            (reference: [8, 20] steps).
        batch_size: episodes per ``sample_batch()`` call.
        seed: RNG seed for reproducible experiment generation.

    Reference hyperparameters (Qian & Burrell 2024 RNN modeling):
        ntrials=10000, ntrials_per_episode=20, rew_times=[8, 8, 8]
        (dt = 0.5 s -> ISI = 4 s), jitter=1, iti_min=20.
    """

    kind = "rl_experiment"
    input_dim = 4
    output_dim = 1

    def __init__(
        self,
        mode: str = "conditioning",
        ntrials: int = 10000,
        ntrials_per_episode: int = 20,
        rew_times: list[int] | tuple = (8, 8, 8),
        rew_sizes: list[float] | tuple = (1.0, 1.0, 1.0),
        rew_probs: list[float] | None = None,
        cue_shown: list[bool] | None = None,
        cue_probs: list[float] | tuple = (0.4, 0.2, 0.4),
        jitter: int = 1,
        iti_min: int = 20,
        iti_p: float = 1.0 / 12,
        iti_trunc: tuple[int, int] = (8, 20),
        batch_size: int = 12,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if (rew_probs is None) != (cue_shown is None):
            raise ValueError("rew_probs and cue_shown must be set together "
                             "(or neither, to use the mode defaults).")
        if mode is not None:
            if mode not in _MODE_PARAMS:
                raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
            if rew_probs is None:
                rew_probs, cue_shown = _MODE_PARAMS[mode]
        self.mode = mode
        self.ntrials = ntrials
        self.ntrials_per_episode = ntrials_per_episode
        self.rew_times = list(rew_times)
        self.rew_sizes = list(rew_sizes)
        self.rew_probs = list(rew_probs)
        self.cue_shown = list(cue_shown)
        self.cue_probs = list(cue_probs)
        self.jitter = jitter
        self.iti_min = iti_min
        self.iti_p = iti_p
        self.iti_trunc = iti_trunc
        self.batch_size = batch_size
        self.seed = seed
        self.n_cues = len(self.cue_probs)
        self.input_dim = self.n_cues + 1

        rng = np.random.RandomState(seed)
        self._make_experiment(rng)
        self._gen = torch.Generator().manual_seed(seed) if seed is not None else None

    # ------------------------------------------------- experiment generation
    def _make_experiment(self, rng: np.random.RandomState) -> None:
        """Draw trial sequence, ITIs, and per-trial arrays; group into episodes."""
        # Trial-type sequence: blocks of 20 trials with fixed per-cue counts,
        # shuffled within block (keeps local proportions close to cue_probs).
        trials_per_block = 20
        nblocks = int(np.ceil(self.ntrials / trials_per_block))
        per_cue = np.round(np.array(self.cue_probs) * trials_per_block).astype(int)
        block = np.concatenate([c * np.ones(n, dtype=int)
                                for c, n in enumerate(per_cue)])
        cues = np.concatenate([block[rng.permutation(len(block))]
                               for _ in range(nblocks)])[: self.ntrials]

        # ITIs: geometric(p) - 1, truncated to iti_trunc, shifted by iti_min.
        itis = np.empty(self.ntrials, dtype=int)
        n = 0
        while n < self.ntrials:
            draw = rng.geometric(p=self.iti_p, size=self.ntrials) - 1
            keep = draw[(draw >= self.iti_trunc[0]) & (draw <= self.iti_trunc[1])]
            m = min(len(keep), self.ntrials - n)
            itis[n:n + m] = keep[:m]
            n += m
        itis = itis + self.iti_min

        # Per-trial arrays and metadata.
        self.trials: list[dict] = []
        for i, (cue, iti) in enumerate(zip(cues, itis)):
            self.trials.append(self._make_trial(int(cue), int(iti), rng))

        # Episodes: consecutive non-overlapping groups of trials.
        self.episodes: list[list[dict]] = []
        for t0 in range(0, self.ntrials - self.ntrials_per_episode + 1,
                        self.ntrials_per_episode):
            episode = self.trials[t0:t0 + self.ntrials_per_episode]
            for ti, trial in enumerate(episode):
                trial["index_in_episode"] = ti
            self.episodes.append(episode)

    def _make_trial(self, cue: int, iti: int, rng: np.random.RandomState) -> dict:
        """Single trial: cue at t=iti, reward (size or 0) at t=iti+isi."""
        rewarded = rng.rand() <= self.rew_probs[cue]
        rew_size = self.rew_sizes[cue] if rewarded else 0.0
        isi = int(self.rew_times[cue])
        if isi > 0 and self.jitter > 0:
            isi += int(rng.choice(np.arange(-self.jitter, self.jitter + 1)))
        assert isi >= 0

        length = iti + isi + 1
        X = np.zeros((length, self.input_dim), dtype=np.float32)
        y = np.zeros((length, 1), dtype=np.float32)
        if self.cue_shown[cue]:
            X[iti, cue] = 1.0
        X[iti + isi, -1] = rew_size   # reward channel is also an input
        y[iti + isi, 0] = rew_size
        return {"cue": cue, "iti": iti, "isi": isi, "reward_size": rew_size,
                "rewarded": rewarded, "show_cue": self.cue_shown[cue],
                "X": X, "y": y, "length": length}

    # ------------------------------------------------------------- interface
    def episode_arrays(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Concatenated (X, y) of one episode, shape (T, 4) / (T, 1)."""
        episode = self.episodes[index]
        return (np.concatenate([t["X"] for t in episode], axis=0),
                np.concatenate([t["y"] for t in episode], axis=0))

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        X, y = self.episode_arrays(index)
        return {"inputs": torch.from_numpy(X), "targets": torch.from_numpy(y)}

    def sample_batch(self) -> dict[str, torch.Tensor]:
        """Random episodes (with replacement), zero-padded + mask.

        Returns {"inputs": (B, T, 4), "targets": (B, T, 1), "mask": (B, T, 1)}.
        """
        idx = torch.randint(0, len(self.episodes), (self.batch_size,),
                            generator=self._gen)
        seqs = [self.episode_arrays(int(i)) for i in idx]
        T = max(X.shape[0] for X, _ in seqs)
        B = len(seqs)
        inputs = torch.zeros(B, T, self.input_dim)
        targets = torch.zeros(B, T, self.output_dim)
        mask = torch.zeros(B, T, self.output_dim)
        for b, (X, y) in enumerate(seqs):
            L = X.shape[0]
            inputs[b, :L] = torch.from_numpy(X)
            targets[b, :L] = torch.from_numpy(y)
            mask[b, :L] = 1.0
        return {"inputs": inputs, "targets": targets, "mask": mask}

    def sample_trials(self, n: int, seed: int | None = None) -> Trials:
        """Return n complete single trials as a ``Trials`` object."""
        n_total = len(self.trials)
        if seed is None:
            idx = np.arange(min(n, n_total))
        else:
            g = np.random.RandomState(seed)
            idx = (g.permutation(n_total)[:n] if n <= n_total
                   else g.randint(0, n_total, n))
        lengths = [self.trials[i]["length"] for i in idx]
        T = max(lengths)
        inputs = torch.zeros(len(idx), T, self.input_dim)
        targets = torch.zeros(len(idx), T, 1)
        mask = torch.zeros(len(idx), T, 1)
        conditions = []
        for row, (i, L) in enumerate(zip(idx, lengths)):
            t = self.trials[int(i)]
            inputs[row, :L] = torch.from_numpy(t["X"])
            targets[row, :L] = torch.from_numpy(t["y"])
            mask[row, :L] = 1.0
            conditions.append({
                "epochs": {"iti": (0, t["iti"]), "isi": (t["iti"], t["iti"] + t["isi"])},
                "n_steps": L,
                "is_catch": not t["rewarded"],
                "cue": t["cue"], "iti": t["iti"], "isi": t["isi"],
                "reward_size": t["reward_size"],
            })
        return Trials(inputs, targets, mask, conditions)

    @classmethod
    def from_params(cls, **kwargs) -> "ContingencyDataset":
        """Registry loader entry point (same as the constructor)."""
        return cls(**kwargs)
