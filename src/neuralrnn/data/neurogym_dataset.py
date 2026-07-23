"""Neurogym task dataset (Paradigm A / task optimization).

Ported from the data-extraction style of RNN_DynamicalSystemAnalysis.ipynb: build a cognitive-task
environment with neurogym, wrap it as a PyTorch dataloader, and yield (inputs, targets) per batch.
Here we convert to the batch-first standard batch dict at the boundary (ARCHITECTURE §3.1 Paradigm A).

neurogym is an optional heavy dependency (see pyproject [project.optional-dependencies] neurogym);
a clear hint is shown when not installed.

Version compatibility (tested with neurogym 1.0.8 and 2.3.1):
  * ``ngym.make`` / ``ngym.Dataset`` exist in both and accept an env *instance*; we always build the
    env ourselves and pass the **unwrapped** TrialEnv. This avoids gymnasium's ``Wrapper.__getattr__``
    fall-through, which emits "env.<attr> to get variables from other wrappers is deprecated"
    warnings (neurogym's Dataset calls ``env.seed(...)`` internally, and analysis code reads
    ``env.dt`` / ``env.trial``). ``self.env`` is therefore always the bare TrialEnv.
  * ``all_envs`` moved between versions (``ngym.all_envs`` in 1.x,
    ``neurogym.envs.registration.all_envs`` in 2.x) — see ``list_neurogym_datasets``.
  * Discrete action spaces yield integer class targets (CrossEntropy); Box action spaces yield
    continuous float targets (regression) instead of being corrupted by a long cast.

Two modes (see NeurogymDataset): streaming ``seq_len`` windows (default), or trial-aligned
pre-generation (``n_trials=N``) exposing the same interface as CognitiveTaskDataset
(``inputs``/``targets``/``mask``/``conditions``/``get_all_trials()``).
"""
from __future__ import annotations

import importlib.metadata
import warnings
from typing import Any

import numpy as np
import torch

from .base import BaseDataset, Trials, subset_trials

# gymnasium emits the first message when an attribute is read through a wrapper chain, and the
# second for neurogym env registrations that declare no render_modes metadata. We build unwrapped
# envs ourselves, but keep scoped filters as a safety net for neurogym internals that may still go
# through wrappers in some versions.
_NOISY_MESSAGES = (
    "to get variables from other wrappers is deprecated",
    "The environment creator metadata doesn't include",
)


def _ignore_noisy_warnings():
    for msg in _NOISY_MESSAGES:
        warnings.filterwarnings("ignore", message=f".*{msg}.*")


def _import_neurogym():
    try:
        import neurogym as ngym
    except ImportError as e:
        raise ImportError(
            "neurogym is required: pip install neurogym"
        ) from e
    return ngym


def neurogym_version() -> str | None:
    """Return the installed neurogym version string, or None if not installed."""
    try:
        return importlib.metadata.version("neurogym")
    except importlib.metadata.PackageNotFoundError:
        return None


def list_neurogym_datasets() -> list[str]:
    """List env ids registered by the installed neurogym (empty list if not installed).

    Works across neurogym versions: 1.x exposes ``ngym.all_envs`` while 2.x moved it to
    ``neurogym.envs.registration.all_envs``. Collection envs (e.g. the yang19 task set, neurogym 2.x
    only) are included when the installed version supports them.
    """
    try:
        ngym = _import_neurogym()
    except ImportError:
        return []
    all_envs = getattr(ngym, "all_envs", None)
    if all_envs is None:
        try:
            from neurogym.envs.registration import all_envs
        except Exception:
            return []
    for kwargs in ({"collections": True}, {}):  # 2.x accepts collections=True; 1.x does not
        try:
            return sorted(all_envs(**kwargs))
        except TypeError:
            continue
        except Exception:
            return []
    return []


def _resolve_task_id(task: str) -> str:
    """Resolve a user-given task name to a registered neurogym env id.

    Accepts the exact id ('PerceptualDecisionMaking-v0'), the bare class name
    ('PerceptualDecisionMaking' -> '-v0' appended), or a case-insensitive match
    ('gonogo' -> 'GoNogo-v0'). Unknown names are returned unchanged so the error raised by
    ``_make_env`` can list the available ids.
    """
    envs = list_neurogym_datasets()
    if task in envs:
        return task
    if f"{task}-v0" in envs:
        return f"{task}-v0"
    lower = task.lower()
    for env_id in envs:
        if env_id.lower() in (lower, f"{lower}-v0"):
            return env_id
    return task


def _make_env(task: str, **env_kwargs: Any):
    """Create an **unwrapped** neurogym env (works with both neurogym 1.x and 2.x).

    Unwrapping matters: reading attributes (``.dt``, ``.trial``, ``.seed()`` ...) through gymnasium
    wrappers triggers deprecation warnings, and *assigning* through them silently shadows the
    attribute on the wrapper. This package and the notebooks therefore always work with the base
    TrialEnv directly.
    """
    ngym = _import_neurogym()
    resolved = _resolve_task_id(task)
    try:
        with warnings.catch_warnings():
            _ignore_noisy_warnings()
            env = ngym.make(resolved, **env_kwargs)
    except Exception as e:
        raise ValueError(
            f"Failed to create neurogym env '{resolved}' (requested '{task}'): {e}\n"
            f"Available env ids: {list_neurogym_datasets()}"
        ) from e
    return getattr(env, "unwrapped", env)


class NeurogymDataset(BaseDataset):
    """Neurogym dataset with two modes (see ``from_task``):

    * **streaming** (default): ``seq_len`` windows sliced from a continuous trial stream via
      ``ngym.Dataset`` — the original behavior. ``mask`` is None; windows may cross trial
      boundaries.
    * **trial-aligned** (``n_trials=N``): pre-generates N complete trials and exposes the same
      interface as ``CognitiveTaskDataset`` — ``inputs`` / ``targets`` / ``mask`` / ``conditions``
      attributes, ``get_all_trials()``, ``__len__``, and whole-trial ``sample_batch()``.
    """

    kind = "neurogym"

    def __init__(self, env, dataset, input_dim: int, output_dim: int,
                 batch_size: int = 16, seq_len: int = 100, output_type: str = "discrete"):
        self.env = env                 # Unwrapped TrialEnv, kept for raw access (see cognitive_tasks.ipynb)
        self._dataset = dataset        # neurogym Dataset (streaming mode); None in trial-aligned mode
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.seq_len = seq_len
        # "discrete": targets are integer class indices (CrossEntropy);
        # "continuous": targets are float action vectors (regression)
        self.output_type = output_type
        self.dt = getattr(env, "dt", None)  # env is unwrapped, so this is a direct attribute
        self._trial_aligned = dataset is None

    @classmethod
    def from_task(cls, task: str, *, batch_size: int = 16, seq_len: int = 100,
                  dt: int = 100, timing: dict | None = None, seed: int | None = None,
                  n_trials: int | None = None, **env_kwargs: Any) -> "NeurogymDataset":
        """Construct dataset from a task name, e.g. task='PerceptualDecisionMaking-v0'.

        Called by load_dataset('perceptual_decision_making') in data/registry.py, and by
        load_dataset's passthrough for any other installed neurogym env id (case-insensitive,
        '-v0' optional). Extra kwargs are forwarded to the neurogym environment
        (dt / timing / task-specific parameters). ``seed`` makes trial generation reproducible.

        ``n_trials=None`` (default) gives the streaming dataset; ``n_trials=N`` pre-generates N
        complete trials and exposes the CognitiveTaskDataset-style interface (see class docstring).
        """
        ngym = _import_neurogym()

        kwargs = {"dt": dt, **env_kwargs}
        if timing is not None:
            kwargs["timing"] = timing
        env = _make_env(task, **kwargs)

        input_dim = int(env.observation_space.shape[0])
        n_actions = getattr(env.action_space, "n", None)
        if n_actions is not None:
            # Discrete action space: n classes (CrossEntropy target)
            output_type, output_dim = "discrete", int(n_actions)
        else:
            # Box action space: continuous regression target
            output_type, output_dim = "continuous", int(env.action_space.shape[0])

        if n_trials is not None:
            ds = cls(env, None, input_dim, output_dim,
                     batch_size=batch_size, seq_len=seq_len, output_type=output_type)
            ds.inputs, ds.targets, ds.mask, ds.conditions = ds._generate_trials(n_trials, seed=seed)
            return ds

        with warnings.catch_warnings():
            _ignore_noisy_warnings()
            try:
                # Pass the env *instance*: both neurogym 1.x and 2.x deepcopy it per batch lane,
                # so their internal env.seed(...) calls land directly on the TrialEnv (no wrapper
                # fall-through, no deprecation warnings).
                dataset = ngym.Dataset(env, batch_size=batch_size, seq_len=seq_len)
            except Exception:
                # Some envs (e.g. perceptualdecisionmaking.ibl20-v0) only work when the Dataset
                # builds wrapped envs itself; fall back to the env-id string path. Warning
                # filters above still cover the wrapper-attribute accesses in that case.
                dataset = ngym.Dataset(_resolve_task_id(task), env_kwargs=kwargs,
                                       batch_size=batch_size, seq_len=seq_len)
        if seed is not None:
            dataset.seed(seed)
            # The initial cache was filled before seeding (Dataset.__init__ reseeds with None);
            # regenerate it so the whole sampled stream is reproducible. _cache is identical in
            # neurogym 1.x/2.x and resets the read pointer.
            try:
                dataset._cache()
            except AttributeError:
                pass
        return cls(env, dataset, input_dim, output_dim,
                   batch_size=batch_size, seq_len=seq_len, output_type=output_type)

    # ------------------------------------------------------------------ trial-aligned mode

    def _generate_trials(self, n_trials: int, seed: int | None = None):
        """Generate ``n_trials`` complete trials from the unwrapped env (zero-padded to the
        longest trial) and return (inputs, targets, mask, conditions) with the same field
        layout as CognitiveTaskDataset."""
        env = self.env
        if seed is not None:
            env.seed(seed)
        obs_list, gt_list, conditions = [], [], []
        with warnings.catch_warnings():
            _ignore_noisy_warnings()
            for _ in range(n_trials):
                env.new_trial()
                ob = np.asarray(env.ob, dtype=np.float32)   # (T_i, input_dim)
                gt = np.asarray(env.gt)                     # (T_i,) or (T_i, act_dim)
                cond = dict(env.trial) if isinstance(env.trial, dict) else {}
                # Unified extras: epoch bounds (multitask_flexible convention) + true trial length
                cond["epochs"] = {p: (int(env.start_ind[p]), int(env.end_ind[p]))
                                  for p in env.start_ind}
                cond["n_steps"] = int(ob.shape[0])
                cond.setdefault("is_catch", False)
                obs_list.append(ob)
                gt_list.append(gt)
                conditions.append(cond)

        t_max = max(ob.shape[0] for ob in obs_list)
        inputs = np.zeros((n_trials, t_max, self.input_dim), dtype=np.float32)
        mask = np.zeros((n_trials, t_max), dtype=np.float32)  # 1 = valid step, 0 = padding
        for i, ob in enumerate(obs_list):
            inputs[i, : ob.shape[0]] = ob
            mask[i, : ob.shape[0]] = 1.0
        if self.output_type == "discrete":
            targets = np.zeros((n_trials, t_max), dtype=np.int64)  # 0 = fixation, masked anyway
        else:
            targets = np.zeros((n_trials, t_max, self.output_dim), dtype=np.float32)
        for i, gt in enumerate(gt_list):
            targets[i, : gt.shape[0]] = gt

        return (torch.from_numpy(inputs),   # (N, T_max, input_dim)
                torch.from_numpy(targets),  # (N, T_max) or (N, T_max, act_dim)
                torch.from_numpy(mask),     # (N, T_max)
                conditions)                 # list of N per-trial dicts

    def sample_trials(self, n: int, seed: int | None = None) -> Trials:
        """Return n complete trials as a ``Trials`` object — in either mode, without
        creating a second dataset (replaces the ``ds_viz = load_dataset(..., n_trials=n)``
        pattern). Streaming mode: fresh trials are generated from the held env.
        Trial-aligned mode: subset of the pre-generated trials (first-n, or seeded
        random when ``seed`` is given)."""
        if not self._trial_aligned:
            inputs, targets, mask, conditions = self._generate_trials(n, seed=seed)
            return Trials(inputs, targets, mask, conditions)
        return subset_trials(self.inputs, self.targets, self.mask,
                             self.conditions, n, seed)

    def _require_trials(self) -> None:
        if not self._trial_aligned:
            raise RuntimeError(
                "This NeurogymDataset is in streaming mode (seq_len windows over a trial stream); "
                "the trial-aligned interface (inputs/targets/mask/conditions/get_all_trials) "
                "requires n_trials=... in from_task/load_dataset."
            )

    def get_all_trials(self) -> dict[str, torch.Tensor]:
        """Return the full pre-generated dataset (trial-aligned mode only)."""
        self._require_trials()
        return {"inputs": self.inputs, "targets": self.targets, "mask": self.mask}

    def __len__(self) -> int:
        self._require_trials()
        return self.inputs.shape[0]

    # ------------------------------------------------------------------ sampling

    def sample_batch(self) -> dict[str, torch.Tensor]:
        """Sample one batch.

        Trial-aligned mode: whole trials sampled with replacement, returns
        {"inputs" (B,T,obs), "targets" (B,T)/(B,T,act), "mask" (B,T)}.
        Streaming mode: concatenated-trial windows from neurogym, returns
        {"inputs" (B,T,obs), "targets" (B,T) discrete or (B,T,act) continuous,
        "mask": None}.
        """
        if self._trial_aligned:
            # CognitiveTaskDataset pattern: sample whole trials with replacement
            idx = torch.randint(0, self.inputs.shape[0], (self.batch_size,))
            return {"inputs": self.inputs[idx], "targets": self.targets[idx],
                    "mask": self.mask[idx]}
        # neurogym Dataset() returns time-first: inputs (T,B,obs), target (T,B) or (T,B,act)
        with warnings.catch_warnings():
            _ignore_noisy_warnings()
            inputs, target = self._dataset()
        inputs = torch.as_tensor(inputs, dtype=torch.float32).permute(1, 0, 2)  # -> (B,T,obs)
        target = np.asarray(target)
        if self.output_type == "discrete":
            t = torch.as_tensor(target, dtype=torch.long)
            if t.ndim == 3 and t.shape[-1] == 1:
                t = t.squeeze(-1)
            targets = t.permute(1, 0)                                             # -> (B,T)
        else:
            targets = torch.as_tensor(target, dtype=torch.float32).permute(1, 0, 2)  # -> (B,T,act)
        return {"inputs": inputs, "targets": targets, "mask": None}

    def task_input(self, kind: str = "stimulus") -> torch.Tensor:
        """Return the 'task-condition input' for fixed-point analysis (e.g., mean input at 0-coherence for decision tasks).
        Defaults to zero input; override per task during porting as needed (see PORTING_GUIDE recipe 1)."""
        return torch.zeros(self.input_dim, dtype=torch.float32)
