"""Bartolo Monkey Probabilistic Reversal Learning dataset.

Loads behavioral data from Bartolo et al.'s monkey reversal learning task.
Two monkeys (V and W) performed a probabilistic reversal learning task with
binary actions and rewards. This is a one-step task where stage2 == action.

Reference:
    Bartolo, R., et al. (2020). "Medial prefrontal cortex controls
    the expression of reward-seeking behaviour via dorsolateral striatum."
    (Data: https://data.mendeley.com/datasets/p7ft2bvphx)

Data format (.mat files):
    X: (neuron_num, total_trial_num, bin_num) — spike counts (not used for behavior)
    Y: (total_trial_num, 13) — behavioral metadata:
        col 0: chosen_image (0 or 1)
        col 1: chosen_loc (0 or 1)
        col 2: outcome (0 or 1)
        col 3: complete_trial
        col 4: best_chosen
        col 5: trial_in_block
        col 6: reversal_trial
        col 7: block_id (1-24)
        col 8: blockorder
        col 9: block_type (1=what, 2=where)
        col 10: CorrectTrialnumberinsession
        col 11: TotalTrialnumberinblock
        col 12: BlockCompleted

Block structure: 80 trials per block, reversal happens between trials 30-50.
Default truncation: trials 10-70 (60 trials) to avoid edge effects.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .base import BaseDataset


class BartoloMonkeyDataset(BaseDataset):
    """Behavioral dataset for the Bartolo monkey probabilistic reversal learning task.

    Returns batch-first tensors for behavioral fitting (Tiny RNN paradigm).
    Input at each trial: [action_{t-1}, stage2_{t-1}, reward_{t-1}] (3 features).
    Target at each trial: action_t (0 or 1).

    Since this is a one-step task, stage2 == action.
    """

    kind = "behavioral"
    input_dim = 3   # [action, stage2, reward]
    output_dim = 2  # binary choice

    # Recording sessions per monkey
    SESSIONS = {
        'V': ['V20161005', 'V20160929', 'V20160930', 'V20161017'],
        'W': ['W20160112', 'W20160113', 'W20160121', 'W20160122'],
    }

    def __init__(self, data_dir: str | Path, animal_name: str = 'V',
                 filter_block_type: str = 'both',
                 block_truncation: tuple[int, int] = (10, 70),
                 verbose: bool = False,
                 dtype: torch.dtype = torch.float32,
                 input_format: str = 'current'):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.animal_name = animal_name
        self.filter_block_type = filter_block_type
        self.block_truncation = block_truncation
        self.dtype = dtype
        if input_format not in ('current', 'shifted'):
            raise ValueError(f"input_format must be 'current' or 'shifted', got {input_format}")
        self.input_format = input_format

        # Load raw behavioral data
        self._raw_behav = self._load_sessions(verbose)

        # Convert to tensors (batch-first)
        self._build_tensors()

        self.input_dim = 3
        self.output_dim = 2

    def _load_sessions(self, verbose: bool = False) -> dict:
        """Load all sessions for the specified animal."""
        import scipy.io

        sessions = self.SESSIONS[self.animal_name]
        behav = {'action': [], 'stage2': [], 'reward': [],
                 'trial_type': [], 'block_type': []}

        for sess_name in sessions:
            mat_path = self.data_dir / f'SPKcounts_{sess_name}cue_MW_250X250ms.mat'
            if not mat_path.exists():
                raise FileNotFoundError(
                    f"Data file not found: {mat_path}\n"
                    f"Please download the BartoloMonkey dataset from "
                    f"https://data.mendeley.com/datasets/p7ft2bvphx "
                    f"and place .mat files in {self.data_dir}"
                )
            mat = scipy.io.loadmat(str(mat_path))

            chosen_image = mat['Y'][:, 0].astype('int64')
            chosen_loc = mat['Y'][:, 1].astype('int64')
            outcome = mat['Y'][:, 2].astype('int64')
            block_id = mat['Y'][:, 7].astype('int64')
            block_type = mat['Y'][:, 9].astype('int64')
            BlockCompleted = mat['Y'][:, 12].astype('bool')

            trial_num = 80

            if self.filter_block_type in ['what', 'where']:
                if self.filter_block_type == 'what':
                    block_type_idx = 1
                    act = chosen_image
                else:
                    block_type_idx = 2
                    act = chosen_loc
                trial_filter = np.all([
                    block_type == block_type_idx,
                    block_id >= 1, block_id <= 24,
                    BlockCompleted
                ], axis=0)
            elif self.filter_block_type == 'both':
                act = chosen_image.copy()
                trial_filter2 = np.all([
                    block_type == 2, block_id >= 1, block_id <= 24, BlockCompleted
                ], axis=0)
                act[trial_filter2] = chosen_loc[trial_filter2]
                trial_filter = np.all([
                    block_id >= 1, block_id <= 24, BlockCompleted
                ], axis=0)
            else:
                raise ValueError(f"Unknown filter_block_type: {self.filter_block_type}")

            assert trial_filter.sum() % trial_num == 0, (
                f"Filtered trials ({trial_filter.sum()}) are not a multiple of "
                f"trial_num ({trial_num}); check filter_block_type="
                f"{self.filter_block_type!r} against the raw data.")
            episode_num = trial_filter.sum() // trial_num

            act = act[trial_filter].reshape((episode_num, trial_num))
            reward = outcome[trial_filter].reshape((episode_num, trial_num))
            bt_block_type = block_type[trial_filter].reshape((episode_num, trial_num)).mean(1).astype('int64')

            # Block truncation
            bt = self.block_truncation
            eff_trials = np.arange(bt[0], bt[1])
            act = act[:, eff_trials]
            reward = reward[:, eff_trials]

            if verbose:
                print(f"Session {sess_name}: {episode_num} blocks, "
                      f"{act.shape[1]} trials/block")

            behav['action'].extend(list(act))
            behav['stage2'].extend(list(act))  # one-step task: stage2 == action
            behav['reward'].extend(list(reward))
            behav['block_type'].extend(list(np.array(['what', 'where'])[bt_block_type - 1]))

        # Compute trial types: action * 2 + reward
        for i in range(len(behav['action'])):
            behav['trial_type'].append(behav['action'][i] * 2 + behav['reward'][i])

        return behav

    def _build_tensors(self):
        """Convert raw behavioral data to batch-first tensors.

        We support two input conventions:

        * ``input_format='current'`` (default, matches original tinyRNN):
            input[t] = [action[t], stage2[t], reward[t]]
            target[t] = action[t]
          When the model uses ``output_h0=True``, the model prepends the h0
          readout so that the effective alignment is:
            readout(h0)       -> action[0]
            readout(h_{t+1})  -> action[t+1], where h_{t+1} = GRU(input[t], h_t)
          This is the convention used in the original tinyRNN codebase.

        * ``input_format='shifted'`` (legacy NeuralRNN convention):
            input[t] = [action[t-1], stage2[t-1], reward[t-1]]
            target[t] = action[t]
          with input[0] = [0, 0, 0]. This was the initial port convention but
          does not reproduce original tinyRNN results because the last trial's
          observation is never fed into the network.
        """
        n_blocks = len(self._raw_behav['action'])
        T = len(self._raw_behav['action'][0])

        action = np.zeros((n_blocks, T))
        stage2 = np.zeros((n_blocks, T))
        reward = np.zeros((n_blocks, T))
        trial_type = np.zeros((n_blocks, T))
        mask = np.ones((n_blocks, T))

        for b in range(n_blocks):
            act_b = self._raw_behav['action'][b]
            rew_b = self._raw_behav['reward'][b]
            if self.input_format == 'current':
                # Match original tinyRNN: current observation as input.
                action[b] = act_b
                stage2[b] = act_b  # stage2 == action in one-step task
                reward[b] = rew_b
            else:
                # Legacy shifted convention: previous observation as input.
                action[b, 1:] = act_b[:-1]
                stage2[b, 1:] = act_b[:-1]
                reward[b, 1:] = rew_b[:-1]
            trial_type[b] = self._raw_behav['trial_type'][b]

        # Store raw actions for target construction
        raw_action = np.zeros((n_blocks, T))
        for b in range(n_blocks):
            raw_action[b] = self._raw_behav['action'][b]

        # Input: [action, stage2, reward]
        self._inputs = torch.tensor(
            np.stack([action, stage2, reward], axis=-1), dtype=self.dtype
        )  # (B, T, 3)
        # Target: current trial's action
        self._targets = torch.tensor(raw_action, dtype=torch.long)  # (B, T)
        self._mask = torch.tensor(mask, dtype=self.dtype)        # (B, T)
        self._trial_type = torch.tensor(trial_type, dtype=torch.long)  # (B, T)

        self.n_blocks = n_blocks
        self.T = T

    @classmethod
    def load(cls, data_dir: str | None = None, animal_name: str = 'V',
             filter_block_type: str = 'both',
             block_truncation: tuple[int, int] = (10, 70),
             verbose: bool = False,
             dtype: torch.dtype = torch.float32,
             input_format: str = 'current',
             **kwargs) -> "BartoloMonkeyDataset":
        """Load the BartoloMonkey dataset.

        If data_dir is None or doesn't contain .mat files, automatically downloads
        from Mendeley and caches to ~/.cache/neuralrnn/datasets/BartoloMonkey/.

        Args:
            data_dir: Path to directory containing .mat files.
                If None, tries common locations, then auto-downloads.
            animal_name: 'V' or 'W'
            filter_block_type: 'both', 'what', or 'where'
            block_truncation: (start, end) trial indices
            verbose: Print loading info
            dtype: torch dtype for input and mask tensors (float32 or float64).
                The original tinyRNN code uses float64.
            input_format: 'current' (default, matches original tinyRNN) or
                'shifted' (legacy NeuralRNN convention).

        Returns:
            BartoloMonkeyDataset instance
        """
        MENDELEY_DATASET_ID = "p7ft2bvphx"
        FILES_API = f"https://data.mendeley.com/api/datasets/{MENDELEY_DATASET_ID}/files"

        if data_dir is None:
            # Try common locations
            candidates = [
                Path(__file__).parent.parent.parent.parent / "data" / "BartoloMonkey",
                Path.home() / ".cache" / "neuralrnn" / "datasets" / "BartoloMonkey",
                # Reference project location
                Path(__file__).parent.parent.parent.parent.parent /
                    "reference_project" / "Tiny_rnn" / "tinyRNN" / "datasets" / "BartoloMonkey",
            ]
            for candidate in candidates:
                if candidate.exists() and any(candidate.glob("*.mat")):
                    data_dir = candidate
                    if verbose:
                        print(f"Found BartoloMonkey data at: {data_dir}")
                    break

        # Auto-download if not found
        if data_dir is None or not Path(data_dir).exists() or not any(Path(data_dir).glob("*.mat")):
            cache_dir = Path.home() / ".cache" / "neuralrnn" / "datasets" / "BartoloMonkey"
            cache_dir.mkdir(parents=True, exist_ok=True)

            if not any(cache_dir.glob("*.mat")):
                print("BartoloMonkey data not found. Downloading from Mendeley...")
                cls._download_from_mendeley(FILES_API, cache_dir, verbose)

            data_dir = cache_dir

        return cls(data_dir, animal_name, filter_block_type,
                   block_truncation, verbose, dtype=dtype, input_format=input_format)

    @staticmethod
    def _download_from_mendeley(files_api: str, cache_dir: Path, verbose: bool = False):
        """Download .mat files from Mendeley dataset API."""
        import json
        import urllib.request

        # Get file list from Mendeley API
        req = urllib.request.Request(files_api, headers={"User-Agent": "NeuralRNN/1.0"})
        try:
            with urllib.request.urlopen(req) as resp:
                files = json.loads(resp.read())
        except Exception as e:
            raise RuntimeError(
                f"Failed to query Mendeley API: {e}\n"
                f"Please manually download from https://data.mendeley.com/datasets/p7ft2bvphx "
                f"and extract .mat files to {cache_dir}"
            ) from e

        # Download each .mat file
        mat_files = [f for f in files if f["filename"].endswith(".mat")]
        if not mat_files:
            raise RuntimeError(
                f"No .mat files found in Mendeley dataset. "
                f"Please manually download from https://data.mendeley.com/datasets/p7ft2bvphx "
                f"and extract .mat files to {cache_dir}"
            )

        for f in mat_files:
            filename = f["filename"]
            download_url = f.get("content_details", {}).get("download_url")
            if not download_url:
                raise RuntimeError(
                    f"No download URL for {filename}. "
                    f"Please manually download from https://data.mendeley.com/datasets/p7ft2bvphx"
                )

            dest = cache_dir / filename
            if dest.exists():
                if verbose:
                    print(f"  Already exists: {filename}")
                continue

            if verbose:
                print(f"  Downloading {filename}...")

            req_file = urllib.request.Request(download_url, headers={"User-Agent": "NeuralRNN/1.0"})
            try:
                with urllib.request.urlopen(req_file) as resp, open(dest, "wb") as f_out:
                    import shutil
                    shutil.copyfileobj(resp, f_out)
            except Exception as e:
                if dest.exists():
                    dest.unlink()
                raise RuntimeError(
                    f"Failed to download {filename}: {e}\n"
                    f"Please manually download from https://data.mendeley.com/datasets/p7ft2bvphx"
                ) from e

        print(f"Downloaded {len(mat_files)} .mat files to {cache_dir}")

    def sample_batch(self, n_blocks: int | None = None) -> dict[str, torch.Tensor]:
        """Sample a random batch of blocks.

        Args:
            n_blocks: Number of blocks to sample. If None, returns all blocks.

        Returns:
            Dict with 'inputs', 'targets', 'mask' tensors (batch-first).
        """
        if n_blocks is None:
            return {
                'inputs': self._inputs,
                'targets': self._targets,
                'mask': self._mask,
            }

        idx = torch.randint(0, self.n_blocks, (n_blocks,))
        return {
            'inputs': self._inputs[idx],
            'targets': self._targets[idx],
            'mask': self._mask[idx],
        }

    def get_all_data(self) -> dict[str, torch.Tensor]:
        """Return all data as a single batch."""
        return {
            'inputs': self._inputs,
            'targets': self._targets,
            'mask': self._mask,
            'trial_type': self._trial_type,
        }

    def __len__(self) -> int:
        return self.n_blocks

    def __getitem__(self, idx):
        return {
            'inputs': self._inputs[idx],
            'targets': self._targets[idx],
            'mask': self._mask[idx],
        }
