"""Checkerboard perceptual decision-making task (Kleinman et al., 2025, eLife).

A 15x15 red/green checkerboard is presented and the subject reports the
majority color by reaching to the matching target. Two targets (left/right)
are shown each trial with randomized colors, so the *color* decision and the
*direction* (left/right reach) decision are statistically independent. The
target configuration (which side shows which color) is a task variable that is
irrelevant for the final report once the mapping is known.

Inputs (4 channels):
    0: left target color  (-1 = red, +1 = green), on during the targets epoch
    1: right target color (-1 = red, +1 = green), on during the targets epoch
    2: red signed coherence   (R-G)/(R+G) in [-1, 1], on during decision
    3: green signed coherence (G-R)/(R+G), on during decision
       (channels 2+3 sum to zero in the noise-free case; each receives
        independent Gaussian noise N(0, sigma_in^2) during decision)

Outputs (2 channels): decision variables [left reach, right reach]. The target
is 0 during center-hold / targets / stimulus-off epochs and 1 on the correct
DV during the decision epoch. The loss mask excludes the first
``loss_ramp_ms`` of the decision epoch (evidence-integration ramp).

Trial epochs (milliseconds; mode="train" randomizes, mode="eval" fixes them):
    center hold ~ N(200, 50^2) -> targets ~ U[600, 1000] -> decision 1500
    -> stimulus off 100. Train-mode trials are zero-padded to the batch max.

Catch trials (fraction ``catch_fraction``): half receive no input at all,
half receive only the targets (no coherence); target output is 0 throughout.

Reference:
    Kleinman, M., Wang, T., Xiao, D., ... Chandrasekaran, C., Kao, J.C., 2025.
    The information bottleneck as a principle underlying multi-area cortical
    representations during decision-making. eLife. DOI: 10.7554/eLife.89369.2
"""
import numpy as np
import torch

from .task_base import Task


CHANNEL_ORDER = ["left_target_color", "right_target_color", "red_coh", "green_coh"]
OUTPUT_ORDER = ["left_dv", "right_dv"]

# 14 signed coherences, symmetric from 90% green to 90% red (paper Methods)
DEFAULT_COHS = np.linspace(-0.9, 0.9, 14)

# Fixed eval-mode epoch durations (ms): center hold, targets, decision, off
EVAL_EPOCHS_MS = (200.0, 800.0, 1500.0, 100.0)

_EPOCH_NAMES = ("center_hold", "targets", "decision", "off")


def _sample_epochs_ms(rng, mode):
    """Return (center_hold, targets, decision, off) durations in ms."""
    if mode == "eval":
        return EVAL_EPOCHS_MS
    center_hold = max(rng.normal(200.0, 50.0), 50.0)
    targets = rng.uniform(600.0, 1000.0)
    return center_hold, targets, 1500.0, 100.0


class CheckerboardTask(Task):
    """Checkerboard majority-color decision task (unified Task interface)."""

    name = "checkerboard"
    input_dim = 4
    output_dim = 2
    default_dt = 10.0
    deprecated_kwargs = {"input_noise_std": "sigma_in"}

    def __init__(self, n_trials=64, *, dt=10.0, mode="train", cohs=None,
                 catch_fraction=0.1, sigma_in=0.1, loss_ramp_ms=200.0,
                 balanced=False, seed=None):
        self.n_trials = n_trials
        self.dt = dt
        self.mode = mode
        self.cohs = cohs
        self.catch_fraction = catch_fraction
        self.sigma_in = sigma_in  # noise on the two coherence channels, decision epoch only
        self.loss_ramp_ms = loss_ramp_ms
        self.balanced = balanced
        self.seed = seed

    def generate_trials(self):
        """Create trials for the checkerboard task.

        Returns:
            inputs: (N, T, 4) float tensor.
            targets: (N, T, 2) float tensor — full-length target sequence.
            mask: (N, T, 2) float tensor — 0 during the decision ramp and padding.
            conditions: list of dicts with keys ``coherence`` (red signed coh),
                ``left_color`` (-1 red / +1 green), ``correct_choice`` (0 left /
                1 right / -1 for catch), ``catch`` (bool, deprecated alias of
                ``is_catch``), ``epoch_bounds`` ((ch_end, tg_end, dec_end, T),
                deprecated tuple form of ``epochs``), plus the unified keys
                ``epochs`` / ``n_steps`` / ``is_catch``.
        """
        n_trials = self.n_trials
        dt = self.dt
        rng = np.random.default_rng(self.seed)
        cohs = DEFAULT_COHS if self.cohs is None else np.asarray(self.cohs, dtype=float)

        # Condition list: (coherence, left_color). left_color=-1 -> red target on left.
        conditions_grid = [(c, lc) for c in cohs for lc in (-1.0, 1.0)]
        n_cond = len(conditions_grid)
        if self.balanced:
            reps = int(np.ceil(n_trials / n_cond))
            cond_indices = np.tile(np.arange(n_cond), reps)[:n_trials]
            rng.shuffle(cond_indices)
        else:
            cond_indices = rng.integers(0, n_cond, size=n_trials)

        n_catch = int(round(self.catch_fraction * n_trials))
        catch_idx = set(rng.choice(n_trials, size=n_catch, replace=False).tolist()) if n_catch else set()

        trials = []
        for t in range(n_trials):
            coh, left_color = conditions_grid[cond_indices[t]]
            is_catch = t in catch_idx
            epochs_ms = _sample_epochs_ms(rng, self.mode)
            bounds = np.cumsum(np.round(np.asarray(epochs_ms) / dt).astype(int))
            T = int(bounds[-1])
            ch_end, tg_end, dec_end, _ = bounds

            inp = np.zeros((T, 4), dtype=np.float32)
            tgt = np.zeros((T, 2), dtype=np.float32)
            msk = np.ones((T, 2), dtype=np.float32)

            if is_catch:
                # Half of catch trials see the targets only, half see nothing.
                if rng.random() < 0.5:
                    inp[ch_end:tg_end, 0] = left_color
                    inp[ch_end:tg_end, 1] = -left_color
                correct_choice = -1
            else:
                inp[ch_end:tg_end, 0] = left_color
                inp[ch_end:tg_end, 1] = -left_color
                noise = rng.normal(0.0, self.sigma_in, size=(dec_end - tg_end, 2)).astype(np.float32)
                inp[tg_end:dec_end, 2] = coh + noise[:, 0]
                inp[tg_end:dec_end, 3] = -coh + noise[:, 1]
                # Majority color determines the correct reach direction
                # (0 = left, 1 = right).
                red_majority = coh > 0
                if red_majority:
                    correct_choice = 0 if left_color < 0 else 1
                else:
                    correct_choice = 0 if left_color > 0 else 1
                tgt[tg_end:dec_end, correct_choice] = 1.0

            # Exclude the evidence-integration ramp from the loss.
            ramp_steps = int(round(self.loss_ramp_ms / dt))
            msk[tg_end:tg_end + ramp_steps, :] = 0.0

            bounds_int = (int(ch_end), int(tg_end), int(dec_end), T)
            epoch_bounds = [0, *bounds_int]  # cumulative bounds incl. 0
            epochs = {
                name: (int(epoch_bounds[k]), int(epoch_bounds[k + 1]))
                for k, name in enumerate(_EPOCH_NAMES)
            }
            trials.append((inp, tgt, msk, self._cond(
                epochs, T, is_catch,
                coherence=float(coh),
                left_color=float(left_color),
                correct_choice=int(correct_choice),
                catch=bool(is_catch),          # deprecated alias of is_catch
                epoch_bounds=bounds_int,       # deprecated tuple form of epochs
            )))

        # Zero-pad train-mode trials (variable lengths) to the batch maximum.
        T_max = max(tr[3]["epoch_bounds"][3] for tr in trials)
        inputs = np.zeros((n_trials, T_max, 4), dtype=np.float32)
        targets = np.zeros((n_trials, T_max, 2), dtype=np.float32)
        mask = np.zeros((n_trials, T_max, 2), dtype=np.float32)
        conditions = []
        for t, (inp, tgt, msk, cond) in enumerate(trials):
            T = cond["epoch_bounds"][3]
            inputs[t, :T] = inp
            targets[t, :T] = tgt
            mask[t, :T] = msk
            conditions.append(cond)

        return (
            torch.from_numpy(inputs),
            torch.from_numpy(targets),
            torch.from_numpy(mask),
            conditions,
        )


def generate_trials(**kwargs):
    """Backward-compatible shim: CheckerboardTask(**kwargs).generate_trials().

    The pre-refactor ``input_noise_std`` parameter is now ``sigma_in``; it is
    still accepted with a DeprecationWarning.
    """
    return CheckerboardTask.from_kwargs(**kwargs).generate_trials()
