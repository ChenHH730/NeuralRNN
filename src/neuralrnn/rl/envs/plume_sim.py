"""Plume simulation data layer for the Singh-2023 plume-tracking environment.

Pure-numpy port of the reference puff/wind simulators
(``reference_project/reinforcement_learning/Singh-2023-plumetracknets/code/sim_utils.py``
and the loading logic of ``sim_analysis.load_plume``).

Model: odor puffs are born at the source (0, 0) with Poisson rate ``birth_rate``
per simulation step (dt = 0.01 s), initial radius 0.01 m growing at
rdot = 0.01 m/s, advected by the wind field plus per-puff Gaussian y-diffusion
(``N(0, wind_y_var) * dt``). Puffs leaving the arena (x > 10, x < -2,
|y| > 10) are dropped. Puff states are snapshotted every ``snapshot_every``
sim steps (default 4 -> 25 Hz, exactly the reference's env_dt=0.04 downsample
which keeps ``tidx % 4 == 0``).

Dataset naming follows the reference: ``{regime}x{X}b{B}`` with wind magnitude
X/10 m/s and birth rate 0.2*B per sim step, e.g. ``constantx5b5`` = constant
wind 0.5 m/s, birth rate 1.0. Regimes: ``constant``, ``switchNN`` (one wind
direction change of NN degrees at the midpoint), ``noisyN`` (direction redrawn
from N(0, 30) every ~100*N steps, clipped to +/-60 deg).

Cached on disk as ``puff_data_{name}.npz`` / ``wind_data_{name}.npz`` +
``{name}_params.json`` (reference filenames, npz instead of pickle); the cache
is regenerated when the parameters change.

Deviations from the reference (statistically equivalent, documented):
  * random streams use dedicated ``np.random.Generator`` instances instead of
    the global ``np.random`` state;
  * floats are stored as float32 (reference: float16);
  * snapshots are taken at 25 Hz during generation instead of storing 100 Hz
    and downsampling at load (identical kept subset for env_dt = 0.04).
"""
from __future__ import annotations

import hashlib
import json
import os
import re

import numpy as np

# Canonical datasets used in the paper (and evaluatable by the notebooks).
PLUME_DATASETS: dict[str, dict] = {
    "constantx5b5": dict(regime="constant", wind_magnitude=0.5, birth_rate=1.0),
    "noisy3x5b5": dict(regime="noisy3", wind_magnitude=0.5, birth_rate=1.0),
    "switch15x5b5": dict(regime="switch15", wind_magnitude=0.5, birth_rate=1.0),
    "switch30x5b5": dict(regime="switch30", wind_magnitude=0.5, birth_rate=1.0),
    "switch45x5b5": dict(regime="switch45", wind_magnitude=0.5, birth_rate=1.0),
    "noisy6x5b5": dict(regime="noisy6", wind_magnitude=0.5, birth_rate=1.0),
}

SIM_DT = 0.01           # s (100 Hz simulation step)
MIN_RADIUS = 0.01       # m, puff radius at birth
RDOT = 0.01             # m/s, puff radius growth rate
WIND_Y_VAR = 0.5        # per-puff y-diffusion std (= wind_magnitude / sqrt(1))
DURATION = 120.0        # s
SNAPSHOT_EVERY = 4      # sim steps between snapshots (25 Hz == env_dt 0.04)
ENV_DT = SIM_DT * SNAPSHOT_EVERY

# Arena bounds for trimming puffs (reference manual_integrator).
_X_MAX, _X_MIN, _Y_MAX, _Y_MIN = 10.0, -2.0, 10.0, -10.0


def parse_dataset_name(name: str) -> dict:
    """Parse ``{regime}x{X}b{B}`` into simulation parameters."""
    m = re.fullmatch(r"(constant|noisy\d|switch\d+)x(\d+)b(\d+)", name)
    if m is None:
        raise ValueError(f"Unrecognized plume dataset name {name!r}; "
                         f"expected e.g. 'constantx5b5', 'noisy3x5b5', 'switch45x5b5'")
    regime, x, b = m.group(1), int(m.group(2)), int(m.group(3))
    return dict(regime=regime, wind_magnitude=x / 10.0, birth_rate=0.2 * b)


def generate_wind(regime: str, duration: float = DURATION, dt: float = SIM_DT,
                  wind_magnitude: float = 0.5, seed: int = 0):
    """Wind vector time series. Port of ``get_wind_vectors_flexible``.

    Returns (times, wind_x, wind_y) float64 arrays of length duration/dt.
    """
    rng = np.random.default_rng(seed)  # reference: np.random.RandomState(0)
    T = np.arange(0, duration, dt)
    wind_degrees = np.zeros(len(T))

    if "switch" in regime:
        how_much = int(regime.replace("switch", ""))
        wind_degrees[len(T) // 2:] += how_much

    if "noisy" in regime:
        repN = 100 * int(regime.replace("noisy", ""))
        degz = 60  # +/- clip
        noise = np.zeros(len(T))
        switch_idxs = np.arange(len(T), step=repN, dtype=int)
        switch_idxs = [s + rng.choice(np.arange(-repN // 10, repN // 10))
                       for s in switch_idxs]
        switch_idxs = np.sort(switch_idxs)
        for idx in switch_idxs:
            noise[idx:] = rng.normal(0, degz / 2)
        noise = np.clip(noise, -degz, degz)
        wind_degrees += noise

    wind_x = np.cos(wind_degrees * np.pi / 180.0) * wind_magnitude
    wind_y = np.sin(wind_degrees * np.pi / 180.0) * wind_magnitude
    return T, wind_x, wind_y


def generate_puffs(wind_x: np.ndarray, wind_y: np.ndarray,
                   birth_rate: float = 1.0, wind_y_var: float = WIND_Y_VAR,
                   dt: float = SIM_DT, seed: int = 137,
                   snapshot_every: int = SNAPSHOT_EVERY,
                   verbose: bool = False) -> dict:
    """Vectorized Euler puff simulator. Port of ``get_puffs_df_vector``.

    Returns CSR-style arrays over 25 Hz snapshots:
        snap_tidx (n_snaps,) int32: simulation step index of each snapshot
        offsets (n_snaps + 1,) int64: row range of each snapshot
        x, y, radius (n_rows,) float32, puff_number (n_rows,) int32
    """
    rng = np.random.default_rng(seed)  # reference: global np.random, seed 137
    n_steps = len(wind_x) - 1  # reference: int((t_max - t_min) * 100)

    # Initial condition: one puff at the source (reference gen_puff_dict tidx=0).
    px = np.array([0.0])
    py = np.array([0.0])
    pr = np.array([MIN_RADIUS])
    pid = np.array([0], dtype=np.int64)
    next_pid = 1

    snap_tidx, offsets = [], [0]
    xs, ys, rs, pids = [], [], [], []
    for i in range(n_steps):
        # Advect + diffuse + grow (reference manual_integrator).
        px = px + wind_x[i] * dt
        py = py + wind_y[i] * dt + rng.normal(0.0, wind_y_var, size=px.shape[0]) * dt
        pr = pr + dt * RDOT
        keep = (pr > 0) & (px < _X_MAX) & (px > _X_MIN) & (py < _Y_MAX) & (py > _Y_MIN)
        px, py, pr, pid = px[keep], py[keep], pr[keep], pid[keep]
        # Births at the source (after advection, matching the reference order).
        n_births = int(rng.poisson(birth_rate))
        if n_births > 0:
            px = np.concatenate([px, np.zeros(n_births)])
            py = np.concatenate([py, np.zeros(n_births)])
            pr = np.concatenate([pr, np.full(n_births, MIN_RADIUS)])
            pid = np.concatenate([pid, np.arange(next_pid, next_pid + n_births,
                                                 dtype=np.int64)])
            next_pid += n_births
        if i % snapshot_every == 0:
            snap_tidx.append(i)
            xs.append(px.astype(np.float32))
            ys.append(py.astype(np.float32))
            rs.append(pr.astype(np.float32))
            pids.append(pid.astype(np.int32))
            offsets.append(offsets[-1] + len(px))
        if verbose and (i + 1) % 2000 == 0:
            print(f"[plume_sim] step {i + 1}/{n_steps}, live puffs {len(px)}")

    return {
        "snap_tidx": np.asarray(snap_tidx, dtype=np.int32),
        "offsets": np.asarray(offsets, dtype=np.int64),
        "x": np.concatenate(xs), "y": np.concatenate(ys),
        "radius": np.concatenate(rs), "puff_number": np.concatenate(pids),
    }


def generate_centerline(wind_x: np.ndarray, wind_y: np.ndarray,
                        dt: float = SIM_DT,
                        snapshot_every: int = SNAPSHOT_EVERY) -> dict:
    """Deterministic plume centerline (port of ``centerline_cli.py``).

    One puff per sim step is born at the source and advected by the wind
    only (no crosswind diffusion, no radius growth); the trail of puffs at
    each snapshot traces the plume centerline. Per-point tangent angles are
    computed as in the reference: rolling(8)-mean of x and y along the
    (snapshot, birth-order)-sorted points, finite-differenced, then
    ``angle = arctan2(dy, dx)`` (NaN for the first 8 points of each
    snapshot, matching the reference's rolling warm-up).

    Returns CSR-style arrays over 25 Hz snapshots:
        snap_tidx (n_snaps,) int32, offsets (n_snaps+1,) int64,
        x, y, angle (n_rows,) float32 (angle in radians, NaN-padded head).
    """
    n_steps = len(wind_x) - 1
    px = np.zeros(1)
    py = np.zeros(1)

    snap_tidx, offsets = [], [0]
    xs, ys, angs = [], [], []
    for i in range(n_steps):
        px = px + wind_x[i] * dt
        py = py + wind_y[i] * dt
        keep = (px < _X_MAX) & (px > _X_MIN) & (py < _Y_MAX) & (py > _Y_MIN)
        px, py = px[keep], py[keep]
        # Exactly one birth per sim step (reference grow_puffs_centerline).
        px = np.concatenate([px, [0.0]])
        py = np.concatenate([py, [0.0]])
        if i % snapshot_every == 0:
            snap_tidx.append(i)
            # Ordered oldest-first (birth order) == reference sort by
            # (tidx, puff_number).
            n = len(px)
            if n >= 2:
                k = min(8, n)
                kern = np.ones(k) / k
                xr = np.convolve(px, kern, mode="valid")  # n-k+1
                yr = np.convolve(py, kern, mode="valid")
                dx = np.diff(xr)
                dy = np.diff(yr)
                # Reference (centerline_cli.py): angle = arctan(dy/dx) -- a
                # line orientation in [-90, 90] deg (0 = horizontal/downwind
                # for constant wind), NOT a directed angle. Rows are ordered
                # oldest-first (downwind -> source), so the forward diff
                # points upwind; arctan(dy/dx) folds that back to the line
                # orientation (constant wind -> 0, switch45 -> +45 deg).
                ang = np.full(n, np.nan)
                # rolling(8).mean().diff() assigns slope to the later row:
                # valid slopes live at rows k..n-1 (0-indexed).
                with np.errstate(divide="ignore", invalid="ignore"):
                    ang[k:] = np.arctan(dy / dx)
            else:
                ang = np.full(n, np.nan)
            xs.append(px.astype(np.float32))
            ys.append(py.astype(np.float32))
            angs.append(ang.astype(np.float32))
            offsets.append(offsets[-1] + n)

    return {
        "snap_tidx": np.asarray(snap_tidx, dtype=np.int32),
        "offsets": np.asarray(offsets, dtype=np.int64),
        "x": np.concatenate(xs), "y": np.concatenate(ys),
        "angle": np.concatenate(angs),
    }


class PlumeData:
    """Loaded (possibly time-truncated) plume + wind data, 25 Hz snapshots.

    Attributes:
        times (n_steps,) float64, wind_x/wind_y (n_steps,) float64,
        tidxs (n_steps,) int32 simulation step indices,
        offsets (n_steps+1,) int64 CSR offsets into the puff arrays,
        puff_x/puff_y/puff_r (n_rows,) float32, puff_pid (n_rows,) int32,
        cl_offsets (n_steps+1,) int64 CSR offsets into the centerline arrays,
        cl_x/cl_y/cl_angle (n_cl_rows,) float32 centerline points and their
        downwind tangent angles (radians; NaN at each snapshot's head).
    """

    def __init__(self, times, wind_x, wind_y, tidxs, offsets,
                 puff_x, puff_y, puff_r, puff_pid,
                 cl_offsets=None, cl_x=None, cl_y=None, cl_angle=None):
        self.times = times
        self.wind_x = wind_x
        self.wind_y = wind_y
        self.tidxs = tidxs
        self.offsets = offsets
        self.puff_x = puff_x
        self.puff_y = puff_y
        self.puff_r = puff_r
        self.puff_pid = puff_pid
        self.cl_offsets = cl_offsets
        self.cl_x = cl_x
        self.cl_y = cl_y
        self.cl_angle = cl_angle

    @property
    def n_steps(self) -> int:
        return len(self.times)

    def puffs_at(self, k: int):
        """(x, y, radius, puff_id) arrays at downsampled step k."""
        s, e = self.offsets[k], self.offsets[k + 1]
        return (self.puff_x[s:e], self.puff_y[s:e],
                self.puff_r[s:e], self.puff_pid[s:e])

    def wind_at(self, k: int):
        return (float(self.wind_x[k]), float(self.wind_y[k]))

    @property
    def has_centerline(self) -> bool:
        return self.cl_offsets is not None

    def centerline_at(self, k: int):
        """(x, y, angle) centerline arrays at downsampled step k."""
        if self.cl_offsets is None:
            raise ValueError("no centerline data (regenerate the cache)")
        s, e = self.cl_offsets[k], self.cl_offsets[k + 1]
        return self.cl_x[s:e], self.cl_y[s:e], self.cl_angle[s:e]

    def centerline_angle_at(self, k: int, x: float,
                            half_width: float = 0.02) -> float:
        """Local centerline tangent angle (radians): median angle of
        centerline points with |x - point.x| <= ``half_width`` at step ``k``
        (reference ``subset_centerline_angle``). NaN if no points."""
        cx, cy, ca = self.centerline_at(k)
        m = np.abs(cx - x) <= half_width
        if not m.any() or np.isnan(ca[m]).all():
            return float("nan")
        return float(np.nanmedian(ca[m]))


def _params_fingerprint(params: dict) -> str:
    blob = json.dumps(params, sort_keys=True).encode()
    return hashlib.md5(blob).hexdigest()[:12]


def load_or_generate_plume(name: str, data_dir: str, *,
                           t_val_min: float | None = None,
                           t_val_max: float | None = None,
                           env_dt: float = ENV_DT,
                           wind_seed: int = 0, puff_seed: int = 137,
                           verbose: bool = False) -> PlumeData:
    """Load a cached plume simulation, generating it on first use.

    Args:
        name: dataset name (see PLUME_DATASETS / parse_dataset_name).
        data_dir: cache directory.
        t_val_min/t_val_max: truncate to this simulation-time window (the
            reference's load_plume truncation).
        env_dt: environment timestep; only 0.04 s is supported (snapshots are
            generated at 25 Hz).
        wind_seed/puff_seed: simulation seeds (reference defaults: 0 / 137).
    """
    if abs(env_dt - ENV_DT) > 1e-9:
        raise ValueError(f"Only env_dt={ENV_DT} is supported (25 Hz snapshots)")
    base = PLUME_DATASETS.get(name, parse_dataset_name(name))
    params = dict(base, duration=DURATION, dt=SIM_DT, wind_y_var=WIND_Y_VAR,
                  snapshot_every=SNAPSHOT_EVERY,
                  wind_seed=wind_seed, puff_seed=puff_seed,
                  cl_version=2)  # bump when centerline conventions change

    os.makedirs(data_dir, exist_ok=True)
    puff_path = os.path.join(data_dir, f"puff_data_{name}.npz")
    wind_path = os.path.join(data_dir, f"wind_data_{name}.npz")
    cl_path = os.path.join(data_dir, f"centerline_data_{name}.npz")
    meta_path = os.path.join(data_dir, f"{name}_params.json")

    cache_ok = (os.path.exists(puff_path) and os.path.exists(wind_path)
                and os.path.exists(cl_path) and os.path.exists(meta_path))
    if cache_ok:
        with open(meta_path) as f:
            cache_ok = json.load(f).get("fingerprint") == _params_fingerprint(params)
    if not cache_ok:
        if verbose:
            print(f"[plume_sim] generating {name}: {params}")
        T, wind_x, wind_y = generate_wind(
            params["regime"], params["duration"], params["dt"],
            params["wind_magnitude"], seed=params["wind_seed"])
        puffs = generate_puffs(wind_x, wind_y, params["birth_rate"],
                               params["wind_y_var"], params["dt"],
                               seed=params["puff_seed"],
                               snapshot_every=params["snapshot_every"],
                               verbose=verbose)
        centerline = generate_centerline(
            wind_x, wind_y, params["dt"], params["snapshot_every"])
        np.savez(wind_path, time=T, wind_x=wind_x, wind_y=wind_y)
        np.savez(puff_path, **puffs)
        np.savez(cl_path, **centerline)
        with open(meta_path, "w") as f:
            json.dump({**params, "fingerprint": _params_fingerprint(params)},
                      f, indent=2)

    wind = np.load(wind_path)
    puffs = np.load(puff_path)
    centerline = np.load(cl_path)

    # Truncate by simulation time (reference load_plume: time >= t_val_min,
    # time <= t_val_max), keeping the tidx % 4 == 0 downsample (already the
    # snapshot grid).
    times = wind["time"]
    snap_times = times[puffs["snap_tidx"]]
    keep_wind = np.ones(len(times), dtype=bool)
    keep_snap = np.ones(len(snap_times), dtype=bool)
    if t_val_min is not None:
        keep_wind &= times >= t_val_min
        keep_snap &= snap_times >= t_val_min
    if t_val_max is not None:
        keep_wind &= times <= t_val_max
        keep_snap &= snap_times <= t_val_max

    snap_idx = np.flatnonzero(keep_snap)
    offsets, rows = [0], []
    cl_offsets, cl_rows = [0], []
    for k in snap_idx:
        s, e = puffs["offsets"][k], puffs["offsets"][k + 1]
        rows.append((s, e))
        offsets.append(offsets[-1] + (e - s))
        cs, ce = centerline["offsets"][k], centerline["offsets"][k + 1]
        cl_rows.append((cs, ce))
        cl_offsets.append(cl_offsets[-1] + (ce - cs))
    row_idx = np.concatenate([np.arange(s, e) for s, e in rows]) if rows \
        else np.zeros(0, dtype=np.int64)
    cl_row_idx = np.concatenate([np.arange(s, e) for s, e in cl_rows]) \
        if cl_rows else np.zeros(0, dtype=np.int64)

    # Wind at the snapshot times only (reference keeps wind on the same
    # downsampled tidx grid).
    wind_snap = puffs["snap_tidx"][snap_idx]
    return PlumeData(
        times=snap_times[snap_idx].copy(),
        wind_x=wind["wind_x"][wind_snap].copy(),
        wind_y=wind["wind_y"][wind_snap].copy(),
        tidxs=puffs["snap_tidx"][snap_idx].copy(),
        offsets=np.asarray(offsets, dtype=np.int64),
        puff_x=puffs["x"][row_idx], puff_y=puffs["y"][row_idx],
        puff_r=puffs["radius"][row_idx], puff_pid=puffs["puff_number"][row_idx],
        cl_offsets=np.asarray(cl_offsets, dtype=np.int64),
        cl_x=centerline["x"][cl_row_idx], cl_y=centerline["y"][cl_row_idx],
        cl_angle=centerline["angle"][cl_row_idx],
    )
