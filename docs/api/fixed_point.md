# Fixed-Point Finder Hyperparameters

This guide explains how the parameters of the fixed-point finders in `neuralrnn.analysis.fixed_points` affect the recovered fixed points: their **number**, their **positions** in state space, and their **stability labels**. It is written primarily for the ring-attractor analyses in the multitask notebooks, but the principles apply to any model.

---

## 1. Optimizer-related parameters

### 1.1 `backend`

| Backend | Optimizer | Objective | Typical use |
|---|---|---|---|
| `"numeric"` | PyTorch Adam | `‖F(z) − z‖²` | Default. Good general-purpose search; tends to pull candidates toward the nearest point on a ring. |
| `"original"` | Adaptive LR gradient descent | `0.5 · ‖F(z) − z‖²` | Reimplementation of Golub & Sussillo (2018). Large initial steps help spread candidates along a ring, but the default `tol_q=1e-9` is very strict. |

**Effects**
- **Number**: `"numeric"` usually returns more points because `speed_tol` is loose. `"original"` with strict `tol_q` often returns only one best candidate per search.
- **Position**: `"numeric"` points are densely distributed along the attractor. `"original"` points are more accurate when they converge, but may collapse if convergence is too strict.
- **Stability**: The physical fixed points are the same; only the candidate pool differs.

**Tip**: Start with `"numeric"`. Switch to `"original"` only after the ring is recovered, and remember to relax `tol_q`.

---

### 1.2 `n_candidates`

Number of initial states sampled for each search.

- **Number**: More candidates directly increase point density, with diminishing returns after deduplication.
- **Position**: More candidates cover more angles around a ring, reducing the chance of missing a direction.
- **Stability**: Does not change stability labels, but a larger pool may include saddle/unstable points.

**Cost**: Linear in runtime and GPU memory (batch size grows).

**Tip**: Use `200` as baseline; raise to `400–800` for difficult periods such as `go1`.

---

### 1.3 `n_iters`

Number of optimizer iterations.

- **Number**: More iterations let more candidates converge, raising the pass rate.
- **Position**: Too few iterations leaves candidates near but not on the attractor; they may be filtered by `q_thresh`.
- **Stability**: Better-converged points yield more reliable Jacobian eigenvalues.

**Tip**: `5000` is usually enough. Increase to `10000–20000` when using `"original"` with strict `tol_q`.

---

### 1.4 `lr` (numeric backend only)

Adam learning rate.

- **Number**: Too small → slow convergence, many candidates fail `speed_tol`. Too large → oscillation around saddles.
- **Position**: Too small leaves candidates far from fixed points; too large may overshoot the true ring points.
- **Stability**: Convergence quality affects eigenvalue reliability.

**Tip**: Default `1e-3`; try `5e-4` or `2e-3` if needed.

---

### 1.5 `speed_tol` (numeric backend only)

Internal speed threshold used by `NumericFixedPointFinder` to accept candidates.

- **Number**: Larger values keep more points; smaller values keep only well-converged points.
- **Position**: Loose values include "slow points" near the ring that are not exact fixed points.
- **Stability**: Does not change the stability rule, but poor points have unreliable eigenvalues.

**Tip**: Default `0.5` is very permissive. Lower to `0.1` or `0.05` only if you want cleaner points.

---

### 1.6 `initial_rate` / `decrease_factor` / `tol_q` (original backend only)

#### `initial_rate`

- **Number**: Larger initial steps help candidates spread along the ring tangent, reducing collapse.
- **Position**: Large steps explore the marginal direction of a ring attractor.
- **Stability**: No direct effect, but better exploration can locate more saddle/unstable points.

**Tip**: Default `1.0`, matching the reference implementation. Try `0.5` or `2.0` for fine control.

#### `decrease_factor`

Factor by which the learning rate is reduced when the objective increases.

- **Number**: Smaller values (e.g. `0.9`) decay faster and stabilize sooner but may get stuck. Larger values (e.g. `0.99`) explore more but may oscillate.
- **Position**: Affects whether the optimizer settles on a fixed point.
- **Stability**: Indirect through convergence quality.

**Tip**: Default `0.95`.

#### `tol_q`

Convergence threshold on `q = 0.5 · ‖F(z) − z‖²`.

- **Number**: Very strict (`1e-9`) keeps almost nothing; relaxed (`1e-3`–`1e-2`) keeps many ring points.
- **Position**: Strict values keep only the most accurate fixed points; loose values keep slow points.
- **Stability**: Strict values produce more reliable Jacobian eigenvalues.

**Important**: `tol_q = 1e-9` corresponds to `‖F(z) − z‖ ≈ 1.4e-4`, which is hard to reach for softplus CTRNNs. This is the main reason the `"original"` backend returns few points by default.

**Tip**: When using `"original"` for ring attractors, start with `tol_q = 1e-3` and let `q_thresh` do the final filtering.

---

## 2. Initialization / noise parameters

### 2.1 `noise_scale`

Standard deviation of Gaussian noise added to trajectory-based initial states.

- **Number**: Moderately larger noise spreads candidates around the ring and can increase point count; too much noise pushes candidates out of the basin and reduces count.
- **Position**: Controls the initial scatter in state space. Ring attractors are marginally stable along the ring, so small noise is enough to cover different angles.
- **Stability**: Excessive noise may move candidates into other basins (e.g. the central fixed point), changing the stable/unstable ratio.

**Tip**: Default `0.05`. Reduce to `0.02` if points collapse; increase to `0.1` if points are too concentrated.

---

## 3. Post-filtering / deduplication parameters

### 3.1 `q_thresh`

Post-search speed filter. Points with `speed >= q_thresh` are discarded.

- **Number**: Larger values keep more points; smaller values keep only the best-converged points.
- **Position**: Filters out slow points that have not fully settled on the attractor.
- **Stability**: Indirect: poorly converged points have unreliable eigenvalues.

**Important**: This was one of the causes of ring collapse in early versions of the 12b notebook. `test_fig1_cdef.py` uses `1e-2`; values such as `1e-3` filter away many ring points.

**Tip**: Start with `1e-2`; lower only if you also increase `n_iters` or reduce `noise_scale` to ensure convergence.

---

### 3.2 `stability_thresh`

Threshold on `max |eig(J)|` used to label a point as stable.

- **Number**: Does not change the number of points, only the stable/unstable label ratio.
- **Position**: No effect.
- **Stability**: Larger values (e.g. `1.1`) label marginally stable ring points as stable. Smaller values (e.g. `0.99`) label them as unstable/saddle.

**Physical meaning**: In discrete-time systems, strict stability requires `max|eig| < 1`. A ring attractor has one eigenvalue very close to 1 along the ring direction, so `1.0` marks ring points as marginal, `1.05`/`1.1` as stable, and `0.99` as unstable.

**Tip**: Use `1.05` or `1.1` to display ring points as stable. Use `0.99` to emphasize their marginal/saddle nature.

---

### 3.3 `dedup_tol`

L2 distance threshold for merging duplicate fixed points.

- **Number**: Larger values merge more points, yielding fewer final points. Smaller values preserve nearby points.
- **Position**: Merges points whose coordinates are closer than the threshold.
- **Stability**: No effect.

**Important**: If the ring is densely populated, a large `dedup_tol` can merge adjacent angles and make the ring look discontinuous.

**Tip**: Default `5e-3` for 256-dimensional states. Lower to `1e-3` for denser rings; raise to `1e-2` to remove noise points.

---

## 4. Recommended tuning roadmap for ring attractors

Start from the defaults that match `test_fig1_cdef.py`:

```python
FP_BACKEND = "numeric"
N_CANDIDATES_PER_ANGLE = 200
N_ITERS = 5000
LR = 1e-3
SPEED_TOL = 5e-1
NOISE_SCALE = 0.05
Q_THRESH = 1e-2
STABILITY_THRESH = 1.1
DEDUP_TOL = 5e-3
```

If a period (usually `go1`) shows too few ring points:

1. Increase `n_candidates` to `400–800`.
2. If still sparse, reduce `noise_scale` to `0.02`.
3. If still sparse, increase `n_iters` to `10000`.
4. If points are too noisy, raise `q_thresh` or lower `dedup_tol`.

If you want to try the `"original"` backend:

```python
FP_PARAMS = {
    "go1": {
        "backend": "original",
        "tol_q": 1e-3,        # relax strict convergence
        "n_candidates": 400,
    }
}
```

To study stability labels, fix all other parameters and vary only `stability_thresh` (`0.99`, `1.05`, `1.1`) and observe how ring points are classified.

---

## 5. Expected behavior per period

| Period | Expected structure | Most sensitive parameters |
|---|---|---|
| `fix1` | Central stable point + large-radius ring | `noise_scale`, `dedup_tol` |
| `stim1` | One fixed point per stimulus direction, output near zero | `q_thresh`, `noise_scale` |
| `delay1` | Central stable point + medium-radius ring | `n_candidates`, `noise_scale` |
| `go1` | Central stable point + small-radius ring with output-potent component | `n_candidates`, `q_thresh`, `noise_scale` |

`go1` is usually the hardest period because the response input compresses the ring and the ring points have a component along the readout direction, making candidates more likely to collapse onto the central fixed point.
