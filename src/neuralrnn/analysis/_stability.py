"""Discrete-map spectral classification shared by analysis entry points."""
import numpy as np


def spectral_stability(eigenvalues, tolerance=1e-6):
    """Return (label, stable, n_unstable, n_neutral, spectral_radius).

    The closed band [1-tolerance, 1+tolerance] is near-neutral. A direction
    strictly outside its upper edge establishes instability even when another
    direction is in the band. This is local linear evidence, not a nonlinear
    stability proof for nonhyperbolic or nondifferentiable points.
    """
    if not np.isfinite(tolerance) or tolerance < 0 or tolerance >= 1:
        raise ValueError('stability_tol must be finite and in [0, 1)')
    if eigenvalues is None:
        return 'unknown', None, 0, 0, None
    mods = np.abs(np.asarray(eigenvalues))
    if not mods.size or not np.isfinite(mods).all():
        return 'unknown', None, 0, 0, None
    unstable = int(np.sum(mods > 1 + tolerance))
    neutral = int(np.sum((mods >= 1 - tolerance) & (mods <= 1 + tolerance)))
    label = 'unstable' if unstable else ('near_neutral' if neutral else 'stable')
    stable = False if unstable else (None if neutral else True)
    return label, stable, unstable, neutral, float(mods.max())
