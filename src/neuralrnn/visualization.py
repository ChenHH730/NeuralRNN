"""Visualization utilities: trajectories / fixed points / vector fields / weight matrices / line attractors / animations / psychometrics.

Design principles:
  - All functions accept **data** (numpy arrays, dataclasses) and do not depend on a model
  - All functions accept an optional ax argument for composing multi-panel figures
  - All functions return (fig, ax) for further customization
  - 3D functions support elev/azim for camera-angle control
  - Multi-trajectory plots allow per-trajectory colors

Corresponds to the plotting functionality in trainRNNbrain's PerformanceAnalyzer + DynamicSystemAnalyzer.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Default color scheme from trainRNNbrain
DEFAULT_COLORS = ['cornflowerblue', 'mediumblue', 'tomato', 'firebrick']


def _ensure_ax(ax, projection=None, figsize=(8, 6)):
    if ax is not None:
        return ax.figure, ax
    fig = plt.figure(figsize=figsize)
    if projection == "3d":
        ax = fig.add_subplot(111, projection="3d")
    else:
        ax = fig.add_subplot(111)
    return fig, ax


# =========================================================================
# Trajectory visualization
# =========================================================================
def plot_trajectories_2d(
    trajectories: list[np.ndarray],
    pca_result=None,
    *,
    colors: list[str] | None = None,
    labels: list[str] | None = None,
    ax=None,
    alpha: float = 0.5,
    linewidth: float = 0.5,
    show_legend: bool = True,
    show_reference_lines: bool = True,
):
    """Project multiple trajectories onto a 2D PCA plane and plot them.

    Args:
        trajectories: list of (T_i, M) arrays; each trajectory may have a different length
        pca_result: PCAResult object; if None, fit PCA on all trajectories automatically
        colors: color for each trajectory; if None, cycle through DEFAULT_COLORS
        labels: label for each trajectory
        ax: matplotlib Axes; created if None
        alpha: line transparency
        linewidth: line width
        show_legend: whether to show the legend
        show_reference_lines: whether to draw x=0 and y=0 reference lines

    Returns:
        (fig, ax)
    """
    from .analysis.dimensionality import fit_pca

    if pca_result is None:
        all_states = np.concatenate(trajectories, axis=0)
        pca_result = fit_pca(all_states, n_components=2)

    fig, ax = _ensure_ax(ax)
    n = len(trajectories)
    colors = colors or [DEFAULT_COLORS[i % len(DEFAULT_COLORS)] for i in range(n)]

    for i, traj in enumerate(trajectories):
        proj = pca_result.transform(traj)
        lbl = labels[i] if labels else None
        ax.plot(proj[:, 0], proj[:, 1], color=colors[i], alpha=alpha,
                linewidth=linewidth, label=lbl)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    if show_reference_lines:
        ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
        ax.axvline(0, color='gray', linewidth=0.5, linestyle='--')
    if show_legend and labels:
        ax.legend()
    return fig, ax


def plot_trajectories_3d(
    trajectories: list[np.ndarray],
    pca_result=None,
    *,
    colors: list[str] | None = None,
    labels: list[str] | None = None,
    ax=None,
    alpha: float = 0.5,
    linewidth: float = 0.5,
    elev: float = 25,
    azim: float = 45,
):
    """Project multiple trajectories onto a 3D PCA space and plot them.

    Args:
        trajectories: list of (T_i, M) arrays
        pca_result: PCAResult object; if None, fit automatically
        colors: color for each trajectory
        labels: label for each trajectory
        ax: matplotlib 3D Axes; created if None
        alpha: line transparency
        linewidth: line width
        elev, azim: camera elevation and azimuth

    Returns:
        (fig, ax)
    """
    from .analysis.dimensionality import fit_pca

    if pca_result is None:
        all_states = np.concatenate(trajectories, axis=0)
        pca_result = fit_pca(all_states, n_components=3)

    fig, ax = _ensure_ax(ax, projection="3d")
    n = len(trajectories)
    colors = colors or [DEFAULT_COLORS[i % len(DEFAULT_COLORS)] for i in range(n)]

    for i, traj in enumerate(trajectories):
        proj = pca_result.transform(traj)
        lbl = labels[i] if labels else None
        ax.plot(proj[:, 0], proj[:, 1], proj[:, 2],
                color=colors[i], alpha=alpha, linewidth=linewidth, label=lbl)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.view_init(elev=elev, azim=azim)
    if labels:
        ax.legend()
    return fig, ax


# =========================================================================
# Fixed-point visualization
# =========================================================================
def plot_fixed_points(
    fixed_points,
    pca_result=None,
    *,
    ax=None,
    colors: dict | None = None,
    markers: dict | None = None,
    size: float = 80,
):
    """Plot fixed points on a 2D PCA plane.

    Args:
        fixed_points: FixedPointSet object or list of FixedPoint
        pca_result: PCAResult object
        ax: matplotlib Axes
        colors: override for {"stable": "blue", "unstable": "red", "saddle": "orange"}
        markers: override for {"stable": "o", "unstable": "x", "saddle": "^"}
        size: marker size

    Returns:
        (fig, ax)
    """
    from .analysis.linearization import classify_fixed_point
    from .analysis.linearization import LinearizationResult

    _colors = {"stable": "#2196F3", "unstable": "#F44336", "saddle": "#FF9800",
               "unknown": "#9E9E9E"}
    _markers = {"stable": "o", "unstable": "X", "saddle": "^", "unknown": "s"}
    if colors:
        _colors.update(colors)
    if markers:
        _markers.update(markers)

    fig, ax = _ensure_ax(ax)
    points = fixed_points.points if hasattr(fixed_points, 'points') else fixed_points

    for fp in points:
        z = fp.z
        if pca_result is not None:
            z = pca_result.transform(z.reshape(1, -1)).ravel()

        # Classification
        if fp.is_stable is None:
            cat = "unknown"
        elif fp.is_stable:
            cat = "stable"
        else:
            cat = "unstable"

        ax.scatter(z[0], z[1], c=_colors[cat], marker=_markers[cat],
                   s=size, zorder=5, edgecolors='black', linewidths=0.5)

    # Legend
    for cat in ["stable", "unstable", "saddle", "unknown"]:
        ax.scatter([], [], c=_colors[cat], marker=_markers[cat], s=size,
                   label=cat, edgecolors='black', linewidths=0.5)
    ax.legend(fontsize=8)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    return fig, ax


def plot_fixed_points_3d(
    fixed_points,
    pca_result=None,
    *,
    ax=None,
    colors: dict | None = None,
    size: float = 80,
    elev: float = 25,
    azim: float = 45,
):
    """Plot fixed points in a 3D PCA space.

    Args:
        fixed_points: FixedPointSet object
        pca_result: PCAResult object (must have n_components >= 3)
        ax: matplotlib 3D Axes
        colors: color override
        size: marker size
        elev, azim: camera angles

    Returns:
        (fig, ax)
    """
    _colors = {"stable": "#2196F3", "unstable": "#F44336", "saddle": "#FF9800",
               "unknown": "#9E9E9E"}
    if colors:
        _colors.update(colors)

    fig, ax = _ensure_ax(ax, projection="3d")
    points = fixed_points.points if hasattr(fixed_points, 'points') else fixed_points

    for fp in points:
        z = fp.z
        if pca_result is not None:
            z = pca_result.transform(z.reshape(1, -1)).ravel()
        if len(z) < 3:
            continue

        cat = "stable" if fp.is_stable else ("unstable" if fp.is_stable is not None else "unknown")
        ax.scatter(z[0], z[1], z[2], c=_colors[cat], marker='o',
                   s=size, zorder=5, edgecolors='black', linewidths=0.5)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.view_init(elev=elev, azim=azim)
    return fig, ax


# =========================================================================
# Vector-field visualization
# =========================================================================
def plot_vector_field(
    vector_field,
    *,
    ax=None,
    color: str = 'gray',
    scale: float | None = None,
    width: float = 0.003,
    alpha: float = 0.7,
    speed_colormap: str = 'YlOrRd',
    show_speed: bool = False,
):
    """Plot a vector field on a 2D plane (quiver plot).

    Args:
        vector_field: VectorField data object (grid_pc, velocity_pc, speed)
        ax: matplotlib Axes
        color: arrow color (when show_speed=False)
        scale: quiver scale parameter; None for auto
        width: arrow width
        alpha: transparency
        speed_colormap: colormap used when show_speed=True
        show_speed: whether to encode speed magnitude with color

    Returns:
        (fig, ax)
    """
    fig, ax = _ensure_ax(ax)
    vf = vector_field

    # vf.grid_pc is (n_grid, n_grid, 2)
    X = vf.grid_pc[:, :, 0]
    Y = vf.grid_pc[:, :, 1]
    U = vf.velocity_pc[:, :, 0]
    V = vf.velocity_pc[:, :, 1]

    if show_speed:
        speed = vf.speed
        ax.quiver(X, Y, U, V, speed, cmap=speed_colormap,
                  scale=scale, width=width, alpha=alpha)
    else:
        ax.quiver(X, Y, U, V, color=color, scale=scale, width=width, alpha=alpha)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    return fig, ax


# =========================================================================
# Combined phase portrait (trajectory + fixed points + vector field)
# =========================================================================
def plot_phase_portrait(
    trajectories: list[np.ndarray] | None = None,
    fixed_points=None,
    vector_field=None,
    pca_result=None,
    *,
    colors: list[str] | None = None,
    labels: list[str] | None = None,
    title: str | None = None,
    figsize: tuple = (8, 6),
):
    """Combined phase portrait: overlay trajectory, fixed points, and vector field on one 2D PCA plane.

    One-stop analysis visualization, similar to trainRNNbrain's plot_fixed_points with overlaid trajectories.

    Returns:
        (fig, ax)
    """
    from .analysis.dimensionality import fit_pca

    # Auto-fit PCA
    if pca_result is None:
        all_parts = []
        if trajectories:
            all_parts.extend(trajectories)
        if fixed_points:
            pts = fixed_points.points if hasattr(fixed_points, 'points') else fixed_points
            all_parts.extend([fp.z.reshape(1, -1) for fp in pts])
        if all_parts:
            pca_result = fit_pca(np.concatenate(all_parts, axis=0), n_components=2)

    fig, ax = plt.subplots(figsize=figsize)

    # Vector field (bottom layer)
    if vector_field is not None:
        plot_vector_field(vector_field, ax=ax, alpha=0.4)

    # Trajectories
    if trajectories:
        plot_trajectories_2d(trajectories, pca_result, ax=ax,
                             colors=colors, labels=labels, alpha=0.6)

    # Fixed points (top layer)
    if fixed_points is not None:
        plot_fixed_points(fixed_points, pca_result, ax=ax)

    if title:
        ax.set_title(title)
    return fig, ax


# =========================================================================
# Weight-matrix visualization
# =========================================================================
def plot_weight_matrix(
    W: np.ndarray,
    *,
    title: str | None = None,
    ax=None,
    cmap: str = 'RdBu_r',
    center_zero: bool = True,
    colorbar: bool = True,
    xlabel: str | None = None,
    ylabel: str | None = None,
):
    """Plot a weight matrix as a heatmap.

    Args:
        W: (N, M) weight matrix
        title: plot title
        ax: matplotlib Axes
        cmap: colormap
        center_zero: whether to center the color scale at 0 (diverging scale)
        colorbar: whether to show the colorbar
        xlabel, ylabel: axis labels

    Returns:
        (fig, ax)
    """
    fig, ax = _ensure_ax(ax)

    vmin = np.min(W) if not center_zero else -np.max(np.abs(W))
    vmax = np.max(W) if not center_zero else np.max(np.abs(W))

    im = ax.imshow(W, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax, interpolation='nearest')
    if colorbar:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    return fig, ax


def plot_connectivity(
    W_inp: np.ndarray,
    W_rec: np.ndarray,
    W_out: np.ndarray,
    *,
    dale_mask: np.ndarray | None = None,
    sort: bool = True,
    figsize: tuple = (15, 4),
    titles: tuple = ("W_inp", "W_rec", "W_out"),
):
    """Plot input / recurrent / output weight matrices side by side as heatmaps.

    If dale_mask is given and sort=True, neurons are sorted by E/I identity.

    Returns:
        (fig, axes)  axes is (ax1, ax2, ax3)
    """
    if sort and dale_mask is not None:
        # Ensure dale_mask is 1D numpy array
        dale_mask = np.asarray(dale_mask).ravel()
        order = np.argsort(-dale_mask)  # E neurons first
        W_rec = W_rec[np.ix_(order, order)]
        W_inp = W_inp[order]
        # W_out may have fewer columns than neurons (e.g. E-only readout)
        if W_out.shape[1] == len(order):
            W_out = W_out[:, order]
        elif W_out.shape[1] == (dale_mask > 0).sum():
            # E-only readout: sort by E-neuron order only
            e_order = np.argsort(-np.arange(len(order))[dale_mask > 0])
            W_out = W_out[:, e_order]

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    plot_weight_matrix(W_inp.T, title=titles[0], ax=axes[0],
                       xlabel="Neuron", ylabel="Input dim")
    plot_weight_matrix(W_rec, title=titles[1], ax=axes[1],
                       xlabel="Neuron (from)", ylabel="Neuron (to)")
    plot_weight_matrix(W_out, title=titles[2], ax=axes[2],
                       xlabel="Neuron", ylabel="Output dim")
    fig.tight_layout()
    return fig, axes


# =========================================================================
# Cluster-averaged responses
# =========================================================================
def plot_averaged_responses(
    responses: np.ndarray,
    dale_mask: np.ndarray | None = None,
    *,
    ax=None,
    cmap: str = 'RdBu_r',
    labels: list[str] | None = None,
    xlabel: str = "Time step",
    ylabel: str = "Cluster",
):
    """Plot a heatmap of firing-rate trajectories averaged by cluster.

    Args:
        responses: (n_clusters, T) array
        dale_mask: (n_clusters,) optional E/I marker for sorting
        labels: cluster labels

    Returns:
        (fig, ax)
    """
    if dale_mask is not None:
        order = np.argsort(-dale_mask)
        responses = responses[order]
        if labels:
            labels = [labels[i] for i in order]

    fig, ax = _ensure_ax(ax)
    vmax = np.max(np.abs(responses))
    im = ax.imshow(responses, cmap=cmap, aspect='auto', vmin=-vmax, vmax=vmax,
                   interpolation='nearest')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if labels:
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
    return fig, ax


# =========================================================================
# Line-attractor visualization
# =========================================================================
def plot_line_attractor(
    la_result,
    pca_result=None,
    *,
    ax=None,
    show_rhs: bool = True,
    color: str = '#2196F3',
    linewidth: float = 2.0,
    marker_size: float = 30,
):
    """Plot a line attractor on a 2D PCA plane.

    Args:
        la_result: LineAttractorResult object
        pca_result: PCAResult object
        ax: matplotlib Axes
        show_rhs: whether to color points by ||RHS|| (speed)
        color: line color (when show_rhs=False)
        linewidth: line width
        marker_size: sampled-point size

    Returns:
        (fig, ax)
    """
    fig, ax = _ensure_ax(ax)

    coords = la_result.coords
    if pca_result is not None:
        coords = pca_result.transform(coords)

    if show_rhs:
        speeds = la_result.speeds
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=speeds, cmap='YlOrRd',
                        s=marker_size, zorder=5, edgecolors='black', linewidths=0.5)
        fig.colorbar(sc, ax=ax, label="||RHS||")
    else:
        ax.plot(coords[:, 0], coords[:, 1], color=color, linewidth=linewidth, zorder=4)
        ax.scatter(coords[:, 0], coords[:, 1], c=color, s=marker_size,
                   zorder=5, edgecolors='black', linewidths=0.5)

    # Endpoint markers
    if la_result.endpoints is not None:
        ep = la_result.endpoints
        for e, lbl in zip(ep, ["Left", "Right"]):
            ep_proj = pca_result.transform(e.reshape(1, -1)).ravel() if pca_result else e
            ax.scatter(ep_proj[0], ep_proj[1], c='black', marker='*', s=150,
                       zorder=6, label=lbl)
        ax.legend()

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    return fig, ax


def plot_line_attractor_3d(
    la_result,
    projection_axes: np.ndarray,
    *,
    ax=None,
    trajectories: list[np.ndarray] | None = None,
    traj_colors: list[str] | None = None,
    elev: float = 25,
    azim: float = 45,
    color: str = '#2196F3',
    linewidth: float = 2.0,
):
    """Plot a line attractor in a 3D subspace.

    projection_axes: (3, M) axes defining the 3D subspace (e.g. choice/context/sensory).
    Optionally overlay projected trajectories.

    Returns:
        (fig, ax)
    """
    fig, ax = _ensure_ax(ax, projection="3d")

    coords = la_result.coords  # (n_points, M)
    proj = coords @ projection_axes.T  # (n_points, 3)

    ax.plot(proj[:, 0], proj[:, 1], proj[:, 2],
            color=color, linewidth=linewidth, zorder=4)

    # Endpoints
    if la_result.endpoints is not None:
        for e, lbl in zip(la_result.endpoints, ["Left", "Right"]):
            e_proj = e @ projection_axes.T
            ax.scatter(e_proj[0], e_proj[1], e_proj[2], c='black', marker='*',
                       s=150, zorder=6, label=lbl)
        ax.legend()

    # Overlay trajectories
    if trajectories:
        n = len(trajectories)
        traj_colors = traj_colors or [DEFAULT_COLORS[i % len(DEFAULT_COLORS)] for i in range(n)]
        for i, traj in enumerate(trajectories):
            t_proj = traj @ projection_axes.T
            ax.plot(t_proj[:, 0], t_proj[:, 1], t_proj[:, 2],
                    color=traj_colors[i], alpha=0.3, linewidth=0.5)

    ax.set_xlabel("Axis 1")
    ax.set_ylabel("Axis 2")
    ax.set_zlabel("Axis 3")
    ax.view_init(elev=elev, azim=azim)
    return fig, ax


# =========================================================================
# Animations
# =========================================================================
def animate_trajectories_3d(
    trajectories: list[np.ndarray],
    pca_result=None,
    *,
    colors: list[str] | None = None,
    labels: list[str] | None = None,
    fps: int = 30,
    duration: float = 10.0,
    elev: float = 25,
    step: int = 2,
    figsize: tuple = (8, 8),
):
    """Create a rotating 3D trajectory animation.

    The camera rotates around the z-axis to show 3D PCA-projected trajectories.

    Args:
        trajectories: list of (T_i, M) arrays
        pca_result: PCAResult object (must have n_components >= 3)
        colors: color for each trajectory
        labels: label for each trajectory
        fps: frames per second
        duration: animation duration in seconds
        elev: fixed camera elevation
        step: rotation step per frame (frame skipping for performance)
        figsize: figure size

    Returns:
        matplotlib.animation.FuncAnimation
    """
    from .analysis.dimensionality import fit_pca

    if pca_result is None:
        all_states = np.concatenate(trajectories, axis=0)
        pca_result = fit_pca(all_states, n_components=3)

    n = len(trajectories)
    colors = colors or [DEFAULT_COLORS[i % len(DEFAULT_COLORS)] for i in range(n)]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    # Pre-compute projections
    projected = [pca_result.transform(traj) for traj in trajectories]

    # Set axis limits
    all_proj = np.concatenate(projected, axis=0)
    mins = all_proj.min(axis=0)
    maxs = all_proj.max(axis=0)
    margin = (maxs - mins) * 0.1

    lines = []
    for i, proj in enumerate(projected):
        lbl = labels[i] if labels else None
        line, = ax.plot([], [], [], color=colors[i], alpha=0.6, linewidth=0.8, label=lbl)
        lines.append(line)

    if labels:
        ax.legend()

    def init():
        ax.set_xlim(mins[0] - margin[0], maxs[0] + margin[0])
        ax.set_ylim(mins[1] - margin[1], maxs[1] + margin[1])
        ax.set_zlim(mins[2] - margin[2], maxs[2] + margin[2])
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        return lines

    def update(frame):
        azim = (frame * step) % 360
        ax.view_init(elev=elev, azim=azim)
        for i, proj in enumerate(projected):
            lines[i].set_data(proj[:, 0], proj[:, 1])
            lines[i].set_3d_properties(proj[:, 2])
        return lines

    n_frames = int(fps * duration)
    anim = animation.FuncAnimation(fig, update, init_func=init,
                                    frames=n_frames, interval=1000 // fps, blit=False)
    return anim


# =========================================================================
# Progressive trajectory animation (reveal evolution frame by frame)
# =========================================================================
def animate_trajectory_progression(
    trajectories: list[np.ndarray],
    pca_result=None,
    *,
    colors: list[str] | None = None,
    labels: list[str] | None = None,
    projection: str = "3d",
    fps: int = 30,
    n_frames: int | None = None,
    elev: float = 25,
    azim: float = 45,
    linewidth: float = 1.0,
    alpha: float = 0.8,
    figsize: tuple = (8, 6),
    trail_alpha: float = 0.15,
):
    """Progressively reveal trajectory evolution frame by frame.

    Unlike animate_trajectories_3d (which only rotates the camera), this function draws the
    trajectory one timestep at a time, showing how the RNN state evolves from the initial
    condition to the final state.

    Args:
        trajectories: list of (T_i, M) arrays; each trajectory may have a different length
        pca_result: PCAResult object; if None, fit automatically (2D or 3D)
        colors: color for each trajectory
        labels: label for each trajectory
        projection: "2d" or "3d"
        fps: frames per second
        n_frames: total number of frames; if None, use the longest trajectory length
        elev, azim: 3D camera angles (3D mode only)
        linewidth: trajectory line width
        alpha: trajectory transparency
        figsize: figure size
        trail_alpha: fade transparency for already-drawn portions

    Returns:
        matplotlib.animation.FuncAnimation
    """
    from .analysis.dimensionality import fit_pca

    n_traj = len(trajectories)
    max_len = max(len(t) for t in trajectories)
    if n_frames is None:
        n_frames = max_len

    # PCA projection
    n_comp = 3 if projection == "3d" else 2
    if pca_result is None:
        all_states = np.concatenate(trajectories, axis=0)
        pca_result = fit_pca(all_states, n_components=n_comp)

    projected = [pca_result.transform(traj) for traj in trajectories]
    colors = colors or [DEFAULT_COLORS[i % len(DEFAULT_COLORS)] for i in range(n_traj)]
    labels = labels or [None] * n_traj

    # Set up figure
    if projection == "3d":
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig, ax = plt.subplots(figsize=figsize)

    # Compute axis limits
    all_proj = np.concatenate(projected, axis=0)
    mins = all_proj.min(axis=0)
    maxs = all_proj.max(axis=0)
    margin = (maxs - mins) * 0.1

    # Initialize line objects: one "trail" (faded) + one "head" (bright) per trajectory
    trail_lines = []
    head_lines = []
    for i in range(n_traj):
        if projection == "3d":
            trail, = ax.plot([], [], [], color=colors[i], alpha=trail_alpha, linewidth=linewidth * 0.5)
            head, = ax.plot([], [], [], color=colors[i], alpha=alpha, linewidth=linewidth, label=labels[i])
        else:
            trail, = ax.plot([], [], color=colors[i], alpha=trail_alpha, linewidth=linewidth * 0.5)
            head, = ax.plot([], [], color=colors[i], alpha=alpha, linewidth=linewidth, label=labels[i])
        trail_lines.append(trail)
        head_lines.append(head)

    if any(labels):
        ax.legend(fontsize=8)

    def init():
        if projection == "3d":
            ax.set_xlim(mins[0] - margin[0], maxs[0] + margin[0])
            ax.set_ylim(mins[1] - margin[1], maxs[1] + margin[1])
            ax.set_zlim(mins[2] - margin[2], maxs[2] + margin[2])
            ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
            ax.view_init(elev=elev, azim=azim)
        else:
            ax.set_xlim(mins[0] - margin[0], maxs[0] + margin[0])
            ax.set_ylim(mins[1] - margin[1], maxs[1] + margin[1])
            ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        return trail_lines + head_lines

    def update(frame):
        # Each frame reveals one more timestep
        t = int(frame * max_len / n_frames)
        t = max(1, min(t, max_len))

        for i, proj in enumerate(projected):
            traj_len = len(proj)
            # How many points to show for this trajectory at this frame
            show_t = min(t, traj_len)

            if projection == "3d":
                # Trail: all points up to show_t
                trail_lines[i].set_data(proj[:show_t, 0], proj[:show_t, 1])
                trail_lines[i].set_3d_properties(proj[:show_t, 2])
                # Head: last few points (bright)
                head_start = max(0, show_t - max(1, traj_len // 20))
                head_lines[i].set_data(proj[head_start:show_t, 0], proj[head_start:show_t, 1])
                head_lines[i].set_3d_properties(proj[head_start:show_t, 2])
            else:
                trail_lines[i].set_data(proj[:show_t, 0], proj[:show_t, 1])
                head_start = max(0, show_t - max(1, traj_len // 20))
                head_lines[i].set_data(proj[head_start:show_t, 0], proj[head_start:show_t, 1])

        return trail_lines + head_lines

    anim = animation.FuncAnimation(fig, update, init_func=init,
                                    frames=n_frames, interval=1000 // fps, blit=False)
    return anim


# =========================================================================
# Psychometric curves
# =========================================================================
def plot_psychometric_curves(
    curves,
    *,
    ax=None,
    colors: list[str] | None = None,
    show_legend: bool = True,
    markers: list[str] | None = None,
):
    """Plot psychometric curves (coherence vs P(right)).

    Args:
        curves: list of PsychometricCurve data objects
        ax: matplotlib Axes
        colors: color for each curve
        show_legend: whether to show the legend
        markers: marker style for each curve

    Returns:
        (fig, ax)
    """
    fig, ax = _ensure_ax(ax)
    n = len(curves)
    colors = colors or [DEFAULT_COLORS[i % len(DEFAULT_COLORS)] for i in range(n)]
    markers = markers or ['o', 's', '^', 'D']

    for i, curve in enumerate(curves):
        m = markers[i % len(markers)]
        ax.plot(curve.coherences, curve.prob_right, marker=m, color=colors[i],
                label=curve.label, markersize=6, linewidth=1.5)

    ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.5)
    ax.set_xlabel("Coherence")
    ax.set_ylabel("P(right)")
    ax.set_ylim(-0.05, 1.05)
    if show_legend:
        ax.legend()
    return fig, ax


# =========================================================================
# Quick look (trial-by-trial predictions vs targets)
# =========================================================================
def plot_trial_predictions(
    predictions: np.ndarray,
    targets: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    n_trials: int = 6,
    figsize: tuple = (12, 8),
    colors: tuple = ('#2196F3', '#F44336'),
):
    """Plot predictions vs target outputs for multiple trials.

    For classification tasks (neurogym), targets are usually integer class labels (B, T),
    and predictions are logits (B, T, n_classes). This function auto-handles dimension mismatches:
    - If targets is 1D (T,) or 2D (B, T) and predictions is 3D (B, T, O),
      targets are expanded to one-hot (B, T, O) for per-dimension comparison.

    Args:
        predictions: (B, T, O) or (B, T) predicted outputs. If 2D, automatically expanded to (B, T, 1).
        targets: (B, T, O), (B, T), or (B,) target outputs.
                 If dimensions do not match predictions, automatic adaptation is attempted:
                 - (B, T) integer labels + predictions (B, T, O) -> one-hot expansion
                 - other cases are expanded to (B, T, 1)
        mask: (B, T) or (B, T, O) valid-region mask
        n_trials: number of trials to display
        figsize: figure size
        colors: (prediction, target) colors

    Returns:
        (fig, axes)
    """
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)

    # Ensure predictions is 3D
    if predictions.ndim == 2:
        predictions = predictions[:, :, np.newaxis]

    B_pred, T, O = predictions.shape

    # Handle targets dimension mismatch
    if targets.ndim == 1:
        # (T,) -> (1, T) -> will be expanded below
        targets = targets[np.newaxis, :]

    if targets.ndim == 2:
        targets_2d = targets  # (B, T) -- integer class labels
        if predictions.ndim == 3 and predictions.shape[2] > 1:
            # Classification: convert integer labels to one-hot
            B_tgt, T_tgt = targets_2d.shape
            # Ensure batch/time dims match
            B_use = min(B_pred, B_tgt)
            T_use = min(T, T_tgt)
            one_hot = np.zeros((B_use, T_use, O), dtype=targets.dtype)
            for b in range(B_use):
                for t in range(T_use):
                    cls = int(targets_2d[b, t])
                    if 0 <= cls < O:
                        one_hot[b, t, cls] = 1
            targets = one_hot
            predictions = predictions[:B_use, :T_use, :]
        else:
            targets = targets_2d[:, :, np.newaxis]
    # Now targets is 3D: (B, T, O_out)
    # Ensure last dim matches predictions
    if targets.shape[-1] != predictions.shape[-1]:
        # Pad or truncate targets to match O
        B_t, T_t, O_t = targets.shape
        B_p, T_p, O_p = predictions.shape
        B_use = min(B_t, B_p)
        T_use = min(T_t, T_p)
        targets_aligned = np.zeros((B_use, T_use, O_p), dtype=targets.dtype)
        for b in range(B_use):
            for t in range(T_use):
                cls = int(targets[b, t, 0]) if targets.ndim == 3 else int(targets[b, t])
                if 0 <= cls < O_p:
                    targets_aligned[b, t, cls] = 1
        targets = targets_aligned
        predictions = predictions[:B_use, :T_use, :]

    B, T, O = predictions.shape
    n_trials = min(n_trials, B)
    n_cols = min(3, n_trials)
    n_rows = (n_trials + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)

    for i in range(n_trials):
        r, c = divmod(i, n_cols)
        ax = axes[r][c]
        for o in range(O):
            ax.plot(targets[i, :, o], color=colors[1], linewidth=1, alpha=0.7,
                    label="Target" if o == 0 else None)
            ax.plot(predictions[i, :, o], color=colors[0], linewidth=1, alpha=0.7,
                    label="Prediction" if o == 0 else None)
        if mask is not None:
            if mask.ndim == 2:
                valid = mask[i]
            else:
                valid = mask[i, :, 0]
            ax.fill_between(range(T), 0, 1, where=valid > 0,
                            alpha=0.1, transform=ax.get_xaxis_transform())
        ax.set_title(f"Trial {i}")
        if i == 0:
            ax.legend(fontsize=8)

    # Hide unused subplots
    for i in range(n_trials, n_rows * n_cols):
        r, c = divmod(i, n_cols)
        axes[r][c].set_visible(False)

    fig.tight_layout()
    return fig, axes


# =========================================================================
# Task trial visualization
# =========================================================================
_EPOCH_COLORS = ("#f4c7c3", "#c7dbf4", "#d6ecc7", "#f4ecc7", "#e3c7f4", "#c7f4ec")

_SKIP_COND_KEYS = {"epochs", "n_steps", "is_catch"}


def _cond_title(cond: dict) -> str:
    """Compact one-line summary of a trial condition dict."""
    parts = []
    for k, v in cond.items():
        if k in _SKIP_COND_KEYS or isinstance(v, (dict, list, tuple)):
            continue
        if isinstance(v, float):
            parts.append(f"{k}={v:.3g}")
        else:
            parts.append(f"{k}={v}")
    if cond.get("is_catch"):
        parts.append("CATCH")
    return ", ".join(parts)


def plot_trials(data, n: int | None = None, dt: float | None = None, *,
                show_epochs: bool = False, show_legend: bool = True,
                figsize: tuple | None = None, title: str | None = None):
    """Plot example trials (inputs + targets per trial) with epoch shading.

    Accepts a ``Trials`` object (e.g. from ``dataset.sample_trials(n)``), a
    trial-aligned dataset (``inputs``/``targets``/``conditions`` attributes),
    or a dict with ``"inputs"``/``"targets"``/``"conditions"`` keys.

    Args:
        data: Trials / dataset / dict as described above.
        n: number of trials to show (default: min(4, available)).
        dt: time step in ms; when given, the x-axis is in seconds.
        show_epochs: shade epoch spans from ``conditions[i]["epochs"]``.
        show_legend: show channel legend on the first subplot.
        figsize: figure size (default: (10, 2.4 * n)).
        title: optional figure suptitle.

    Returns:
        (fig, axes)
    """
    if hasattr(data, "inputs") and hasattr(data, "conditions"):
        inputs, targets, conditions = data.inputs, data.targets, data.conditions
    elif isinstance(data, dict):
        inputs, targets, conditions = data["inputs"], data["targets"], data["conditions"]
    else:
        raise TypeError(
            "plot_trials expects a Trials object, a trial-aligned dataset, or a dict "
            "with 'inputs'/'targets'/'conditions' keys."
        )
    inputs = np.asarray(inputs, dtype=float)
    targets = np.asarray(targets, dtype=float)

    n_total = inputs.shape[0]
    n = min(n, n_total) if n is not None else min(4, n_total)
    if figsize is None:
        figsize = (10, max(2.4 * n, 2.4))
    fig, axes = plt.subplots(n, 1, figsize=figsize, sharex=True, sharey=True, squeeze=False)
    axes = axes[:, 0]

    for row in range(n):
        ax = axes[row]
        cond = conditions[row] if row < len(conditions) else {}
        n_steps = int(cond.get("n_steps", inputs.shape[1]))
        inp = inputs[row, :n_steps]
        tgt = targets[row, :n_steps]
        t = np.arange(n_steps) * (dt / 1000.0 if dt else 1.0)

        if show_epochs and isinstance(cond.get("epochs"), dict):
            for k, (phase, bounds) in enumerate(cond["epochs"].items()):
                if bounds is None or bounds[0] is None or bounds[1] is None:
                    continue
                x0 = bounds[0] * (dt / 1000.0 if dt else 1.0)
                x1 = bounds[1] * (dt / 1000.0 if dt else 1.0)
                ax.axvspan(x0, x1, color=_EPOCH_COLORS[k % len(_EPOCH_COLORS)],
                           alpha=0.35, lw=0)
                ax.text((x0 + x1) / 2, 1.02, phase, transform=ax.get_xaxis_transform(),
                        ha="center", va="bottom", fontsize=10, color="0.35")

        for ch in range(inp.shape[-1]):
            ax.plot(t, inp[:, ch], lw=0.9, label=f"in{ch}")
        if tgt.ndim == 1:
            ax.step(t, tgt, where="post", color="k", lw=1.2, ls="--", label="target")
        else:
            for ch in range(tgt.shape[-1]):
                ax.step(t, tgt[:, ch], where="post", lw=1.2, ls="--", label=f"target{ch}")

        ax.set_ylabel("value", fontsize=10)
        cond_title = _cond_title(cond)
        if cond_title:
            ax.set_title(f"trial {row}: {cond_title}", fontsize=10, loc="left")
        ax.tick_params(labelsize=10)
        if show_legend and row == 0:
            ax.legend(fontsize=10, ncol=4, loc="upper right")

    axes[-1].set_xlabel("time (s)" if dt else "time (steps)", fontsize=10)
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return fig, axes
