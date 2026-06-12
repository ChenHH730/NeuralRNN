"""可视化工具层：轨迹 / 不动点 / 向量场 / 权重矩阵 / 线吸引子 / 动画 / 心理测量。

设计原则：
  - 所有函数接受**数据**（numpy 数组、dataclass），不依赖模型
  - 所有函数接受可选 ax 参数用于组合多图
  - 所有函数返回 (fig, ax) 以便进一步自定义
  - 3D 函数支持 elev/azim 控制相机角度
  - 多轨迹支持逐轨迹指定颜色

对应 trainRNNbrain 的 PerformanceAnalyzer + DynamicSystemAnalyzer 中的绘图功能。
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# trainRNNbrain 默认配色
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
# 轨迹可视化
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
    """将多条轨迹投影到 2D PCA 平面并绘制。

    Args:
        trajectories: list of (T_i, M) 数组，每条轨迹可不同长度
        pca_result: PCAResult 对象；None 时自动对所有轨迹拟合 PCA
        colors: 每条轨迹的颜色；None 时用 DEFAULT_COLORS 循环
        labels: 每条轨迹的标签
        ax: matplotlib Axes；None 时新建
        alpha: 线条透明度
        linewidth: 线条宽度
        show_legend: 是否显示图例
        show_reference_lines: 是否显示 x=0, y=0 参考线

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
    """将多条轨迹投影到 3D PCA 空间并绘制。

    Args:
        trajectories: list of (T_i, M) 数组
        pca_result: PCAResult 对象；None 时自动拟合
        colors: 每条轨迹的颜色
        labels: 每条轨迹的标签
        ax: matplotlib 3D Axes；None 时新建
        alpha: 线条透明度
        linewidth: 线条宽度
        elev, azim: 相机仰角和方位角

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
# 不动点可视化
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
    """在 2D PCA 平面上绘制不动点。

    Args:
        fixed_points: FixedPointSet 对象或 list of FixedPoint
        pca_result: PCAResult 对象
        ax: matplotlib Axes
        colors: {"stable": "blue", "unstable": "red", "saddle": "orange"} 的覆盖
        markers: {"stable": "o", "unstable": "x", "saddle": "^"} 的覆盖
        size: 标记大小

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

        # 分类
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
    """在 3D PCA 空间中绘制不动点。

    Args:
        fixed_points: FixedPointSet 对象
        pca_result: PCAResult 对象（需 n_components >= 3）
        ax: matplotlib 3D Axes
        colors: 颜色覆盖
        size: 标记大小
        elev, azim: 相机角度

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
# 向量场可视化
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
    """在 2D 平面上绘制向量场（quiver plot）。

    Args:
        vector_field: VectorField 数据对象（grid_pc, velocity_pc, speed）
        ax: matplotlib Axes
        color: 箭头颜色（当 show_speed=False 时）
        scale: quiver scale 参数；None 自动
        width: 箭头宽度
        alpha: 透明度
        speed_colormap: show_speed=True 时的颜色映射
        show_speed: 是否用颜色编码速度范数

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
# 组合相图（轨迹 + 不动点 + 向量场）
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
    """组合相图：轨迹 + 不动点 + 向量场叠加在同一个 2D PCA 平面上。

    一站式分析可视化，类似 trainRNNbrain 的 plot_fixed_points 叠加轨迹。

    Returns:
        (fig, ax)
    """
    from .analysis.dimensionality import fit_pca

    # 自动 PCA
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

    # 向量场（最底层）
    if vector_field is not None:
        plot_vector_field(vector_field, ax=ax, alpha=0.4)

    # 轨迹
    if trajectories:
        plot_trajectories_2d(trajectories, pca_result, ax=ax,
                             colors=colors, labels=labels, alpha=0.6)

    # 不动点（最上层）
    if fixed_points is not None:
        plot_fixed_points(fixed_points, pca_result, ax=ax)

    if title:
        ax.set_title(title)
    return fig, ax


# =========================================================================
# 权重矩阵可视化
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
    """绘制权重矩阵热力图。

    Args:
        W: (N, M) 权重矩阵
        title: 图标题
        ax: matplotlib Axes
        cmap: 颜色映射
        center_zero: 是否将色标中心设为 0（对称色标）
        colorbar: 是否显示色标
        xlabel, ylabel: 轴标签

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
    """并排绘制输入/循环/输出权重矩阵热力图。

    如果 dale_mask 给定且 sort=True，按 E/I 身份排序神经元。

    Returns:
        (fig, axes)  axes 为 (ax1, ax2, ax3)
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
# 聚类平均响应
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
    """绘制按聚类平均的发放率轨迹热力图。

    Args:
        responses: (n_clusters, T) 数组
        dale_mask: (n_clusters,) 可选 E/I 标记用于排序
        labels: 聚类标签

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
# 线吸引子可视化
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
    """在 2D PCA 平面上绘制线吸引子。

    Args:
        la_result: LineAttractorResult 对象
        pca_result: PCAResult 对象
        ax: matplotlib Axes
        show_rhs: 是否用颜色编码 ‖RHS‖（速度）
        color: 线条颜色（show_rhs=False 时）
        linewidth: 线条宽度
        marker_size: 采样点大小

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
        fig.colorbar(sc, ax=ax, label="‖RHS‖")
    else:
        ax.plot(coords[:, 0], coords[:, 1], color=color, linewidth=linewidth, zorder=4)
        ax.scatter(coords[:, 0], coords[:, 1], c=color, s=marker_size,
                   zorder=5, edgecolors='black', linewidths=0.5)

    # 端点标记
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
    """在 3D 子空间中绘制线吸引子。

    projection_axes: (3, M) 定义 3D 子空间的轴（如 choice/context/sensory）。
    可选叠加轨迹投影。

    Returns:
        (fig, ax)
    """
    fig, ax = _ensure_ax(ax, projection="3d")

    coords = la_result.coords  # (n_points, M)
    proj = coords @ projection_axes.T  # (n_points, 3)

    ax.plot(proj[:, 0], proj[:, 1], proj[:, 2],
            color=color, linewidth=linewidth, zorder=4)

    # 端点
    if la_result.endpoints is not None:
        for e, lbl in zip(la_result.endpoints, ["Left", "Right"]):
            e_proj = e @ projection_axes.T
            ax.scatter(e_proj[0], e_proj[1], e_proj[2], c='black', marker='*',
                       s=150, zorder=6, label=lbl)
        ax.legend()

    # 叠加轨迹
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
# 动画
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
    """创建 3D 轨迹旋转动画。

    相机围绕 z 轴旋转，展示 3D PCA 投影的轨迹。

    Args:
        trajectories: list of (T_i, M) 数组
        pca_result: PCAResult 对象（需 n_components >= 3）
        colors: 每条轨迹的颜色
        labels: 每条轨迹的标签
        fps: 帧率
        duration: 动画时长（秒）
        elev: 相机仰角（固定）
        step: 每帧旋转步数（跳帧以提高性能）
        figsize: 图大小

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

    # 预计算投影
    projected = [pca_result.transform(traj) for traj in trajectories]

    # 设置轴范围
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
# 轨迹渐进动画（逐帧绘制轨迹演变过程）
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
    """渐进绘制轨迹的动画——逐帧揭示轨迹随时间的演变过程。

    与 animate_trajectories_3d（仅旋转相机）不同，此函数逐时间步绘制轨迹，
    展示 RNN 状态如何从初始状态演化到最终状态。

    Args:
        trajectories: list of (T_i, M) 数组，每条轨迹可不同长度
        pca_result: PCAResult 对象；None 时自动拟合（2D 或 3D）
        colors: 每条轨迹的颜色
        labels: 每条轨迹的标签
        projection: "2d" 或 "3d"
        fps: 帧率
        n_frames: 总帧数；None 时取最长轨迹的长度
        elev, azim: 3D 相机角度（仅 3D 模式）
        linewidth: 轨迹线宽
        alpha: 轨迹透明度
        figsize: 图大小
        trail_alpha: 已绘制部分的淡出透明度

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
# 心理测量曲线
# =========================================================================
def plot_psychometric_curves(
    curves,
    *,
    ax=None,
    colors: list[str] | None = None,
    show_legend: bool = True,
    markers: list[str] | None = None,
):
    """绘制心理测量曲线（coherence vs P(right)）。

    Args:
        curves: list of PsychometricCurve 数据对象
        ax: matplotlib Axes
        colors: 每条曲线的颜色
        show_legend: 是否显示图例
        markers: 每条曲线的标记样式

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
# 快速查看（trial-by-trial 预测 vs 目标）
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
    """绘制多个 trial 的预测 vs 目标输出。

    对于分类任务（neurogym），targets 通常是整数类别标签 (B, T)，
    predictions 是 logits (B, T, n_classes)。此函数自动处理维度不匹配：
    - 若 targets 是 1D (T,) 或 2D (B, T) 且 predictions 是 3D (B, T, O)，
      则将 targets 扩展为 one-hot (B, T, O) 以便逐维度对比。

    Args:
        predictions: (B, T, O) 或 (B, T) 预测输出。2D 时自动扩展为 (B, T, 1)。
        targets: (B, T, O), (B, T), 或 (B,) 目标输出。
                 若维度与 predictions 不匹配，自动尝试适配：
                 - (B, T) 整数标签 + predictions (B, T, O) → one-hot 展开
                 - 其余情况自动扩展为 (B, T, 1)
        mask: (B, T) 或 (B, T, O) 有效区域掩码
        n_trials: 展示的 trial 数
        figsize: 图大小
        colors: (prediction, target) 颜色

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
        # (T,) → (1, T) → will be expanded below
        targets = targets[np.newaxis, :]

    if targets.ndim == 2:
        targets_2d = targets  # (B, T) — integer class labels
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

    # 隐藏多余子图
    for i in range(n_trials, n_rows * n_cols):
        r, c = divmod(i, n_cols)
        axes[r][c].set_visible(False)

    fig.tight_layout()
    return fig, axes
