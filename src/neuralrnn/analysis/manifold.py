"""流形 / 轨迹几何分析包装（MARBLE、neuralflow）—— 占位骨架。

设计决定（ARCHITECTURE §3.3 / §5）：MARBLE 与 neuralflow **不是** RNN 模型，
而是作用在"轨迹/向量场数据"上的分析方法，因此归入 analysis/ 而非 model zoo。
它们是可选重依赖（pip install 'neuralrnn[manifold]'）。

统一约定：上游用 model.generate / collect_states 得到轨迹与速度（位置+速度对），
本模块把它们喂给 MARBLE/neuralflow 的 API，产出嵌入/距离/比较结果。
具体接线见 PORTING_GUIDE 配方6（MARBLE）与配方8（neuralflow）。
"""
from __future__ import annotations

import numpy as np


def trajectories_to_pos_vel(traj: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """把轨迹 (B,T,M) 或 (T,M) 转成 MARBLE 需要的 (position, velocity) 对。
    velocity[t] = x[t+1] − x[t]，位置取前 T−1 步。"""
    arr = np.asarray(traj)
    if arr.ndim == 3:
        arr = arr.reshape(-1, arr.shape[-1])
    pos = arr[:-1]
    vel = np.diff(arr, axis=0)
    return pos, vel


def marble_embedding(pos: np.ndarray, vel: np.ndarray, **marble_kwargs):
    """用 MARBLE 对 (pos, vel) 学习无监督流形嵌入。

    移植（配方6）时实现：
        from MARBLE import construct_dataset, net
        data = construct_dataset(pos, features=vel)
        model = net(data, **marble_kwargs); model.fit()
        return model.transform(data)
    """
    raise NotImplementedError(
        "MARBLE 流形嵌入：请按 PORTING_GUIDE 配方6 接入 MARBLE，并 "
        "pip install 'neuralrnn[manifold]'。"
    )


def neuralflow_analysis(spike_data, **kwargs):
    """neuralflow（连续时间潜流场）分析入口。移植见配方8。"""
    raise NotImplementedError(
        "neuralflow 分析：请按 PORTING_GUIDE 配方8 接入 neuralflow。"
    )
