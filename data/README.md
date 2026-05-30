# data/ —— 数据缓存目录

本目录用于存放下载/解压后的开源数据集（由 `src/neuralrnn/data/download.py` 写入）。

- 默认缓存目录：`~/.cache/neuralrnn/datasets`（可用环境变量 `NEURALRNN_CACHE` 覆盖到本目录）。
- 数据集的 URL / 文件名 / 解压方式登记在 `src/neuralrnn/data/registry.py` 的 `DATASET_REGISTRY`。
- **无网环境**：把对应文件手动放到缓存目录即可命中，`download.py` 会给出期望路径提示。

## 自定义数据导入

除了内置的 neurogym 任务数据和开源数据集（如 lorenz63），本框架支持导入用户自生成的数据集。
使用 `CustomDataset` 可以将 numpy 数组、torch 张量、.npz 文件或 .mat 文件导入框架：

```python
from neuralrnn.data.custom_dataset import CustomDataset

# 从 numpy 数组导入（范式 B：时间序列重构）
ds = CustomDataset.from_arrays(trajectory, mode="timeseries", sequence_length=200)

# 从 .npz 文件导入（含内部状态，用于 teacher forcing）
ds = CustomDataset.from_npz("my_data.npz", sequence_length=150, batch_size=8)

# 从 MATLAB .mat 文件导入
ds = CustomDataset.from_mat("neural_data.mat", variable_map={"inputs": "stim", "targets": "spikes"})
```

支持的数据模式：
- **supervised**（范式 A）：输入-输出对，用于任务优化训练
- **timeseries**（范式 B）：观测时间序列，用于动力学重构；可附带内部状态用于 teacher forcing

详见 [api/reference.md](api/reference.md) 中 `CustomDataset` 的完整文档。

> 大文件不建议提交进版本库；本目录默认只跟踪本说明。
