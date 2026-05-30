# 教程 Notebook

每个 notebook 是一篇论文方法的**端到端可运行复现**，共享同一套框架 API
（`AutoModel` / `Trainer` / `analysis`），范式差异只体现在所用的 `Objective`。

## 已就绪

| Notebook | 范式 | 复现 | 关键 API |
|---|---|---|---|
| [01_plrnn_reconstruction_paradigmB.ipynb](01_plrnn_reconstruction_paradigmB.ipynb) | B 重构 | CNS2023 / shallowPLRNN | `TeacherForcingObjective` · `find_fixed_points`(解析) · `max_lyapunov_exponent` · D_stsp/D_H |
| [02_ctrnn_fixedpoints_paradigmA.ipynb](02_ctrnn_fixedpoints_paradigmA.ipynb) | A 任务 | nn-brain / CTRNN | `SupervisedObjective` · `find_fixed_points`(数值) · `fit_pca` · `linearize` · `dominant_direction` · 线吸引子 · ParametricWorkingMemory |
| [03_custom_dataset_paradigmA.ipynb](03_custom_dataset_paradigmA.ipynb) | A 任务 | 自定义数据导入 | `CustomDataset.from_arrays` · `SupervisedObjective`("regression") · 训练+评估 |

## 待补（随移植推进添加，配方见 PORTING_GUIDE）

- 低秩 RNN：DMS 任务的低秩结构与动力学（配方3）
- Latent Circuit：把高维 RNN 投影到低维可解释回路（配方4）
- LFADS：用 `VariationalObjective` 做尖峰序列的潜动力推断（配方5）
- MARBLE：对轨迹/向量场做无监督流形几何比较（配方6，`analysis/manifold`）
- Tiny RNN：行为拟合 + 嵌套交叉验证选模型（配方7，`train/cv`）
- neuralflow：连续时间潜流场分析（配方8，`analysis/manifold`）

## 运行前提

```bash
pip install -e .            # 核心依赖（torch/numpy/scipy/safetensors）
pip install -e '.[viz]'     # 可视化（matplotlib/jupyter）
pip install -e '.[neurogym]'  # 教程 02 需要 neurogym
```

无网环境下，需要下载的数据集（如 Lorenz63）请先按 `src/neuralrnn/data/download.py`
的提示把文件手动放入缓存目录（默认 `~/.cache/neuralrnn/datasets`，或设 `NEURALRNN_CACHE`）。
