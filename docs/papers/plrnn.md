# Durstewitz lab：shallowPLRNN 动力学重构（DSR）

> **范式**：B（动力学重构）
> **原始来源**：CNS 2023 tutorial（`CNS2023_tutorial.ipynb`），Durstewitz lab DSR 系列
> **框架落点**：`models/plrnn` + `analysis`（解析不动点 / Lyapunov / D_stsp / D_H）
> **状态**：✅ 参考实现（框架中范式B 与解析分析能力最完整的样例）

## 1. 解决什么问题
给定一段观测时间序列（仿真系统如 Lorenz63，或实测神经/行为信号），训练一个生成式 RNN，
使其**自由运行**产生的轨迹在长期统计上与真实系统一致——再现吸引子形状、功率谱、
甚至最大 Lyapunov 指数。这叫**动力系统重构**（Dynamical Systems Reconstruction, DSR）。

## 2. 核心方法
浅层分段线性 RNN（shallowPLRNN）：

$$
z_t = A\,z_{t-1} + W_1\,\mathrm{ReLU}(W_2 z_{t-1} + h_2) + h_1\;(+\,C s_t),\qquad A=\mathrm{diag}.
$$

- 读出：identity（直接观测潜状态，DSR 标准设定，$M$ 常等于观测维 $N$）；
- 训练目标：**广义教师强制**（GTF）——前向时把观测按强度 $\alpha$ 注入预测
  $z \leftarrow \alpha z^{\text{obs}} + (1-\alpha) z^{\text{pred}}$，再算 MSE。小 $\alpha$（如 0.1）
  对混沌系统的梯度训练至关重要（`TeacherForcingObjective`）；
- 解析能力：分段线性结构 ⇒ 雅可比闭式
  $J(z)=\mathrm{diag}(A)+W_1\mathrm{diag}(\mathbb{1}[W_2 z+h_2>0])W_2$，
  且不动点 / $k$-环可由环方程**线性求解**（`scy_fi`）。

## 3. 落到框架里
| 原始代码 | 框架 API | 说明 |
|---|---|---|
| `shallowPLRNN.forward(z, s)` | `models/plrnn/modeling_plrnn.py: recurrence` | 数值逐位对齐 |
| 解析雅可比 | `ShallowPLRNNModel.jacobian` | 与自动微分对拍（ReLU 边界除外） |
| `predict_sequence_using_gtf` + `generalized_teacher_forcing` | `train/objectives/teacher_forcing.py` | 推广到 $M\neq N$ 的部分强制 |
| `TimeSeriesDataset` + `sample_batch` | `data/timeseries_dataset.py` | time-first → batch-first，targets 右移一位 |
| `max_lyapunov_exponent`（QR 重正交） | `analysis/lyapunov.py` | 用 `generate` + `jacobian` 契约 |
| `state_space_divergence_*` / `power_spectrum_error` | `analysis/stsp_metrics.py` | D_stsp（binning/gmm）、D_H |
| `scy_fi` / `main` / `construct_relu_matrix` | `analysis/fixed_points.py: AnalyticPLRNNFixedPointFinder` | 严格保留原 `A=diag` 约定 |

- 新增 config 字段：`latent_dim`（=$M$）、`hidden_dim`（=$L$）、`autonomous`（是否含外部输入 $C$）。
- 暴露 `analytic_parameters()` 给解析后端取 $(A, W_1, W_2, h_1, h_2)$。
- 数据：`data/registry.py` 的 `lorenz63`（CNS-2023 zip，含 train/test）。

## 4. 对拍要点
- **数值一致**：用原 notebook 的 PLRNN 权重，对同一 $z$ 比较框架 `recurrence` 与原 `forward`（容差 1e-6）；
- **解析 vs 自动微分雅可比**：在远离 ReLU 边界的随机 $z$ 上 `allclose`；
- **解析不动点**：`AnalyticPLRNNFixedPointFinder` 的结果应与原 `main(np.diag(A), …)` 一致
  （我们已按原实现保留 `np.diag(A)` 广播这一约定，确保特征值数值可复现）；
- **整体重构**：训练后 $\lambda_{\max}$ 应接近 0.9（Lorenz63），D_stsp / D_H 显著下降。

## 5. 复现实验
对应 `notebook/`（范式B Lorenz63 教程）。流程：`load_dataset("lorenz63")` →
`ShallowPLRNNModel` + `TeacherForcingObjective(alpha=0.1)` + `Trainer` 训练 →
`generate` 自由运行 → 算 D_stsp / D_H / λ_max、用 `find_fixed_points` 解析求不动点并判稳定性。
