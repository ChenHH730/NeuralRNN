# nn-brain：CTRNN 任务训练 + 不动点动力学分析

> **范式**：A（任务优化 + 可解释分析）
> **原始来源**：nn-brain 教程（`RNN_DynamicalSystemAnalysis.ipynb`，基于 neurogym）
> **框架落点**：`models/ctrnn` + `analysis`（数值不动点 / 线性化 / 降维 / 向量场）
> **状态**：✅ 参考实现

## 1. 解决什么问题
认知神经科学常用 RNN 在认知任务（如感知决策 PerceptualDecisionMaking）上训练，
再把训练好的网络当作"模型大脑"，用动力系统工具反推它是**怎么完成计算的**——
例如决策的证据积累是否对应某条线吸引子。

## 2. 核心方法
连续时间 RNN（CTRNN），由 ODE 欧拉离散：

$$
z_t = (1-\alpha)\,z_{t-1} + \alpha\, f\!\big(W_{\text{rec}} z_{t-1} + W_{\text{in}} x_t + b\big),\qquad \alpha=\Delta t/\tau.
$$

- 读出 $G_\phi(z)=W_{\text{out}} z + b_{\text{out}}$，分类任务取 argmax；
- 训练目标：交叉熵（监督，`SupervisedObjective`）；
- 分析：在固定输入条件下，最小化 $\|F(z)-z\|^2$ 搜不动点，对雅可比做特征分解判稳定性，
  PCA 投影把活动、不动点、线吸引子画在同一平面。

## 3. 落到框架里
| 原始代码 | 框架 API | 说明 |
|---|---|---|
| `CTRNN.recurrence(input, hidden)` | `models/ctrnn/modeling_ctrnn.py: recurrence` | 欧拉步，含 $\alpha=\Delta t/\tau$ |
| neurogym `Dataset(task)` | `data/neurogym_dataset.py: NeurogymDataset.from_task` | time-first → batch-first |
| 训练 loop（CE 损失） | `SupervisedObjective` + `Trainer` | |
| `optim.Adam([hidden])` 最小化 $\|F(z)-z\|^2$ | `analysis/fixed_points.py: NumericFixedPointFinder` | 模型无关 |
| `np.linalg.eig(jac)` + PCA 画图 | `analysis/linearization.py` + `analysis/dimensionality.py` | 取主特征方向画线吸引子 |

- config 字段：`tau`、`dt`、`activation`、`dale`/`ei_ratio`（E-I 版）、`sigma_rec`、`trainable_h0`。
- 雅可比：默认自动微分兜底即可（CTRNN 无需解析）。
- 数据：`data/registry.py` 的 `perceptual_decision_making`（neurogym，无需下载）。

## 4. 对拍要点
把原 notebook 训练出的 CTRNN 权重塞进框架 `CTRNNModel`，对同一 `input` 比较
`recurrence(x, h)` 与原 `net.rnn.recurrence(input, hidden)` 是否 `allclose`（容差 1e-5）。
不动点搜索的收敛点集合应与原 notebook 数量级一致、落在 PCA 同一区域。

## 5. 复现实验
对应 `notebook/`（范式A 教程）。预期：感知决策任务正确率随相干性单调上升；
不动点沿 PC1 排成近似线吸引子，最大特征值方向与积累方向一致。
