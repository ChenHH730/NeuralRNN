# Recurrent neural network (RNN) as discrete dynamical system


This part set a dynamical system framework for NeuralRNN by introducing (1) RNN structure, (2) fixed point and trajectory, and (3) dynamical reconstruction.

---

## 1. 统一对象：带读出的离散动力系统

无论是任务优化的 CTRNN、低秩 RNN，还是用于重构的 PLRNN、LFADS，它们都可以写成同一个三元组

$$
z_t = F_\theta(z_{t-1},\, x_t), \qquad y_t = G_\phi(z_t), \qquad z_0 = z_{\text{init}}
$$

- $z_t \in \mathbb{R}^M$：潜状态（latent / hidden state），$M$ 为潜维度；
- $x_t \in \mathbb{R}^{K}$：外部输入（任务刺激或外生驱动），自治系统中 $x_t \equiv 0$；
- $y_t \in \mathbb{R}^{D}$：读出（观测预测 / 动作 logits）；
- $F_\theta$：**单步转移**，框架里就是 `recurrence`；$G_\phi$：**读出**，就是 `readout`。

> **关键**：模型只负责定义 $F_\theta$ 与 $G_\phi$。损失（范式差异）不写进模型——
> 它由 `Objective` 决定。分析（不动点、Lyapunov……）只是对 $F_\theta$ 的几何/谱研究，
> 因此只要拿到 `recurrence` 就够了，不必知道模型是哪一类。

### Example

**连续时间 RNN（CTRNN，范式A）** 由 ODE 欧拉离散得到：

$$
\tau \dot r = -r + f(W_{\text{rec}} r + W_{\text{in}} x + b)
\;\;\Longrightarrow\;\;
z_t = (1-\alpha)\,z_{t-1} + \alpha\, f(W_{\text{rec}} z_{t-1} + W_{\text{in}} x_t + b),\quad \alpha = \tfrac{\Delta t}{\tau}.
$$

**分段线性 RNN（PLRNN，范式B）** 用 ReLU 把状态空间切成线性子区：

$$
z_t = A\,z_{t-1} + W_1\,\mathrm{ReLU}(W_2 z_{t-1} + h_2) + h_1\;(+\,C x_t),\quad A=\mathrm{diag}.
$$

PLRNN 的分段线性结构带来一个巨大好处：不动点与 $k$-环可以**解析求解**（见 §4）。

---

## 2. 不动点与轨迹

**不动点** $z^*$ 满足 $F_\theta(z^*, x) = z^*$ （给定输入条件 $x$）。等价地，速度场

$$
v(z) = F_\theta(z, x) - z
$$

为零。框架的数值不动点搜索就是并行地最小化 $\|v(z)\|^2$（`analysis/fixed_points.py`
的 `NumericFixedPointFinder`）。围绕一群不动点常能看出任务的几何结构，
例如决策任务里沿某方向排成一条**线吸引子**（line attractor），其方向由主特征向量给出。

---

## 3. 线性化、雅可比与稳定性

在不动点附近，动力学由**雅可比矩阵** $J = \partial F_\theta / \partial z\big|_{z^*}$ 主导：

$$
\delta z_{t} \approx J\,\delta z_{t-1}.
$$

对**离散**系统，稳定性看 $J$ 的特征值的模长  $|\lambda_i|$ ：

- 所有 $|\lambda_i| < 1$：**稳定**不动点（吸引子）；
- 存在 $|\lambda_i| > 1$：**不稳定**方向，个数即鞍点的不稳定维数；
- $|\lambda_i| = 1$：临界（分岔边界）。

框架里 `analysis/linearization.py` 做特征分解与分类；`dominant_direction` 取最大特征值
方向用于画慢流形/线吸引子。雅可比来自模型契约 `model.jacobian`：默认用自动微分兜底，
解析模型（如 PLRNN）覆盖为闭式 $J(z) = \mathrm{diag}(A) + W_1\,\mathrm{diag}\!\big(\mathbb{1}[W_2 z + h_2 > 0]\big)\,W_2$。

---

## 4. PLRNN 的解析不动点 / k-环

因为 ReLU 在每个线性子区里是常值对角矩阵 $D$（对角元为 $\mathbb{1}[W_2 z + h_2 > 0]$），
在该子区内系统是**仿射**的：

$$
z_t = (A + W_1 D W_2)\,z_{t-1} + (W_1 D h_2 + h_1).
$$

令其为不动点（或递推 $k$ 步得 $k$-环的环方程）即可**线性求解** $z^*$，再验证解是否真的落在
假设的子区里。`analysis/fixed_points.py` 的 `AnalyticPLRNNFixedPointFinder`（移植自 CNS2023
的 `scy_fi`/`main`）就是启发式地枚举子区组合、解环方程、回代验证，并用 `get_eigvals`
给出每个环点的稳定性。统一入口 `find_fixed_points` 在模型支持解析时优先走这条路，否则回退数值法。

---

## 5. 混沌判据：最大 Lyapunov 指数

最大 Lyapunov 指数刻画相邻轨迹的指数分离率：

$$
\lambda_{\max} = \lim_{T\to\infty} \frac{1}{T} \sum_{t=1}^{T} \log \big\| J_t\, q \big\|,
$$

数值上沿轨迹累乘雅可比并周期做 QR 重正交（避免数值溢出），累加 $\log|R_{11}|/T$
（`analysis/lyapunov.py`，移植自 CNS2023）。$\lambda_{\max} > 0$ 表征混沌——例如 Lorenz63
的 $\lambda_{\max} \approx 0.9$，一个重构成功的模型应当复现出相近的值。

---

## 6. 重构质量：状态空间散度 D_stsp 与功率谱距离 D_H

重构（范式B）的目标不是逐点预测，而是**再现长期统计与几何**。两个标准指标：

- **D_stsp**（state space divergence）：把生成轨迹与真实轨迹在状态空间的占据分布做散度。
  低维（≲4）用直方图 + Laplace 平滑 + KL；高维用高斯混合的蒙特卡洛 KL。衡量"吸引子形状"是否一致。
- **D_H**（power spectrum distance）：逐维比较（平滑后的）功率谱的 Hellinger 距离均值，
  衡量"时间结构/频率成分"是否一致。

二者在 `analysis/stsp_metrics.py` 实现（移植自 CNS2023），可用于训练中评估或论文复现。

---

## 7. 训练范式与本视角的关系

- **范式A / 监督**：给定 $x_{1:T}$，让 $G_\phi(z_t)$ 拟合任务目标（分类/回归）。损失是任务损失。
- **范式B / 教师强制重构**：观测 $X_{1:T}$ 即（近似）潜轨迹，用广义教师强制
  $z \leftarrow \alpha\, x^{\text{obs}} + (1-\alpha)\, z^{\text{pred}}$ 稳定混沌系统的训练（`TeacherForcingObjective`）。
- **范式B / 变分（LFADS）**：把 $z_0$（与输入）当隐变量，最大化 ELBO（`VariationalObjective`）。
- **行为拟合（Tiny RNN）**：$y_t$ 是动作分布，最大化选择的对数似然，配合嵌套交叉验证。

所有这些只是"在同一个 $(F_\theta, G_\phi)$ 上换一个 $\mathcal{L}$"，这正是框架把范式差异收敛到
`Objective`、而让 `Trainer` 与 `analysis` 完全通用的根本原因。

---

### 参考实现对照

| 概念 | 代码位置 |
|---|---|
| $F_\theta$ / $G_\phi$ 契约 | `modeling_utils.py: recurrence / readout` |
| 数值不动点 | `analysis/fixed_points.py: NumericFixedPointFinder` |
| 解析不动点 / k-环 | `analysis/fixed_points.py: AnalyticPLRNNFixedPointFinder` |
| 线性化 / 稳定性 | `analysis/linearization.py` |
| 最大 Lyapunov 指数 | `analysis/lyapunov.py` |
| D_stsp / D_H | `analysis/stsp_metrics.py` |
| 向量场 | `analysis/vector_field.py` |
