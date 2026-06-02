# E-I RNN: Training Excitatory-Inhibitory Recurrent Neural Networks for Cognitive Tasks

> Song, H.F., Yang, G.R. and Wang, X.J., 2016.
> Training excitatory-inhibitory recurrent neural networks for cognitive tasks:
> a simple and flexible framework.
> PLoS computational biology, 12(2).

---

## 1. 核心问题与动机（Why E-I RNN?）

### 1.1 传统 RNN 的生物学缺陷

用 RNN 建模认知任务的核心思路是：让 RNN 在同样的任务上训练，然后分析训练好的网络如何完成任务，从而推断大脑可能使用的计算机制。但传统 RNN 存在几个严重的生物学缺陷：

| 缺陷 | 生物学事实 | 后果 |
|---|---|---|
| 发放率可正可负 | 真实神经元发放率 ≥ 0 | 网络动力学与真实回路不对应 |
| 不区分兴奋/抑制神经元 | Dale 原则：每个神经元只释放兴奋性或抑制性递质 | 无法产生非正规（non-normal）动力学等关键特性 |
| 所有连接模式相同 | E→E、E→I、I→E、I→I 连接的稀疏度和特异性不同 | 无法建模局部回路的真实结构 |
| 读出可来自所有单元 | 长程投射主要是兴奋性的 | 与皮层架构不一致 |

**最关键的问题是 Dale 原则**：在哺乳动物皮层中，神经元要么是纯兴奋性的，要么是纯抑制性的。这一约束对网络动力学有深远影响——例如，它能产生 **非正规（non-normal）动力学**，这是选择性放大神经活动模式的关键机制（Murphy & Miller, 2009）。

### 1.2 同一任务，多种解

另一个关键问题是：**训练可以产生多个行为表现相同但结构和动力学截然不同的网络**。约束和正则化的选择决定了训练算法找到哪种解。因此，问题不再是"RNN 能否完成任务"（答案几乎总是"能"），而是"什么样的架构和约束能产生与真实神经记录最相似的网络活动"。

### 1.3 本文的贡献

本文提出了一个**灵活的、基于梯度下降的 E-I RNN 训练框架**，能够：
- 将 Dale 原则等生物学知识硬编码进网络结构
- 通过约束连接模式（稀疏性、E/I 比例、长程投射规则）引导训练产生生物学合理的解
- 用统一的方法在多种认知任务上训练和分析网络

---

## 2. 方法核心

### 2.1 网络动力学

E-I RNN 的连续时间方程：

$$
\tau \dot{\mathbf{x}} = -\mathbf{x} + W^{\text{rec}} \mathbf{r} + W^{\text{in}} \mathbf{u} + \sqrt{2\tau \sigma_{\text{rec}}^2} \xi
$$

$$
\mathbf{r} = [\mathbf{x}]_+ = \max(\mathbf{x}, 0)
$$

$$
\mathbf{z} = W^{\text{out}} \mathbf{r}
$$

关键点：
- **阈值线性（ReLU）激活**：发放率严格非负，与生物学一致
- **连续时间**：通过 Euler 离散化（$\alpha = \Delta t / \tau$）实现
- **噪声**：输入噪声 $\sigma_{\text{in}}$（共享）和递归噪声 $\sigma_{\text{rec}}$（私有）

### 2.2 Dale 原则的实现

将单元分为兴奋性（E）和抑制性（I）两群：

- **E 单元**：所有传出权重 ≥ 0
- **I 单元**：所有传出权重 ≤ 0
- **E:I 比例**：通常 4:1（80% E, 20% I）
- **读出**：仅从 E 单元读出（长程投射是纯兴奋性的）

权重矩阵参数化：$W^{\text{rec}} = W^{\text{rec},+} \cdot D$，其中 $W^{\text{rec},+}$ 非负，$D$ 是对角符号矩阵（+1 for E, -1 for I）。

### 2.3 连接模式约束

除了 Dale 原则，还可以约束连接模式：
- **无自连接**：对角线为零
- **稀疏连接**：通过硬约束（mask）或 L1 正则化
- **区域间连接**：局部抑制、长程兴奋
- **输入特化**：不同输入投射到不同的 E 单元群

### 2.4 训练方法

使用带修正的 SGD（Pascanu et al., 2013）：
1. **梯度裁剪**：防止梯度爆炸（最大范数 $G$）
2. **梯度保持正则化**：防止梯度消失（$\lambda_\Omega$）
3. **目标函数**：监督学习（交叉熵/MSE）+ 正则项

---

## 3. 分析方法详解（Tutorial 对应内容）

### 3.1 选择性分析（Selectivity / $d'$ 分析）

**Motivation**：在实验中，神经科学家通过记录单个神经元在不同条件下的活动来判断它是否"调谐"于某个任务变量。$d'$（选择性指数）是衡量这种调谐的标准指标。

**定义**：

$$
d' = \frac{\mu_1 - \mu_2}{\sqrt{(\sigma_1^2 + \sigma_2^2)/2}}
$$

其中 $\mu_1, \sigma_1^2$ 是单元在 choice-1 试次刺激期的均值和方差，$\mu_2, \sigma_2^2$ 同理。

**物理含义**：
- $d' > 0$：单元在 choice-1 试次中活动更高 → "选择性于 choice 1"
- $d' < 0$：单元在 choice-2 试次中活动更高 → "选择性于 choice 2"
- $|d'|$ 越大：选择性越强

**为什么重要**：
- 揭示了训练后网络中哪些单元编码了任务相关信息
- 为后续的连接可视化提供了排序依据
- 与实验中对单神经元的分析方法一致，便于与真实数据对比

**Tutorial 对应**：Section 5 — 计算每个神经元的 $d'$，分别绘制 E 和 I 单元的分布。

### 3.2 连接可视化（Connectivity Visualization）

**Motivation**：训练后的网络连接矩阵包含了"网络如何计算"的结构信息。通过按选择性排序单元，可以揭示训练过程中涌现的聚类结构。

**方法**：
1. 计算有效递归权重 $W^{\text{rec}} = |W| \cdot D$（Dale 约束下的实际权重）
2. 按 $d'$ 排序单元（从最选择性于 choice 1 到最选择性于 choice 2）
3. 用热力图可视化排序后的权重矩阵

**关键发现**（论文 Fig 3）：
- **无 Dale 约束**：相似 $d'$ 的单元之间有强兴奋连接，不同 $d'$ 的单元之间有强抑制连接
- **有 Dale 约束**：不同 $d'$ 的单元通过 **I 单元** 间接交互；训练后 E→E 和 I→I 连接自然分化
- **有结构约束**：训练进一步强化了预设的连接模式

**为什么重要**：
- 直接展示了网络的"布线图"
- 揭示了 Dale 约束如何影响网络的计算策略
- 与实验中用光遗传学或电生理绘制的连接图可直接对比

**Tutorial 对应**：Section 6 — 绘制按 $d'$ 排序的权重矩阵热力图，标注 E/I 边界。

### 3.3 PCA 降维分析

**Motivation**：高维神经活动（如 50 个单元）难以直接可视化。PCA 将活动投影到方差最大的低维平面，揭示群体动力学的本质结构。

**方法**：
1. 收集所有试次、所有时间步的神经活动
2. 拟合 PCA，取前 2-3 个主成分
3. 将每个试次的轨迹投影到 PC 空间

**关键发现**：
- 知觉决策任务中，不同 coherence 的轨迹在 PC 空间中形成**扇形展开**的模式
- 选择 1 和选择 2 的轨迹分别向不同方向发散
- 零 coherence 的轨迹位于中间，反映了随机决策

**为什么重要**：
- 将不可可视化的高维动力学变成可理解的几何结构
- 揭示了网络在做决策时的"神经状态空间"轨迹
- 与实验中对群体神经活动的降维分析方法完全一致

**Tutorial 对应**：Section 7 — 拟合 PCA，按 coherence 和 ground truth 着色绘制轨迹。

### 3.4 不动点分析（Fixed-Point Analysis）

**Motivation**：不动点（$F(z^*) = z^*$）是理解动力系统行为的关键。稳定的不动点对应吸引子——网络状态会被"吸引"到这些点附近。在决策任务中，不动点对应于网络的"选择"状态。

**方法**：
1. 在 0-coherence 刺激条件下（$[1, 0.5, 0.5]$），从多个随机初值出发
2. 用梯度下降最小化 $\|F(z) - z\|^2$
3. 收敛后检查速度 $\|F(z) - z\|$ 是否低于阈值
4. 将找到的不动点投影到 PCA 空间

**关键发现**：
- 知觉决策网络中存在一条**近似稳定的不动点链**（line attractor）
- 这条链沿 PC1 方向延伸，不同位置对应不同的"证据积累"状态
- 网络轨迹沿着这条链滑动，直到收敛到端点（选择 1 或选择 2）

**为什么重要**：
- 揭示了网络完成任务的**动力学机制**：决策 = 沿吸引子滑动
- 不动点的分布和稳定性解释了网络的精度-速度权衡
- 与实验中用动力系统理论分析神经数据的方法一致

**Tutorial 对应**：Section 8 — 数值不动点搜索 + PCA 空间可视化。

### 3.5 Jacobian 特征值分析（线性化）

**Motivation**：不动点本身只告诉我们网络"停在哪里"，但 Jacobian 特征值告诉我们网络"在不动点附近如何行为"——是被吸引（稳定）还是被排斥（不稳定），以及沿哪些方向。

**方法**：
1. 在不动点 $z^*$ 处计算 Jacobian $J = \partial F / \partial z$
2. 求特征值 $\lambda_i$ 和特征向量
3. 特征值在单位圆内 → 稳定方向；单位圆外 → 不稳定方向
4. 主导特征向量揭示了不动点附近的主动力学方向

**关键发现**：
- 不动点链上的点有**一个接近 1 的实特征值**（对应线吸引子方向）
- 其他特征值都在单位圆内（稳定方向）
- 主导特征向量沿 PC1 方向，与不动点链对齐

**为什么重要**：
- 区分了"真正的吸引子"和"伪不动点"
- 主导特征向量揭示了网络的**计算方向**——证据积累的方向
- 特征值谱完整刻画了不动点附近的动力学

**Tutorial 对应**：Section 9 — Jacobian 特征值在复平面上的分布 + 主导方向可视化。

---

## 4. 论文中其他重要分析方法

### 4.1 心理测量函数（Psychometric Function）

**定义**：选择 1 的百分比作为 signed coherence 的函数。

**意义**：与猴子实验中的心理测量曲线直接对比，验证网络的行为与动物一致。S 形曲线反映了噪声对决策的影响。

### 4.2 反应时间分析

**定义**：输出达到阈值所需的时间作为 coherence 的函数。

**意义**：反应时间是决策任务的核心行为指标。网络应表现出与动物一致的 speed-accuracy tradeoff。

### 4.3 混合选择性分析（Mixed Selectivity）

**定义**：单个单元同时对多个任务变量（如 choice、motion、color、context）有选择性。

**意义**：混合选择性是前额叶皮层的标志性特征，使网络能够在高维状态空间中实现灵活的计算。

### 4.4 状态空间回归分析

**方法**：用线性回归将群体活动投影到任务变量轴（choice、motion、color、context）上。

**意义**：揭示了不同任务变量如何在群体活动中编码，以及它们之间的相互作用。

---

## 5. E-I RNN vs. 其他 RNN 的优势

| 对比维度 | 普通 RNN | E-I RNN |
|---|---|---|
| 生物学合理性 | ❌ 发放率可正可负，不满足 Dale 原则 | ✅ 发放率非负，满足 Dale 原则 |
| 动力学类型 | 对称权重 → 正规动力学 | 非对称权重（E/I 不对称）→ 非正规动力学 |
| 选择性放大 | 无自然机制 | E/I 平衡产生选择性放大（Murphy & Miller, 2009） |
| 连接结构 | 全对称，无生物学意义 | E→E、E→I、I→E、I→I 可分别约束 |
| 与实验对比 | 需要后处理才能与真实数据对比 | 直接与实验数据对比（发放率、连接、选择性） |
| 长程投射 | 所有单元都可读出 | 仅 E 单元读出，符合皮层架构 |
| 训练约束 | 无生物学约束 | 可引入多种生物学知识（稀疏性、层结构、区域间连接） |

**核心优势**：E-I RNN 不仅能完成任务，还能产生与真实神经回路**结构和动力学**相似的解。这使得训练好的 E-I RNN 可以作为"计算假说"的来源——分析网络如何计算，可以推断大脑可能如何计算。

---

## 6. 原始代码地图

| 组件 | 文件/函数 | 描述 |
|---|---|---|
| `PosWLinear` | `EI_RNN.ipynb` | 非负权重线性层（Dale 约束） |
| `EIRecLinear` | `EI_RNN.ipynb` | 带 Dale 掩码的递归层 |
| `EIRNN` | `EI_RNN.ipynb` | E-I RNN 单元（状态 + 输出） |
| `Net` | `EI_RNN.ipynb` | 完整网络（EIRNN + E 单元读出） |
| 训练循环 | `EI_RNN.ipynb` | Adam + CrossEntropy |
| $d'$ 计算 | `EI_RNN.ipynb` | 选择性指数 |
| 连接可视化 | `EI_RNN.ipynb` | 权重矩阵热力图 |

---

## 7. 映射到 NeuralRNN API

### 模型

```python
from neuralrnn import AutoConfig, AutoModel

cfg = AutoConfig.for_model('ei_rnn',
                           input_dim=3, latent_dim=50, output_dim=3,
                           dt=20, sigma_rec=0.15, relu_after_blend=True)
model = AutoModel.from_config(cfg)
# model.e_size = 40, model.i_size = 10
# model._recurrent_weight() → Dale 约束下的有效权重
```

### 分析

```python
from neuralrnn.analysis import fit_pca, find_fixed_points, linearize, dominant_direction

# 选择性分析（$d'$）— 直接用 numpy 计算
d_prime = (mean_0 - mean_1) / np.sqrt((std_0**2 + std_1**2) / 2)

# PCA
pca = fit_pca(activity_all, n_components=2)

# 不动点
fps = find_fixed_points(model, backend='numeric',
                        task_input=torch.tensor([1, 0.5, 0.5]),
                        n_candidates=128, n_iters=10000)

# Jacobian
lin = linearize(model, fps.points[0].z, task_input=task_input)
d = dominant_direction(lin)
```

---

## 8. 复现 Notebook

`notebook/04_EIRNN_paradigmA.ipynb` 复现了论文的核心分析流程：

| Section | 内容 | 对应论文 |
|---|---|---|
| 2 | E-I RNN 训练 | Methods: Training |
| 4 | 单元活动可视化 | Fig 2G/H |
| 5 | $d'$ 选择性分析 | Eq 30, Fig 3 |
| 6 | 连接可视化 | Fig 3 |
| 7 | PCA 降维 | Fig 4B |
| 8 | 不动点分析 | Fig 7 (line attractor) |
| 9 | Jacobian 特征值 | 稳定性分析 |

---

## 9. 参考文献

1. 原始论文: https://doi.org/10.1371/journal.pcbi.1004792
2. 代码: https://github.com/xjwanglab/pycog
3. 参考实现: `reference_project/Neural_network_for_brain_2020/EI_RNN.ipynb`
4. Murphy, B.K. & Miller, K.D. (2009). Balanced amplification: A new mechanism of selective amplification of neural activity patterns. Neuron.
5. Pascanu, R. et al. (2013). On the difficulty of training recurrent neural networks. ICML.
