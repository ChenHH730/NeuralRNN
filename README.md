# NeuralRNN

**认知神经科学 RNN 方法的统一框架** —— 把两类前沿范式统一到同一套 *transformers 风格* 接口下：

- **paradigm A 基于任务的优化（Task-based optimization，TBO）**[^1]：在认知任务上训练 RNN，再用不动点 / 向量场 / 降维等工具反推它如何完成计算。其目的是用RNN作为认知计算的替代
- **paradigm B 动力学重构（Dynamical system reconstruction，DSR）**[^2][^3]：直接从神经/行为时间序列里拟合出能再现其吸引子、功率谱、Lyapunov 谱的生成式 RNN。

二者共享一套模型设计、`Trainer` 与 `analysis` 分析器；**两个paradigm的区别只是 `Objective` 不同**：paradigm A的目标是尽可能地优化输出完成认知任务，paradigm B的目标是构造与目标神经活动同构的动力学系统。此外，DSR也可以用于对TBO训练好的模型进行动力学重构以进行可解释性分析[^4]。

---

## Core concept

所有模型都被看作"带读出的离散动力系统" $z_t=F_\theta(z_{t-1},x_t),\;y_t=G_\phi(z_t)$。
一个模型只要实现两个方法——

```python
def recurrence(self, x_t, z_prev, *, inputs=None): ...  # 单步转移 F
def readout(self, z_t): ...                              # 读出 G
```

——就能自动接入统一的训练器与全部分析工具。详见 [docs/theory/dynamical_systems.md](docs/theory/dynamical_systems.md)。

## Install

```bash
pip install -e .                  # 核心：torch / numpy / scipy / safetensors
pip install -e '.[neurogym]'      # 范式A 认知任务
pip install -e '.[lfads]'         # LFADS（lightning）
pip install -e '.[manifold]'      # MARBLE / neuralflow 流形分析
pip install -e '.[viz]'           # 教程可视化
pip install -e '.[all]'           # 全家桶
```

## Quickstart

```python
from neuralrnn import AutoConfig, AutoModel, Trainer, TrainingArguments
from neuralrnn import TeacherForcingObjective, load_dataset

# 1) dataset
# use registered dataset or custom dataset
ds = load_dataset("lorenz63", sequence_length=200, batch_size=16, normalize=True) 

# 2) model (config) + objective (based on the paradigm) + training
cfg = AutoConfig.for_model("shallow_plrnn", input_dim=0, latent_dim=3,
        output_dim=3, hidden_dim=50, autonomous=True) # model config
model = AutoModel.from_config(cfg) # load model
Trainer(model, ds, TeacherForcingObjective(alpha=0.1),
        TrainingArguments(max_steps=2000)).train() # train model

# 3) save and load（config.json + model.safetensors）
model.save_pretrained("ckpt/")
model = AutoModel.from_pretrained("ckpt/")

# 4) analysis (model agnostic)
from neuralrnn.analysis import find_fixed_points, max_lyapunov_exponent
fps = find_fixed_points(model)
```

See full pipeline in [`notebook/README.md`](notebook/README.md) and API in [`docs/api/referce.md`](docs/api/referce.md).

## Content structure

```
src/neuralrnn/
  configuration_utils.py   modeling_utils.py     # 核心契约（Config / Model 基类）
  auto/                    # AutoConfig / AutoModel 注册分发
  models/                  # 模型库：ctrnn(范式A)、plrnn(范式B)；其余待移植
  data/                    # 统一 batch、数据集、开源数据注册表 + 下载缓存
  train/                   # 通用 Trainer + 四个 Objective + 嵌套交叉验证
  analysis/                # 不动点/线性化/向量场/降维/Lyapunov/D_stsp,D_H/流形
  losses/  inputs/  tools/ # 预留
docs/                      # ARCHITECTURE.md · PORTING_GUIDE.md · theory/ · papers/
notebook/                  # 逐篇论文的端到端教程
```

## 内置 vs 待移植

| 模型 | 范式 | 状态 |
|---|---|---|
| CTRNN / Vanilla / E-I RNN | A | ✅ 参考实现 |
| shallow / dend / AL-PLRNN | B | ✅ shallow 参考实现（dend/AL） |
| 低秩 RNN · Latent Circuit · LFADS · Tiny RNN | A/B/行为 | ⬜ 待移植 |
| MARBLE · neuralflow（分析方法，非模型） | 分析 | ⬜ 待移植（→ `analysis/`） |

## 把新论文纳入框架

用 AI 辅助移植的完整手册见 **[docs/PORTING_GUIDE.md](docs/PORTING_GUIDE.md)**：四种适配器契约、
通用 8 步流程与铁律、8 篇论文逐篇配方、可直接复制的 AI 提示词模板、移植看板与常见坑表。

核心原则：**移植 = 写适配器（包装 + 对拍），不重写数学**。任何模型实现 `recurrence/readout`
即插即用；分析层只通过模型公共契约工作，绝不 import 具体模型类。

## 设计文档

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) —— 总体方案与架构（**先读**）
- [docs/PORTING_GUIDE.md](docs/PORTING_GUIDE.md) —— AI 辅助移植手册
- [docs/theory/dynamical_systems.md](docs/theory/dynamical_systems.md) —— 统一数学视角
- [docs/papers/](docs/papers/) —— 逐篇论文方法笔记


## License

MIT，见 [LICENSE](LICENSE)。各被移植论文的原始代码版权归原作者所有，移植时请遵循其各自许可证。


## Reference

[^1]: [Training Excitatory-Inhibitory Recurrent Neural Networks for Cognitive Tasks](https://doi.org/10.1371/journal.pcbi.1004792). 
project: https://github.com/gyyang/nn-brain

[^2]: [Reconstructing computational dynamics from neural measurements with RNN](https://www.nature.com/articles/s41583-023-00740-7)
project: https://github.com/DurstewitzLab/CNS-2023
[^3]: [Discovering cognitive strategies with tiny-RNN](https://www.nature.com/articles/s41586-025-09142-4) 
project: https://github.com/jil095/tinyRNN
[^4]: https://github.com/engellab/latentcircuit
[^5]: https://github.com/Dynamics-of-Neural-Systems-Lab/MARBLE
[]: https://github.com/NN4Neurosim/nn4n https://nn4n.org/
[]: 