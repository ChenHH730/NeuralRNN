# NeuralRNN 文档

本目录是 NeuralRNN 的文档中心。如果你关心项目是如何构建的，请看 [ARCHITECTURE.md](ARCHITECTURE.md)和[PORTING_GUIDE.md](PORTING_GUIDE.md)。如果你关心如何使用本项目，请看 [api/reference.md](api/reference.md)。如果你关心项目的理论背景，请看 theory 文件夹。如果你关心每一篇文章的具体方法，请看 papers 文件夹。

1. **[ARCHITECTURE.md](ARCHITECTURE.md)** —— 总体方案与架构设计（先读这个）。
   设计哲学、目录结构、五大模块（数据 / 模型+config / 训练 / 分析 / 文档与教程）的
   详细设计、张量约定、两条端到端范式示例、实现路线图。

2. **[PORTING_GUIDE.md](PORTING_GUIDE.md)** —— 用 AI 把各论文开源代码纳入本框架的实操手册。
   四种适配器契约（模型 / 数据 / 目标 / 分析）、通用 8 步移植流程与"移植铁律"、
   8 篇论文的逐篇配方、可直接复制的 AI 提示词模板、移植看板与常见坑表。

3. **[api/reference.md](api/reference.md)** —— 完整 API 参考文档。
   覆盖所有模块、类、函数的签名、参数说明与用法示例。查阅具体 API 时优先看这里。

4. **[theory/](theory/)** —— 理论背景。
   - [dynamical_systems.md](theory/dynamical_systems.md)：把"带读出的离散动力系统"作为
     统一视角，讲清不动点 / 雅可比 / 稳定性 / Lyapunov / 状态空间散度等分析的数学基础。

5. **[papers/](papers/)** —— 逐篇论文的方法笔记。
   - [README.md](papers/README.md)：论文清单与状态。
   - [_TEMPLATE.md](papers/_TEMPLATE.md)：每纳入一篇论文，复制此模板填写。
   - 已就绪示例：[ctrnn.md](papers/ctrnn.md)（范式A 参考）、[plrnn.md](papers/plrnn.md)（范式B 参考）。

教程（notebook）见仓库根的 [`notebook/`](../notebook/)。

## 两条范式

- **范式 A（任务优化 + 可解释分析）**：用 RNN 在认知任务上训练出能解任务的网络，再用
  不动点 / 向量场 / 降维等工具反向理解它"怎么算"。代表：nn-brain、低秩 RNN、Latent Circuit。
- **范式 B（动力学重构）**：直接从观测到的神经/行为时间序列里，拟合一个能再现其动力学
  （吸引子、分岔、Lyapunov 谱）的生成式 RNN。代表：PLRNN/dendPLRNN/ALRNN、LFADS、Tiny RNN。

两者在本框架里共享同一套模型契约与 Trainer，差异仅由 **Objective** 决定。


