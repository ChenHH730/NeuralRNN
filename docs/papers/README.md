# 论文方法笔记

每篇纳入 NeuralRNN 的论文，在此放一份"方法笔记"：说明它解决什么问题、核心方法、
落到框架里对应哪些模块、以及与原始开源实现的对拍要点。逐篇移植配方在
[../PORTING_GUIDE.md](../PORTING_GUIDE.md)；本目录侧重**方法本身的讲解**。

## 状态总览

| 论文 / 项目 | 范式 | 框架落点 | 笔记 | 代码状态 |
|---|---|---|---|---|
| nn-brain（CTRNN + 不动点分析） | A | `models/ctrnn` + `analysis` | [ctrnn.md](ctrnn.md) | ✅ 参考实现 |
| Durstewitz lab（shallowPLRNN / DSR） | B | `models/plrnn` + `analysis` | [plrnn.md](plrnn.md) | ✅ 参考实现 |
| 低秩 RNN（low-rank RNN） | A | `models/lowrank` | _待写_ | ⬜ 待移植 |
| Latent Circuit | A | `models/latent_circuit` | _待写_ | ⬜ 待移植 |
| LFADS | B | `models/lfads` + `VariationalObjective` | _待写_ | ⬜ 待移植 |
| MARBLE（流形几何） | 分析 | `analysis/manifold` | _待写_ | ⬜ 待移植 |
| Tiny RNN（行为拟合） | 行为 | `models/tiny_rnn` + CV | _待写_ | ⬜ 待移植 |
| neuralflow（连续时间潜流场） | 分析 | `analysis/manifold` | _待写_ | ⬜ 待移植 |

新增论文时：复制 [_TEMPLATE.md](_TEMPLATE.md) → 重命名 → 填写，并在上表加一行。
