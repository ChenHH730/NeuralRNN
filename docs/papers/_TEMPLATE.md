# <论文标题 / 项目名>

> **范式**：A（任务优化）/ B（动力学重构）/ 行为 / 分析
> **原始仓库**：<链接>
> **框架落点**：`models/<family>`（或 `analysis/<...>`） + `<Objective>`
> **状态**：⬜ 待移植 / 🟡 移植中 / ✅ 已就绪

## 1. 解决什么问题
<一两段说明研究动机与要回答的科学/方法问题。>

## 2. 核心方法
<模型形式（写出 recurrence 的数学式）、训练目标、关键技巧。配图或公式。>

- 转移 $F_\theta$：
- 读出 $G_\phi$：
- 训练目标：
- 关键超参：

## 3. 落到框架里
| 原始代码 | 框架 API | 说明 |
|---|---|---|
| `<file>:<fn>` | `models/<family>/modeling_*.py: recurrence` | |
| `<file>:<fn>` | `train/objectives/<...>` | |
| `<file>:<fn>` | `analysis/<...>` | |

- 新增的 config 字段：
- 是否支持解析不动点 / 解析雅可比：
- 数据来源与 `data/registry.py` 条目：

## 4. 对拍要点（与原实现）
<diff-test 怎么做：用什么权重、喂什么输入、比较哪个张量、容差多少。见 PORTING_GUIDE 契约A。>

## 5. 复现实验
<对应 notebook、关键指标（如 D_stsp / D_H / λ_max / 任务正确率）的预期数值。>
