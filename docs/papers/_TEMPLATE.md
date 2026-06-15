# <Paper Title / Project Name>

> **Paradigm**: A (task optimization) / B (dynamical systems reconstruction) / behavior / analysis
> **Original repository**: <link>
> **Framework target**: `models/<family>` (or `analysis/<...>`) + `<Objective>`
> **Status**: ⬜ Pending port / 🟡 Porting / ✅ Ready

## 1. What problem it solves

<One or two paragraphs explaining the scientific/methodological motivation and the question the work addresses.>

## 2. Core method

<Model form (write the recurrence equation), training objective, and key tricks. Include figures or formulas.>

- Transition $F_\theta$:
- Readout $G_\phi$:
- Training objective:
- Key hyperparameters:

## 3. How to use this method in our framework

| Original code | Framework API | Note |
|---|---|---|
| `<file>:<fn>` | `models/<family>/modeling_*.py: recurrence` | |
| `<file>:<fn>` | `train/objectives/<...>` | |
| `<file>:<fn>` | `analysis/<...>` | |

- New config fields:
- Whether analytic fixed points / analytic Jacobian are supported:
- Data source and `data/registry.py` entry:

## 4. Consistency with the original implementation

<How to diff-test: which weights, what input, which tensor to compare, and what tolerance. See PORTING_GUIDE contract A.>

## 5. Reproduction experiments

<Corresponding notebook and expected values for key metrics.>
