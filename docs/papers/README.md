# Paper Method Notes

For each paper integrated into NeuralRNN, we keep a "method note": it explains what problem the work addresses, the core method, which framework modules it maps to, and the key diff-test points against the original open-source implementation. The step-by-step porting recipe is in [../PORTING_GUIDE.md](../PORTING_GUIDE.md); this directory focuses on **explaining the methods themselves**.

## Status Overview

| Paper / Project | Paradigm | Framework Target | Note | Code Status |
|---|---|---|---|---|
| nn-brain (CTRNN + fixed-point analysis) | A | `models/ctrnn` + `analysis` | [ctrnn.md](ctrnn.md) | ✅ Reference implementation |
| Durstewitz lab (shallowPLRNN / DSR) | B | `models/plrnn` + `analysis` | [plrnn.md](plrnn.md) | ✅ Reference implementation |
| Low-rank RNN | A | `models/lowrank` | _to be written_ | ⬜ Pending port |
| Latent Circuit | A | `models/latent_circuit` | [latent_circuit.md](latent_circuit.md) | ✅ Ready |
| LFADS | B | `models/lfads` + `VariationalObjective` | _to be written_ | ⬜ Pending port |
| MARBLE (manifold geometry) | Analysis | `analysis/manifold` | _to be written_ | ⬜ Pending port |
| Tiny RNN (behavior fitting) | Behavior | `models/tiny_rnn` + CV | [tiny_rnn.md](tiny_rnn.md) | ✅ Ready |
| neuralflow (continuous-time latent flow field) | Analysis | `analysis/manifold` | _to be written_ | ⬜ Pending port |

When adding a new paper: copy [_TEMPLATE.md](_TEMPLATE.md) → rename → fill in, and add a row to the table above.
