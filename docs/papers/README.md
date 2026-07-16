# Paper Method Notes

For each paper integrated into NeuralRNN, we keep a "method note": it explains what problem the work addresses, the core method, which framework modules it maps to, and the key diff-test points against the original open-source implementation. The step-by-step porting recipe is in [../PORTING_GUIDE.md](../PORTING_GUIDE.md); this directory focuses on **explaining the methods themselves**.

## Status Overview

| Paper / Project | Paradigm | Framework Target | Note | Code Status |
|---|---|---|---|---|
| nn-brain (CTRNN + fixed-point analysis) | A | `models/ctrnn` + `analysis` | [ctrnn.md](ctrnn.md) | ✅ Reference implementation |
| Durstewitz lab (shallowPLRNN / DSR) | B | `models/plrnn` + `analysis` | [plrnn.md](plrnn.md) | ✅ Reference implementation |
| dendPLRNN (Brenner et al. 2022) | B | `models/plrnn` + `analysis` | [plrnn.md](plrnn.md) | ✅ Reference implementation |
| ALRNN-DSR (Brenner et al. 2024) | B | `models/plrnn` + `analysis` | [plrnn.md](plrnn.md) | ✅ Reference implementation |
| Low-rank RNN | A/B | `models/lowrank` | [lowrank_rnn.md](lowrank_rnn.md) | ✅ Ready |
| Tiny RNN (behavior fitting) | Behavior | `models/tiny_rnn` + CV | [tiny_rnn.md](tiny_rnn.md) | ✅ Ready |
| Latent Circuit | A | `models/latent_circuit` | [latent_circuit.md](latent_circuit.md) | ✅ Ready |
| Echo-State / Critical Initialization | A | `models/ctrnn` + `freeze_*` | [esn.md](esn.md) | ✅ Ready |
| Constrained RNN (seRNN / sparse / modular) | A | `models/constrained_rnn` | [constrained_rnn.md](constrained_rnn.md) | ✅ Ready |
| Connectome-Constrained RNN | B | `models/connectome_rnn` | [connectome_constrained_rnn.md](connectome_constrained_rnn.md) | ✅ Ready |
| Short-Term Plasticity RNN | A | inline (`notebook/11_STP_RNN_paradigmA.ipynb`) | [stp_rnn.md](stp_rnn.md) | 🚧 Notebook port |
| Neural Sequence Models (Orhan 2019; Zhou 2023) | A | `models/ctrnn` + `models/ctrnn.ei_rnn` + `analysis/sequentiality` | [neural_sequence.md](neural_sequence.md) | ✅ Ready |
| Multitask RNN (Yang et al., 2019) | A | `models/ctrnn` + `data/tasks/multitask_yang_task.py` | [multitask.md](multitask.md) | ✅ Ready |

When adding a new paper: copy [_TEMPLATE.md](_TEMPLATE.md) → rename → fill in, and add a row to the table above.
