# Third-party source and license notices

This ledger applies only to the installable `src/neuralrnn/` package in
NeuralRNN 0.5.0. NeuralRNN is distributed as a combined work under
`GPL-3.0-only`. Files derived from third-party projects remain subject to their
original copyright notices and license terms. The complete texts referenced
below are included in `LICENSE` and `licenses/`.

"Reimplementation" in this document means that the cited paper, public task
protocol, interface, equations, or numerical behavior was consulted, but the
NeuralRNN implementation was written independently. Numerical equality under
fixed test inputs describes output validation and does not mean source-code
identity.

## Adapted or translated implementations

| NeuralRNN files or components | Upstream project | Relationship | Upstream license | Notice file |
|---|---|---|---|---|
| `src/neuralrnn/train/objectives/teacher_forcing.py`; shallow-PLRNN portions of `src/neuralrnn/models/plrnn/` | [CNS-2023](https://github.com/DurstewitzLab/CNS-2023) | Python/PyTorch adaptation of published GPL implementation | GPL-3.0-only | `LICENSE` |
| `src/neuralrnn/analysis/lyapunov.py`; SCYFI-related analytical fixed-point routines | [SCYFI](https://github.com/DurstewitzLab/SCYFI) | Adapted numerical algorithms | GPL-3.0-only | `LICENSE` |
| dendritic and almost-linear PLRNN options in `src/neuralrnn/models/plrnn/` | [dendPLRNN](https://github.com/DurstewitzLab/dendPLRNN) and [ALRNN-DSR](https://github.com/DurstewitzLab/ALRNN-DSR) | Adapted model equations and implementation conventions | GPL-3.0-only | `LICENSE` |
| line-attractor routines and related visualization/training code under `src/neuralrnn/` | [trainRNNbrain](https://github.com/engellab/trainRNNbrain) | Modified Python implementation | MIT | `licenses/MIT.txt` |
| `src/neuralrnn/models/latent_circuit/` | [latentcircuit](https://github.com/engellab/latentcircuit) | Modified PyTorch implementation | MIT | `licenses/MIT.txt` |
| E-I model conventions and NeuroGym extraction code under `src/neuralrnn/` | [nn-brain](https://github.com/gyyang/nn-brain) | Modified Python implementation | MIT | `licenses/MIT.txt` |
| PyCog-compatible E-I constraints and initialization options | [PyCog](https://github.com/frsong/pycog) | Modified implementation of model conventions | MIT | `licenses/MIT.txt` |
| `src/neuralrnn/analysis/demixed.py` | [dPCA](https://github.com/machenslab/dPCA) | Modified implementation of demixed PCA | MIT | `licenses/MIT.txt` |
| `src/neuralrnn/rl/envs/echoice.py` | [neural-circuit-economic-choice](https://github.com/aldobattista/neural-circuit-economic-choice) | Gymnasium/PyTorch adaptation | MIT | `licenses/MIT.txt` |
| plume environment, simulation, agents and evaluation code under `src/neuralrnn/rl/` | [plumetracknets](https://github.com/BruntonUWBio/plumetracknets) | Modified Python implementation | MIT | `licenses/MIT.txt` |
| PPO utilities used by the plume workflow | [pytorch-a2c-ppo-acktr-gail](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail) | Modified implementation conventions | MIT | `licenses/MIT.txt` |
| Go/No-Go task and related activation-analysis conventions under `src/neuralrnn/` | [ActivationMattersRNN](https://github.com/engellab/ActivationMattersRNN) | Modified task and analysis conventions | MIT | `licenses/MIT.txt` |
| `src/neuralrnn/data/neurogym_dataset.py` and NeuroGym task adapters | [NeuroGym](https://github.com/neurogym/neurogym) | Adapter and modified extraction conventions; NeuroGym is also a runtime dependency | Apache-2.0 | `licenses/Apache-2.0.txt`, `licenses/NOTICE-NeuroGym.txt` |
| STP-RNN model and task code under `src/neuralrnn/` | [Short-term-plasticity-RNN](https://github.com/nmasse/Short-term-plasticity-RNN) | PyTorch adaptation of model and task implementation | Apache-2.0 | `licenses/Apache-2.0.txt` |

All listed upstream projects are also cited in the relevant documentation or
manuscript sections. Scientific citation and software-license compliance are
treated as separate requirements.

## Independent reimplementations based on unlicensed reference repositories

The repositories below did not contain an explicit software license when they
were audited for the 0.5.0 release. NeuralRNN therefore does not redistribute
their source code. The listed components were independently implemented from
the papers, task specifications, equations, interfaces, or externally visible
numerical behavior.

| NeuralRNN component | Consulted reference | Material consulted |
|---|---|---|
| `src/neuralrnn/data/contingency_dataset.py` and `src/neuralrnn/rl/probe.py` | [value_rnn](https://github.com/mobeets/valuernn) and Qian, Burrell et al. | Experimental protocol, event conventions, array layout and expected numerical outputs |
| `src/neuralrnn/data/tasks/multitask_yang_task.py` | [gyyang/multitask](https://github.com/gyyang/multitask) | Task definitions, timing and input/output layout |
| `src/neuralrnn/rl/planning.py` and `src/neuralrnn/rl/envs/maze.py` | [KrisJensen/planning_code](https://github.com/KrisJensen/planning_code) | Maze/planning protocol, interface conventions and expected behavior |
| `src/neuralrnn/models/lowrank/`, `src/neuralrnn/analysis/population_structure.py` and low-rank task helpers | [fmastrogiuseppe/LowRank](https://github.com/fmastrogiuseppe/LowRank) and [adrian-valente/populations_paper_code](https://github.com/adrian-valente/populations_paper_code) | Published equations, parameter conventions, task formats and analysis definitions |
| `src/neuralrnn/train/cv.py` and Tiny-RNN data interfaces | Tiny RNN reference repository accompanying the cited study | Cross-validation protocol, configuration naming and dataset interfaces |

These references receive academic attribution, but no license is asserted for
their repositories and no permission from those repositories is used as the
basis for licensing the independently written NeuralRNN code.

## Copyright notices for permissively licensed material

The MIT copyright and permission notices are reproduced in
`licenses/MIT.txt`. Apache-2.0 material is accompanied by the full Apache
license and the NeuroGym NOTICE file. The top-level `LICENSE` contains GPLv3,
which governs distribution of the combined NeuralRNN work.
