# Tutorial Notebook

These notebook implement previous works with the shared architecture of NeuralRNN (using `AutoModel` / `Trainer` / `analysis`).

| Notebook | Paradigm | Reference | Key API |
|---|---|---|---|
| [01_ctrnn](01_ctrnn_fixedpoints_paradigmA.ipynb) | Task | Yang (2020) | `SupervisedObjective` · `find_fixed_points` |
| [02_latent_circuit](02_latent_circuit_paradigmB.ipynb) | Reconstruction | Langdon & Engel (2025) | `latent_circuit` · `ReconstructionObjective` · `connection analysis` |
| [03_custom_pipeline](03_custom_pipeline.ipynb) | Task | None | `CustomDataset.from_arrays` · `SupervisedObjective`|
| [04_EIRNN](04_EIRNN_paradigmA.ipynb) | Task | Song et al. (2016) | `ei_rnn` |
| [05_plrnn](05_plrnn_reconstruction_paradigmB.ipynb) | Reconstruction | Durstewitz PLRNN series | `TeacherForcingObjective` · `plrnn` · `dend_plrnn` · `alrnn` |
| [06_tiny_RNN](06_tiny_RNN_paradigmB.ipynb) | Reconstruction | Ji-An et al. (2025) | `tiny_rnn` · inference |
| [07_lowrank_RNN](07_lowrank_RNN_paradigmA.ipynb) | Task | Dubreuil et al. (2022) | `lowrank_rnn` · poulation structure` |
| [08_lowrank_RNN](08_lowrank_RNN_paradigmB.ipynb) | Reconstruction | Valente et al. (2022) | `lowrank_rnn` · LINT  |
| [09_echo_state_network](09_echo_state_network_paradigmA.ipynb) | Task | Pachitariu et al., (2026) | `freeze_*` · critical init · reservoir |
| [10_constrained_RNN](10_constrained_RNN_paradigmA.ipynb) | Task | seRNN; critical_init sparse/modular | `constrained_rnn` · `se_rnn` · `sparse_rnn` · `modular_rnn` · spatial regularizer  |
| [11_STP_RNN](11_STP_RNN_paradigmA.ipynb) | Task | Masse et al. (2019) | STP-RNN · activity-silent maintenance · manipulation-driven persistent activity |
| [12_multitask](12_multitask_paradigmA.ipynb) | Task | Yang et al. (2019) | `multitask_yang` dataset · 20-task CTRNN · task variance clusters |
| [13_flexible_multitask](13_flexible_multitask_paradigmA.ipynb) | Task | Driscoll et al. (2024) | dynamic motifs |
| [14_activation](14_activation_paradigmA.ipynb) | Task | Tolmachev & Engel (2025) | activation function comparison |
| [15_neural_sequence](15_neural_sequence_paradigmA.ipynb) | Task | Orhan & Ma (2019); Zhou et al. (2023) Figure 3 | inline Orhan/Zhou T+WM tasks · `ctrnn` · `ei_rnn` · sequentiality index · effective dimensionality · ramp-to-sequence transition |
| [16_connectome_rnn](16_connectome_rnn_paradigmB.ipynb) | Reconstruction | Beiran & Litwin-Kumar (2025) | `gain_rnn` · cycling task · teacher-student (shared J, gain/bias only) · Fig.1 readout-trained student · Fig.2 recorded-activity students (M=20/40/80/120, `ReconstructionObjective`) |
| [17_multi_area_rnn](17_multi_area_rnn_paradigmA.ipynb) | Task | Kleinman et al. (2025) | `multiarea_rnn` (+ manual `constrained_rnn` masks) · checkerboard task · information bottleneck across 3 areas · dPCA · W21/W32 SVD alignment |
| [cognitive_tasks](cognitive_tasks.ipynb) | Tutorial | — | Visualize all built-in cognitive tasks (inputs / targets / masks) |
| [objectives](objectives.ipynb) | Tutorial | — | `Objective` layer, built-in objectives, loss terms, custom objectives, `build_objective` |
| [quickstart](quickstart.ipynb) | Tutorial | — | quick start of two paradigms |

In an offline environment, for datasets that need to be downloaded (such as Lorenz63), please manually place the files into the cache directory as instructed by `src/neuralrnn/data/download.py` (default: `~/.cache/neuralrnn/datasets`, or set `NEURALRNN_CACHE`).
