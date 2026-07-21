# Tutorial Notebook

Every notebook is to implement a proposed method from previous works, with the shared architecture of NeuralRNN (using `AutoModel` / `Trainer` / `analysis`).

| Notebook | Paradigm | Reference | Key API |
|---|---|---|---|
| [01_ctrnn_fixedpoints_paradigmA.ipynb](01_ctrnn_fixedpoints_paradigmA.ipynb) | Task | Song et al. (2016) | `SupervisedObjective` · `find_fixed_points` · `fit_pca` · `linearize` · `dominant_direction` · `Line attractor`|
| [02_latent_circuit_paradigmB.ipynb](02_latent_circuit_paradigmB.ipynb) | Reconstruction | Langdon & Engel (2025) | `latent_circuit` · `LatentCircuitObjective` · `connection analysis` |
| [03_custom_pipeline.ipynb](03_custom_pipeline.ipynb) | Task | None | `CustomDataset.from_arrays` · `SupervisedObjective`|
| [04_EIRNN_paradigmA.ipynb](04_EIRNN_paradigmA.ipynb) | Task | Song et al. (2016) | `ei_rnn` |
| [05_plrnn_reconstruction_paradigmB.ipynb](05_plrnn_reconstruction_paradigmB.ipynb) | Reconstruction | Durstewitz PLRNN series | `TeacherForcingObjective` · `plrnn` · `dend_plrnn` · `alrnn` |
| [06_tiny_RNN_paradigmB.ipynb](06_tiny_RNN_paradigmB.ipynb) | Behavioral Fitting | Ji-An et al. (2025) | `tiny_rnn` · `BehavioralObjective` · `output_h0=True` · original color scheme |
| [06a_test_gru.ipynb](06a_test_gru.ipynb) | Behavioral Fitting | Ji-An et al. (2025) | architecture validation with original training style + float64 experiment |
| [07_lowrank_RNN_paradigmA.ipynb](07_lowrank_RNN_paradigmA.ipynb) | Task | Dubreuil et al. (2022) | `lowrank_rnn` · `vector field` · `GMM clustering` |
| [08_lowrank_RNN_paradigmB.ipynb](08_lowrank_RNN_paradigmB.ipynb) | Reconstruction | Valente et al. (2022) | `lowrank_rnn` · LINT |
| [09_echo_state_network_paradigmA.ipynb](09_echo_state_network_paradigmA.ipynb) | Task / Reservoir | critical_init (Pachitariu et al., 2026) | `freeze_*` · critical recurrent init · readout-only training · `find_fixed_points` · `emax` comparison |
| [10_constrained_RNN_paradigmA.ipynb](10_constrained_RNN_paradigmA.ipynb) | Task | seRNN; critical_init sparse/modular | `constrained_rnn` · `se_rnn` · `sparse_rnn` · `modular_rnn` · spatial regularizer · structural masks · Mante · DelayComparison line attractor |
| [11_STP_RNN_paradigmA.ipynb](11_STP_RNN_paradigmA.ipynb) | Task | Masse et al. (2019) | STP-RNN · activity-silent maintenance · manipulation-driven persistent activity |
| [12_multitask_paradigmA.ipynb](12_multitask_paradigmA.ipynb) | Task | Yang et al. (2019) | `multitask_yang` dataset · 20-task CTRNN · task variance clusters |
| [13_flexible_multitask_paradigmA.ipynb](13_flexible_multitask_paradigmA.ipynb) | Task / Integration | Driscoll et al. (2024) | dynamic motifs |
| [14_activation_paradigmA.ipynb](14_activation_paradigmA.ipynb) | Task | Tolmachev & Engel (2025) | activation function comparison |
| [15_neural_sequence_paradigmA.ipynb](15_neural_sequence_paradigmA.ipynb) | Task | Orhan & Ma (2019); Zhou et al. (2023) Figure 3 | inline Orhan/Zhou T+WM tasks · `ctrnn` · `ei_rnn` · sequentiality index · effective dimensionality · ramp-to-sequence transition |
| [16_connectome_rnn_paradigmB.ipynb](16_connectome_rnn_paradigmB.ipynb) | Reconstruction | Beiran & Litwin-Kumar (2025) | `gain_rnn` · cycling task · teacher-student (shared J, gain/bias only) · Fig.1 readout-trained student · Fig.2 recorded-activity students (M=30/60/90/180) |
| [17_multi_area_rnn_paradigmA.ipynb](17_multi_area_rnn_paradigmA.ipynb) | Task | Kleinman et al. (2025) | `multiarea_rnn` (+ manual `constrained_rnn` masks) · checkerboard task · information bottleneck across 3 areas · dPCA · W21/W32 SVD alignment |
| [cognitive_tasks.ipynb](cognitive_tasks.ipynb) | Gallery | Built-in tasks | Visualize all built-in cognitive tasks (inputs / targets / masks) |
| [objectives](objectives.ipynb) | Tutorial | — | `Objective` layer, built-in objectives, loss terms, custom objectives, `build_objective` |

## environment

```bash
pip install -e .            # torch/numpy/scipy/safetensors
pip install -e '.[viz]'     # matplotlib/jupyter
pip install -e '.[neurogym]'  # neurogym
```

无网环境下，需要下载的数据集（如 Lorenz63）请先按 `src/neuralrnn/data/download.py`
的提示把文件手动放入缓存目录（默认 `~/.cache/neuralrnn/datasets`，或设 `NEURALRNN_CACHE`）。
