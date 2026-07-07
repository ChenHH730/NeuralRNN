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
| [10a_constrained_RNN_paradigmB.ipynb](10a_constrained_RNN_paradigmB.ipynb) | Reconstruction | Beiran & Litwin-Kumar (2025) | `connectome_rnn` · teacher-student · recordings break degeneracy · stiff/sloppy modes · optimal neuron selection |
| [11_STP_RNN_paradigmA.ipynb](11_STP_RNN_paradigmA.ipynb) | Task | Masse et al. (2019) | inline STP-RNN · DMS · DRMS · ABBA · activity-silent maintenance · manipulation-driven persistent activity |

## environment

```bash
pip install -e .            # torch/numpy/scipy/safetensors
pip install -e '.[viz]'     # matplotlib/jupyter
pip install -e '.[neurogym]'  # neurogym
```

无网环境下，需要下载的数据集（如 Lorenz63）请先按 `src/neuralrnn/data/download.py`
的提示把文件手动放入缓存目录（默认 `~/.cache/neuralrnn/datasets`，或设 `NEURALRNN_CACHE`）。
