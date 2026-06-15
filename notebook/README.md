# Tutorial Notebook

Every notebook is to implement a proposed method from previous works, with the shared architecture of NeuralRNN (using `AutoModel` / `Trainer` / `analysis`).

| Notebook | Paradigm | Reference | Key API |
|---|---|---|---|
| [01_ctrnn_fixedpoints_paradigmA.ipynb](02_ctrnn_fixedpoints_paradigmA.ipynb) | Task | Song et al. (2016) | `SupervisedObjective` · `find_fixed_points` · `fit_pca` · `linearize` · `dominant_direction` · `Line attractor`|
| [02_plrnn_reconstruction_paradigmB.ipynb](01_plrnn_reconstruction_paradigmB.ipynb) | Reconstruction | Durstewitz et al. (2023) | `TeacherForcingObjective` · `find_fixed_points` · `max_lyapunov_exponent`|
| [03_custom_pipeline.ipynb](03_custom_pipeline.ipynb) | Task | None | `CustomDataset.from_arrays` · `SupervisedObjective`|
| [04_EIRNN_paradigmA.ipynb](04_EIRNN_paradigmA.ipynb) | Task | Song et al. (2016) | `ei_rnn` |
| [05_latent_circuit_paradigmA.ipynb](05_latent_circuit_paradigmA.ipynb) | Reconstruction | Langdon & Engel (2025) | `latent_circuit` · `LatentCircuitObjective` · `connection analysis` |
| [06_tiny_RNN_paradigmB.ipynb](06_tiny_RNN_paradigmB.ipynb) | Reconstruction | Ji-An et al. (2025) | `tiny_rnn` · `BehavioralObjective` |
| [07_lowrank_RNN_paradigmA.ipynb](07_lowrank_RNN_paradigmA.ipynb) | Task | Dubreuil et al. (2022) | `lowrank_rnn` · `vector field` · `GMM clustering` |

## ongoing notebook (配方见 PORTING_GUIDE)

- LFADS：用 `VariationalObjective` 做尖峰序列的潜动力推断（配方5）

## environment

```bash
pip install -e .            # torch/numpy/scipy/safetensors
pip install -e '.[viz]'     # matplotlib/jupyter
pip install -e '.[neurogym]'  # neurogym
```

无网环境下，需要下载的数据集（如 Lorenz63）请先按 `src/neuralrnn/data/download.py`
的提示把文件手动放入缓存目录（默认 `~/.cache/neuralrnn/datasets`，或设 `NEURALRNN_CACHE`）。
