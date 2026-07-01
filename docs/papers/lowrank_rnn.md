# Low-Rank RNN

> **Paradigm**: A/B (task optimization + LINT inference from trajectories)  
> **Original repositories**: [Dubreuil et al. (2022)](https://github.com/immersive-limit/low_rank_rnns) / Valente et al. (2022) low-rank inference code  
> **Framework target**: `models/lowrank` + `SupervisedObjective`  
> **Status**: ✅ Ready

## 1. What problem it solves

Low-rank RNNs constrain the recurrent weight matrix to a low-rank structure $W^{rec} = m \, n^T / N$, where $m, n \in \mathbb{R}^{N \times R}$ and $R \ll N$. This:

- Reduces recurrent parameters from $N^2$ to $2NR$.
- Makes the dynamics transparent: activity projected onto the $R$ columns of $m$ reveals the low-dimensional computation.
- Enables population-structure analysis: neurons cluster by their connectivity vectors $(m_i, n_i, w^{in}_i, w^{out}_i)$.
- Allows resampling of new networks from fitted population statistics.

Two papers are covered:

- **Dubreuil et al. (2022), *Nature Neuroscience***: task-optimized low-rank RNNs for context-dependent decision making (Mante, RDM, Romo, Raposo, DMS tasks).
- **Valente et al. (2022), *NeurIPS***: inferring low-rank dynamics from neural data (Paradigm B).

This port covers **both paradigms**. Paradigm A uses the framework `LowrankRNNModel` directly. Paradigm B additionally requires a full-rank "teacher" network. The reference `FullRankRNN` keeps separate membrane potentials and firing rates, uses `scale_by_hidden_size=False`, and has per-channel input/output scaling (`si`/`so`). These details differ from the existing `CTRNNModel`, so the full-rank network is implemented notebook-locally in `08_lowrank_RNN_paradigmB.ipynb` to guarantee an exact reproduction of the teacher dynamics. The LINT fitting protocol also uses two small notebook-local training loops: one for the task-trained full-rank teacher and one for fitting the low-rank network to the teacher's firing-rate trajectories with an identity readout.

A companion notebook, `08b_lowrank_RNN_paradigmB.ipynb`, demonstrates that the **same generic `Trainer` + `SupervisedObjective(regression)` pipeline** can perform LINT when the teacher is a standard framework `CTRNNModel`. The trick is to set the low-rank network's `output_dim = latent_dim`, fix its readout to the identity (`wo = N * I`), and supply the teacher's firing-rate trajectories as regression targets. No custom training loop and no framework modifications are required.

## 2. Core method

### Dynamics (Euler discretization)

$$
\begin{aligned}
r_t &= \phi(z_t + b) \\
\text{rec}_t &= r_t \, n \, m^T / N \\
z_{t+1} &= z_t + \sigma \xi_t + \alpha\left(-z_t + \text{rec}_t + x_t W^{in}_{\text{full}}\right) \\
y_t &= \phi_{\text{out}}(z_t) \, W^{out}_{\text{full}} / N
\end{aligned}
$$

where $\alpha = dt/\tau$, $\phi$ is `tanh` or `relu`, and $W^{in}_{\text{full}}$, $W^{out}_{\text{full}}$ apply per-channel scaling (`si`, `so`).

### Key hyperparameters

| Param | Default | Meaning |
|---|---|---|
| `latent_dim` (`N`) | 500 | Number of hidden neurons |
| `rank` (`R`) | 1 | Rank of $W^{rec}$ |
| `alpha` | 0.2 | Euler step $dt/\tau$ |
| `noise_std` | 0.05 | Recurrent Gaussian noise |
| `activation` | `"tanh"` | Hidden activation |
| `output_activation` | `"tanh"` | Readout activation |
| `scale_by_hidden_size` | `True` | Divide recurrent/output by $N$ |

## 3. How to use this method in our framework

| Original code | Framework API | Note |
|---|---|---|
| `low_rank_rnns.modules.LowRankRNN` | `models/lowrank/modeling_lowrank.py: LowrankRNNModel` | Direct port |
| Task generators in `low_rank_rnns/tasks/` | `neuralrnn.data.tasks.*` | All 5 tasks registered |
| MSE regression training | `SupervisedObjective(task_type='regression')` + `Trainer` | |
| `make_vecs()`, `gmm_fit()` | `analysis/population_structure.py` | Population clustering |
| `overlap_matrix()`, `gram_factorization()` | `analysis/linalg_utils.py` | Linear algebra helpers |
| `SupportLowRankRNN` | **Not ported** | `to_support_net()` in notebook is a stub |

### Quick example

```python
from neuralrnn import AutoConfig, AutoModel, Trainer, TrainingArguments
from neuralrnn import SupervisedObjective, load_dataset

# Load task data
ds = load_dataset('rdm', batch_size=32, num_trials=800, seed=42)

# Create rank-1 low-rank RNN
cfg = AutoConfig.for_model('lowrank_rnn',
    input_dim=1, latent_dim=256, output_dim=1,
    rank=1, alpha=0.2, noise_std=0.05, activation='tanh')
model = AutoModel.from_config(cfg)

# Train
Trainer(model, ds, SupervisedObjective(task_type='regression'),
        TrainingArguments(max_steps=500, learning_rate=5e-3)).train()

# Save / load
model.save_pretrained('models/lowrank_rdm/')
model2 = AutoModel.from_pretrained('models/lowrank_rdm/')
```

## 4. Consistency with the original implementation

- Architecture verified by comparing forward rollouts on identical task inputs.
- `forward(return_dynamics=True)` returns `(output, trajectories)` to match reference analysis code.
- `scale_by_hidden_size=True` matches the reference scaling.
- `svd_reparametrization()` orthogonalizes $m$ and $n$ via SVD of $m \, n^T$ before vector-field / fixed-point analysis.
- Tensors are forced contiguous in `save_pretrained()` to avoid safetensors errors.

## 5. Reproduction experiments

### 5.1 Paradigm A: task-optimized low-rank RNNs

See `notebook/07_lowrank_RNN_paradigmA.ipynb`:

1. **Tutorial**: Rank-1 RDM training, connectivity-vector scatter plots, m-subspace trajectory projection, save/load.
2. **Mante full analysis**: GMM population clustering, psychometric matrices, gain analysis, effective input, overlap matrices, state-space trajectories.
3. **All 5 tasks**: RDM, Romo, Raposo, DMS, Mante — accuracy should be near ceiling.
4. **Rank comparison**: Rank 1/2/3 on the oscillator dataset from `03_custom_pipeline.ipynb`.

### 5.2 Paradigm B: LINT inference from full-rank trajectories

See `notebook/08_lowrank_RNN_paradigmB.ipynb`:

1. **Full-rank teacher**: Train a 1024-unit `FullRankRNN` on the Mante task. The reference `FullRankRNN` is implemented notebook-locally because its dynamics (separate membrane potential and firing rate with `scale_by_hidden_size=False`) are not identical to the existing CTRNN implementation.
2. **LINT fitting**: Fit rank-$r$ `LowrankRNNModel` networks to $\tanh(h^{FR})$ trajectories using an identity readout, then plug in the full-rank readout for task evaluation.
3. **Rank scan**: Compute state-space $R^2$ and accuracy for both truncated-SVD and LINT-fitted networks for ranks 1 to 14, reproducing the first panel of Valente et al. (2022) Figure 3. Ranks 1–14 are scanned in the notebook so the curves match the reference exactly.
4. **Downstream analysis**: Population clustering, connectivity overlap, PCA/TDR comparisons, inactivation experiments, $\kappa_1$ trajectory comparisons, and gain distributions.

See `notebook/08b_lowrank_RNN_paradigmB.ipynb` for the **standard-Trainer** variant:

1. **CTRNN teacher**: Train a 512-unit `CTRNNModel` on the Mante task using the framework `Trainer` + `SupervisedObjective(regression)`.
2. **LINT with `Trainer`**: Fit rank-$r$ `LowrankRNNModel` networks to the CTRNN's firing-rate trajectories using the *same* `Trainer` + `SupervisedObjective(regression)`. The only adjustments are setting `output_dim = latent_dim` and freezing the readout to the identity (`wo = N \cdot I`).
3. **Rank scan**: Compute state-space $R^2$ and task accuracy after transferring the CTRNN readout. The notebook keeps `rank_max = 1` as requested, and the rank-1 LINT network already reaches high $R^2$ and near-ceiling accuracy.
4. **Downstream analysis**: The same full suite as `08_lowrank_RNN_paradigmB.ipynb` is reproduced: population clustering, SVD-vs-fitted and TDR-vs-fitted connectivity overlap matrices, PCA cumulative variance and $R^2$ vs. kept PCs, TDR projections (both TDR axes and LINT connectivity axes for both contexts), inactivation experiments on populations 0/1/2 in both networks, single-trial $\kappa_1$ trajectory comparisons, and context-dependent gain distributions.

## 6. Known limitations

- `SupportLowRankRNN` / network resampling is **not yet ported** into the neuralrnn framework. The notebook stub `to_support_net()` in `07_lowrank_RNN_paradigmA.ipynb` explains this and raises `NotImplementedError` if called.

## 7. References

- Dubreuil, A., Valente, A., Beiran, M., Mastrogiuseppe, F., & Ostojic, S. (2022). The role of population structure in computations through neural dynamics. *Nature Neuroscience*, 25, 783-794.
- Valente, A., Ostojic, S., & Bhatt, D. (2022). Extracting computational mechanisms from neural data using low-rank RNNs. *NeurIPS*.
