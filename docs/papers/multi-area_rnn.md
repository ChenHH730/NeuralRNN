# Kleinman et al. 2025 — The Information Bottleneck as a Principle Underlying Multi-Area Cortical Representations

**Paper:** Kleinman, M., Wang, T., Xiao, D., Feghhi, E., Lee, K., Carr, N., Li, Y., Hadidi, N., Chandrasekaran, C., Kao, J.C. (2025). *The information bottleneck as a principle underlying multi-area cortical representations during decision-making.* eLife (Reviewed Preprint v2). DOI: 10.7554/eLife.89369.2

**Reference material:** `reference_project/multi-area_rnn/Kleinman_2025/` (PDF + Chinese summary). **No public code or neural data** — model-only reproduction.

**Framework entry points:**
- Model: `models/multiarea_rnn/` (`MultiAreaRNNConfig` / `MultiAreaRNNModel`), or equivalent hand-built masks on `constrained_rnn`.
- Task: `data/tasks/checkerboard_task.py` (`TASK_REGISTRY["checkerboard"]`, dataset `"checkerboard"`).
- Analysis: `analysis/demixed.py` (`fit_dpca`, `axis_svd_alignment`, `potent_null_projection`).
- Notebook: `notebook/17_multi_area_rnn_paradigmA.ipynb`.

## 1. Core question and finding

Why does cortex distribute decision-making over multiple areas? Recordings in monkeys (DLPFC -> PMd) during a checkerboard decision task show DLPFC keeps all task variables while PMd holds a **minimal sufficient representation** (only the reach direction). A multi-area RNN trained on the same task spontaneously reproduces this: the last area drops color and target-configuration information. Mechanism: the direction axis partially orthogonalizes from other variables in Area 1 and preferentially aligns with the top right singular vectors of the inter-area weights W21/W32, so it is propagated while other variables (random alignment) are attenuated across the cascade. Sufficient conditions: **>= 3 areas + Dale's law**; feedback not required.

## 2. Checkerboard task (as implemented)

- Inputs (4): left target color (-1 red / +1 green), right target color, red signed coherence (R-G)/(R+G), green signed coherence.
- 14 coherences symmetric in [-0.9, 0.9] x 2 target configurations = 28 conditions.
- Epochs: center hold ~N(200, 50^2) ms -> targets ~U[600, 1000] ms -> decision 1500 ms -> stimulus off. Eval mode fixes durations (200/800/1500/100) for alignment.
- Noise: N(0, 0.1^2) on both coherence channels during decision; recurrent noise N(0, 0.05^2).
- Targets (2 DVs, left/right reach): 0 except during decision, where the correct DV is 1. Loss mask excludes the first 200 ms of the decision epoch.
- 10% catch trials (half no input, half targets only), target 0 throughout.

## 3. Multi-area RNN (as implemented)

| Hyperparameter | Value (paper, Table 1) | Framework mapping |
|---|---|---|
| Units | 300 = 3 areas x 100 | `area_sizes=[100,100,100]` |
| tau / dt | 50 ms / 10 ms | `tau=50, dt=10` (alpha=0.2) |
| Activation | ReLU | `activation="relu"` |
| Dale | yes, 80/20 per area | `dale_signs` (per-area split) |
| Feedforward | 10% E->E, 2-5% E->I (scanned) | `ff_ee_density=0.10`, `ff_ei_density=0.02` |
| Feedback | 5% E->E | `fb_density=0.05` |
| Inter-area sources | E units only | mask construction in `masks.py` |
| Input / readout | Area 1 / Area 3 | `input_areas=[0]`, `output_area=-1`, `output_e_only=True` |
| Recurrent noise | N(0, 0.05^2) | `sigma_rec=0.1` (adjusted) |
| Optimizer / lr | Adam, 5e-5 | Adam, lr 1e-3 (5e-5 converges too slowly for a tutorial budget) |
| Loss | MSE + L2 (l=1 each, normalized) + l_Omega=2 | plain masked MSE (`SupervisedObjective`); the paper shows (its Fig. 5e) the phenomenon is robust to these regularizers |
| Stop criterion | >=65% correct per direction on 2800 CV trials | fixed 2000 steps, batch 128; decision threshold 0.7 |

## 4. INFERRED details (not stated in the paper)

These were chosen as reasonable defaults and are flagged as inferences:

1. **Recurrent initialization / spectral radius.** The paper does not state the within-area init. Framework-default init explodes under Dale (the |W| magnitudes have positive mean, producing a large outlier eigenvalue), so `MultiAreaRNNModel` rescales the effective recurrent matrix to spectral radius 1.0 at build time (`rec_spectral_radius`).
2. **L_Omega and weight-L2 terms omitted.** The paper's loss includes normalized weight L2 terms (l=1) and an Omega anti-vanishing-gradient regularizer (l_Omega=2) whose exact form is not spelled out (cited from Song et al. 2016 / Pascanu et al. 2013). The notebook trains with plain masked MSE: the paper's own robustness scan (its Fig. 5e) shows the information-bottleneck phenomenon does not depend on these terms, and in practice the network learns fine without them.
3. **Batch size, step budget, decision threshold.** The paper specifies neither batch size nor step count; the notebook uses batch 128, 2000 steps, and a 0.7 DV threshold for the decision readout.
4. **Readout restricted to Area-3 E units** (`output_e_only=True`), following the Song 2016 convention used by the group's earlier work.
5. **Decoding for usable information** uses logistic regression with 5-fold CV instead of the paper's 3-layer MLP; accuracies are similar in practice but not identical.

## 5. Reproduction scope

Model-only: Fig. 3 (psychometric/RT, per-area PC trajectories, dPCA relative variance, decoding + usable information) and Fig. 4 (axis overlap in Area 1, axis alignment with W21/W32 SVD, potent/null projections onto Area-1 W_rec). Fig. 2 and all DLPFC/PMd quantitative comparisons require the neural data, which is not available in the reference folder.

Expected qualitative results (paper): direction information is preserved across areas while color and target-config variance/decoding collapse by Area 3; the Area-1 direction axis aligns with the top W21 right singular vectors above the random baseline, the color axis does not.
