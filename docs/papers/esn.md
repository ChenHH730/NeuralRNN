# Echo-State Network / Critical Initialization

**Reference**: Pachitariu, Zhong, Gracias, Minisi, Lopez & Stringer (2026). *A critical initialization for biological neural networks*. Nature. https://doi.org/10.1038/s41586-026-10528-1

**Code reference**: `reference_project/randomRNN/critical_init/`

**Framework target**: `models/ctrnn` + unified `freeze_*` API.

---

## What problem it addresses

Echo-state networks (ESNs) and reservoir computing show that a randomly initialized recurrent network (the *reservoir*) can solve interesting temporal tasks if only its readout weights are trained. The practical question is how to initialize the reservoir so that its hidden dynamics are rich enough to encode the task, while remaining stable enough to be read out reliably.

The `critical_init` paper demonstrates that large random recurrent networks, when their recurrent connectivity is scaled so that the largest eigenvalue is just below 1, produce power-law covariance spectra and long intrinsic timescales. These properties make them strong reservoirs.

---

## Core method

1. **Freeze input and recurrent parameters** via the framework's `freeze_*` config flags:
   ```python
   cfg = AutoConfig.for_model(
       "ctrnn", ...,
       freeze_input=True, freeze_recurrent=True, freeze_output=False
   )
   ```
2. **Initialize the recurrent weight matrix at criticality**:
   - Draw a random matrix $A$.
   - Symmetrize, zero the diagonal.
   - Scale by $\sqrt{N} \cdot \mathrm{std}(A)$ and divide by 2 (symmetric case).
   - Normalize so $\max_i \mathrm{Re}[\lambda_i(A)] \approx 0.998$.
3. **Train only the readout layer** with `SupervisedObjective` and the generic `Trainer`.

The framework's `Trainer` optimizes `model.parameters()`; PyTorch automatically skips parameters whose `requires_grad=False`, so no special trainer logic is required.

---

## Framework mapping

| Concept | NeuralRNN component |
|---|---|
| Reservoir | `CTRNNModel.h2h` (frozen) |
| Input projection | `CTRNNModel.input2h` (frozen) |
| Readout | `CTRNNModel.readout_layer` (trainable) |
| Freezing API | `NeuralRNNConfig.freeze_*` + `NeuralDynamicsModel.freeze_parameters` |
| Critical init helper | Notebook-local `critical_init_recurrent()` (see `09_echo_state_network_paradigmA.ipynb`) |

---

## Key diff-test points

- With a properly critical reservoir, readout-only training reaches high accuracy on perceptual decision-making.
- If the recurrent spectral radius is too small, activity decays quickly and accuracy drops toward chance; if it is too large, dynamics become unstable and readout noise dominates (see the `emax = 0.1, 0.5, 1, 2, 5` comparison in the tutorial).
- Fixed-point analysis under the 0-coherence input reveals the dynamical scaffold used by the readout.
- The `freeze_*` flags must survive `save_pretrained` / `from_pretrained` roundtrips.

---

## Files

- Tutorial: `notebook/09_echo_state_network_paradigmA.ipynb` (all English; includes readout-only training, fixed-point analysis, and the `emax` comparison bar plot)
- Summary of the reference project: `reference_project/randomRNN/critical_init/SUMMARY.md`
- Freeze API tests: `test/test_freeze.py`
