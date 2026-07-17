# NeuralRNN

---

**NeuralRNN is A unified framework for implementing RNN methods in cognitive neuroscience** — bringing two major paradigms under a single [Transformers](https://github.com/huggingface/transformers)-style interface:

- **Paradigm A: Task-Based Optimization (TBO)**[^1]: Train RNNs on cognitive tasks, then reverse-engineer how they perform computation using analyses including fixed points, vector fields, dimensionality reduction, etc. The goal is to use RNNs as a proxy for cognitive computation.
- **Paradigm B: Dynamical System Reconstruction (DSR)**[^2][^3]: Fit generative RNNs directly from neural/behavioral time series that can reproduce attractors, power spectra, and Lyapunov spectra of the target system.

Both paradigms share a unified set of `model config`, `Trainer`, and `analysis` tools. **The only difference between the two paradigms is the `Objective`**: Paradigm A aims to optimize output for cognitive task performance, while Paradigm B aims to construct a dynamical system isomorphic to the target neural activity. Moreover, DSR can also be applied to reconstruct the dynamics of TBO-trained models for interpretability analysis[^4].

---

## Core Concept

All models are viewed as "discrete dynamical systems with downstream readout" $z_t=F_\theta(z_{t-1},x_t),\;y_t=G_\phi(z_t)$.

A model only needs to implement two methods:

```python
def recurrence(self, x_t, z_prev, *, inputs=None): ...  # single-step transition F
def readout(self, z_t): ...                              # readout G
```

NeuralRNN provides an interface to automatically connect the model to the unified trainer and all analysis tools.

## What NeuralRNN contains and not

NeuralRNN contains the pipeline to model RNN for neuroscience research, including (1) constructing dataset, (2) building and configuring RNN models, (3) training, (4) model analysis, and (5) **selective layer freezing for echo-state / reservoir-computing style training** (see the full pipeline in **[`custom pipeline`](notebook/03_custom_pipeline.ipynb)**  and the documents in **[`docs`](docs/README.md)**). We also provide the guide to implement each built-in model through this framework (see **[`notebook`](notebook/README.md)**).

However, there are others methods using dynamical system methods as well, including [MARBLE](https://www.nature.com/articles/s41592-024-02582-2), [FINDR](https://www.nature.com/articles/s41586-025-09528-4), [neuralflow](https://www.nature.com/articles/s41586-025-09199-1), and [SSMLearn](https://arxiv.org/abs/2510.13519). These model-agnostic methods aim to inference interpretable representations of neural population dynamics exactly from the neural response, which are not included in NeuralRNN but can be suitably combined for the further analysis of RNN models.


## Install

```bash
$ git clone https://github.com/ChenHH730/NeuralRNN.git
$ cd NeuralRNN
$ pip install -e .
```


## Quickstart

```python
from neuralrnn import AutoConfig, AutoModel, Trainer, TrainingArguments
from neuralrnn import TeacherForcingObjective, load_dataset

# 1) dataset
# use registered dataset or custom dataset
ds = load_dataset("lorenz63", sequence_length=200, batch_size=16, normalize=True) 

# 2) model (config) + objective (based on the paradigm) + training
cfg = AutoConfig.for_model("shallow_plrnn", input_dim=0, latent_dim=3,
        output_dim=3, hidden_dim=50, autonomous=True)  # model config
model = AutoModel.from_config(cfg)  # load model
Trainer(model, ds, TeacherForcingObjective(alpha=0.1),
        TrainingArguments(max_steps=2000)).train()  # train model

# 3) save and load (config.json + model.safetensors)
model.save_pretrained("ckpt/")
model = AutoModel.from_pretrained("ckpt/")

# 4) analysis (model agnostic)
from neuralrnn.analysis import find_fixed_points, max_lyapunov_exponent
fps = find_fixed_points(model)
```

## Content Structure

```
src/neuralrnn/
  configuration_utils.py   modeling_utils.py     # core contracts (Config / Model base classes)
  auto/                    # AutoConfig / AutoModel registration & dispatch
  models/                  # model zoo: ctrnn, ei_rnn, lowrank_rnn (Paradigm A), plrnn (Paradigm B), latent_circuit, tiny_rnn
  data/                    # unified batching, datasets, open data registry + download cache
  train/                   # generic Trainer + paradigm Objectives + reusable loss terms /
                           #   regularizers / metrics + nested cross-validation
  analysis/                # fixed points / linearization / vector fields / dim reduction /
                           #   Lyapunov / D_stsp, D_H / manifold / sequentiality
  inputs/  tools/          # reserved
docs/                      # ARCHITECTURE.md · PORTING_GUIDE.md · theory/ · papers/
notebook/                  # end-to-end tutorials for each paper
```

## Built-in Models

| Model | Paradigm | Status (mostly used) |
|---|---|---|
| continuous time RNN | A | ✅ |
| E-I RNN (Dale's principle) | A | ✅ |
| Latent Circuit | B | ✅ |
| piecewise linear RNN | B | ✅ |
| Tiny RNN | B | ✅ |
| low-rank RNN | AB | ✅ |
| constrained RNN | A | ✅ |
| seRNN | A | ✅ |
| connectome-constrained RNN | B | ✅ |

## Porting New Papers into the Framework

See [docs/PORTING_GUIDE.md](docs/PORTING_GUIDE.md) for the complete AI-assisted porting manual: four adapter contracts, the universal 8-step workflow with hard rules, per-paper recipes for 8 papers, copy-paste-ready AI prompt templates, a porting kanban board, and a table of common pitfalls.

Core principle: **Porting = writing adapters (wrapping + verification), not rewriting mathematics**. Any model that implements `recurrence/readout` is plug-and-play; the analysis layer works only through the model's public contract and never imports specific model classes.

## Design Docs

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — overall design and architecture (**read first**)
- [docs/PORTING_GUIDE.md](docs/PORTING_GUIDE.md) — AI-assisted porting manual
- [docs/theory/dynamical_systems.md](docs/theory/dynamical_systems.md) — unified mathematical perspective
- [docs/papers/](docs/papers/) — per-paper method notes

## License

MIT, see [LICENSE](LICENSE). Original code of ported papers belongs to their respective authors; please follow their individual licenses when porting.

## References

[^1]: [Training Excitatory-Inhibitory Recurrent Neural Networks for Cognitive Tasks](https://doi.org/10.1371/journal.pcbi.1004792). 
Project: https://github.com/gyyang/nn-brain

[^2]: [Reconstructing computational dynamics from neural measurements with RNN](https://www.nature.com/articles/s41583-023-00740-7)
Project: https://github.com/DurstewitzLab/CNS-2023

[^3]: [Discovering cognitive strategies with tiny-RNN](https://www.nature.com/articles/s41586-025-09142-4) 
Project: https://github.com/jil095/tinyRNN

[^4]: https://github.com/engellab/latentcircuit

[^5]: https://github.com/Dynamics-of-Neural-Systems-Lab/MARBLE

[^6]: https://github.com/NN4Neurosim/nn4n https://nn4n.org/
[^7]: 