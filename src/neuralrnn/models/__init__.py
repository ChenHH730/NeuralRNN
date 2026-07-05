"""Model zoo.

To keep optional heavy dependencies lazily loaded, this package does not import
individual model modules at import time. Use either:
  - cross-family unified construction: neuralrnn.AutoModel / AutoConfig (recommended);
  - or explicit family imports: from neuralrnn.models.ctrnn import CTRNNModel, CTRNNConfig.

Built-in reference implementations: ctrnn (Paradigm A), plrnn (Paradigm B),
latent_circuit (Paradigm A), lowrank (Paradigm A).
To be ported: lfads (see docs/PORTING_GUIDE.md).
"""
