"""Low-rank RNN model family.

Low-rank recurrent neural networks where the recurrent connectivity matrix
is parametrized as W_rec = m @ n^T / N (rank-r factorization).

References:
    - Dubreuil, A., Valente, A., Beiran, M., Mastrogiuseppe, F., & Ostojic, S. (2022).
      The role of population structure in computations through neural dynamics.
      Nature Neuroscience, 25, 783-794.
    - Valente, A., Ostojic, S., & Bhatt, D. (2022).
      Extracting computational mechanisms from neural data using low-rank RNNs.
      NeurIPS.
"""
from .configuration_lowrank import LowrankRNNConfig
from .modeling_lowrank import LowrankRNNModel

__all__ = ["LowrankRNNConfig", "LowrankRNNModel"]
