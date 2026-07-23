"""Constrained RNN family configurations.

Provides a base ConstrainedRNNConfig that lets users supply hard structural masks
on the input, recurrent, and output weights, plus configs for three common
constraint families: spatial embedding (se_rnn), sparse connectivity (sparse_rnn),
and modular connectivity (modular_rnn).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..ctrnn.configuration_ctrnn import CTRNNConfig


def _mask_to_list(x: Any) -> Any:
    """Convert a mask array/tensor/list to a JSON-serializable nested list."""
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        return list(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.generic,)):
        return x.item()
    # torch.Tensor is not imported at module load in some contexts
    if hasattr(x, "tolist"):
        return x.tolist()
    return x


class ConstrainedRNNConfig(CTRNNConfig):
    """Base config for constrained CTRNN variants.

    Extends CTRNNConfig with optional hard masks on input/recurrent/output weights.
    A mask entry of 0 means "no connection": the corresponding weight is forced to
    zero and receives no gradient. Masks are stored as buffers; None means no mask.

    Args:
        rec_mask: Optional (M, M) mask applied to recurrent weights h2h.weight.
        in_mask:  Optional (input_dim, M) mask applied to input2h.weight.
        out_mask: Optional (M, output_dim) mask applied to readout_layer.weight.
        nonlinearity_mode: See CTRNNConfig ("pre_activation" default, "post_blend", "rate").
        All CTRNNConfig arguments are inherited.
    """

    model_type = "constrained_rnn"

    def __init__(
        self,
        input_dim: int = 3,
        latent_dim: int = 64,
        output_dim: int = 3,
        dt: float | None = None,
        tau: float = 100.0,
        alpha: float | None = None,
        activation: str = "relu",
        dale: bool = False,
        ei_ratio: float = 0.8,
        trainable_h0: bool = False,
        sigma_rec: float = 0.0,
        noise_alpha_scaling: bool = False,
        rec_mask: list | Any | None = None,
        in_mask: list | Any | None = None,
        out_mask: list | Any | None = None,
        nonlinearity_mode: str = "pre_activation",
        **kwargs,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            latent_dim=latent_dim,
            output_dim=output_dim,
            dt=dt,
            tau=tau,
            alpha=alpha,
            activation=activation,
            dale=dale,
            ei_ratio=ei_ratio,
            trainable_h0=trainable_h0,
            sigma_rec=sigma_rec,
            noise_alpha_scaling=noise_alpha_scaling,
            nonlinearity_mode=nonlinearity_mode,
            **kwargs,
        )
        self.rec_mask = rec_mask
        self.in_mask = in_mask
        self.out_mask = out_mask

    def to_dict(self) -> dict[str, Any]:
        """Serialize, converting mask arrays to JSON-safe nested lists."""
        d = super().to_dict()
        d["rec_mask"] = _mask_to_list(d.get("rec_mask"))
        d["in_mask"] = _mask_to_list(d.get("in_mask"))
        d["out_mask"] = _mask_to_list(d.get("out_mask"))
        return d


class SERNNConfig(ConstrainedRNNConfig):
    """Spatially-embedded RNN (seRNN) config.

    Units are placed on a regular grid in ``embedding_dim`` dimensions (2 or 3).
    Recurrent connectivity is biased toward short-range connections via a
    distance-weighted L1 regularizer, optionally combined with weighted
    communicability as in the original seRNN paper.

    Args:
        grid_shape: Tuple defining the grid, e.g. (5, 5, 4) for 100 units in 3D
            or (10, 10) for 100 units in 2D. The product must equal latent_dim.
        embedding_dim: 2 or 3. Inferred from len(grid_shape) when None.
        distance_power: Exponent applied to distances before weighting (default 1).
        se1_weight: Coefficient lambda for the distance-weighted L1 term.
        comms_factor: If > 0, include unbiased weighted communicability term
            with this exponent. Set to 0 to use distance only.
        distance_metric: Metric passed to scipy.spatial.distance.pdist.
        orthogonal_init: If True, initialize recurrent weights with orthogonal init.
    """

    model_type = "se_rnn"

    def __init__(
        self,
        grid_shape: tuple | list | None = None,
        embedding_dim: int | None = 3,
        distance_power: float = 1.0,
        se1_weight: float = 0.5,
        comms_factor: float = 1.0,
        distance_metric: str = "euclidean",
        orthogonal_init: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.grid_shape = grid_shape
        self.embedding_dim = embedding_dim
        self.distance_power = distance_power
        self.se1_weight = se1_weight
        self.comms_factor = comms_factor
        self.distance_metric = distance_metric
        self.orthogonal_init = orthogonal_init


class SparseRNNConfig(ConstrainedRNNConfig):
    """Sparse RNN config.

    A random subset of recurrent connections is kept; the rest are hard-masked
    to zero throughout training.

    Args:
        sparsity: Fraction of recurrent connections that exist, e.g. 0.05 means
            5% connectivity. The complement (1 - sparsity) is masked to zero.
        allow_self_connections: If False, the diagonal of the recurrent mask is
            always zero (no autapses).
        seed: RNG seed for sampling the sparse mask.
    """

    model_type = "sparse_rnn"

    def __init__(
        self,
        sparsity: float = 0.1,
        allow_self_connections: bool = False,
        seed: int | None = 42,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.sparsity = sparsity
        self.allow_self_connections = allow_self_connections
        self.seed = seed


class ModularRNNConfig(ConstrainedRNNConfig):
    """Modular RNN config.

    Hidden units are partitioned into ``n_modules`` modules. Intra-module
    connections are dense (density ``intra_density``); inter-module connections
    exist with probability ``p_inter``.

    Args:
        n_modules: Number of modules. latent_dim must be divisible by n_modules.
        p_inter: Probability of an inter-module connection.
        intra_density: Density of connections inside each module (default 1.0).
        allow_self_connections: If False, removes autapses from the mask.
        seed: RNG seed for sampling the modular mask.
    """

    model_type = "modular_rnn"

    def __init__(
        self,
        n_modules: int = 4,
        p_inter: float = 0.05,
        intra_density: float = 1.0,
        allow_self_connections: bool = False,
        seed: int | None = 42,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.n_modules = n_modules
        self.p_inter = p_inter
        self.intra_density = intra_density
        self.allow_self_connections = allow_self_connections
        self.seed = seed
