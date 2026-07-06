"""Constrained RNN family implementations.

All models inherit from ConstrainedRNNModel (which itself inherits CTRNNModel).
The base class supports arbitrary hard masks on input/recurrent/output weights;
derived classes generate common masks (spatial, sparse, modular) and expose
specialized regularizers / utilities.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...auto.modeling_auto import register_model
from ..ctrnn.modeling_ctrnn import CTRNNModel
from .configuration_constrained_rnn import (
    ConstrainedRNNConfig,
    ModularRNNConfig,
    SERNNConfig,
    SparseRNNConfig,
)


def _to_tensor(x: Any, dtype: torch.dtype = torch.float32) -> torch.Tensor | None:
    """Convert mask config value (list/array/Tensor/None) to a torch tensor."""
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.to(dtype)
    return torch.as_tensor(np.asarray(x), dtype=dtype)


@register_model("constrained_rnn")
class ConstrainedRNNModel(CTRNNModel):
    """CTRNN with hard structural masks on input/recurrent/output weights.

    Masks are registered as non-trainable buffers. In recurrence, the recurrent
    weight is first multiplied by ``rec_mask``; similarly ``in_mask`` and
    ``out_mask`` gate the input and readout weights. Because masking is an
    elementwise multiplication by zero, masked positions receive no gradient and
    remain zero.
    """

    config_class = ConstrainedRNNConfig

    def __init__(self, config: ConstrainedRNNConfig) -> None:
        # CTRNNModel __init__ builds layers and calls apply_freeze_config
        super().__init__(config)
        M = config.latent_dim

        # Register optional masks as buffers (None mask handled by helper below)
        rec_mask = _to_tensor(config.rec_mask)
        in_mask = _to_tensor(config.in_mask)
        out_mask = _to_tensor(config.out_mask)

        if rec_mask is not None:
            if rec_mask.shape != (M, M):
                raise ValueError(
                    f"rec_mask must have shape ({M}, {M}), got {tuple(rec_mask.shape)}"
                )
            self.register_buffer("rec_mask", rec_mask)
        else:
            self.rec_mask = None

        if in_mask is not None:
            expected = (config.input_dim, M)
            if in_mask.shape != expected:
                raise ValueError(
                    f"in_mask must have shape {expected}, got {tuple(in_mask.shape)}"
                )
            self.register_buffer("in_mask", in_mask)
        else:
            self.in_mask = None

        if out_mask is not None:
            expected = (M, config.output_dim)
            if out_mask.shape != expected:
                raise ValueError(
                    f"out_mask must have shape {expected}, got {tuple(out_mask.shape)}"
                )
            self.register_buffer("out_mask", out_mask)
        else:
            self.out_mask = None

        # Zero out initial weights according to masks so that the network starts
        # with the correct connectivity pattern.
        self._apply_masks_to_weights()

    def _apply_masks_to_weights(self) -> None:
        """Project existing weights onto the allowed mask support."""
        with torch.no_grad():
            if self.rec_mask is not None:
                self.h2h.weight.copy_(self.h2h.weight * self.rec_mask)
            if self.in_mask is not None:
                # input2h.weight shape is (M, input_dim); in_mask is (input_dim, M)
                self.input2h.weight.copy_(
                    self.input2h.weight * self.in_mask.t()
                )
            if self.out_mask is not None:
                # readout_layer.weight shape is (output_dim, M); out_mask is (M, output_dim)
                self.readout_layer.weight.copy_(
                    self.readout_layer.weight * self.out_mask.t()
                )

    def _recurrent_weight(self) -> torch.Tensor:
        W = self.h2h.weight
        if self.rec_mask is not None:
            W = W * self.rec_mask
        if self.dale_mask is not None:
            W = W.abs() @ self.dale_mask
        return W

    def recurrence(self, x_t, z_prev, *, inputs=None):
        W = self._recurrent_weight()
        in_weight = self.input2h.weight
        if self.in_mask is not None:
            in_weight = in_weight * self.in_mask.t()
        pre = F.linear(x_t, in_weight, self.input2h.bias) + F.linear(
            z_prev, W, self.h2h.bias
        )
        if self.config.sigma_rec > 0 and self.training:
            if getattr(self.config, "noise_alpha_scaling", False):
                noise_std = (2 * self.alpha * self.config.sigma_rec ** 2) ** 0.5
            else:
                noise_std = self.config.sigma_rec
            pre = pre + noise_std * torch.randn_like(pre)
        if self.config.relu_after_blend:
            z = self.act((1 - self.alpha) * z_prev + self.alpha * pre)
        else:
            z = (1 - self.alpha) * z_prev + self.alpha * self.act(pre)
        return z

    def readout(self, z_t):
        out_weight = self.readout_layer.weight
        if self.out_mask is not None:
            out_weight = out_weight * self.out_mask.t()
        return F.linear(z_t, out_weight, self.readout_layer.bias)

    def constraint_loss(self) -> torch.Tensor:
        """Optional structural regularizer. Base class returns zero.

        Subclasses override to add e.g. spatial-embedding regularization.
        """
        return torch.tensor(0.0, device=next(self.parameters()).device)


@register_model("se_rnn")
class SERNNModel(ConstrainedRNNModel):
    """Spatially-embedded RNN with distance-weighted recurrent regularization.

    Units are placed on a regular grid. The recurrent weight matrix is
    regularized by a distance-weighted L1 term, optionally combined with the
    unbiased weighted communicability matrix as in the original seRNN work.
    """

    config_class = SERNNConfig

    def __init__(self, config: SERNNConfig) -> None:
        # Determine grid shape / embedding dimension
        M = config.latent_dim
        if config.grid_shape is None:
            if config.embedding_dim == 2:
                s = int(np.sqrt(M))
                if s * s != M:
                    raise ValueError(
                        f"latent_dim={M} is not a perfect square; provide grid_shape explicitly"
                    )
                grid_shape = (s, s)
            elif config.embedding_dim == 3:
                # Try a compact 3D factorization: find dims close to cube root
                grid_shape = _nearest_3d_grid(M)
            else:
                raise ValueError("embedding_dim must be 2 or 3")
        else:
            grid_shape = tuple(config.grid_shape)
            if np.prod(grid_shape) != M:
                raise ValueError(
                    f"grid_shape {grid_shape} product ({np.prod(grid_shape)}) must equal latent_dim ({M})"
                )
        self.grid_shape = grid_shape
        self.embedding_dim = len(grid_shape)

        # Build coordinate grid and distance matrix
        coords = _build_grid_coordinates(grid_shape)
        distance_matrix = _pairwise_distance_matrix(coords, config.distance_metric)
        distance_matrix = distance_matrix ** config.distance_power

        # The base config stores the distance matrix so the base __init__ registers it.
        # We pass it via a fresh ConstrainedRNN-compatible config object.
        base_config_dict = config.to_dict()
        base_config_dict["rec_mask"] = None  # seRNN does not hard-mask by default
        base_config_dict["in_mask"] = None
        base_config_dict["out_mask"] = None
        base_config = ConstrainedRNNConfig(**base_config_dict)

        super().__init__(base_config)

        # Restore original variant config so save_pretrained serializes the variant
        # hyperparameters (grid_shape, se1_weight, ...) rather than the base config.
        self.config = config

        self.se1_weight = config.se1_weight
        self.comms_factor = config.comms_factor
        self.distance_power = config.distance_power
        self.distance_metric = config.distance_metric

        self.register_buffer("coordinates", torch.from_numpy(coords).float())
        self.register_buffer("distance_matrix", torch.from_numpy(distance_matrix).float())

        if config.orthogonal_init:
            nn.init.orthogonal_(self.h2h.weight)
            self._apply_masks_to_weights()

    def constraint_loss(self) -> torch.Tensor:
        """Distance-weighted L1 (+ optional communicability) on recurrent weights."""
        W = self.h2h.weight.abs()
        if self.comms_factor > 0:
            with torch.no_grad():
                absW = W.detach()
                row_sum = absW.sum(dim=1).clamp_min(1e-12)
                D_inv_sqrt = torch.diag(row_sum ** -0.5)
                # Unbiased weighted communicability: exp(D^{-1/2} |W| D^{-1/2})
                comms = torch.linalg.matrix_exp(D_inv_sqrt @ absW @ D_inv_sqrt)
                comms.fill_diagonal_(0)
                comms = comms ** self.comms_factor
            W = W * comms.detach()
        loss = self.se1_weight * (W * self.distance_matrix).sum()
        return loss

    def get_neuron_positions(self) -> np.ndarray:
        """Return (M, embedding_dim) array of unit coordinates."""
        return self.coordinates.cpu().numpy()


@register_model("sparse_rnn")
class SparseRNNModel(ConstrainedRNNModel):
    """Sparse RNN: only a fraction ``sparsity`` of recurrent connections exist."""

    config_class = SparseRNNConfig

    def __init__(self, config: SparseRNNConfig) -> None:
        M = config.latent_dim
        rng = np.random.default_rng(config.seed)
        n_total = M * M
        n_keep = int(round(config.sparsity * n_total))
        flat_mask = np.zeros(n_total, dtype=np.float32)
        flat_mask[:n_keep] = 1.0
        rng.shuffle(flat_mask)
        rec_mask = flat_mask.reshape(M, M)
        if not config.allow_self_connections:
            np.fill_diagonal(rec_mask, 0.0)

        base_config_dict = config.to_dict()
        base_config_dict["rec_mask"] = rec_mask
        base_config_dict["in_mask"] = None
        base_config_dict["out_mask"] = None
        base_config = ConstrainedRNNConfig(**base_config_dict)

        super().__init__(base_config)
        # Restore original variant config for serialization.
        self.config = config
        self.sparsity = config.sparsity


@register_model("modular_rnn")
class ModularRNNModel(ConstrainedRNNModel):
    """Modular RNN: dense intra-module + sparse inter-module recurrent connectivity."""

    config_class = ModularRNNConfig

    def __init__(self, config: ModularRNNConfig) -> None:
        M = config.latent_dim
        n_modules = config.n_modules
        if M % n_modules != 0:
            raise ValueError(f"latent_dim ({M}) must be divisible by n_modules ({n_modules})")
        module_size = M // n_modules

        rng = np.random.default_rng(config.seed)
        rec_mask = np.zeros((M, M), dtype=np.float32)

        for i in range(n_modules):
            start = i * module_size
            end = start + module_size
            # Intra-module block
            block = rng.random((module_size, module_size)) < config.intra_density
            block = block.astype(np.float32)
            rec_mask[start:end, start:end] = block
            # Inter-module connections
            other_idx = np.concatenate(
                [np.arange(j * module_size, (j + 1) * module_size) for j in range(n_modules) if j != i]
            )
            inter = rng.random((module_size, len(other_idx))) < config.p_inter
            rec_mask[start:end, other_idx] = inter.astype(np.float32)

        if not config.allow_self_connections:
            np.fill_diagonal(rec_mask, 0.0)

        base_config_dict = config.to_dict()
        base_config_dict["rec_mask"] = rec_mask
        base_config_dict["in_mask"] = None
        base_config_dict["out_mask"] = None
        base_config = ConstrainedRNNConfig(**base_config_dict)

        super().__init__(base_config)
        # Restore original variant config for serialization.
        self.config = config
        self.n_modules = n_modules
        self.module_size = module_size


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _build_grid_coordinates(grid_shape: tuple[int, ...]) -> np.ndarray:
    """Return (M, D) array of integer grid coordinates."""
    axes = [np.arange(s) for s in grid_shape]
    grids = np.meshgrid(*axes, indexing="ij")
    coords = np.stack([g.ravel() for g in grids], axis=-1).astype(np.float32)
    return coords


def _pairwise_distance_matrix(coords: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    """Return (M, M) distance matrix from (M, D) coordinates."""
    from scipy.spatial.distance import pdist, squareform

    dist_vec = pdist(coords, metric=metric)
    return squareform(dist_vec).astype(np.float32)


def _nearest_3d_grid(M: int) -> tuple[int, int, int]:
    """Find a 3D grid shape whose product equals M and is close to cubic."""
    best = None
    best_score = float("inf")
    # Search integer factor triplets
    for a in range(1, int(np.cbrt(M)) + 2):
        if M % a != 0:
            continue
        M_ab = M // a
        for b in range(1, int(np.sqrt(M_ab)) + 2):
            if M_ab % b != 0:
                continue
            c = M_ab // b
            score = abs(a - b) + abs(b - c) + abs(a - c)
            if score < best_score:
                best_score = score
                best = (a, b, c)
    if best is None:
        raise ValueError(f"Could not find a 3D grid for latent_dim={M}")
    return best
