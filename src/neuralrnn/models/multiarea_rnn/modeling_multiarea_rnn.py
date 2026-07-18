"""Multi-area RNN model implementation.

All dynamics are inherited from ConstrainedRNNModel (Euler CTRNN with hard
masks and Dale constraints). This class only translates the area-level
hyperparameters of MultiAreaRNNConfig into block-structured masks plus a
per-area Dale sign vector, following the same base-config injection pattern as
SparseRNNModel / ModularRNNModel.

Reference:
    Kleinman, M., Wang, T., Xiao, D., ... Chandrasekaran, C., Kao, J.C., 2025.
    The information bottleneck as a principle underlying multi-area cortical
    representations during decision-making. eLife (reviewed preprint).
    DOI: 10.7554/eLife.89369.2
"""
from __future__ import annotations

import torch

from ...auto.modeling_auto import register_model
from ..constrained_rnn.configuration_constrained_rnn import ConstrainedRNNConfig
from ..constrained_rnn.modeling_constrained_rnn import ConstrainedRNNModel
from .configuration_multiarea_rnn import MultiAreaRNNConfig
from .masks import area_slices, build_multiarea_masks


@torch.no_grad()
def rescale_effective_spectral_radius(model, target: float, iters: int = 100) -> None:
    """Rescale model.h2h so the effective recurrent matrix has spectral radius ~target.

    Under Dale's law the effective matrix is |W| @ diag(signs), whose
    positive-mean magnitudes create a large outlier eigenvalue (framework
    default init explodes over long trials); rescaling keeps the initial
    dynamics in the stable-but-active regime. Works with any ConstrainedRNNModel
    (used by MultiAreaRNNModel and by the manual-mask construction path).
    """
    W_eff = model._recurrent_weight()
    M = W_eff.shape[0]
    v = torch.ones(M, 1, device=W_eff.device, dtype=W_eff.dtype)
    v = v / v.norm()
    for _ in range(iters):
        v = W_eff @ v
        v = v / v.norm().clamp_min(1e-12)
    sr = (v.T @ W_eff @ v).item()
    if sr > 1e-12:
        model.h2h.weight.mul_(target / sr)
        if hasattr(model, "_apply_masks_to_weights"):
            model._apply_masks_to_weights()


@register_model("multiarea_rnn")
class MultiAreaRNNModel(ConstrainedRNNModel):
    """Cascaded multi-area CTRNN (dense intra-area, sparse E-source inter-area)."""

    config_class = MultiAreaRNNConfig

    def __init__(self, config: MultiAreaRNNConfig) -> None:
        rec_mask, in_mask, out_mask, dale_signs = build_multiarea_masks(
            area_sizes=config.area_sizes,
            input_dim=config.input_dim,
            output_dim=config.output_dim,
            ei_ratio=config.ei_ratio,
            intra_density=config.intra_density,
            ff_ee_density=config.ff_ee_density,
            ff_ei_density=config.ff_ei_density,
            fb_density=config.fb_density,
            fb_ei_density=config.fb_ei_density,
            input_areas=config.input_areas,
            input_e_only=config.input_e_only,
            output_area=config.output_area,
            output_e_only=config.output_e_only,
            allow_self_connections=config.allow_self_connections,
            mask_seed=config.mask_seed,
        )

        base_config_dict = config.to_dict()
        base_config_dict["rec_mask"] = rec_mask
        base_config_dict["in_mask"] = in_mask
        base_config_dict["out_mask"] = out_mask
        base_config_dict["dale_signs"] = dale_signs.tolist()
        base_config = ConstrainedRNNConfig(**base_config_dict)

        super().__init__(base_config)

        # Restore the variant config so save_pretrained serializes the
        # area-level hyperparameters; masks are rebuilt from mask_seed on load.
        self.config = config
        self.area_slices = area_slices(config.area_sizes)

        if config.rec_spectral_radius is not None:
            rescale_effective_spectral_radius(
                self, float(config.rec_spectral_radius),
                config.spectral_norm_iters)

    def area_states(self, states: torch.Tensor, area: int) -> torch.Tensor:
        """Slice (..., M) states down to one area's units."""
        return states[..., self.area_slices[int(area) % len(self.area_slices)]]
