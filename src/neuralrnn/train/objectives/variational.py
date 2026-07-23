"""Variational objective ELBO (LFADS paradigm, placeholder skeleton).

Corresponds to reference_project/LFADS-torch: sequential variational autoencoder with
loss = reconstruction log-likelihood (Poisson / Gaussian) - KL(posterior || prior).
When porting (PORTING_GUIDE recipe 5), let the LFADS model return posterior distribution
parameters and reconstruction rates in DynamicsModelOutput.extras, and this objective assembles the ELBO.

Convention: model(batch["inputs"]).extras must contain at least
    {"rates": (B,T,N), "posterior": <distribution or its parameters>, "prior": <distribution or its parameters>}
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import Objective
from .registry import register_objective
from ...modeling_utils import NeuralDynamicsModel


@register_objective("variational")
class VariationalObjective(Objective):
    """ELBO objective (LFADS paradigm, placeholder — model side not yet ported).

    loss = recon_NLL(rates, targets) + kl_weight * KL. Expects the model to
    return "rates" (B,T,N) and optionally a scalar "kl" in output.extras.
    """

    def __init__(self, kl_weight: float = 1.0, likelihood: str = "poisson"):
        """kl_weight: KL annealing weight. likelihood: "poisson" / "gaussian"."""
        self.kl_weight = float(kl_weight)
        self.likelihood = likelihood          # "poisson" / "gaussian"

    def _recon_nll(self, rates: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Reconstruction negative log-likelihood; (B,T,N) tensors -> scalar."""
        if self.likelihood == "poisson":
            # Negative Poisson log-likelihood (rates are intensities)
            return F.poisson_nll_loss(rates, target, log_input=False, full=False,
                                      reduction="mean")
        return F.mse_loss(rates, target)

    def compute_loss(self, model: NeuralDynamicsModel, batch):
        """Batch keys: "inputs" (B,T,K), "targets" (B,T,N).
        Returns (loss, {"loss", "recon", "kl", "kl_weight"})."""
        out = model(batch["inputs"])
        extras = out.extras or {}
        if "rates" not in extras:
            raise RuntimeError(
                "VariationalObjective requires the model to provide 'rates'/'kl' in output.extras. "
                "Follow PORTING_GUIDE recipe 5 to implement the LFADS model forward."
            )
        target = batch["targets"]
        recon = self._recon_nll(extras["rates"], target)
        # KL: prefer a scalar kl already computed by the model; otherwise expect posterior/prior to compute here (fill in when porting)
        kl = extras.get("kl")
        if kl is None:
            kl = torch.zeros((), device=recon.device)  # TODO(port): compute from posterior / prior
        loss = recon + self.kl_weight * kl
        return loss, {"loss": loss.item(), "recon": float(recon),
                      "kl": float(kl), "kl_weight": self.kl_weight}
