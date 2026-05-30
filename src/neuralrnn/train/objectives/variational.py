"""变分目标 ELBO（LFADS 范式，占位骨架）。

对应 reference_project/LFADS-torch：序列变分自编码器，损失 = 重构对数似然(泊松/高斯)
- KL(后验 || 先验)。移植时（PORTING_GUIDE 配方5）让 LFADS 模型在 forward 的
DynamicsModelOutput.extras 里返回后验分布参数与重构率，由本目标组装 ELBO。

约定：model(batch["inputs"]).extras 至少含
    {"rates": (B,T,N), "posterior": <分布或其参数>, "prior": <分布或其参数>}
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import Objective
from ...modeling_utils import NeuralDynamicsModel


class VariationalObjective(Objective):
    def __init__(self, kl_weight: float = 1.0, likelihood: str = "poisson"):
        self.kl_weight = float(kl_weight)
        self.likelihood = likelihood          # "poisson" / "gaussian"

    def _recon_nll(self, rates: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.likelihood == "poisson":
            # 负泊松对数似然（rates 为强度）
            return F.poisson_nll_loss(rates, target, log_input=False, full=False,
                                      reduction="mean")
        return F.mse_loss(rates, target)

    def compute_loss(self, model: NeuralDynamicsModel, batch):
        out = model(batch["inputs"])
        extras = out.extras or {}
        if "rates" not in extras:
            raise RuntimeError(
                "VariationalObjective 需要模型在 output.extras 提供 'rates'/'kl'。"
                "请按 PORTING_GUIDE 配方5 实现 LFADS 模型的 forward。"
            )
        target = batch["targets"]
        recon = self._recon_nll(extras["rates"], target)
        # KL：优先用模型已算好的标量 kl；否则期望提供 posterior/prior 让此处计算（移植时补全）
        kl = extras.get("kl")
        if kl is None:
            kl = torch.zeros((), device=recon.device)  # TODO(移植): 由 posterior/prior 计算
        loss = recon + self.kl_weight * kl
        return loss, {"loss": loss.item(), "recon": float(recon),
                      "kl": float(kl), "kl_weight": self.kl_weight}
