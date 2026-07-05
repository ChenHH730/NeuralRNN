"""PLRNN family configurations (piecewise-linear RNN for dynamical-system reconstruction, DSR).

Reference implementation: ported from Durstewitz lab's CNS2023_tutorial.ipynb (shallowPLRNN).
This serves as the Contract-A copy template for "Paradigm B (dynamics reconstruction)"
and demonstrates configuration for models with analytic Jacobians.
dend_plrnn and alrnn share this package; each overrides recurrence/jacobian per its paper.
"""
from __future__ import annotations

from ...configuration_utils import NeuralRNNConfig


class ShallowPLRNNConfig(NeuralRNNConfig):
    """shallowPLRNN state equation:
        z_t = A z_{t-1} + W1 ReLU(W2 z_{t-1} + h2) + h1 + C s_t

    Args:
        latent_dim: Latent dimension M (usually equals observation dimension N in DSR)
        hidden_dim: Hidden dimension L (expressivity hyperparameter)
        input_dim:  External input dimension K; 0 means autonomous (C s_t omitted)
        output_dim: Readout dimension; default None -> latent_dim (DSR identity readout)
        autonomous: Whether the system is autonomous (no external input).
                    Default None -> inferred from input_dim==0;
                    explicitly setting True forces input_dim=0 for consistency.
        observation: Observation model; "identity" means direct observation of latent state (x_t = z_t)
    """

    model_type = "shallow_plrnn"

    def __init__(
        self,
        latent_dim: int = 3,
        hidden_dim: int = 50,
        input_dim: int = 0,
        output_dim: int | None = None,
        observation: str = "identity",
        autonomous: bool | None = None,
        **kwargs,
    ) -> None:
        kwargs.pop("activation", None)          # PLRNN is intrinsically ReLU; avoid conflict with fixed value below
        if autonomous is None:
            autonomous = (input_dim == 0)
        if autonomous:
            input_dim = 0                       # autonomous <=> no external input; keep invariants consistent
        if output_dim is None:
            output_dim = latent_dim             # DSR readout is usually identity
        super().__init__(input_dim=input_dim, latent_dim=latent_dim,
                         output_dim=output_dim, activation="relu", **kwargs)
        self.hidden_dim = hidden_dim
        self.observation = observation
        self.autonomous = bool(autonomous)      # store as attribute (saved in config.json for correct reload)


class DendPLRNNConfig(ShallowPLRNNConfig):
    """dendritic PLRNN (linear spline basis expansion).

    State equation:
        z_t = A z_{t-1} + W sum_b alpha_b ReLU(z_{t-1} - h_b) + h_0 + C s_t

    Args:
        n_bases: Number of basis functions B per latent unit.
        use_clipping: Whether to use the clipped basis expansion that guarantees
            bounded orbits when ||A||_2 < 1.
        clip_range: Optional hard clip range for latent states.
    """
    model_type = "dend_plrnn"

    def __init__(
        self,
        n_bases: int = 20,
        use_clipping: bool = False,
        clip_range: float | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_bases = n_bases
        self.use_clipping = use_clipping
        self.clip_range = clip_range


class ALRNNConfig(ShallowPLRNNConfig):
    """almost-linear RNN.

    State equation:
        z_t = A z_{t-1} + W Phi*(z_{t-1}) + h + C s_t
        Phi*(z) = [z_1, ..., z_{M-P}, ReLU(z_{M-P+1}), ..., ReLU(z_M)]

    Args:
        n_linear: Number of linear (non-ReLU) units; the remaining
            latent_dim - n_linear units are ReLU.
        use_clipping: Whether to clip latent states.
        clip_range: Optional hard clip range for latent states.
    """
    model_type = "alrnn"

    def __init__(
        self,
        n_linear: int = 1,
        use_clipping: bool = False,
        clip_range: float | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_linear = n_linear
        self.use_clipping = use_clipping
        self.clip_range = clip_range
