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
        use_clipping: Whether to use the clipped basis expansion
            sum_b alpha_b (ReLU(z - H_b) - ReLU(z)) that guarantees
            bounded orbits when ||A||_2 < 1 (reference clipped-PLRNN).
        clip_range: Optional hard clip range for latent states (unclipped variant only).
        threshold_range: Optional (min, max) range for initializing the basis
            thresholds H; pass the (normalized) data range so the bases cover the
            observations (reference init_thetas_uniform). None -> U(±1/sqrt(M)).
        init_scheme: "default" (uniform inits) or "paper" (Talathi-Vartak AW split,
            alphas ~ U(±1/sqrt(B)), h ~ U(±1/sqrt(M))).
        learn_z0: Learn the observation->latent lift B (output_dim, latent_dim)
            used by init_state_from_obs (reference Z0Model). Default False -> zero-hidden init.
    """
    model_type = "dend_plrnn"

    def __init__(
        self,
        n_bases: int = 20,
        use_clipping: bool = False,
        clip_range: float | None = None,
        threshold_range: tuple[float, float] | None = None,
        init_scheme: str = "default",
        learn_z0: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if init_scheme not in ("default", "paper"):
            raise ValueError(f"Unknown init_scheme: {init_scheme}")
        self.n_bases = n_bases
        self.use_clipping = use_clipping
        self.clip_range = clip_range
        self.threshold_range = tuple(threshold_range) if threshold_range is not None else None
        self.init_scheme = init_scheme
        self.learn_z0 = bool(learn_z0)


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
        init_scheme: "default" (uniform inits) or "paper" (AL-RNN reference:
            A = diag of normalized positive-definite matrix, W ~ N(0, 0.01^2), h = 0).
        learn_z0: Learn the observation->latent lift B (output_dim, latent_dim)
            used by init_state_from_obs (AL-RNN reference: z0 = x0 @ B, with the
            first output_dim dims then hard-set to x0). Default False -> zero-hidden init.
    """
    model_type = "alrnn"

    def __init__(
        self,
        n_linear: int = 1,
        use_clipping: bool = False,
        clip_range: float | None = None,
        init_scheme: str = "default",
        learn_z0: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if init_scheme not in ("default", "paper"):
            raise ValueError(f"Unknown init_scheme: {init_scheme}")
        self.n_linear = n_linear
        self.use_clipping = use_clipping
        self.clip_range = clip_range
        self.init_scheme = init_scheme
        self.learn_z0 = bool(learn_z0)
