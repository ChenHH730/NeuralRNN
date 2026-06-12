"""可解释性分析层：不动点 / 线性化 / 向量场 / 降维 / 重构指标 / Lyapunov / 流形 / 线吸引子。

铁律：本层只通过模型公共契约工作，不 import 任何具体模型类（见 PORTING_GUIDE 契约D）。
"""
from .fixed_points import (
    find_fixed_points,
    NumericFixedPointFinder,
    ScipyFixedPointFinder,
    AnalyticPLRNNFixedPointFinder,
    FixedPoint,
    FixedPointSet,
)
from .linearization import (
    linearize,
    classify_fixed_point,
    dominant_direction,
    LinearizationResult,
)
from .vector_field import compute_vector_field, VectorField
from .dimensionality import fit_pca, collect_states, PCAResult
from .lyapunov import max_lyapunov_exponent
from .stsp_metrics import (
    state_space_divergence,
    state_space_divergence_binning,
    state_space_divergence_gmm,
    power_spectrum_error,
    hellinger_distance,
)
from .manifold import trajectories_to_pos_vel
from .connectivity import analyze_connectivity, ConnectivityResult
from .perturbation import (
    analyze_perturbation,
    latent_to_rnn_perturbation,
    apply_perturbation,
    compute_choice,
    PerturbationSpec,
    PerturbationResult,
)
from .psychometric import compute_psychometric, PsychometricCurve, PsychometricResult
from .line_attractor import (
    find_line_attractor_endpoints,
    walk_line_attractor,
    compute_line_attractor,
    LineAttractorPoint,
    LineAttractorResult,
)

__all__ = [
    "find_fixed_points",
    "NumericFixedPointFinder",
    "ScipyFixedPointFinder",
    "AnalyticPLRNNFixedPointFinder",
    "FixedPoint",
    "FixedPointSet",
    "linearize",
    "classify_fixed_point",
    "dominant_direction",
    "LinearizationResult",
    "compute_vector_field",
    "VectorField",
    "fit_pca",
    "collect_states",
    "PCAResult",
    "max_lyapunov_exponent",
    "state_space_divergence",
    "state_space_divergence_binning",
    "state_space_divergence_gmm",
    "power_spectrum_error",
    "hellinger_distance",
    "trajectories_to_pos_vel",
    "analyze_connectivity",
    "ConnectivityResult",
    "analyze_perturbation",
    "latent_to_rnn_perturbation",
    "apply_perturbation",
    "compute_choice",
    "PerturbationSpec",
    "PerturbationResult",
    "compute_psychometric",
    "PsychometricCurve",
    "PsychometricResult",
    "find_line_attractor_endpoints",
    "walk_line_attractor",
    "compute_line_attractor",
    "LineAttractorPoint",
    "LineAttractorResult",
]
