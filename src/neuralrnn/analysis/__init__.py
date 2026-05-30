"""可解释性分析层：不动点 / 线性化 / 向量场 / 降维 / 重构指标 / Lyapunov / 流形。

铁律：本层只通过模型公共契约工作，不 import 任何具体模型类（见 PORTING_GUIDE 契约D）。
"""
from .fixed_points import (
    find_fixed_points,
    NumericFixedPointFinder,
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

__all__ = [
    "find_fixed_points",
    "NumericFixedPointFinder",
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
]
