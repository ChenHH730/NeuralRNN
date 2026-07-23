"""Interpretability analysis layer: fixed points / linearization / vector field / dimensionality reduction / reconstruction metrics / Lyapunov / manifolds / line attractor / population structure.

Golden rule: this layer only works through the model's public contract and does not import any concrete model classes (see PORTING_GUIDE contract D).
"""
from .linalg_utils import (
    phi_prime,
    gram_schmidt,
    gram_schmidt_pt,
    gram_factorization,
    overlap_matrix,
    corrvecs,
    project,
    angle_vectors,
    angle_vec_subsp,
    flatten_trajectory,
    unflatten_trajectory,
    map_device,
)
from .population_structure import (
    make_vecs,
    gmm_fit,
    compute_population_means,
    compute_population_covariances,
)
from .fixed_points import (
    find_fixed_points,
    NumericFixedPointFinder,
    ScipyFixedPointFinder,
    GolubFixedPointFinder,
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
from .dimensionality import fit_pca, collect_states, effective_dimensionality, PCAResult
from .sequentiality import (
    compute_sequentiality_index,
    sort_neurons_by_peak_time,
    weight_profile_by_peak_order,
    split_ei_weight_submatrices,
)
from .lyapunov import max_lyapunov_exponent
from .stsp_metrics import (
    state_space_divergence,
    state_space_divergence_binning,
    state_space_divergence_gmm,
    power_spectrum_error,
    hellinger_distance,
)
from .manifolds import (
    PLRNNManifoldTracer,
    compute_manifold,
    ManifoldSegment,
    ManifoldTrace,
)
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
from .demixed import (
    fit_dpca,
    DPCAResult,
    axis_overlap_matrix,
    axis_svd_alignment,
    potent_null_projection,
)

__all__ = [
    # -- linalg utils --
    "phi_prime",
    "gram_schmidt",
    "gram_schmidt_pt",
    "gram_factorization",
    "overlap_matrix",
    "corrvecs",
    "project",
    "angle_vectors",
    "angle_vec_subsp",
    "flatten_trajectory",
    "unflatten_trajectory",
    "map_device",
    # -- population structure --
    "make_vecs",
    "gmm_fit",
    "compute_population_means",
    "compute_population_covariances",
    # -- fixed points --
    "find_fixed_points",
    "NumericFixedPointFinder",
    "ScipyFixedPointFinder",
    "GolubFixedPointFinder",
    "AnalyticPLRNNFixedPointFinder",
    "FixedPoint",
    "FixedPointSet",
    # -- linearization --
    "linearize",
    "classify_fixed_point",
    "dominant_direction",
    "LinearizationResult",
    # -- vector field --
    "compute_vector_field",
    "VectorField",
    # -- dimensionality --
    "fit_pca",
    "collect_states",
    "effective_dimensionality",
    "PCAResult",
    # -- sequentiality --
    "compute_sequentiality_index",
    "sort_neurons_by_peak_time",
    "weight_profile_by_peak_order",
    "split_ei_weight_submatrices",
    # -- lyapunov --
    "max_lyapunov_exponent",
    # -- stsp metrics --
    "state_space_divergence",
    "state_space_divergence_binning",
    "state_space_divergence_gmm",
    "power_spectrum_error",
    "hellinger_distance",
    # -- PLRNN invariant manifolds --
    "PLRNNManifoldTracer",
    "compute_manifold",
    "ManifoldSegment",
    "ManifoldTrace",
    # -- connectivity --
    "analyze_connectivity",
    "ConnectivityResult",
    # -- perturbation --
    "analyze_perturbation",
    "latent_to_rnn_perturbation",
    "apply_perturbation",
    "compute_choice",
    "PerturbationSpec",
    "PerturbationResult",
    # -- psychometric --
    "compute_psychometric",
    "PsychometricCurve",
    "PsychometricResult",
    # -- line attractor --
    "find_line_attractor_endpoints",
    "walk_line_attractor",
    "compute_line_attractor",
    "LineAttractorPoint",
    "LineAttractorResult",
    # -- demixed PCA / axis alignment --
    "fit_dpca",
    "DPCAResult",
    "axis_overlap_matrix",
    "axis_svd_alignment",
    "potent_null_projection",
]
