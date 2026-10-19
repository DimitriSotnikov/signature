"""Expected-signature matching MFC: dual optimisation over the cone B ∩ T^N.

See ``spec/esig_mfc_spec.md`` and ``doc/expected_signature_matching.md``. Requires
``jax_enable_x64=True``.
"""
from .esig_mfc import (
    path_signatures, brownian_paths, target_paths, expected_signature, word_weights,
    ConeSpec, make_b_cone, project, in_cone,
    Tilt, DualProblem,
    DualResult, optimize_dual, continuation,
    control_tensor, optimal_policy, rollout, resample,
)

__all__ = [
    "path_signatures", "brownian_paths", "target_paths", "expected_signature", "word_weights",
    "ConeSpec", "make_b_cone", "project", "in_cone",
    "Tilt", "DualProblem",
    "DualResult", "optimize_dual", "continuation",
    "control_tensor", "optimal_policy", "rollout", "resample",
]
