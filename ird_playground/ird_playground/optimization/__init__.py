"""Offline reachability-guided trajectory optimisation."""

from ird_playground.optimization.differentiable_energy import (
    DecodedTrajectory,
    DifferentiableTrajectoryEnergy,
    GuidanceResult,
    TrajectoryEnergyConfig,
    TrajectoryEnergyOutput,
    cubic_bspline_basis,
    cubic_bspline_matrices,
    encode_reference_controls,
    gauss_legendre_path_samples,
    optimize_guidance_controls,
)

from ird_playground.optimization.trajectory_sqp import (
    Pinocchio8DofAdapter,
    TrajectoryOptimizationConfig,
    TrajectoryOptimizationProblem,
    TrajectoryOptimizationResult,
    WorldConstraint,
    generate_ird_rail_warm_starts,
    optimize_trajectory,
    retime_trajectory,
    validate_trajectory,
)
from ird_playground.optimization.srs_trajectory_dp import (
    SrsTrajectoryDpConfig,
    SrsTrajectoryDpResult,
    solve_srs_trajectory_dp,
)
from ird_playground.optimization.ellipsoid_sdf import (
    ellipsoid_radial_signed_distance,
    ellipsoid_surface_mesh,
    exact_ellipsoid_signed_distance,
)

__all__ = [
    "DecodedTrajectory",
    "DifferentiableTrajectoryEnergy",
    "GuidanceResult",
    "Pinocchio8DofAdapter",
    "SrsTrajectoryDpConfig",
    "SrsTrajectoryDpResult",
    "TrajectoryOptimizationConfig",
    "TrajectoryOptimizationProblem",
    "TrajectoryOptimizationResult",
    "TrajectoryEnergyConfig",
    "TrajectoryEnergyOutput",
    "WorldConstraint",
    "generate_ird_rail_warm_starts",
    "ellipsoid_radial_signed_distance",
    "ellipsoid_surface_mesh",
    "exact_ellipsoid_signed_distance",
    "cubic_bspline_basis",
    "cubic_bspline_matrices",
    "encode_reference_controls",
    "gauss_legendre_path_samples",
    "optimize_guidance_controls",
    "optimize_trajectory",
    "retime_trajectory",
    "solve_srs_trajectory_dp",
    "validate_trajectory",
]
