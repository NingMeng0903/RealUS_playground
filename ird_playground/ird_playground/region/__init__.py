"""Region package: legacy SE(3) aggregate + Sprint-0A local ellipsoid/cone."""

from ird_playground.region.aggregate import (
    OrientationExtent,
    PositionExtent,
    RegionScore,
    aggregate_mean_softmin,
    aggregate_mq,
    region_score_a,
    sample_anisotropic_xi,
)
from ird_playground.region.local_region import (
    LocalExtent,
    RegionAggConfig,
    local_region_cost,
    make_joint_sobol_ellipsoid_cone,
    robust_region_cost,
)

__all__ = [
    "LocalExtent",
    "OrientationExtent",
    "PositionExtent",
    "RegionAggConfig",
    "RegionScore",
    "aggregate_mean_softmin",
    "aggregate_mq",
    "local_region_cost",
    "make_joint_sobol_ellipsoid_cone",
    "region_score_a",
    "robust_region_cost",
    "sample_anisotropic_xi",
]
