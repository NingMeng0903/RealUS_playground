"""Differentiable robust-region queries over the signed IRD field."""

from ird_playground.region.operator import (
    RegionA,
    RegionAConfig,
    RegionAResult,
    base_from_rail_torch,
    normalized_softmin,
)
__all__ = [
    "RegionA",
    "RegionAConfig",
    "RegionAResult",
    "base_from_rail_torch",
    "normalized_softmin",
]
