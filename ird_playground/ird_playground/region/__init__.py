"""Differentiable robust-region queries over the signed IRD field."""

from ird_playground.region.direction_lobe import (
    DirectionLobeResult,
    ascend_direction,
    chart_from_direction,
    direction_lobe,
)
from ird_playground.region.operator import (
    RegionA,
    RegionAConfig,
    RegionAResult,
    base_from_rail_torch,
    normalized_softmin,
)
from ird_playground.region.rail_query import RailQueryResult, rail_inverse_query
from ird_playground.region.set_query import (
    SetQueryConfig,
    SetQueryOperator,
    SetQueryResult,
    rockafellar_uryasev_cvar,
)
from ird_playground.region.trajectory_operator import (
    TrajectoryTaskConfig,
    TrajectoryTaskOperator,
    TrajectoryTaskResult,
)

__all__ = [
    "DirectionLobeResult",
    "RailQueryResult",
    "RegionA",
    "RegionAConfig",
    "RegionAResult",
    "SetQueryConfig",
    "SetQueryOperator",
    "SetQueryResult",
    "TrajectoryTaskConfig",
    "TrajectoryTaskOperator",
    "TrajectoryTaskResult",
    "ascend_direction",
    "base_from_rail_torch",
    "chart_from_direction",
    "direction_lobe",
    "normalized_softmin",
    "rail_inverse_query",
    "rockafellar_uryasev_cvar",
]
