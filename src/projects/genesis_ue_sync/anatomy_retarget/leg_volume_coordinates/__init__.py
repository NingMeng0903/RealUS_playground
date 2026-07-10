"""Canonical SMPL leg volumetric coordinate utilities."""

from .atlas import (
    LegVolumeAtlas,
    LegVolumeConfig,
    VesselSkinProjection,
    bake_leg_volume_atlases,
    load_leg_volume_atlas,
    save_leg_volume_atlas,
)
from .lbs_bridge import LbsKinematicState, PoseBatch, apply_lbs_pose, inverse_lbs_pose
from .pose_bundle import query_pose_aware_coordinates
from .projection import project_vessel_centerlines_to_skin, remap_vessel_projection_to_skin
from .surface_refine import SurfaceAtlasRefiner
from .volume_refine import VolumeTetRefiner
from .butterfly import ButterflySurface, make_butterfly_surface

__all__ = [
    "LegVolumeAtlas",
    "LegVolumeConfig",
    "LbsKinematicState",
    "PoseBatch",
    "ButterflySurface",
    "SurfaceAtlasRefiner",
    "VolumeTetRefiner",
    "VesselSkinProjection",
    "apply_lbs_pose",
    "bake_leg_volume_atlases",
    "inverse_lbs_pose",
    "load_leg_volume_atlas",
    "make_butterfly_surface",
    "project_vessel_centerlines_to_skin",
    "query_pose_aware_coordinates",
    "remap_vessel_projection_to_skin",
    "save_leg_volume_atlas",
]
