"""Source-free binding repair for production analytic distance-ratio depth.

This preserves the existing h/theta queries and the existing depth operator,
but supplies actual production skin points. It does not change source files,
infer new anatomy, or claim agreement with the distinct baked layered d field.
"""
from __future__ import annotations
import numpy as np
from adapter import SOURCE
from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates.atlas import query_atlas_coordinates
from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates.harmonic import compute_volume_d_distance_ratio

def corrected_distance_ratio_depth(atlas, points):
    return compute_volume_d_distance_ratio(np.asarray(points),atlas.skin_vertices,atlas.core_points)

def query_with_corrected_skin_depth(atlas, points):
    xi,skin=query_atlas_coordinates(atlas,points)
    xi=xi.copy()
    xi[:,2]=corrected_distance_ratio_depth(atlas,points)
    return xi,skin
