"""PyVista scenes matching Zacharias 2013 + Vahrenkamp 2013 paper figures."""

from rm75_control.tools.reachability.viz.colormap import (
    ZACHARIAS_DIR_FACE_MISSING,
    ZACHARIAS_DIR_FACE_REACHABLE,
    ZACHARIAS_ROBOT_GRAY,
    make_vahrenkamp_irm_cmap,
    make_zacharias_d_cmap,
)
from rm75_control.tools.reachability.viz.inversion_scene import (
    render_base_candidates,
    render_best_placement_pose,
    render_irm_ground,
)
from rm75_control.tools.reachability.viz.orientation_glyph import render_direction_spheres
from rm75_control.tools.reachability.viz.robot_scene import build_robot_pv
from rm75_control.tools.reachability.viz.sphere_glyphs import render_reachability_index, render_slice
from rm75_control.tools.reachability.viz.trajectory_overlay import render_trajectory_overlay

__all__ = [
    "ZACHARIAS_DIR_FACE_MISSING",
    "ZACHARIAS_DIR_FACE_REACHABLE",
    "ZACHARIAS_ROBOT_GRAY",
    "build_robot_pv",
    "make_vahrenkamp_irm_cmap",
    "make_zacharias_d_cmap",
    "render_base_candidates",
    "render_best_placement_pose",
    "render_direction_spheres",
    "render_irm_ground",
    "render_reachability_index",
    "render_slice",
    "render_trajectory_overlay",
]
