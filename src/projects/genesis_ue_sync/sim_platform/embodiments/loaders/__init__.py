"""Generic embodiment loaders."""

from projects.genesis_ue_sync.sim_platform.embodiments.loaders.urdf_loader import (
    URDFToolFrames,
    build_embodiment_from_urdf,
    cached_scaled_urdf_path,
    parse_collapsed_kinematic_edges,
    parse_kinematic_edges,
    parse_root_link_name,
    parse_revolute_joint_limits,
    patch_urdf_zero_inertial_links,
    resolve_shape_specific_proxy_urdf,
    resolve_urdf_with_limb_group_scale,
    resolve_urdf_with_uniform_scale,
    write_limb_group_scaled_urdf,
    write_uniform_scaled_urdf,
)

__all__ = [
    "URDFToolFrames",
    "build_embodiment_from_urdf",
    "cached_scaled_urdf_path",
    "parse_collapsed_kinematic_edges",
    "parse_kinematic_edges",
    "parse_root_link_name",
    "parse_revolute_joint_limits",
    "patch_urdf_zero_inertial_links",
    "resolve_shape_specific_proxy_urdf",
    "resolve_urdf_with_limb_group_scale",
    "resolve_urdf_with_uniform_scale",
    "write_limb_group_scaled_urdf",
    "write_uniform_scaled_urdf",
]
