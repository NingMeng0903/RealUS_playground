"""Embodiment profiles, generic loaders, and project presets."""

from projects.genesis_ue_sync.sim_platform.embodiments.loaders import (
    URDFToolFrames,
    build_embodiment_from_urdf,
    parse_root_link_name,
    parse_revolute_joint_limits,
    patch_urdf_zero_inertial_links,
    resolve_shape_specific_proxy_urdf,
    resolve_urdf_with_limb_group_scale,
    resolve_urdf_with_uniform_scale,
    write_uniform_scaled_urdf,
)
from projects.genesis_ue_sync.sim_platform.embodiments.capsule_drive_canonical import (
    capsule_packed_q_from_amongus_human,
    capsule_packed_q_from_smpl_axis_angle,
    smpl_pose_axis_angle_row_from_amongus_human,
)
from projects.genesis_ue_sync.sim_platform.embodiments.smpl2urdf import (
    ProxyBodyGeometry,
    ProxyGeometry,
    ProxyPointCloudConfig,
    build_proxy_cloud_sequence,
    build_proxy_geometry_for_sequence,
    human_sequence_from_smpl_pkl,
    resolve_smpl_proxy_urdf,
    sample_proxy_surface_points,
    shape_joints_from_sequence,
)
from projects.genesis_ue_sync.sim_platform.embodiments.presets import build_panda_ultrasound_preset
from projects.genesis_ue_sync.sim_platform.embodiments.profiles import (
    CameraRigProfile,
    EmbodimentProfile,
    EndEffectorProfile,
    JointLimit,
    RobotProfile,
    SensorProfile,
    ToolProfile,
)

__all__ = [
    "CameraRigProfile",
    "EmbodimentProfile",
    "EndEffectorProfile",
    "JointLimit",
    "RobotProfile",
    "SensorProfile",
    "ToolProfile",
    "ProxyBodyGeometry",
    "ProxyGeometry",
    "ProxyPointCloudConfig",
    "URDFToolFrames",
    "build_embodiment_from_urdf",
    "capsule_packed_q_from_amongus_human",
    "capsule_packed_q_from_smpl_axis_angle",
    "build_proxy_cloud_sequence",
    "build_proxy_geometry_for_sequence",
    "human_sequence_from_smpl_pkl",
    "build_panda_ultrasound_preset",
    "parse_root_link_name",
    "parse_revolute_joint_limits",
    "patch_urdf_zero_inertial_links",
    "resolve_shape_specific_proxy_urdf",
    "resolve_smpl_proxy_urdf",
    "resolve_urdf_with_limb_group_scale",
    "resolve_urdf_with_uniform_scale",
    "sample_proxy_surface_points",
    "shape_joints_from_sequence",
    "smpl_pose_axis_angle_row_from_amongus_human",
    "write_uniform_scaled_urdf",
]
