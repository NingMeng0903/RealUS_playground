"""Adapters between canonical Genesis-aligned math and external systems."""

from bridge.adapters.blender_bedlam import bedlam_unreal_to_smpl_translation, smpl_yup_to_blender
from bridge.adapters.genesis import genesis_quat_wxyz_from_xyzw, xyzw_from_genesis_quat_wxyz
from bridge.adapters.opencv import canonical_camera_from_scene_camera, opencv_camera_matrices_from_scene_camera
from bridge.adapters.ue import (
    apply_camera_basis,
    quaternion_xyzw_from_order,
    ue_camera_payload_from_spec,
    ue_camera_world_pose_from_location_quaternion_m,
    ue_rotation_matrix_from_quat_xyzw,
    ue_rotator_deg_from_camera_spec,
)
from bridge.adapters.urdf import rotation3_from_quat_xyzw, root_transform_from_pose

__all__ = [
    'apply_camera_basis',
    'bedlam_unreal_to_smpl_translation',
    'canonical_camera_from_scene_camera',
    'genesis_quat_wxyz_from_xyzw',
    'opencv_camera_matrices_from_scene_camera',
    'quaternion_xyzw_from_order',
    'rotation3_from_quat_xyzw',
    'root_transform_from_pose',
    'smpl_yup_to_blender',
    'ue_camera_payload_from_spec',
    'ue_camera_world_pose_from_location_quaternion_m',
    'ue_rotation_matrix_from_quat_xyzw',
    'ue_rotator_deg_from_camera_spec',
    'xyzw_from_genesis_quat_wxyz',
]
