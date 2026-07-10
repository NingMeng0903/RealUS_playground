"""Core canonical transform, rotation, and camera utilities."""

from bridge.core.camera import CanonicalCamera, build_intrinsics_from_fov, opencv_camera_matrices_from_lookat
from bridge.core.convention import (
    BEDLAM_WORLD_CONVENTION,
    BLENDER_WORLD_CONVENTION,
    CANONICAL_GENESIS_CONVENTION,
    OPENCV_CAMERA_CONVENTION,
    UE_WORLD_CONVENTION,
    CanonicalConvention,
)
from bridge.core.rotation import (
    axis_angle_rotation,
    lookat_frame,
    opencv_camera_rotation_from_lookat,
    quaternion_wxyz_to_xyzw,
    quaternion_xyzw_to_matrix,
    quaternion_xyzw_to_wxyz,
    rotation_matrix_to_quaternion_xyzw,
    ue_rotator_deg_from_lookat,
    ue_rotator_deg_from_matrix,
)
from bridge.core.transform import CanonicalTransform, mat4_inv, mat4_mul

__all__ = [
    'BEDLAM_WORLD_CONVENTION',
    'BLENDER_WORLD_CONVENTION',
    'CANONICAL_GENESIS_CONVENTION',
    'CanonicalCamera',
    'CanonicalConvention',
    'CanonicalTransform',
    'OPENCV_CAMERA_CONVENTION',
    'UE_WORLD_CONVENTION',
    'axis_angle_rotation',
    'build_intrinsics_from_fov',
    'lookat_frame',
    'mat4_inv',
    'mat4_mul',
    'opencv_camera_matrices_from_lookat',
    'opencv_camera_rotation_from_lookat',
    'quaternion_wxyz_to_xyzw',
    'quaternion_xyzw_to_matrix',
    'quaternion_xyzw_to_wxyz',
    'rotation_matrix_to_quaternion_xyzw',
    'ue_rotator_deg_from_lookat',
    'ue_rotator_deg_from_matrix',
]
