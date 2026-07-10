from __future__ import annotations

import json
import time
from collections.abc import Sequence

import numpy as np

from bridge.core.camera import opencv_camera_matrices_from_lookat
from bridge.core.rotation import (
    quaternion_xyzw_to_matrix,
    rotation_matrix_to_quaternion_xyzw,
    ue_rotator_deg_from_lookat,
    ue_rotator_deg_from_matrix,
)
from bridge.core.transform import mat4_inv, mat4_mul

_AGENT_DEBUG_LOG = "/home/camp/.cursor/debug-logs/debug-05706c.log"


def _agent_debug_log(*, hypothesis_id: str, location: str, message: str, data: dict, run_id: str = "post-fix") -> None:
    payload = {
        "sessionId": "05706c",
        "runId": str(run_id),
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    try:
        with open(_AGENT_DEBUG_LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=True) + "\n")
    except Exception:
        pass

UE_WORLD_FROM_GENESIS = np.diag([1.0, -1.0, 1.0, 1.0]).astype(np.float64)
UE_ROTATION_FROM_GENESIS = UE_WORLD_FROM_GENESIS[:3, :3]


def quaternion_xyzw_from_order(quaternion: Sequence[float], order: str = 'xyzw') -> np.ndarray:
    values = [float(v) for v in quaternion]
    if str(order) == 'xyzw':
        return np.asarray(values, dtype=np.float64).reshape(4)
    if str(order) == 'wxyz':
        w, x, y, z = values
        return np.asarray([x, y, z, w], dtype=np.float64)
    raise ValueError(f'Unsupported quaternion order: {order}')


def ue_rotation_matrix_from_quat_xyzw(quat_xyzw: Sequence[float]) -> np.ndarray:
    return quaternion_xyzw_to_matrix(quat_xyzw)


def ue_world_point_from_genesis_m(point_m: Sequence[float]) -> np.ndarray:
    point = np.asarray([float(v) for v in point_m], dtype=np.float64).reshape(3)
    return UE_ROTATION_FROM_GENESIS @ point


def ue_world_rotation_from_genesis(rotation: np.ndarray) -> np.ndarray:
    rot = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    return UE_ROTATION_FROM_GENESIS @ rot @ UE_ROTATION_FROM_GENESIS


def ue_world_quat_xyzw_from_genesis(quat_xyzw: Sequence[float] | None) -> np.ndarray | None:
    if quat_xyzw is None:
        return None
    rot = quaternion_xyzw_to_matrix(quat_xyzw)
    return rotation_matrix_to_quaternion_xyzw(ue_world_rotation_from_genesis(rot))


def apply_camera_basis(world_from_camera: np.ndarray, camera_basis: np.ndarray | None) -> np.ndarray:
    if camera_basis is None:
        return np.asarray(world_from_camera, dtype=np.float64).reshape(4, 4)
    pose = np.asarray(world_from_camera, dtype=np.float64).reshape(4, 4).copy()
    pose[:3, :3] = pose[:3, :3] @ np.asarray(camera_basis, dtype=np.float64).reshape(3, 3)
    return pose


def ue_camera_world_pose_from_location_quaternion_m(
    *,
    location_m: Sequence[float],
    quaternion: Sequence[float],
    quat_order: str = 'xyzw',
    world_from_ue: np.ndarray | None = None,
    camera_basis: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = ue_rotation_matrix_from_quat_xyzw(quaternion_xyzw_from_order(quaternion, quat_order))
    pose[:3, 3] = np.asarray([float(v) for v in location_m], dtype=np.float64).reshape(3)
    if world_from_ue is not None:
        pose = mat4_mul(world_from_ue, pose)
    pose = apply_camera_basis(pose, camera_basis)
    return pose, mat4_inv(pose)


def ue_rotator_deg_from_camera_spec(camera_spec) -> tuple[float, float, float]:
    roll_deg_extra = float(getattr(camera_spec, "roll_deg", 0.0) or 0.0)
    pos = tuple(float(v) for v in ue_world_point_from_genesis_m(camera_spec.pos).tolist())
    lookat = tuple(float(v) for v in ue_world_point_from_genesis_m(camera_spec.lookat).tolist())
    up = tuple(float(v) for v in ue_world_point_from_genesis_m(camera_spec.up).tolist())
    roll, pitch, yaw = ue_rotator_deg_from_lookat(pos, lookat, up)
    roll += roll_deg_extra
    return roll, pitch, yaw


def ue_camera_payload_from_spec(camera_spec) -> dict:
    roll, pitch, yaw = ue_rotator_deg_from_camera_spec(camera_spec)
    # #region agent log
    if str(getattr(camera_spec, "name", "")) == "cam_top":
        _, wfc = opencv_camera_matrices_from_lookat(
            camera_spec.pos,
            camera_spec.lookat,
            camera_spec.up,
            roll_deg=float(getattr(camera_spec, "roll_deg", 0.0) or 0.0),
        )
        matrix_roll, matrix_pitch, matrix_yaw = ue_rotator_deg_from_matrix(
            ue_world_rotation_from_genesis(np.asarray(wfc[:3, :3], dtype=np.float64))
        )
        lookat_roll, lookat_pitch, lookat_yaw = ue_rotator_deg_from_lookat(
            tuple(float(v) for v in ue_world_point_from_genesis_m(camera_spec.pos).tolist()),
            tuple(float(v) for v in ue_world_point_from_genesis_m(camera_spec.lookat).tolist()),
            tuple(float(v) for v in ue_world_point_from_genesis_m(camera_spec.up).tolist()),
        )
        _agent_debug_log(
            hypothesis_id="A",
            location="bridge/adapters/ue.py:ue_camera_payload_from_spec",
            message="cam_top rotator path selection",
            data={
                "chosen_rotator_deg": [float(roll), float(pitch), float(yaw)],
                "chosen_path": "lookat",
                "legacy_lookat_rotator_deg": [float(lookat_roll), float(lookat_pitch), float(lookat_yaw)],
                "opencv_matrix_rotator_deg": [float(matrix_roll), float(matrix_pitch), float(matrix_yaw)],
                "scene_pos_m": [float(v) for v in camera_spec.pos],
                "scene_up": [float(v) for v in camera_spec.up],
            },
        )
    # #endregion
    pos = tuple(float(v) * 100.0 for v in ue_world_point_from_genesis_m(camera_spec.pos).tolist())
    lookat_cm = tuple(float(v) * 100.0 for v in ue_world_point_from_genesis_m(camera_spec.lookat).tolist())
    return {
        'name': str(camera_spec.name),
        'x': pos[0],
        'y': pos[1],
        'z': pos[2],
        'roll': roll,
        'pitch': pitch,
        'yaw': yaw,
        'fov': float(camera_spec.fov),
        'lookat_cm': lookat_cm,
        'res': tuple(int(v) for v in camera_spec.res),
    }
