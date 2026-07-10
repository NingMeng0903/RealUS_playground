from __future__ import annotations

import math
from typing import Iterable

import numpy as np


ArrayLike3 = Iterable[float]
ArrayLike4 = Iterable[float]


def normalize3(vec: ArrayLike3, *, fallback: ArrayLike3 | None = None) -> np.ndarray:
    arr = np.asarray(tuple(float(v) for v in vec), dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(arr))
    if norm < 1e-12:
        if fallback is None:
            raise ValueError('Cannot normalize a near-zero vector without a fallback.')
        return np.asarray(tuple(float(v) for v in fallback), dtype=np.float64).reshape(3)
    return arr / norm


def axis_angle_rotation(axis: ArrayLike3, angle_rad: float) -> np.ndarray:
    axis = normalize3(axis)
    x, y, z = axis.tolist()
    c = float(math.cos(float(angle_rad)))
    s = float(math.sin(float(angle_rad)))
    t = 1.0 - c
    return np.asarray(
        [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
        ],
        dtype=np.float64,
    )


def quaternion_xyzw_to_matrix(quat_xyzw: ArrayLike4) -> np.ndarray:
    x, y, z, w = (float(v) for v in quat_xyzw)
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z, w = x / n, y / n, z / n, w / n
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.asarray(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def rotation_matrix_to_quaternion_xyzw(matrix: np.ndarray) -> np.ndarray:
    r = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(r))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (r[2, 1] - r[1, 2]) / s
        y = (r[0, 2] - r[2, 0]) / s
        z = (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        w = (r[2, 1] - r[1, 2]) / s
        x = 0.25 * s
        y = (r[0, 1] + r[1, 0]) / s
        z = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        w = (r[0, 2] - r[2, 0]) / s
        x = (r[0, 1] + r[1, 0]) / s
        y = 0.25 * s
        z = (r[1, 2] + r[2, 1]) / s
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        w = (r[1, 0] - r[0, 1]) / s
        x = (r[0, 2] + r[2, 0]) / s
        y = (r[1, 2] + r[2, 1]) / s
        z = 0.25 * s
    quat = np.asarray([x, y, z, w], dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    return quat if norm < 1e-12 else quat / norm


def quaternion_xyzw_to_wxyz(quat_xyzw: ArrayLike4) -> tuple[float, float, float, float]:
    x, y, z, w = (float(v) for v in quat_xyzw)
    return (w, x, y, z)


def quaternion_wxyz_to_xyzw(quat_wxyz: ArrayLike4) -> tuple[float, float, float, float]:
    w, x, y, z = (float(v) for v in quat_wxyz)
    return (x, y, z, w)


def lookat_frame(
    pos: ArrayLike3,
    lookat: ArrayLike3,
    up: ArrayLike3 = (0.0, 0.0, 1.0),
    *,
    roll_deg: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pos_v = np.asarray(tuple(float(v) for v in pos), dtype=np.float64).reshape(3)
    lookat_v = np.asarray(tuple(float(v) for v in lookat), dtype=np.float64).reshape(3)
    up_hint = normalize3(up, fallback=(0.0, 0.0, 1.0))
    forward = normalize3(lookat_v - pos_v, fallback=(1.0, 0.0, 0.0))
    right = normalize3(np.cross(forward, up_hint), fallback=(1.0, 0.0, 0.0))
    true_up = normalize3(np.cross(right, forward), fallback=(0.0, 0.0, 1.0))
    if abs(float(roll_deg)) > 1e-9:
        roll = axis_angle_rotation(forward, math.radians(float(roll_deg)))
        right = normalize3(roll @ right, fallback=(1.0, 0.0, 0.0))
        true_up = normalize3(roll @ true_up, fallback=(0.0, 0.0, 1.0))
    return right, true_up, forward


def opencv_camera_rotation_from_lookat(
    pos: ArrayLike3,
    lookat: ArrayLike3,
    up: ArrayLike3 = (0.0, 0.0, 1.0),
    *,
    roll_deg: float = 0.0,
) -> np.ndarray:
    right, true_up, forward = lookat_frame(pos, lookat, up, roll_deg=roll_deg)
    down = -true_up
    return np.stack([right, down, forward], axis=0)


def ue_rotator_deg_from_matrix(matrix: np.ndarray) -> tuple[float, float, float]:
    mat = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    sy = math.sqrt(mat[0, 0] ** 2 + mat[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        yaw = math.degrees(math.atan2(mat[1, 0], mat[0, 0]))
        pitch = math.degrees(math.atan2(-mat[2, 0], sy))
        roll = math.degrees(math.atan2(mat[2, 1], mat[2, 2]))
    else:
        yaw = math.degrees(math.atan2(-mat[0, 1], mat[1, 1]))
        pitch = math.degrees(math.atan2(-mat[2, 0], max(sy, 1e-10)))
        roll = math.degrees(math.atan2(mat[0, 2], mat[1, 2]))
    return -roll, -pitch, yaw


def ue_rotator_deg_from_lookat(
    pos: ArrayLike3,
    lookat: ArrayLike3,
    up: ArrayLike3 = (0.0, 0.0, 1.0),
) -> tuple[float, float, float]:
    pos_v = np.asarray(tuple(float(v) for v in pos), dtype=np.float64).reshape(3)
    lookat_v = np.asarray(tuple(float(v) for v in lookat), dtype=np.float64).reshape(3)
    forward = normalize3(lookat_v - pos_v, fallback=(0.0, 0.0, -1.0))
    up_hint = normalize3(up, fallback=(0.0, 0.0, 1.0))
    up_proj = up_hint - float(np.dot(forward, up_hint)) * forward
    if float(np.linalg.norm(up_proj)) < 1e-8:
        fallback_up = np.asarray((1.0, 0.0, 0.0) if abs(float(forward[2])) < 0.95 else (0.0, 1.0, 0.0), dtype=np.float64)
        up_proj = fallback_up - float(np.dot(forward, fallback_up)) * forward
    up_proj = normalize3(up_proj, fallback=(0.0, 1.0, 0.0))
    right = normalize3(np.cross(forward, up_proj), fallback=(0.0, 1.0, 0.0))
    true_up = normalize3(np.cross(right, forward), fallback=(0.0, 0.0, 1.0))

    hyp_xy = math.sqrt(max(float(forward[0] * forward[0] + forward[1] * forward[1]), 0.0))
    yaw = math.degrees(math.atan2(float(up_proj[1]), float(up_proj[0]))) if hyp_xy < 1e-10 else math.degrees(math.atan2(float(forward[1]), float(forward[0])))
    pitch = math.degrees(math.atan2(float(forward[2]), hyp_xy))

    ref_world_up = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
    ref_proj = ref_world_up - float(np.dot(ref_world_up, forward)) * forward
    ref_proj = normalize3(ref_proj, fallback=(1.0, 0.0, 0.0))
    roll = math.degrees(math.atan2(float(np.dot(np.cross(ref_proj, true_up), forward)), float(np.dot(ref_proj, true_up))))
    return roll, pitch, yaw
