from __future__ import annotations

import numpy as np


def as_pose_array(pose: list[float] | tuple[float, ...] | np.ndarray) -> np.ndarray:
    pose_arr = np.asarray(pose, dtype=np.float32).reshape(-1)
    if pose_arr.size != 7:
        raise ValueError(f"Expected pose with 7 values (xyz + quaternion wxyz), got {pose_arr.size}.")
    pose_arr[3:] = normalize_quaternion_wxyz(pose_arr[3:])
    return pose_arr


def quat_wxyz_to_rotation_matrix(q: list[float] | tuple[float, ...] | np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=np.float64).reshape(4).tolist()
    return np.asarray(
        [
            [w * w + x * x - y * y - z * z, 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), w * w - x * x + y * y - z * z, 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), w * w - x * x - y * y + z * z],
        ],
        dtype=np.float32,
    )


def normalize_quaternion_wxyz(quaternion: list[float] | tuple[float, ...] | np.ndarray) -> np.ndarray:
    quat = np.asarray(quaternion, dtype=np.float32).reshape(4)
    norm = float(np.linalg.norm(quat))
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return quat / norm


def quaternion_conjugate_wxyz(quaternion: list[float] | tuple[float, ...] | np.ndarray) -> np.ndarray:
    quat = normalize_quaternion_wxyz(quaternion)
    return np.array([quat[0], -quat[1], -quat[2], -quat[3]], dtype=np.float32)


def quaternion_multiply_wxyz(
    lhs: list[float] | tuple[float, ...] | np.ndarray,
    rhs: list[float] | tuple[float, ...] | np.ndarray,
) -> np.ndarray:
    w1, x1, y1, z1 = normalize_quaternion_wxyz(lhs)
    w2, x2, y2, z2 = normalize_quaternion_wxyz(rhs)
    return normalize_quaternion_wxyz(
        np.array(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dtype=np.float32,
        )
    )


def quaternion_from_rotvec_wxyz(rotvec: list[float] | tuple[float, ...] | np.ndarray) -> np.ndarray:
    rotvec_arr = np.asarray(rotvec, dtype=np.float32).reshape(3)
    angle = float(np.linalg.norm(rotvec_arr))
    if angle < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    axis = rotvec_arr / angle
    half_angle = 0.5 * angle
    sin_half = np.sin(half_angle)
    return normalize_quaternion_wxyz(
        np.array([np.cos(half_angle), *(axis * sin_half)], dtype=np.float32)
    )


def pose_error_wxyz(
    target_pose: list[float] | tuple[float, ...] | np.ndarray,
    current_pose: list[float] | tuple[float, ...] | np.ndarray,
) -> np.ndarray:
    target = as_pose_array(target_pose)
    current = as_pose_array(current_pose)
    position_error = target[:3] - current[:3]
    q_error = quaternion_multiply_wxyz(target[3:], quaternion_conjugate_wxyz(current[3:]))
    if q_error[0] < 0.0:
        q_error = -q_error
    orientation_error = 2.0 * q_error[1:]
    return np.concatenate([position_error, orientation_error], dtype=np.float32)


def apply_pose_delta_wxyz(
    base_pose: list[float] | tuple[float, ...] | np.ndarray,
    delta_twist: list[float] | tuple[float, ...] | np.ndarray,
) -> np.ndarray:
    pose = as_pose_array(base_pose)
    delta = np.asarray(delta_twist, dtype=np.float32).reshape(6)
    result = pose.copy()
    result[:3] += delta[:3]
    delta_quat = quaternion_from_rotvec_wxyz(delta[3:])
    result[3:] = quaternion_multiply_wxyz(delta_quat, pose[3:])
    return result


def damped_pseudoinverse(matrix: np.ndarray, damping: float) -> np.ndarray:
    mat = np.asarray(matrix, dtype=np.float32)
    rows, cols = mat.shape
    if rows <= cols:
        regularized = mat @ mat.T + (damping**2) * np.eye(rows, dtype=np.float32)
        return mat.T @ np.linalg.inv(regularized)
    regularized = mat.T @ mat + (damping**2) * np.eye(cols, dtype=np.float32)
    return np.linalg.inv(regularized) @ mat.T


def clip_norm(vector: list[float] | tuple[float, ...] | np.ndarray, max_norm: float) -> np.ndarray:
    vec = np.asarray(vector, dtype=np.float32).reshape(-1)
    if max_norm <= 0.0:
        return vec
    norm = float(np.linalg.norm(vec))
    if norm <= max_norm or norm < 1e-8:
        return vec
    return vec * (max_norm / norm)
