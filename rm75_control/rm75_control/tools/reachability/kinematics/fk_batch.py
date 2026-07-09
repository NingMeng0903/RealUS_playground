"""Batched forward kinematics for Monte-Carlo capability-map sampling.

Pinocchio's FK is inherently per-configuration; the "batch" here is just a
tight Python loop with re-used ``pin.Data``. We keep the API vectorised
(np arrays in / np arrays out) so the caller can chunk across processes.
"""

from __future__ import annotations

import numpy as np
import pinocchio as pin

from rm75_control.tools.reachability.data_model.frames import tool_axis_from_quat
from rm75_control.tools.reachability.kinematics.model_locked_rail import LockedRailModel


def _quat_from_matrix(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation to unit quaternion (qx, qy, qz, qw).

    Local implementation (Shepperd's method) to avoid scipy overhead per call.
    """
    m = R
    t = m[0, 0] + m[1, 1] + m[2, 2]
    if t > 0.0:
        s = 0.5 / np.sqrt(t + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return np.array([x, y, z, w], dtype=np.float64)


def fk_position_quat_batch(
    lm: LockedRailModel, Q: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """(K, 7) joint config → (positions (K, 3), quats (K, 4)) at the TCP.

    ``lm`` and its ``.data`` are mutated in place — do NOT share across threads;
    hand each worker its own :meth:`LockedRailModel.clone`.
    """
    Q = np.asarray(Q, dtype=np.float64)
    if Q.ndim != 2 or Q.shape[1] != lm.model.nq:
        raise ValueError(f"Q shape must be (K, {lm.model.nq}), got {Q.shape}")
    K = Q.shape[0]
    positions = np.empty((K, 3), dtype=np.float64)
    quats = np.empty((K, 4), dtype=np.float64)
    model, data, fid = lm.model, lm.data, lm.tcp_id
    for i in range(K):
        pin.forwardKinematics(model, data, Q[i])
        pin.updateFramePlacement(model, data, fid)
        M = data.oMf[fid]
        positions[i] = M.translation
        quats[i] = _quat_from_matrix(np.asarray(M.rotation))
    return positions, quats


def fk_tool_axis_batch(lm: LockedRailModel, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convenience: positions + TCP +Z axis (unit) in the arm-base frame."""
    pos, quat = fk_position_quat_batch(lm, Q)
    axis = tool_axis_from_quat(quat)
    axis /= np.clip(np.linalg.norm(axis, axis=1, keepdims=True), 1e-12, None)
    return pos, axis
