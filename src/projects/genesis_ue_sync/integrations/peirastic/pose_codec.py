"""Pose encoding for PEIRASTIC ``FrankaRobotStateMessage`` (libfranka-compatible layout)."""

from __future__ import annotations

import numpy as np

from bridge.core.rotation import quaternion_xyzw_to_matrix, quaternion_wxyz_to_xyzw


def tcp_pose_to_homogeneous(pose7: np.ndarray) -> np.ndarray:
    """Build 4x4 ``base_T_tcp`` from Genesis tcp pose ``[x,y,z, qw,qx,qy,qz]``."""
    p = np.asarray(pose7, dtype=np.float64).reshape(7)
    pos = p[:3]
    quat_wxyz = p[3:7]
    quat_xyzw = quaternion_wxyz_to_xyzw(tuple(float(x) for x in quat_wxyz))
    rot = quaternion_xyzw_to_matrix(quat_xyzw)
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = rot
    out[:3, 3] = pos
    return out


def o_t_ee_flat_from_homogeneous(T: np.ndarray) -> list[float]:
    """Serialize ``O_T_EE`` to match ``FrankaInterface.last_eef_pose`` round-trip."""
    mat = np.asarray(T, dtype=np.float64).reshape(4, 4)
    flat = mat.T.ravel(order="C")
    return [float(x) for x in flat.tolist()]
