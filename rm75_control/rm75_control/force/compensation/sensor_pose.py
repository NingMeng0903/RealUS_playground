"""Regressor pose: link_7 (sensor) vs active TCP.

φ gravity compensation must use the physical sensor / flange orientation,
not the teach-pendant tool frame (gripper RY+90° etc.).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as Rsc

DEFAULT_EULER_ORDER = "xyz"


def pose_to_se3(pose6: np.ndarray, euler_order: str = DEFAULT_EULER_ORDER) -> tuple[np.ndarray, np.ndarray]:
    pose6 = np.asarray(pose6, dtype=float).reshape(6)
    t = pose6[:3].copy()
    R = Rsc.from_euler(euler_order, pose6[3:6], degrees=False).as_matrix()
    return t, R


def se3_to_pose(t: np.ndarray, R: np.ndarray, euler_order: str = DEFAULT_EULER_ORDER) -> np.ndarray:
    pose = np.zeros(6, dtype=float)
    pose[:3] = t
    pose[3:6] = Rsc.from_matrix(R).as_euler(euler_order, degrees=False)
    return pose


def se3_inv(t: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    Rt = R.T
    return -Rt @ t, Rt


def se3_mul(ta, Ra, tb, Rb):
    return ta + Ra @ tb, Ra @ Rb


def link7_pose_from_tcp(
    pose_tcp: np.ndarray,
    tool_offset_link7_to_tcp: np.ndarray,
    *,
    euler_order: str = DEFAULT_EULER_ORDER,
) -> np.ndarray:
    """base→link_7 from base→tool and link_7→tool (RealMan tool frame offset)."""
    tb, Rb = pose_to_se3(pose_tcp, euler_order)
    to, Ro = pose_to_se3(tool_offset_link7_to_tcp, euler_order)
    ti, Ri = se3_inv(to, Ro)
    tf, Rf = se3_mul(tb, Rb, ti, Ri)
    return se3_to_pose(tf, Rf, euler_order)


def regressor_pose6(
    pose_tcp: np.ndarray,
    *,
    frame: str,
    tool_offset: np.ndarray | None = None,
    pose_link7: np.ndarray | None = None,
    euler_order: str = DEFAULT_EULER_ORDER,
) -> np.ndarray:
    """
    Pose passed to ``R_base_sensor`` for φ regressor / observer.

    ``link_7``: prefer ``pose_link7`` (Pin FK) else recover from TCP + offset.
    ``tcp``: active tool TCP (legacy).
    """
    if frame == "tcp":
        return np.asarray(pose_tcp, dtype=float).reshape(6).copy()
    if pose_link7 is not None:
        return np.asarray(pose_link7, dtype=float).reshape(6).copy()
    if tool_offset is None:
        raise ValueError("link_7 regressor pose needs pose_link7 or tool_offset")
    return link7_pose_from_tcp(pose_tcp, tool_offset, euler_order=euler_order)
