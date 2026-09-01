"""Shared Cartesian pose error utilities (base frame + tool-frame tracking norms)."""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial.transform import Rotation as Rsc


def wrap_pi(angle: float) -> float:
    return float(math.atan2(math.sin(angle), math.cos(angle)))


def pose_error(
    desired: np.ndarray,
    current: np.ndarray,
    euler_order: str = "xyz",
) -> np.ndarray:
    """Base-frame 6D pose error: linear diff + SO(3) log (rotvec of R_des @ R_cur^T)."""
    err = np.zeros(6, dtype=float)
    err[:3] = np.asarray(desired[:3], dtype=float) - np.asarray(current[:3], dtype=float)
    r_des = Rsc.from_euler(euler_order, desired[3:6], degrees=False).as_matrix()
    r_cur = Rsc.from_euler(euler_order, current[3:6], degrees=False).as_matrix()
    err[3:6] = Rsc.from_matrix(r_des @ r_cur.T).as_rotvec()
    return err


def pose_track_error_mm_deg(
    desired: np.ndarray,
    current: np.ndarray,
    *,
    track_axes: np.ndarray,
    euler_order: str = "xyz",
) -> tuple[float, float]:
    """Tool-frame tracking error on position/velocity-controlled axes only."""
    err_base = pose_error(desired, current, euler_order)
    r_cur = Rsc.from_euler(euler_order, np.asarray(current[3:6], dtype=float), degrees=False).as_matrix()
    err_tool = np.zeros(6, dtype=float)
    err_tool[:3] = r_cur.T @ err_base[:3]
    err_tool[3:6] = r_cur.T @ err_base[3:6]
    ta = np.asarray(track_axes, dtype=float)
    err_tool *= ta
    pos_mm = float(np.linalg.norm(err_tool[:3]) * 1000.0)
    rot_deg = float(np.degrees(np.linalg.norm(err_tool[3:6])))
    return pos_mm, rot_deg


def mask_base_pose_error(
    err_base: np.ndarray,
    current_pose: np.ndarray,
    track_axes: np.ndarray,
    euler_order: str = "xyz",
) -> np.ndarray:
    """Zero force-axis components of a base-frame pose error (tool selection)."""

    err = np.asarray(err_base, dtype=float).reshape(6).copy()
    r_cur = Rsc.from_euler(
        euler_order, np.asarray(current_pose[3:6], dtype=float), degrees=False
    ).as_matrix()
    tool = np.zeros(6, dtype=float)
    tool[:3] = r_cur.T @ err[:3]
    tool[3:6] = r_cur.T @ err[3:6]
    tool *= np.clip(np.asarray(track_axes, dtype=float).reshape(6), 0.0, 1.0)
    out = np.zeros(6, dtype=float)
    out[:3] = r_cur @ tool[:3]
    out[3:6] = r_cur @ tool[3:6]
    return out
