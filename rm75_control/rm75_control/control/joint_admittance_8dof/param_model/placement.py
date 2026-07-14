"""World placement from base-coordinate calibration (rail_y = 0).

The assembly is NOT placed by specifying the frame pose.  Instead the
user calibrates the **base coordinate** (``base_link`` / ``arm_mount`` origin)
in world when ``rail_y = 0``, optionally with a small tilt.  The Genesis entity
pose (``rail_base`` root) is back-solved so rail, slider, frame, and arm all
follow from the URDF kinematic tree.

Orientation in yaml uses **quaternion** ``base_quat_wxyz`` (Genesis convention:
w, x, y, z).  ``base_euler_deg`` is accepted as a fallback only when quat is
omitted.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as Rsc


def base_offset_in_rail_base(spec: dict[str, Any], layout: dict[str, float]) -> np.ndarray:
    """``base_link`` origin in ``rail_base`` frame when ``rail_y = 0`` (meters)."""
    m = float(layout["m"])
    arm = spec["arm_mount"]
    return np.array(
        [
            float(arm["offset_x_mm"]) * m,
            float(arm["offset_y_mm"]) * m,
            float(layout["slider_top_z"]),
        ],
        dtype=np.float64,
    )


def _normalize_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64).reshape(4)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / n


def _quat_wxyz_to_R(quat_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = _normalize_quat_wxyz(quat_wxyz)
    # scipy uses x,y,z,w
    return Rsc.from_quat([x, y, z, w]).as_matrix()


def _R_to_quat_wxyz(R: np.ndarray) -> tuple[float, float, float, float]:
    x, y, z, w = Rsc.from_matrix(R).as_quat()
    return (float(w), float(x), float(y), float(z))


def _pose_to_T(
    pos: np.ndarray,
    *,
    quat_wxyz: np.ndarray | None = None,
    euler_deg: np.ndarray | None = None,
) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    if quat_wxyz is not None:
        T[:3, :3] = _quat_wxyz_to_R(quat_wxyz)
    elif euler_deg is not None:
        T[:3, :3] = Rsc.from_euler("xyz", euler_deg, degrees=True).as_matrix()
    T[:3, 3] = np.asarray(pos, dtype=np.float64).reshape(3)
    return T


def _parse_quat_wxyz(wc: dict[str, Any], key: str, default: tuple[float, float, float, float]) -> np.ndarray:
    if key not in wc:
        return np.array(default, dtype=np.float64)
    return _normalize_quat_wxyz(np.asarray(wc[key], dtype=np.float64))


def resolve_world_calib(spec: dict[str, Any], layout: dict[str, float]) -> dict[str, Any]:
    """Merge ``world_calib`` yaml with defaults derived from geometry."""
    wc = dict(spec.get("world_calib") or {})
    base_in_rb = base_offset_in_rail_base(spec, layout)
    default_pos = base_in_rb.copy()
    pos = np.asarray(wc.get("base_pos_m", default_pos), dtype=np.float64).reshape(3)

    identity_q = (1.0, 0.0, 0.0, 0.0)
    base_quat = _parse_quat_wxyz(wc, "base_quat_wxyz", identity_q)
    rb_quat = _parse_quat_wxyz(wc, "base_in_rail_quat_wxyz", identity_q)

    # Fallback: euler only when quat key absent
    base_euler = None
    if "base_quat_wxyz" not in wc and "base_euler_deg" in wc:
        base_euler = np.asarray(wc["base_euler_deg"], dtype=np.float64).reshape(3)
    rb_euler = None
    if "base_in_rail_quat_wxyz" not in wc and "base_in_rail_euler_deg" in wc:
        rb_euler = np.asarray(wc["base_in_rail_euler_deg"], dtype=np.float64).reshape(3)

    return {
        "base_pos_m": pos,
        "base_quat_wxyz": base_quat,
        "base_euler_deg": base_euler,
        "base_in_rail_base_pos": base_in_rb,
        "base_in_rail_quat_wxyz": rb_quat,
        "base_in_rail_euler_deg": rb_euler,
    }


def entity_pose_from_calib(calib: dict[str, Any]) -> dict[str, Any]:
    """Return Genesis entity pose for URDF root ``rail_base``.

    Output keys: ``pos`` (m), ``quat_wxyz`` (w,x,y,z).  At ``rail_y = 0`` the
    ``base_link`` world pose matches the calibrated base pose.
    """
    T_world_base = _pose_to_T(
        calib["base_pos_m"],
        quat_wxyz=calib["base_quat_wxyz"],
        euler_deg=calib.get("base_euler_deg"),
    )
    T_railbase_base = _pose_to_T(
        calib["base_in_rail_base_pos"],
        quat_wxyz=calib["base_in_rail_quat_wxyz"],
        euler_deg=calib.get("base_in_rail_euler_deg"),
    )
    T_world_railbase = T_world_base @ np.linalg.inv(T_railbase_base)
    pos = tuple(float(x) for x in T_world_railbase[:3, 3])
    quat_wxyz = _R_to_quat_wxyz(T_world_railbase[:3, :3])
    return {"pos": pos, "quat_wxyz": quat_wxyz}
