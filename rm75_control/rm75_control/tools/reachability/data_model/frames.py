"""Frame conventions and the map's translation invariance.

The capability map is built with the 8-DOF URDF frozen at ``rail_y = 0``, so
map coordinates live in the ``rail_base`` = ``base_link`` frame (``arm_mount``
is a 0-offset fixed joint). Sliding the rail during a scan is equivalent to
translating the whole map by ``rail_y`` along +Y in the rail_base frame — and
placing the rail base at world y ``y_b`` is an outer +Y translation of the same
frame in world coordinates.

The single knob we care about at query time is therefore::

    y_shift = y_b + rail_y     (bounded by rail_y ∈ [-0.18, +0.18])

and the transform from world point ``p_w`` to the arm-base frame is::

    p_ab = p_w - (x_b, y_b + rail_y, z_b)

This module keeps that arithmetic in one place so nothing else in the package
has to know the convention.
"""

from __future__ import annotations

import numpy as np


def arm_base_from_world(
    p_world: np.ndarray,
    rail_base_world_xyz: np.ndarray,
    rail_y: float,
) -> np.ndarray:
    """Convert a world point (or batch (N,3)) into the arm-base frame."""
    p = np.asarray(p_world, dtype=np.float64)
    b = np.asarray(rail_base_world_xyz, dtype=np.float64).reshape(3)
    offset = b.copy()
    offset[1] = offset[1] + float(rail_y)
    return p - offset


def apply_yshift_world_to_arm_base(
    p_world: np.ndarray,
    xz_base_world: tuple[float, float],
    y_shift: float,
) -> np.ndarray:
    """Convenience wrapper for the 1-D placement case (only y_shift is free)."""
    p = np.asarray(p_world, dtype=np.float64)
    offset = np.array([xz_base_world[0], float(y_shift), xz_base_world[1]], dtype=np.float64)
    if p.ndim == 1:
        return p - offset
    return p - offset[None, :]


def tool_axis_from_quat(quat_xyzw: np.ndarray) -> np.ndarray:
    """Return the +Z body axis expressed in the parent frame from quaternion(s).

    Accepts shape (4,) or (N,4) with ``[qx, qy, qz, qw]`` convention (matches
    :mod:`scipy.spatial.transform.Rotation` and the rest of this codebase).
    Skips scipy to stay dependency-light and vectorises over the batch.
    """
    q = np.asarray(quat_xyzw, dtype=np.float64)
    single = q.ndim == 1
    if single:
        q = q[None, :]
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    ax = 2.0 * (x * z + w * y)
    ay = 2.0 * (y * z - w * x)
    az = 1.0 - 2.0 * (x * x + y * y)
    v = np.stack([ax, ay, az], axis=-1)
    return v[0] if single else v
