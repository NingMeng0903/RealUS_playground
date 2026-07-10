from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from bridge.core.rotation import quaternion_xyzw_to_matrix


def rotation3_from_quat_xyzw(quat_xyzw: Sequence[float]) -> list[list[float]]:
    return quaternion_xyzw_to_matrix(quat_xyzw).tolist()


def root_transform_from_pose(
    base_pos_m: Sequence[float],
    base_quat_xyzw: Sequence[float] | None,
) -> list[list[float]]:
    out = np.eye(4, dtype=np.float64)
    out[:3, 3] = np.asarray([float(v) for v in base_pos_m], dtype=np.float64).reshape(3)
    if base_quat_xyzw is not None:
        out[:3, :3] = quaternion_xyzw_to_matrix(base_quat_xyzw)
    return out.tolist()
