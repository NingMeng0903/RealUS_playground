from __future__ import annotations

from typing import Iterable

from bridge.core.rotation import quaternion_wxyz_to_xyzw, quaternion_xyzw_to_wxyz


ArrayLike4 = Iterable[float]


def genesis_quat_wxyz_from_xyzw(quat_xyzw: ArrayLike4) -> tuple[float, float, float, float]:
    return quaternion_xyzw_to_wxyz(quat_xyzw)


def xyzw_from_genesis_quat_wxyz(quat_wxyz: ArrayLike4) -> tuple[float, float, float, float]:
    return quaternion_wxyz_to_xyzw(quat_wxyz)
