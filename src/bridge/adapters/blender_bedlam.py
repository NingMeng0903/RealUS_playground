from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def smpl_yup_to_blender(vec: Sequence[float]) -> np.ndarray:
    x, y, z = (float(v) for v in vec)
    return np.asarray([x, z, -y], dtype=np.float64)


def bedlam_unreal_to_smpl_translation(vec_xyz_m: Sequence[float]) -> np.ndarray:
    x, y, z = (float(v) for v in vec_xyz_m)
    return np.asarray([x, z, y], dtype=np.float64)
