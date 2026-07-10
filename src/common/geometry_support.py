from __future__ import annotations

import numpy as np


def support_plane_shift(vertices_world: np.ndarray, bed_z: float) -> float:
    support_z = float(np.percentile(vertices_world[:, 2], 2.0))
    return bed_z - support_z


def lower_shell_mask(vertices_world: np.ndarray, lower_quantile: float = 0.22) -> np.ndarray:
    z = vertices_world[:, 2].astype(np.float64)
    threshold = float(np.quantile(z, float(np.clip(lower_quantile, 0.05, 0.45))))
    return z <= threshold


def support_plane_shift_masked(
    vertices_world: np.ndarray,
    bed_z: float,
    mask: np.ndarray,
    percentile: float = 4.0,
) -> float:
    z = vertices_world[:, 2].astype(np.float64)
    masked_z = z[mask.astype(bool)]
    if masked_z.size < 12:
        return support_plane_shift(vertices_world, bed_z)
    support_z = float(np.percentile(masked_z, percentile))
    return float(bed_z - support_z)
