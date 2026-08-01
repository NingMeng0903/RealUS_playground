"""Shared figure style for capability / global-IRD cross-probe compare."""

from __future__ import annotations

# Zacharias D fraction → colour bar ticks 0 … BAR_MAX (same units as capability figures).
PROBE_COMPARE_CLIM: tuple[float, float] = (0.0, 0.18)
PROBE_COMPARE_BAR_MAX: float = 18.0
PROBE_COMPARE_N_LEVELS: int = 8
PROBE_COMPARE_D_MIN: float = 0.02
SPHERE_RADIUS_FACTOR: float = 0.48

# Fixed display framing after shoulder Y-centering (robot on y=0 cut).
# Identical for every mount so the arm sits at the same pixel size/position.
MOUNT_COMPARE_BOUNDS: tuple[float, float, float, float, float, float] = (
    -1.05,
    1.05,
    -0.12,
    1.20,
    -0.35,
    1.20,
)
# Focus ON the cut plane (y=0) through the arm — not inside the hemisphere,
# otherwise the left 45° panel parks the robot on the rim.
MOUNT_COMPARE_FOCUS: tuple[float, float, float] = (0.0, 0.0, 0.45)
MOUNT_COMPARE_PARALLEL_SCALE: float = 1.05
MOUNT_COMPARE_OBLIQUE_SPAN: float = 2.0

# IRD figures: world origin = TCP, arm extends toward −Z.
MOUNT_COMPARE_IRD_BOUNDS: tuple[float, float, float, float, float, float] = (
    -1.05,
    1.05,
    -0.12,
    1.20,
    -1.15,
    0.55,
)
MOUNT_COMPARE_IRD_FOCUS: tuple[float, float, float] = (0.0, 0.0, -0.15)
