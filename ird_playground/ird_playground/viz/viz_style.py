"""Shared figure style for capability / global-IRD cross-probe compare."""

from __future__ import annotations

# Zacharias D fraction → colour bar ticks 0 … BAR_MAX (same units as capability figures).
PROBE_COMPARE_CLIM: tuple[float, float] = (0.0, 0.18)
PROBE_COMPARE_BAR_MAX: float = 18.0
PROBE_COMPARE_N_LEVELS: int = 8
PROBE_COMPARE_D_MIN: float = 0.02
SPHERE_RADIUS_FACTOR: float = 0.48
