"""Map bed contact stiffness (N/m) to equilibrium support sink depth."""

from __future__ import annotations

import numpy as np

REF_STIFFNESS_N_PER_M = 6.0e5
REF_EQUILIBRIUM_SINK_M = 0.028
MIN_SINK_M = 0.003
MAX_SINK_M = 0.055


def target_support_sink_m(stiffness_n_per_m: float) -> float:
    """Equilibrium sink depth (m, positive = into mattress). Higher k -> less sink."""

    k = max(1.0e4, float(stiffness_n_per_m))
    sink = REF_EQUILIBRIUM_SINK_M * REF_STIFFNESS_N_PER_M / k
    return float(np.clip(sink, MIN_SINK_M, MAX_SINK_M))


def target_support_phi_m(stiffness_n_per_m: float) -> float:
    """Target support phi = z - plane_z (negative means below plane / sunk in)."""

    return -target_support_sink_m(stiffness_n_per_m)
