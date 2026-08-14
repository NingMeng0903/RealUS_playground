"""Pluggable rail "goodness" metric g(q) and ∂g/∂y_rail.

Used by :class:`RailExtensionTask` as a singularity / reachability guardrail
(and, in scan mode, as a soft preference).  Default and production hot path
is σ_min (Yoshikawa / SVD of J).  IRD is one-shot ``d*`` at scan start only;
do not put ``IrdRailGoodness`` autograd on the 5 ms thread.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from rm75_control.control.joint_admittance_8dof.model import RobotKinematics
from rm75_control.control.joint_admittance_8dof.solver.sigma_grad import (
    sigma_min_grad_rail,
)


@runtime_checkable
class RailGoodness(Protocol):
    """Scalar configuration quality + rail directional derivative."""

    def g(self, q_rad: np.ndarray) -> float:
        """Higher is better (e.g. σ_min, μ, RegionA clearance)."""
        ...

    def dg_dy_rail(self, q_rad: np.ndarray) -> float:
        """∂g/∂y_rail under TCP-preserving coordinated motion (1/m)."""
        ...


class SigmaMinGoodness:
    """Default goodness: minimum singular value of the world Jacobian.

    ``dg_dy_rail`` is the TCP-preserving directional derivative from
    :func:`sigma_min_grad_rail` (naive ∂σ/∂q_rail is identically zero).
    """

    def __init__(self, kin: RobotKinematics) -> None:
        self.kin = kin

    def g(self, q_rad: np.ndarray) -> float:
        J = self.kin.jacobian(np.asarray(q_rad, dtype=float))
        return float(self.kin.singular_values(J).min())

    def dg_dy_rail(self, q_rad: np.ndarray) -> float:
        return float(sigma_min_grad_rail(self.kin, np.asarray(q_rad, dtype=float)))


class CachedRailGoodness:
    """Throttle expensive g / ∂g evaluations (e.g. ~20 Hz at 200 Hz control)."""

    def __init__(self, inner: RailGoodness, *, period_ticks: int = 10) -> None:
        self.inner = inner
        self.period_ticks = max(1, int(period_ticks))
        self._tick = 0
        self._g = 0.0
        self._dg = 0.0
        self._g_target = 0.0
        self._dg_target = 0.0
        self._slew_left = 0

    def refresh(self, q_rad: np.ndarray, *, force: bool = False) -> tuple[float, float]:
        self._tick += 1
        if force or self._tick == 1 or (self._tick % self.period_ticks == 0):
            self._g_target = float(self.inner.g(q_rad))
            self._dg_target = float(self.inner.dg_dy_rail(q_rad))
            self._slew_left = max(1, self.period_ticks)
        if self._slew_left > 0:
            alpha = 1.0 / float(self._slew_left)
            self._g += alpha * (self._g_target - self._g)
            self._dg += alpha * (self._dg_target - self._dg)
            self._slew_left -= 1
        else:
            self._g = float(self._g_target)
            self._dg = float(self._dg_target)
        return self._g, self._dg

    def g(self, q_rad: np.ndarray) -> float:
        return float(self.refresh(q_rad)[0])

    def dg_dy_rail(self, q_rad: np.ndarray) -> float:
        return float(self.refresh(q_rad)[1])
