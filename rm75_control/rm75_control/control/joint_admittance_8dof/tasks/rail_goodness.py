"""Pluggable rail "goodness" metric g(q) and ∂g/∂y_rail.

Used by :class:`RailExtensionTask` as a singularity / reachability guardrail
(and, in scan mode, as a soft preference).  Default implementation is σ_min
(Yoshikawa / SVD of J).  Swap in ``RegionARailGoodness`` (IRD RegionA adapter)
when ``ird_playground`` is installed — the rail task does not change.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

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


class RegionARailGoodness:
    """Thin optional adapter: IRD ``RegionA`` robust clearance as rail goodness.

    Imports ``ird_playground`` lazily so ``rm75_control`` still loads without
    IRD installed.  ``dg_dy_rail`` is a finite-difference under rail motion
    (RegionA re-bases the axis frame via ``query_tcp_rail``).
    """

    def __init__(
        self,
        kin: RobotKinematics,
        field: Any,
        *,
        region_a: Any | None = None,
        T_world_rail: np.ndarray | None = None,
        T_rail_base0: np.ndarray | None = None,
        fd_eps_m: float = 1.0e-4,
        device: str = "cpu",
    ) -> None:
        try:
            import torch
            from ird_playground.region.operator import RegionA, RegionAConfig
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "RegionARailGoodness requires ird_playground (optional dependency)"
            ) from exc
        self._torch = torch
        self.kin = kin
        self.field = field
        self.region_a = region_a or RegionA(RegionAConfig())
        self.T_world_rail = (
            np.eye(4, dtype=np.float64)
            if T_world_rail is None
            else np.asarray(T_world_rail, dtype=np.float64).reshape(4, 4)
        )
        self.T_rail_base0 = (
            np.eye(4, dtype=np.float64)
            if T_rail_base0 is None
            else np.asarray(T_rail_base0, dtype=np.float64).reshape(4, 4)
        )
        self.fd_eps_m = float(fd_eps_m)
        self.device = device

    def _pose_tensor(self, q_rad: np.ndarray):
        pose = self.kin.fk_pose(np.asarray(q_rad, dtype=float))
        from scipy.spatial.transform import Rotation as Rsc

        T = np.eye(4, dtype=np.float64)
        T[:3, 3] = pose[:3]
        T[:3, :3] = Rsc.from_euler(
            self.kin.euler_order, pose[3:6], degrees=False
        ).as_matrix()
        return self._torch.as_tensor(T, dtype=self._torch.float32, device=self.device)

    def g(self, q_rad: np.ndarray) -> float:
        q = np.asarray(q_rad, dtype=float)
        rail = float(q[0])
        T_tcp = self._pose_tensor(q)
        Tw = self._torch.as_tensor(
            self.T_world_rail, dtype=self._torch.float32, device=self.device
        )
        Tb = self._torch.as_tensor(
            self.T_rail_base0, dtype=self._torch.float32, device=self.device
        )
        rail_t = self._torch.as_tensor(rail, dtype=self._torch.float32, device=self.device)
        with self._torch.no_grad():
            result = self.region_a.query_tcp_rail(
                self.field, T_tcp, rail_t, T_world_rail=Tw, T_rail_base0=Tb
            )
            val = result.robust_clearance
        return float(np.asarray(val.detach().cpu()).reshape(-1)[0])

    def dg_dy_rail(self, q_rad: np.ndarray) -> float:
        q = np.asarray(q_rad, dtype=float).copy()
        eps = self.fd_eps_m
        q_hi = q.copy()
        q_lo = q.copy()
        q_hi[0] += eps
        q_lo[0] -= eps
        return (self.g(q_hi) - self.g(q_lo)) / (2.0 * eps)


class CachedRailGoodness:
    """Throttle expensive g / ∂g evaluations (e.g. ~20 Hz at 200 Hz control)."""

    def __init__(self, inner: RailGoodness, *, period_ticks: int = 10) -> None:
        self.inner = inner
        self.period_ticks = max(1, int(period_ticks))
        self._tick = 0
        self._g = 0.0
        self._dg = 0.0

    def refresh(self, q_rad: np.ndarray, *, force: bool = False) -> tuple[float, float]:
        self._tick += 1
        if force or self._tick == 1 or (self._tick % self.period_ticks == 0):
            self._g = float(self.inner.g(q_rad))
            self._dg = float(self.inner.dg_dy_rail(q_rad))
        return self._g, self._dg

    def g(self, q_rad: np.ndarray) -> float:
        return float(self.refresh(q_rad)[0])

    def dg_dy_rail(self, q_rad: np.ndarray) -> float:
        return float(self.refresh(q_rad)[1])
