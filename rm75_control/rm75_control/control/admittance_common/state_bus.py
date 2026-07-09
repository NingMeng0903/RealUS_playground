"""Shared robot state fan-out: one UDP push observer, many readers."""

from __future__ import annotations

from typing import Any

import numpy as np

from rm75_control.control.admittance_common.async_state import (
    AsyncStateSnapshot,
    RealtimeStateObserver,
    create_state_observer,
)
from rm75_control.control.joint_admittance_8dof.model import deg2rad, full_q_from_arm


def expand_q_meas_8dof(q_deg_or_rad: np.ndarray, rail_m: float) -> np.ndarray:
    """Realman feedback is 7 arm joints; prepend rail for 8-DOF FK / viz."""
    q = np.asarray(q_deg_or_rad, dtype=float)
    if q.size >= 8:
        return q[:8].copy()
    if q.size == 7:
        if np.max(np.abs(q)) > 2.0 * np.pi:
            q = deg2rad(q)
        return full_q_from_arm(q, rail_m)
    raise ValueError(f"expected 7 or 8 joint values, got {q.size}")


class RobotStateBus:
    """Owns exactly one ``RealtimeStateObserver``; WBC and digital twin share ``read()``."""

    def __init__(
        self,
        robot,
        raw_config: dict[str, Any] | None = None,
        *,
        robot_ip: str | None = None,
        observer: RealtimeStateObserver | None = None,
    ) -> None:
        if observer is not None:
            self._obs = observer
            self._external = True
        else:
            self._obs = create_state_observer(robot, raw_config, robot_ip=robot_ip)
            self._external = False

    @property
    def observer(self) -> RealtimeStateObserver:
        return self._obs

    @property
    def push_period_ms(self) -> float:
        return float(self._obs.push_period_ms)

    def start(self) -> None:
        self._obs.start()

    def stop(self) -> None:
        if not self._external:
            self._obs.stop()

    def read(self) -> AsyncStateSnapshot:
        return self._obs.read()

    def wait_first_pose(self, timeout_s: float = 5.0) -> np.ndarray:
        return self._obs.wait_first_pose(timeout_s=timeout_s)

    def q_meas_8dof(self, rail_m: float) -> np.ndarray | None:
        snap = self.read()
        if snap.q_deg is None:
            return None
        return expand_q_meas_8dof(snap.q_deg, rail_m)
