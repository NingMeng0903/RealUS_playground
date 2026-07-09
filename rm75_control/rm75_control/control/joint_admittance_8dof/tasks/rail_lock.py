"""Rail prismatic DOF hold task (used only in RailMode.LOCKED + LockedStyle.HOLD).

The other LOCKED styles (RAIL_ONLY / TCP_FIXED) do not use this task: they let
the external plan drive ``qdot_ff[0]`` and the QP box pin the rail velocity to
that value.  RailMode.COUPLED lets the QP decide rail motion itself (subject to
reg / v_max / a_max / resync from the standard SafetyLimits path).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rm75_control.control.joint_admittance_8dof.model import RAIL_INDEX
from rm75_control.control.joint_admittance_8dof.tasks.rail_mode import LockedStyle, RailMode


@dataclass
class RailLockConfig:
    """Rail control configuration.

    Fields under "lock_*" only take effect in ``LOCKED + HOLD``.  ``v_max_m_s``
    and travel/visual metadata apply to all modes.
    """

    mode: RailMode = RailMode.LOCKED
    locked_style: LockedStyle = LockedStyle.HOLD
    q_ref_m: float | None = None
    # HOLD-only knobs
    lock_gain: float = 200.0
    lock_reg_scale: float = 100.0  # multiply qp.reg[0] when HOLD-locked
    lock_vel_eps_m_s: float = 0.0  # rail velocity box in HOLD (m/s)
    lock_hard_pin: bool = True     # after QP, pin q_cmd[0] = q_ref every tick
    # Rail speed / geometry (used by planners and safety limits)
    v_max_m_s: float | None = None
    travel_m: float = 0.50         # ±0.25 m effective travel (500 mm)

    def __post_init__(self) -> None:
        if isinstance(self.mode, str):
            self.mode = RailMode(self.mode)
        if isinstance(self.locked_style, str):
            self.locked_style = LockedStyle(self.locked_style)

    @property
    def is_locked_hold(self) -> bool:
        return self.mode == RailMode.LOCKED and self.locked_style == LockedStyle.HOLD


class RailLockTask:
    """When ``LOCKED + HOLD``, pull rail_y toward q_ref (m/s per m error)."""

    def __init__(self, cfg: RailLockConfig | None = None) -> None:
        self.cfg = cfg or RailLockConfig()
        self.q_ref = self.cfg.q_ref_m

    def reset(self, q_rad: np.ndarray) -> None:
        if self.q_ref is None:
            self.q_ref = float(np.asarray(q_rad, dtype=float)[RAIL_INDEX])

    def set_reference(self, q_ref_m: float) -> None:
        self.q_ref = float(q_ref_m)

    @property
    def active(self) -> bool:
        """Task is only meaningful in LOCKED + HOLD."""
        return self.cfg.is_locked_hold and self.q_ref is not None

    def __call__(self, q_rad: np.ndarray) -> np.ndarray:
        qdot0 = np.zeros_like(np.asarray(q_rad, dtype=float))
        if not self.active:
            return qdot0
        err = float(q_rad[RAIL_INDEX]) - float(self.q_ref)
        qdot0[RAIL_INDEX] = -self.cfg.lock_gain * err
        return qdot0
