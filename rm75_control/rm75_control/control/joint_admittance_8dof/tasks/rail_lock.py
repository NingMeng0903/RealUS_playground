"""Configuration for ownership of the continuous prismatic rail DOF."""

from __future__ import annotations

from dataclasses import dataclass

from rm75_control.control.joint_admittance_8dof.tasks.rail_mode import LockedStyle, RailMode


@dataclass
class RailLockConfig:
    """Rail control configuration.

    Fields under "lock_*" only take effect in ``LOCKED + HOLD``.  ``v_max_m_s``
    and travel/visual metadata apply to all modes.
    """

    mode: RailMode = RailMode.LOCKED
    locked_style: LockedStyle = LockedStyle.HOLD
    q_ref_m: float | None = None  # legacy/offline rail reference; HOLD moves still use set_locked()
    lock_vel_eps_m_s: float = 0.0  # rail velocity box in HOLD (m/s)
    # Rail speed / geometry (used by planners and safety limits)
    v_max_m_s: float | None = None
    travel_m: float = 0.80         # mechanical [0, travel_m] m (rail_y=0 at -Y end)
    soft_min_m: float = 0.01       # usable command band (host soft limits)
    soft_max_m: float = 0.78

    def __post_init__(self) -> None:
        if isinstance(self.mode, str):
            self.mode = RailMode(self.mode)
        if isinstance(self.locked_style, str):
            self.locked_style = LockedStyle(self.locked_style)

    @property
    def is_locked_hold(self) -> bool:
        return self.mode == RailMode.LOCKED and self.locked_style == LockedStyle.HOLD
