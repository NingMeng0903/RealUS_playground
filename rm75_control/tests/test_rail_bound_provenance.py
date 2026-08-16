"""Low-cost provenance diagnostics for the rail velocity box."""

from __future__ import annotations

import numpy as np

from rm75_control.control.joint_admittance_8dof.solver.constraint_mgr import (
    VelocityBoxConstraints,
)
from rm75_control.control.joint_admittance_8dof.utils.safety import SafetyLimits


def _limits(*, acceleration: float | None = None) -> SafetyLimits:
    return SafetyLimits(
        q_lower=np.full(8, -5.0),
        q_upper=np.full(8, 5.0),
        v_max=np.array([0.1, *([1.0] * 7)]),
        a_max=(None if acceleration is None else np.full(8, acceleration)),
        position_margin=np.full(8, 0.05),
    )


def test_velocity_provenance_reports_only_final_active_side() -> None:
    box = VelocityBoxConstraints(_limits(), damper_band_rad=0.0)
    _lo, hi = box.bounds(np.zeros(8), dt=0.1)

    assert box.active_rail_bounds(hi[0]) == {"velocity_upper"}
    assert box.active_rail_bounds(0.0) == set()
    assert box.last_rail_bounds == (float(_lo[0]), float(hi[0]))


def test_acceleration_and_jerk_sources_are_distinguished() -> None:
    box = VelocityBoxConstraints(_limits(acceleration=0.2), damper_band_rad=0.0)
    qdot_prev = np.zeros(8)
    qdot_prev[0] = 0.02
    qdot_prev2 = qdot_prev.copy()
    _lo, hi = box.bounds(
        np.zeros(8),
        dt=0.1,
        qdot_prev=qdot_prev,
    )
    assert box.active_rail_bounds(hi[0]) == {"acceleration_upper"}

    _lo, hi = box.bounds(
        np.zeros(8),
        dt=0.1,
        qdot_prev=qdot_prev,
        qdot_prev2=qdot_prev2,
        j_max=np.full(8, 0.1),
    )
    assert box.active_rail_bounds(hi[0]) == {"jerk_upper"}


def test_position_lead_and_pin_are_reported_at_final_boundary() -> None:
    box = VelocityBoxConstraints(_limits(), damper_band_rad=0.0)
    _lo, hi = box.bounds(np.array([4.95, *([0.0] * 7)]), dt=0.1)
    assert hi[0] == 0.0
    assert box.active_rail_bounds(0.0) == {"position_upper"}

    _lo, hi = box.bounds(
        np.zeros(8),
        dt=0.1,
        q_meas=np.zeros(8),
        q_cmd=np.array([0.019, *([0.0] * 7)]),
        resync_err=np.array([0.02, *([0.0] * 7)]),
    )
    assert box.active_rail_bounds(hi[0]) == {"lead_upper"}

    _lo, hi = box.bounds(np.zeros(8), dt=0.1, rail_vel_pin_m_s=0.03)
    assert _lo[0] == hi[0] == 0.03
    assert box.active_rail_bounds(0.03) == {"pin_lower", "pin_upper"}

