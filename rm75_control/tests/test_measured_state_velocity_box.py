"""The P0 safety geometry and command integrator use distinct state roles."""

import numpy as np
import pytest

from rm75_control.control.joint_admittance_8dof.solver.constraint_mgr import (
    VelocityBoxConstraints,
    stopping_velocity,
)
from rm75_control.control.joint_admittance_8dof.utils.safety import SafetyLimits


def test_position_damper_uses_measured_state_but_lead_uses_command_state() -> None:
    limits = SafetyLimits(
        q_lower=np.full(8, -1.0),
        q_upper=np.full(8, 1.0),
        v_max=np.ones(8),
        a_max=None,
        position_margin=np.full(8, 0.1),
    )
    box = VelocityBoxConstraints(limits, damper_band_rad=0.2)
    q_meas = np.zeros(8)
    q_cmd = np.zeros(8)
    q_cmd[3] = 0.25

    lo, hi = box.bounds(
        q_meas,
        dt=0.01,
        q_meas=q_meas,
        q_cmd=q_cmd,
        resync_err=np.full(8, 0.2),
    )

    # Measured q is centred, so position damping leaves negative motion free.
    assert lo[3] == -1.0
    # Command is ahead of measurement, so only further positive lead is blocked.
    assert hi[3] == 0.0


def test_rail_lead_exempt_keeps_coupled_velocity_box_open() -> None:
    limits = SafetyLimits(
        q_lower=np.full(8, 0.0),
        q_upper=np.full(8, 0.8),
        v_max=np.full(8, 0.30),
        a_max=None,
        position_margin=np.zeros(8),
    )
    box = VelocityBoxConstraints(limits, damper_band_rad=0.0)
    q_meas = np.zeros(8)
    q_meas[0] = 0.686
    q_cmd = np.zeros(8)
    q_cmd[0] = 0.666
    resync = np.r_[0.02, np.full(7, 0.10)]
    lo, hi = box.bounds(
        q_meas,
        dt=0.005,
        q_meas=q_meas,
        q_cmd=q_cmd,
        resync_err=resync,
    )
    assert lo[0] >= -1.0e-12
    lo_open, hi_open = box.bounds(
        q_meas,
        dt=0.005,
        q_meas=q_meas,
        q_cmd=q_cmd,
        resync_err=resync,
        rail_lead_exempt=True,
    )
    assert lo_open[0] < -0.05
    assert hi_open[0] > 0.05


def test_legacy_call_without_q_cmd_keeps_previous_semantics() -> None:
    limits = SafetyLimits(
        q_lower=np.full(2, -1.0),
        q_upper=np.full(2, 1.0),
        v_max=np.ones(2),
        a_max=None,
        position_margin=np.zeros(2),
    )
    box = VelocityBoxConstraints(limits, damper_band_rad=0.0)
    q = np.array([0.15, 0.0])
    lo, hi = box.bounds(q, 0.01, q_meas=np.zeros(2), resync_err=0.2)
    assert np.all(lo <= hi)
    assert hi[0] < 1.0


def test_command_lead_braking_remains_acceleration_feasible() -> None:
    limits = SafetyLimits(
        q_lower=np.full(8, -2.0),
        q_upper=np.full(8, 2.0),
        v_max=np.ones(8),
        a_max=np.full(8, 18.0),
        position_margin=np.zeros(8),
    )
    box = VelocityBoxConstraints(limits, damper_band_rad=0.0)
    q_meas = np.zeros(8)
    q_cmd = np.zeros(8)
    q_cmd[5] = np.deg2rad(5.95)
    previous = np.zeros(8)
    previous[5] = 0.30

    lo, hi = box.bounds(
        q_meas,
        dt=0.005,
        qdot_prev=previous,
        q_meas=q_meas,
        q_cmd=q_cmd,
        resync_err=np.full(8, np.deg2rad(6.0)),
    )

    assert np.all(lo <= hi)
    # The command is already outside the stopping-distance viability kernel.
    # Use maximum feasible braking rather than raising VelocityBoxInfeasible.
    assert lo[5] == hi[5]
    assert hi[5] == previous[5] - limits.a_max[5] * 0.005


def test_command_lead_viability_property_over_signs_and_distances() -> None:
    limits = SafetyLimits(
        q_lower=np.full(8, -3.0),
        q_upper=np.full(8, 3.0),
        v_max=np.ones(8),
        a_max=np.full(8, 18.0),
        position_margin=np.zeros(8),
    )
    box = VelocityBoxConstraints(limits, damper_band_rad=0.0)
    q_meas = np.zeros(8)
    resync = np.full(8, np.deg2rad(6.0))
    for lead in np.linspace(-1.2 * resync[3], 1.2 * resync[3], 49):
        for previous in np.linspace(-0.8, 0.8, 33):
            q_cmd = np.zeros(8)
            q_cmd[3] = lead
            qdot_prev = np.zeros(8)
            qdot_prev[3] = previous
            lo, hi = box.bounds(
                q_meas,
                dt=0.005,
                qdot_prev=qdot_prev,
                q_meas=q_meas,
                q_cmd=q_cmd,
                resync_err=resync,
            )
            assert np.all(lo <= hi + 1.0e-12)
            assert lo[3] >= previous - limits.a_max[3] * 0.005 - 1.0e-12
            assert hi[3] <= previous + limits.a_max[3] * 0.005 + 1.0e-12


@pytest.mark.parametrize(
    ("lead", "previous"),
    (
        (0.063405, 0.9818717760492348),
        (-0.063673, -1.167957179249451),
        (0.067031, 0.5735964001943359),
        (-0.064026, -0.9184248983451805),
        (0.065912, 0.7663564078561553),
    ),
)
def test_hardware_prefault_command_lead_snapshots_brake_without_empty_box(
    lead: float,
    previous: float,
) -> None:
    """Reconstruct the next box from each final successful hardware tick."""

    limits = SafetyLimits(
        q_lower=np.full(8, -3.0),
        q_upper=np.full(8, 3.0),
        v_max=np.full(8, 2.0),
        a_max=np.full(8, 18.0),
        position_margin=np.zeros(8),
    )
    box = VelocityBoxConstraints(limits, damper_band_rad=0.0)
    q_meas = np.zeros(8)
    q_cmd = np.zeros(8)
    qdot_prev = np.zeros(8)
    q_cmd[3] = lead
    qdot_prev[3] = previous

    lo, hi = box.bounds(
        q_meas,
        dt=0.005,
        qdot_prev=qdot_prev,
        q_meas=q_meas,
        q_cmd=q_cmd,
        resync_err=np.full(8, np.deg2rad(6.0)),
    )

    assert np.all(lo <= hi)
    remaining = np.deg2rad(6.0) - abs(lead)
    viable_speed = float(stopping_velocity(remaining, 18.0, 0.005))
    assert hi[3] <= viable_speed + 1.0e-12
    assert lo[3] >= -viable_speed - 1.0e-12
    assert lo[3] >= previous - 18.0 * 0.005 - 1.0e-12
    assert hi[3] <= previous + 18.0 * 0.005 + 1.0e-12


@pytest.mark.parametrize(
    ("lead", "previous"),
    (
        (0.063405, 0.9818717760492348),
        (-0.063673, -1.167957179249451),
        (0.067031, 0.5735964001943359),
        (-0.064026, -0.9184248983451805),
        (0.065912, 0.7663564078561553),
    ),
)
def test_hardware_prefault_snapshot_stops_at_lead_limit_with_frozen_feedback(
    lead: float,
    previous: float,
) -> None:
    """Even the most adverse admissible command must remain lead-viable."""

    limits = SafetyLimits(
        q_lower=np.full(8, -3.0),
        q_upper=np.full(8, 3.0),
        v_max=np.full(8, 2.0),
        a_max=np.full(8, 18.0),
        position_margin=np.zeros(8),
    )
    box = VelocityBoxConstraints(limits, damper_band_rad=0.0)
    q_meas = np.zeros(8)
    q_cmd = np.zeros(8)
    qdot_prev = np.zeros(8)
    q_cmd[3] = lead
    qdot_prev[3] = previous
    threshold = np.deg2rad(6.0)

    for _ in range(100):
        lo, hi = box.bounds(
            q_meas,
            dt=0.005,
            qdot_prev=qdot_prev,
            q_meas=q_meas,
            q_cmd=q_cmd,
            resync_err=np.full(8, threshold),
        )
        chosen = hi[3] if lead > 0.0 else lo[3]
        q_cmd[3] += chosen * 0.005
        qdot_prev[3] = chosen
        assert abs(q_cmd[3]) <= threshold + 1.0e-12

    assert abs(qdot_prev[3]) <= 1.0e-12
    assert abs(q_cmd[3]) == pytest.approx(threshold, abs=1.0e-12)
