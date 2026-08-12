"""The P0 safety geometry and command integrator use distinct state roles."""

import numpy as np

from rm75_control.control.joint_admittance_8dof.solver.constraint_mgr import (
    VelocityBoxConstraints,
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
