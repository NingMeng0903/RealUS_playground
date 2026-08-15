"""SDK joint_speed is preferred over finite-differenced q_meas."""

from __future__ import annotations

import numpy as np
import pytest

from rm75_control.control.admittance_common.async_state import AsyncStateSnapshot
from rm75_control.control.joint_admittance_8dof.loop import _qdot_meas_8dof


class _Rail:
    def __init__(self, v_m_s: float) -> None:
        self.measured_speed_m_s = float(v_m_s)


def test_qdot_meas_uses_sdk_arm_speed_not_delta_q():
    q0 = np.zeros(8)
    q1 = np.array([0.01, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    snap = AsyncStateSnapshot(
        qdot_deg_s=np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0, -5.0])
    )
    qdot = _qdot_meas_8dof(q1, q0, 0.005, snap, _Rail(0.03))
    assert qdot is not None
    assert qdot[0] == pytest.approx(0.03)
    np.testing.assert_allclose(qdot[1], np.deg2rad(10.0))
    np.testing.assert_allclose(qdot[7], np.deg2rad(-5.0))
    # Finite-diff of J1 would be 0.2/0.005 = 40 rad/s; SDK must win.
    assert abs(qdot[1] - 40.0) > 1.0


def test_qdot_meas_falls_back_to_finite_diff_without_sdk():
    q0 = np.zeros(8)
    q1 = np.array([0.0, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    qdot = _qdot_meas_8dof(q1, q0, 0.005, AsyncStateSnapshot())
    assert qdot is not None
    np.testing.assert_allclose(qdot[1], 2.0)


def test_qdot_meas_sdk_without_rail_keeps_arm():
    q1 = np.zeros(8)
    snap = AsyncStateSnapshot(qdot_deg_s=np.full(7, 3.0))
    qdot = _qdot_meas_8dof(q1, None, None, snap, None)
    assert qdot is not None
    np.testing.assert_allclose(qdot[1:], np.deg2rad(3.0))
    assert qdot[0] == 0.0
