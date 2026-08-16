from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from rm75_control.control.joint_admittance_8dof.loop import (
    _qdot_meas_8dof,
    _rail_execution_velocity_estimate,
)


class _Bridge:
    enabled = True
    config = SimpleNamespace(poll_hz=40.0, vel_max_m_s=0.30)

    def __init__(self, feedback) -> None:
        self.execution_feedback = feedback


def test_rail_execution_estimate_extrapolates_at_most_one_poll() -> None:
    feedback = SimpleNamespace(
        position_m=0.401,
        sample_mono_s=10.0,
        v_meas_m_s=0.04,
        v_cmd_m_s=0.06,
        a_cmd_m_s2=0.8,
        command_mode=SimpleNamespace(value="coupled_velocity"),
    )
    estimate = _rail_execution_velocity_estimate(
        _Bridge(feedback), now_s=10.04, freshness_s=0.05
    )
    assert estimate is not None
    assert estimate.position_m == pytest.approx(0.401)
    assert estimate.extrapolation_age_s == pytest.approx(0.025)
    assert estimate.velocity_m_s == pytest.approx(0.06)
    assert estimate.command_mode == "coupled_velocity"


def test_rail_execution_estimate_rejects_stale_feedback() -> None:
    feedback = SimpleNamespace(
        position_m=0.401,
        sample_mono_s=10.0,
        v_meas_m_s=0.04,
        v_cmd_m_s=0.06,
        a_cmd_m_s2=0.0,
        command_mode="coupled_velocity",
    )
    with pytest.raises(RuntimeError, match="stale"):
        _rail_execution_velocity_estimate(
            _Bridge(feedback), now_s=10.051, freshness_s=0.05
        )


def test_rail_execution_estimate_rejects_encoder_gated_sample() -> None:
    feedback = SimpleNamespace(
        position_m=0.401,
        sample_mono_s=10.0,
        v_meas_m_s=0.04,
        v_cmd_m_s=0.0,
        a_cmd_m_s2=0.0,
        valid=False,
        command_mode="coupled_velocity",
    )
    with pytest.raises(RuntimeError, match="encoder gate"):
        _rail_execution_velocity_estimate(
            _Bridge(feedback), now_s=10.001, freshness_s=0.05
        )


def test_rail_execution_estimate_is_none_without_active_rail() -> None:
    bridge = SimpleNamespace(enabled=False)
    assert (
        _rail_execution_velocity_estimate(
            bridge, now_s=10.0, freshness_s=0.05
        )
        is None
    )


def test_qdot_fallback_keeps_same_snapshot_rail_velocity() -> None:
    q_old = np.array([0.400, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7])
    q_new = q_old.copy()
    q_new[0] += 0.004  # finite difference would incorrectly report 0.8 m/s
    q_new[1] += 0.001
    qdot = _qdot_meas_8dof(
        q_new,
        q_old,
        0.005,
        SimpleNamespace(qdot_deg_s=None),
        rail_velocity_m_s=0.031,
    )
    assert qdot is not None
    assert qdot[0] == pytest.approx(0.031)
    assert qdot[1] == pytest.approx(0.2)
