"""Rail velocity-box bind-stage telemetry: isolate one stage, then reset."""

from __future__ import annotations

import numpy as np
import pytest

from rm75_control.control.joint_admittance_8dof.collision_model import CollisionConfig
from rm75_control.control.joint_admittance_8dof.loop import JointIkConfig, JointIkController
from rm75_control.control.joint_admittance_8dof.model import RobotKinematics, full_q_from_arm
from rm75_control.control.joint_admittance_8dof.solver.constraint_mgr import (
    RAIL_BIND_ACCEL,
    RAIL_BIND_JERK,
    RAIL_BIND_LEAD,
    RAIL_BIND_LOCKED,
    RAIL_BIND_NONE,
    RAIL_BIND_PIN,
    RAIL_BIND_VMAX_DAMPER,
    RAIL_BIND_WALL_CAP,
    VelocityBoxConstraints,
)
from rm75_control.control.joint_admittance_8dof.solver.qp_builder import QpConfig
from rm75_control.control.joint_admittance_8dof.utils.safety import SafetyLimits


Q_SAFE = full_q_from_arm(
    np.deg2rad([5.0, -30.0, 10.0, 60.0, -5.0, 45.0, 0.0]), 0.40
)
_DT = 0.005
_V_MAX = 0.12


def _mid_q(q0: float = 0.40) -> np.ndarray:
    q = np.zeros(8)
    q[0] = float(q0)
    return q


def _rail_box(*, a_max=None, reaction_s=0.0, band=0.0, v_max=_V_MAX):
    nv = 8
    a = None if a_max is None else np.concatenate(([float(a_max)], np.full(nv - 1, 3.0)))
    lim = SafetyLimits(
        q_lower=np.concatenate(([0.005], np.full(nv - 1, -2.0))),
        q_upper=np.concatenate(([0.78], np.full(nv - 1, 2.0))),
        v_max=np.concatenate(([float(v_max)], np.full(nv - 1, 1.0))),
        a_max=a,
        position_margin=np.zeros(nv),
    )
    bands = np.full(nv, 0.15)
    bands[0] = float(band)
    return VelocityBoxConstraints(
        lim, damper_band_rad=bands, rail_reaction_s=float(reaction_s)
    )


def test_vmax_damper_binds_at_mid_rail_without_rate_limits() -> None:
    box = _rail_box(a_max=None, reaction_s=0.0, band=0.0, v_max=_V_MAX)
    q = _mid_q()
    lo, hi = box.bounds(q, _DT, qdot_prev=None, q_meas=q, q_cmd=q)
    assert box.last_rail_bind_lo == RAIL_BIND_VMAX_DAMPER
    assert box.last_rail_bind_hi == RAIL_BIND_VMAX_DAMPER
    assert lo[0] == pytest.approx(-_V_MAX)
    assert hi[0] == pytest.approx(_V_MAX)


def test_accel_stage_binds_when_a_max_is_tiny() -> None:
    box = _rail_box(a_max=0.001, reaction_s=0.0, band=0.0, v_max=_V_MAX)
    q = _mid_q()
    prev = np.zeros(8)
    prev[0] = 0.05
    lo, hi = box.bounds(
        q,
        _DT,
        qdot_prev=prev,
        q_meas=q,
        q_cmd=q,
        qdot_prev2=prev,
        j_max=np.full(8, 1.0e6),
        box_h1=_DT,
        box_h2=_DT,
    )
    assert box.last_rail_bind_lo == RAIL_BIND_ACCEL
    assert box.last_rail_bind_hi == RAIL_BIND_ACCEL
    assert hi[0] == pytest.approx(0.05 + 0.001 * _DT, abs=1.0e-12)
    assert lo[0] == pytest.approx(0.05 - 0.001 * _DT, abs=1.0e-12)


def test_jerk_stage_binds_when_j_max_is_tiny() -> None:
    box = _rail_box(a_max=0.60, reaction_s=0.0, band=0.0, v_max=_V_MAX)
    q = _mid_q()
    prev = np.zeros(8)
    prev2 = np.zeros(8)
    prev[0] = 0.050
    prev2[0] = 0.049
    j_max = np.full(8, 10.0)
    lo, hi = box.bounds(
        q,
        _DT,
        qdot_prev=prev,
        q_meas=q,
        q_cmd=q,
        qdot_prev2=prev2,
        j_max=j_max,
        box_h1=_DT,
        box_h2=_DT,
    )
    centre = 0.050 + (0.050 - 0.049)
    span = 10.0 * _DT * _DT
    assert box.last_rail_bind_lo == RAIL_BIND_JERK
    assert box.last_rail_bind_hi == RAIL_BIND_JERK
    assert lo[0] == pytest.approx(centre - span, abs=1.0e-12)
    assert hi[0] == pytest.approx(centre + span, abs=1.0e-12)


def test_wall_cap_binds_near_travel_end_without_damper() -> None:
    box = _rail_box(a_max=0.60, reaction_s=0.06, band=0.0, v_max=_V_MAX)
    q = _mid_q(0.775)
    lo, hi = box.bounds(q, _DT, qdot_prev=None, q_meas=q, q_cmd=q)
    assert box.last_rail_bind_hi == RAIL_BIND_WALL_CAP
    assert hi[0] < _V_MAX - 1.0e-6


def test_lead_resync_binds_when_command_is_ahead() -> None:
    box = _rail_box(a_max=0.60, reaction_s=0.06, band=0.0, v_max=_V_MAX)
    q = _mid_q(0.40)
    cmd = _mid_q(0.43)
    re = np.full(8, 0.15)
    re[0] = 0.020
    lo, hi = box.bounds(
        q, _DT, qdot_prev=None, q_meas=q, q_cmd=cmd, resync_err=re
    )
    assert box.last_rail_bind_hi == RAIL_BIND_LEAD
    assert hi[0] < 1.0e-9


def test_pin_and_lock_overwrite_bind_stage() -> None:
    box = _rail_box(a_max=None, reaction_s=0.0, band=0.0, v_max=_V_MAX)
    q = _mid_q()
    box.bounds(q, _DT, qdot_prev=None, q_meas=q, q_cmd=q, rail_vel_pin_m_s=0.01)
    assert box.last_rail_bind_lo == RAIL_BIND_PIN
    assert box.last_rail_bind_hi == RAIL_BIND_PIN
    box.bounds(q, _DT, qdot_prev=np.zeros(8), q_meas=q, q_cmd=q, rail_locked=True)
    assert box.last_rail_bind_lo == RAIL_BIND_LOCKED
    assert box.last_rail_bind_hi == RAIL_BIND_LOCKED


def test_qp_core_reset_clears_rail_box_attribution() -> None:
    qp = QpConfig(backend="proxqp", collision=CollisionConfig(enabled=False))
    cfg = JointIkConfig(control_frame="base", qp=qp, collision=CollisionConfig(enabled=False))
    controller = JointIkController(RobotKinematics(), cfg)
    controller.reset(Q_SAFE)
    core = controller.core
    assert core.last_rail_bind_lo == RAIL_BIND_NONE
    assert core.last_rail_bind_hi == RAIL_BIND_NONE
    assert core.last_rail_box_lo == 0.0
    assert core.last_rail_box_hi == 0.0
    assert core.last_rail_task_vel_used == 0.0
    J = core.kin.jacobian(Q_SAFE)
    core.step(
        Q_SAFE,
        np.array([0.0, 0.05, 0.0, 0.0, 0.0, 0.0]),
        0.005,
        q_meas=Q_SAFE,
        rail_exec_vel_m_s=0.0,
        rail_task_vel_m_s=0.05,
        rail_task_weight=64.0,
        zero_secondary_rail=True,
        jacobian=J,
        sigma=core.kin.singular_values(J),
        box_h1=0.005,
        box_h2=0.005,
    )
    assert np.isfinite(core.last_rail_box_lo)
    assert np.isfinite(core.last_rail_box_hi)
    assert int(core.last_rail_bind_hi) != RAIL_BIND_NONE or int(core.last_rail_bind_lo) != RAIL_BIND_NONE
    controller.reset(Q_SAFE)
    assert core.last_rail_bind_lo == RAIL_BIND_NONE
    assert core.last_rail_bind_hi == RAIL_BIND_NONE
    assert core.last_rail_box_lo == 0.0
    assert core.last_rail_box_hi == 0.0
    assert core.last_rail_task_vel_used == 0.0
    assert core.last_rail_h1 == 0.0
    assert core.last_rail_qdot_prev == 0.0
    assert core.last_rail_qdot_prev2 == 0.0
