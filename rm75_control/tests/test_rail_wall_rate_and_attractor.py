"""Rail hard box 5/780 mm: stopping envelope + linear taper, no Cartesian Y clip.

One-tick accel/jerk from rest is ~7.5e-5 m/s, so live qdot[0] magnitude is
not the wall contract.  Geometry is the velocity box; the controller checks
are unclipped v_cmd Y and into-wall qdot[0] ≈ 0 at the hard wall.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from rm75_control.control.joint_admittance_8dof.collision_model import CollisionConfig
from rm75_control.control.joint_admittance_8dof.config import build_joint_ik_config
from rm75_control.control.joint_admittance_8dof.loop import JointIkConfig, JointIkController
from rm75_control.control.joint_admittance_8dof.model import RobotKinematics, full_q_from_arm
from rm75_control.control.joint_admittance_8dof.solver.constraint_mgr import (
    VelocityBoxConstraints,
    stopping_velocity,
)
from rm75_control.control.joint_admittance_8dof.solver.qp_builder import QpConfig
from rm75_control.control.joint_admittance_8dof.tasks.rail_extension import (
    rail_vel_ff_from_reference,
)
from rm75_control.control.joint_admittance_8dof.tasks.rail_mode import RailMode
from rm75_control.control.joint_admittance_8dof.utils.safety import SafetyLimits


Q_SAFE = full_q_from_arm(
    np.deg2rad([5.0, -30.0, 10.0, 60.0, -5.0, 45.0, 0.0]), 0.40
)
_CFG = Path(__file__).resolve().parents[1] / "configs" / "joint_admittance_8dof.yaml"
_INTO_HI = np.array([0.0, 0.08, 0.0, 0.0, 0.0, 0.0])
_INTO_LO = np.array([0.0, -0.08, 0.0, 0.0, 0.0, 0.0])
_V_MAX = 0.08
_DT = 0.005
_A_RAIL = 0.30
_T_RAIL = 0.15
_BAND = 0.025


def _rail_box(*, a_max=None, reaction_s=0.0, band=0.01, v_max=_V_MAX):
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


def _bounds(q0: float, *, q_cmd: float | None = None, box=None):
    q = np.zeros(8)
    q[0] = float(q0)
    cmd = None if q_cmd is None else np.array([float(q_cmd), *q[1:]])
    return (box or _rail_box()).bounds(q, _DT, qdot_prev=None, q_meas=q, q_cmd=cmd)


def _controller() -> JointIkController:
    qp = QpConfig(backend="proxqp", collision=CollisionConfig(enabled=False))
    cfg = JointIkConfig(
        control_frame="base",
        qp=qp,
        collision=CollisionConfig(enabled=False),
    )
    cfg.rail.mode = RailMode.COUPLED
    cfg.rail.v_max_m_s = _V_MAX
    cfg.rail.hard_min_m = 0.005
    cfg.rail.hard_max_m = 0.78
    cfg.rail.soft_min_m = 0.030
    cfg.rail.soft_max_m = 0.755
    cfg.qp.limit_damper_band_rail_m = _BAND
    cfg.qp.limit_damper_rail_reaction_s = _T_RAIL
    inner = JointIkController(RobotKinematics(), cfg)
    inner.reset(Q_SAFE)
    return inner


def _step(inner: JointIkController, q0: float, twist: np.ndarray):
    q = Q_SAFE.copy()
    q[0] = float(q0)
    inner.reset(q)
    return inner.update(twist, q_meas=q, vel_ff=twist)


def test_yaml_hard_soft_and_no_far_wln() -> None:
    raw = yaml.safe_load(_CFG.read_text(encoding="utf-8"))
    cfg = build_joint_ik_config(raw)
    assert not hasattr(cfg.qp, "wln")
    assert not hasattr(cfg.qp, "limit_reaction_rail_s")
    assert cfg.qp.limit_damper_rail_reaction_s == pytest.approx(0.15)
    assert cfg.rail.hard_min_m == pytest.approx(0.005)
    assert cfg.rail.hard_max_m == pytest.approx(0.78)
    assert cfg.rail.soft_min_m == pytest.approx(0.030)
    assert cfg.rail.soft_max_m == pytest.approx(0.755)
    assert cfg.qp.limit_damper_band_rail_m == pytest.approx(0.025)
    from rm75_control.control.joint_admittance_8dof.hw.rail_servo import (
        parse_rail_servo_config,
    )

    servo = parse_rail_servo_config(raw)
    assert servo.hard_max_m == pytest.approx(0.78)
    assert servo.soft_max_m == pytest.approx(0.755)
    assert cfg.rail_extension.soft_max_m == pytest.approx(0.78)
    assert cfg.rail_extension.soft_min_m == pytest.approx(0.005)


def test_mid_stroke_keeps_full_into_wall_speed() -> None:
    lo, hi = _bounds(0.40)
    assert hi[0] == pytest.approx(_V_MAX, abs=1e-9)
    assert lo[0] == pytest.approx(-_V_MAX, abs=1e-9)
    # Linear 10 mm band: 760 mm is still the inner edge (full speed).
    lo760, hi760 = _bounds(0.76)
    assert hi760[0] == pytest.approx(_V_MAX, abs=1e-9)
    assert lo760[0] == pytest.approx(-_V_MAX, abs=1e-9)
    inner = _controller()
    step = _step(inner, 0.40, _INTO_HI)
    assert rail_vel_ff_from_reference(step.v_cmd, inner.kin, inner.q_cmd) == pytest.approx(
        0.08, abs=1e-6
    )


def test_linear_taper_has_no_enter_step() -> None:
    lo, hi = _bounds(0.77)
    assert float(hi[0]) == pytest.approx(_V_MAX, abs=1e-9)
    assert lo[0] == pytest.approx(-_V_MAX, abs=1e-9)
    _lo_mid, hi_mid = _bounds(0.775)
    assert float(hi_mid[0]) == pytest.approx(0.5 * _V_MAX, abs=1e-9)
    inner = _controller()
    step = _step(inner, 0.77, _INTO_HI)
    assert rail_vel_ff_from_reference(step.v_cmd, inner.kin, inner.q_cmd) == pytest.approx(
        0.08, abs=1e-6
    )


def test_damper_and_one_tick_box_taper_into_wall() -> None:
    box = _rail_box(a_max=_A_RAIL, reaction_s=_T_RAIL, band=_BAND, v_max=0.15)
    _lo, hi_far = _bounds(0.40, box=box)
    assert float(hi_far[0]) == pytest.approx(0.15, abs=1e-9)

    _lo, hi_edge = _bounds(0.755, box=box)
    assert float(hi_edge[0]) == pytest.approx(0.15, abs=1e-9)

    _lo, hi_mid = _bounds(0.765, box=box)
    assert float(hi_mid[0]) == pytest.approx(0.15 * 0.015 / _BAND, abs=1e-8)

    _lo, hi_wall = _bounds(0.78, box=box)
    assert float(hi_wall[0]) == pytest.approx(0.0, abs=1e-9)
    assert float(hi_wall[0]) < float(hi_mid[0]) < float(hi_edge[0])

    q = np.zeros(8)
    q[0] = 0.779
    _lo, hi_tick = box.bounds(q, _DT)
    assert float(hi_tick[0]) <= (0.78 - 0.779) / _DT + 1.0e-12


def test_hard_wall_zeros_rail_not_cartesian_y() -> None:
    lo_hi, hi_hi = _bounds(0.78)
    assert hi_hi[0] == pytest.approx(0.0, abs=1e-9)
    assert lo_hi[0] == pytest.approx(-_V_MAX, abs=1e-9)
    lo_lo, hi_lo = _bounds(0.005)
    assert lo_lo[0] == pytest.approx(0.0, abs=1e-9)
    assert hi_lo[0] == pytest.approx(_V_MAX, abs=1e-9)

    inner = _controller()
    hi = _step(inner, 0.78, _INTO_HI)
    assert rail_vel_ff_from_reference(hi.v_cmd, inner.kin, hi.q_send) == pytest.approx(
        0.08, abs=1e-6
    )
    assert float(hi.qdot[0]) == pytest.approx(0.0, abs=1e-4)
    lo = _step(inner, 0.005, _INTO_LO)
    assert rail_vel_ff_from_reference(lo.v_cmd, inner.kin, lo.q_send) == pytest.approx(
        -0.08, abs=1e-6
    )
    assert float(lo.qdot[0]) == pytest.approx(0.0, abs=1e-4)


def test_damper_uses_leading_rail_state() -> None:
    """Linear taper sees the state closer to the wall."""
    _lo, hi_cmd = _bounds(0.74, q_cmd=0.775)
    assert float(hi_cmd[0]) == pytest.approx(0.5 * _V_MAX, abs=1e-9)
    _lo, hi_meas = _bounds(0.775, q_cmd=0.74)
    assert float(hi_meas[0]) == pytest.approx(0.5 * _V_MAX, abs=1e-9)
    _lo, hi_mid = _bounds(0.74, q_cmd=0.74)
    assert float(hi_mid[0]) == pytest.approx(_V_MAX, abs=1e-9)

    box = _rail_box(a_max=_A_RAIL, reaction_s=_T_RAIL, band=_BAND, v_max=0.15)
    _lo, hi_lead = _bounds(0.74, q_cmd=0.765, box=box)
    _lo, hi_both = _bounds(0.765, q_cmd=0.765, box=box)
    assert float(hi_lead[0]) == pytest.approx(float(hi_both[0]), abs=1e-9)
    _lo, hi_far = _bounds(0.74, q_cmd=0.74, box=box)
    assert float(hi_lead[0]) < float(hi_far[0])


def test_leave_wall_is_not_reduced() -> None:
    _lo, hi = _bounds(0.005)
    assert hi[0] == pytest.approx(_V_MAX, abs=1e-9)
    lo, _hi = _bounds(0.78)
    assert lo[0] == pytest.approx(-_V_MAX, abs=1e-9)
    lo_band, _ = _bounds(0.77)
    assert lo_band[0] == pytest.approx(-_V_MAX, abs=1e-9)
    box = _rail_box(a_max=_A_RAIL, reaction_s=_T_RAIL, band=_BAND, v_max=0.15)
    lo_env, _ = _bounds(0.765, box=box)
    assert float(lo_env[0]) == pytest.approx(-0.15, abs=1e-9)


def test_overshoot_kills_into_wall_not_leave() -> None:
    lo, hi = _bounds(0.782)
    assert float(hi[0]) == pytest.approx(0.0, abs=1e-9)
    assert float(lo[0]) == pytest.approx(-_V_MAX, abs=1e-9)
    lo_lo, hi_lo = _bounds(0.002)
    assert float(lo_lo[0]) == pytest.approx(0.0, abs=1e-9)
    assert float(hi_lo[0]) == pytest.approx(_V_MAX, abs=1e-9)

    inner = _controller()
    q = Q_SAFE.copy()
    q[0] = 0.782
    inner.reset(q)
    into = inner.update(_INTO_HI, q_meas=q, vel_ff=_INTO_HI)
    assert float(into.qdot[0]) == pytest.approx(0.0, abs=1e-4)
    assert rail_vel_ff_from_reference(into.v_cmd, inner.kin, into.q_send) == pytest.approx(
        0.08, abs=1e-6
    )
    leave_lo, _ = _bounds(0.782)
    assert float(leave_lo[0]) < -0.04
