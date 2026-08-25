"""Side-lying attractor, minus escape, and rail host-loop contracts."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

from rm75_control.control.joint_admittance_8dof.config import (
    assert_design_attractor_consistent,
    build_joint_ik_config,
)
from rm75_control.control.joint_admittance_8dof.hw.rail_servo import (
    RailServoBridge,
    live_host_accel_m_s2,
    next_poll_deadline,
)
from rm75_control.control.joint_admittance_8dof.loop import JointIkController
from rm75_control.control.joint_admittance_8dof.model import RobotKinematics
from rm75_control.control.joint_admittance_8dof.tasks.psi_retarget import (
    PostureRetarget,
    PsiRetargetConfig,
    d_from_q,
    fold_psi_to_positive,
    psi_err_avoiding_zero,
)
from rm75_control.control.joint_admittance_8dof.tasks.rail_extension import (
    RailExtensionConfig,
    RailExtensionTask,
)
from rm75_control.kinematics.srs_ik import Q_LOWER, Q_UPPER, psi_from_q


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "joint_admittance_8dof.yaml"
_Q_NOM_DEG = np.array([0.0, -89.5, -94.5, 65.2, 96.0, 89.3, 61.0, 94.6])


def _raw() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def test_design_q_nominal_matches_psi_attr_and_comfort() -> None:
    cfg = build_joint_ik_config(_raw())
    kin = RobotKinematics()
    qn = np.asarray(cfg.nullspace.q_nominal_rad, dtype=float)
    assert fold_psi_to_positive(psi_from_q(qn)) == pytest.approx(
        float(cfg.psi_retarget.psi_attr_rad), abs=np.deg2rad(1.0)
    )
    assert d_from_q(kin, qn) == pytest.approx(float(cfg.psi_retarget.d_attr_m), abs=0.005)
    q_arm = qn[1:]
    margin = float(np.min(np.minimum(q_arm - Q_LOWER, Q_UPPER - q_arm)))
    assert margin >= float(cfg.qp.joint_comfort.activate_rad) - 1.0e-9
    assert_design_attractor_consistent(cfg, kin=kin)


def test_inconsistent_yaml_is_rejected() -> None:
    raw = deepcopy(_raw())
    raw["inner"]["nullspace"]["q_nominal_deg"] = [
        0.0, 0.0, -45.0, 0.0, 90.0, 40.0, 60.0, 0.0
    ]
    with pytest.raises(ValueError, match="q_nominal"):
        build_joint_ik_config(raw)

    raw = deepcopy(_raw())
    raw["inner"]["psi_retarget"]["psi_attr_deg"] = 0.0
    with pytest.raises(ValueError, match="psi_attr"):
        build_joint_ik_config(raw)


def test_centering_and_q_star_keep_signed_nominal() -> None:
    cfg = build_joint_ik_config(_raw())
    cfg.ird.enabled = False
    cfg.collision.enabled = False
    cfg.qp.collision.enabled = False
    inner = JointIkController(RobotKinematics(), cfg)
    q = np.deg2rad(_Q_NOM_DEG).copy()
    q[0] = 0.40
    q[1] = abs(float(q[1]))
    inner.reset(q)
    assert inner.core.q_star_signs is not None
    assert float(inner.core.q_star_signs[1]) < 0.0
    assert inner.core.q_star is not None
    assert float(inner.core.q_star[1]) > 0.0
    assert float(inner.centering_task.q_target[1]) > 0.0
    assert inner._family_ok is False


def test_planar_start_keeps_design_j1_sign() -> None:
    cfg = build_joint_ik_config(_raw())
    cfg.ird.enabled = False
    cfg.collision.enabled = False
    cfg.qp.collision.enabled = False
    inner = JointIkController(RobotKinematics(), cfg)
    q = np.array(
        [0.31, 0.0, np.deg2rad(-30.0), 0.0, np.pi / 2.0, 0.0, np.pi / 2.0, np.pi / 2.0]
    )
    inner.reset(q)
    yaml_j1 = float(cfg.nullspace.q_nominal_rad[1])
    assert inner.core.q_star_signs is not None
    assert float(inner.core.q_star_signs[1]) < 0.0
    assert inner.core.q_star is not None
    assert abs(float(inner.core.q_star[1]) - yaml_j1) > np.deg2rad(20.0)
    assert abs(float(inner.core.q_star[1])) < np.deg2rad(10.0)
    assert abs(float(inner.centering_task.q_target[1])) < np.deg2rad(10.0)


def test_homotopy_done_pins_centering_to_yaml() -> None:
    """s=1 still publishes last-valid SRS q*; yaml is signs-only."""
    cfg = build_joint_ik_config(_raw())
    cfg.ird.enabled = False
    cfg.collision.enabled = False
    cfg.qp.collision.enabled = False
    inner = JointIkController(RobotKinematics(), cfg)
    q = np.array(
        [0.31, 0.0, np.deg2rad(-30.0), 0.0, np.pi / 2.0, 0.0, np.pi / 2.0, np.pi / 2.0]
    )
    inner.reset(q)
    yanked = q.copy()
    yanked[1] = 0.0
    assert inner.posture_retarget is not None
    inner.posture_retarget.q_star_rad = yanked
    inner.posture_retarget.homotopy_s = 0.4
    inner._publish_homotopy_centering()
    assert float(inner.centering_task.q_target[1]) == pytest.approx(0.0)
    inner.posture_retarget.homotopy_s = 1.0
    inner._publish_homotopy_centering()
    yaml_j1 = float(cfg.nullspace.q_nominal_rad[1])
    assert float(inner.centering_task.q_target[1]) == pytest.approx(0.0)
    assert inner.core.q_star is not None
    assert float(inner.core.q_star[1]) == pytest.approx(0.0)
    assert abs(float(inner.core.q_star[1]) - yaml_j1) > np.deg2rad(20.0)
    assert inner.core.q_star_signs is not None
    assert float(inner.core.q_star_signs[1]) < 0.0
    # Invalid SRS keeps the last valid vector instead of snapping to yaml.
    inner.posture_retarget.q_star_rad = yanked + np.nan
    inner._publish_homotopy_centering()
    assert float(inner.centering_task.q_target[1]) == pytest.approx(0.0)


def test_psi_star_returns_home_after_healthy_dwell() -> None:
    kin = RobotKinematics()
    cfg = PsiRetargetConfig(
        enabled=True,
        psi_rate_rad_s=np.deg2rad(25.0),
        psi_return_dwell_s=1.0,
        psi_replan_period_s=0.1,
    )
    rt = PostureRetarget(kin, cfg)
    q = np.deg2rad(_Q_NOM_DEG).copy()
    q[0] = 0.40
    rt.reset(q)
    assert rt.psi_star_rad == pytest.approx(float(cfg.psi_attr_rad), abs=1e-9)
    hijack = float(np.deg2rad(100.0))
    rt._psi_star = hijack
    rt.psi_star_rad = hijack
    rt._healthy_dwell_s = 0.0
    dt = 0.05
    prev = float(rt._psi_cmd)
    for _ in range(10):
        psi, _d = rt.step(q, dt, rail_lo=0.005, rail_hi=0.78)
        assert abs(psi_err_avoiding_zero(prev, psi)) <= cfg.psi_rate_rad_s * dt + 1e-9
        assert psi * prev >= -1e-9 or abs(prev) > 0.5 * np.pi
        prev = psi
    assert rt.psi_star_rad == pytest.approx(hijack, abs=1e-9)
    for _ in range(16):
        psi, _d = rt.step(q, dt, rail_lo=0.005, rail_hi=0.78)
        assert abs(psi_err_avoiding_zero(prev, psi)) <= cfg.psi_rate_rad_s * dt + 1e-9
        assert psi * prev >= -1e-9 or abs(prev) > 0.5 * np.pi
        prev = psi
    assert rt.psi_star_rad == pytest.approx(float(cfg.psi_attr_rad), abs=1e-9)


def test_collapsed_wrist_search_then_home() -> None:
    kin = RobotKinematics()
    cfg = PsiRetargetConfig(
        enabled=True,
        psi_replan_period_s=0.0,
        psi_return_dwell_s=1.0,
        psi_rate_rad_s=np.deg2rad(25.0),
    )
    rt = PostureRetarget(kin, cfg)
    q_bad = np.array(
        [0.360018, 2.534646, -0.341951, -2.812693, 2.084567, 2.844237, 0.329491, -1.621615]
    )
    q_good = np.deg2rad(_Q_NOM_DEG).copy()
    q_good[0] = 0.36
    rt.reset(q_bad)
    rt.step(q_bad, 0.1, rail_lo=0.005, rail_hi=0.78)
    assert rt.last_psi_search_count >= 1
    prev = float(rt._psi_cmd)
    dt = 0.05
    for _ in range(25):
        psi, _d = rt.step(q_good, dt, rail_lo=0.005, rail_hi=0.78)
        assert abs(psi_err_avoiding_zero(prev, psi)) <= cfg.psi_rate_rad_s * dt + 1e-9
        assert not (prev * psi < 0.0 and abs(prev) < 0.5 * np.pi)
        prev = psi
    assert rt.psi_star_rad == pytest.approx(float(cfg.psi_attr_rad), abs=1e-9)


def test_preferred_escape_sign_is_minus_except_min_pin() -> None:
    kin = RobotKinematics()
    task = RailExtensionTask(
        kin,
        RailExtensionConfig(
            enabled=True,
            soft_min_m=0.10,
            soft_max_m=0.70,
            pin_margin_m=0.008,
            escape_leave_m=0.04,
            escape_sign_policy="minus",
        ),
    )
    for y in np.linspace(0.15, 0.69, 28):
        assert task._preferred_escape_sign(float(y)) <= 0.0
    assert task._preferred_escape_sign(0.105) == pytest.approx(1.0)


def test_step_reference_velocity_step_lag_is_at_most_3_2_mm() -> None:
    a_max = live_host_accel_m_s2(
        vel_max_m_s=0.15, accel_ms=120.0, configured_m_s2=1.2, lead_mm=10.0
    )
    assert a_max == pytest.approx(min(1.2, 0.85 * (1000.0 / 60.0) * 0.010 / 0.12))
    dt = 0.02
    x_ref = 0.400
    v_ref = 0.0
    v_goal = 0.08
    max_lag = 0.0
    for i in range(80):
        now = i * dt
        x_goal = 0.400 + v_goal * now
        x_ref, v_ref, _a = RailServoBridge._step_reference(
            x_ref,
            v_ref,
            x_goal,
            v_goal,
            stationary=False,
            dt=dt,
            v_max=0.15,
            a_max=a_max,
        )
        max_lag = max(max_lag, abs(x_goal - x_ref))
    assert max_lag <= 0.0032 + 1.0e-6


def test_next_poll_deadline_does_not_accumulate_overrun_debt() -> None:
    period = 0.023
    next_t = 0.0
    now = 0.0
    for _ in range(5):
        now += 0.040
        next_t = next_poll_deadline(next_t, now, period)
        assert next_t == pytest.approx(now)
    now += 0.010
    next_t = next_poll_deadline(next_t, now, period)
    assert next_t == pytest.approx(0.200 + period)
    on_time = next_poll_deadline(1.0, 1.010, period)
    assert on_time == pytest.approx(1.0 + period)
