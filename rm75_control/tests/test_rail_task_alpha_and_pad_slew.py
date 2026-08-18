"""Gates for rail_task_alpha, margin escape unmute, pad slew, and sat scale."""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from rm75_control.control.joint_admittance_8dof.config import build_joint_ik_config
from rm75_control.control.joint_admittance_8dof.loop import (
    JointIkController,
    Phase,
    _reference_governor_scale,
)
from rm75_control.control.joint_admittance_8dof.model import RobotKinematics
from rm75_control.control.joint_admittance_8dof.saturation_latch import (
    SaturationConfig,
    secondary_scale_from_slack,
)
from rm75_control.control.joint_admittance_8dof.tasks.rail_extension import (
    RailExtensionConfig,
    RailExtensionTask,
)
from rm75_control.control.joint_admittance_8dof.teleop.gamepad_twist import (
    GamepadTwistConfig,
    GamepadTwistOuterLoop,
    slew_vec,
)
from rm75_control.control.joint_admittance_8dof.teleop.xbox_pad import FakePad


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "joint_admittance_8dof.yaml"


def _inner(*, rail_task_alpha: float, q: np.ndarray) -> JointIkController:
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    cfg = build_joint_ik_config(raw)
    cfg.collision.enabled = False
    cfg.qp.collision.enabled = False
    cfg.ird.enabled = False
    cfg.qp.rail_task_alpha = float(rail_task_alpha)
    kin = RobotKinematics()
    inner = JointIkController(kin, cfg)
    inner.reset(q)
    return inner


def _safe_q() -> np.ndarray:
    return np.array([0.40, 0.2, -0.5, 0.1, 1.6, -0.4, 0.6, 0.0], dtype=float)


def test_secondary_scale_from_slack_is_smoothstep() -> None:
    cfg = SaturationConfig(slack_enter=0.03, slack_exit=0.015, secondary_scale=0.15)
    assert secondary_scale_from_slack(0.0, cfg) == pytest.approx(1.0)
    assert secondary_scale_from_slack(0.015, cfg) == pytest.approx(1.0)
    assert secondary_scale_from_slack(0.03, cfg) == pytest.approx(0.15)
    mid = secondary_scale_from_slack(0.0225, cfg)
    assert 0.15 < mid < 1.0


def test_secondary_scale_lpf_does_not_jump() -> None:
    inner = _inner(rail_task_alpha=0.0, q=_safe_q())
    inner.last_slack_norm = 0.20
    a = inner._secondary(_safe_q(), None, dt_s=0.005)
    s1 = float(inner.last_sat_scale)
    inner.last_slack_norm = 0.20
    inner._secondary(_safe_q(), None, dt_s=0.005)
    s2 = float(inner.last_sat_scale)
    assert 0.15 < s1 < 1.0
    assert s2 < s1
    assert s1 - s2 < 0.20


def test_governor_crawls_at_floor_when_saturated() -> None:
    phase = Phase(
        outer=SimpleNamespace(),
        governor_err_ok_mm=10.0,
        governor_err_max_mm=40.0,
        governor_scale_min=0.25,
        governor_crawl_floor=0.05,
        governor_joint_err_max_deg=0.0,
    )
    raw = _reference_governor_scale(
        phase, outer_err_mm=80.0, joint_err_deg=None, physical_saturated=True
    )
    assert raw == pytest.approx(0.05)


def test_rail_task_alpha_zero_matches_affine() -> None:
    q = _safe_q()
    inner = _inner(rail_task_alpha=0.0, q=q)
    J = inner.kin.jacobian(q)
    rail_exec = 0.04
    twist = np.array([0.0, 0.05, 0.0, 0.0, 0.0, 0.0])
    inner.core.step(
        q,
        twist,
        0.005,
        q_meas=q,
        rail_exec_vel_m_s=rail_exec,
        jacobian=J,
        kinematics_ready=True,
    )
    np.testing.assert_allclose(inner.core.last_task_jacobian[:, 0], 0.0, atol=1e-12)
    np.testing.assert_allclose(
        inner.core.last_rail_exec_contrib, J[:, 0] * rail_exec, atol=1e-12
    )
    assert inner.core.last_rail_task_alpha == pytest.approx(0.0)


def test_rail_task_alpha_one_uses_full_jacobian() -> None:
    q = _safe_q()
    inner = _inner(rail_task_alpha=1.0, q=q)
    J = inner.kin.jacobian(q)
    rail_exec = 0.04
    twist = np.array([0.0, 0.05, 0.0, 0.0, 0.0, 0.0])
    inner.core.step(
        q,
        twist,
        0.005,
        q_meas=q,
        rail_exec_vel_m_s=rail_exec,
        jacobian=J,
        kinematics_ready=True,
    )
    np.testing.assert_allclose(inner.core.last_task_jacobian, J, atol=1e-12)
    np.testing.assert_allclose(inner.core.last_rail_exec_contrib, 0.0, atol=1e-12)
    np.testing.assert_allclose(inner.core.last_task_target, twist, atol=1e-12)


def test_rail_task_alpha_does_not_slam_without_macro() -> None:
    """A free rail column at alpha=0.07 is a 1/alpha amplifier; pin to 0."""
    q = _safe_q()
    inner0 = _inner(rail_task_alpha=0.0, q=q)
    inner1 = _inner(rail_task_alpha=0.07, q=q)
    J = inner0.kin.jacobian(q)
    twist = np.array([0.0, 0.08, 0.0, 0.0, 0.0, 0.0])
    kwargs = dict(
        q_meas=q,
        rail_exec_vel_m_s=0.0,
        jacobian=J,
        kinematics_ready=True,
        rail_open_travel=True,
    )
    r0 = inner0.core.step(q, twist, 0.005, **kwargs)
    r1 = inner1.core.step(q, twist, 0.005, **kwargs)
    assert abs(float(r0.qdot[0])) < 1.0e-3
    assert abs(float(r1.qdot[0])) < 1.0e-3
    np.testing.assert_allclose(
        inner1.core.last_task_jacobian[:, 0], 0.07 * J[:, 0], atol=1e-12
    )


def test_rail_task_alpha_follows_macro_and_mixes_arm() -> None:
    q = _safe_q()
    inner0 = _inner(rail_task_alpha=0.0, q=q)
    inner1 = _inner(rail_task_alpha=0.07, q=q)
    J = inner0.kin.jacobian(q)
    twist = np.array([0.0, 0.08, 0.0, 0.0, 0.0, 0.0])
    kwargs = dict(
        q_meas=q,
        rail_exec_vel_m_s=0.0,
        jacobian=J,
        kinematics_ready=True,
        rail_open_travel=True,
        rail_task_vel_m_s=0.08,
        rail_task_weight=1.0,
    )
    r0 = inner0.core.step(q, twist, 0.005, **kwargs)
    r1 = inner1.core.step(q, twist, 0.005, **kwargs)
    # Published rail follows the plant-scale pref (accel-limited first tick).
    assert float(inner0.core.last_qdot_qp1[0]) > 1.0e-4
    assert float(inner1.core.last_qdot_qp1[0]) > 1.0e-4
    assert float(r1.qdot[0]) == pytest.approx(
        float(inner1.core.last_qdot_qp1[0]), abs=1.0e-6
    )
    np.testing.assert_allclose(
        inner1.core.last_task_jacobian[:, 0], 0.07 * J[:, 0], atol=1e-12
    )


def test_lagged_meas_cannot_publish_past_hard_max() -> None:
    """051639: q_cmd ran to 0.81 while meas lagged and the stick was leaving."""
    q = _safe_q()
    q[0] = 0.772
    inner = _inner(rail_task_alpha=0.07, q=q)
    inner.core.sync_applied(np.array([0.12, 0, 0, 0, 0, 0, 0, 0], dtype=float))
    twist = np.array([0.0, -0.10, 0.0, 0.0, 0.0, 0.0])
    q_meas = q.copy()
    q_meas[0] = 0.766
    hi = float(inner.cfg.rail.hard_max_m)
    for _ in range(40):
        step = inner.update(twist, q_meas=q_meas.copy(), vel_ff=twist)
        assert float(inner.q_cmd[0]) <= hi + 1.0e-9
        if float(inner.q_cmd[0]) >= hi - 1.0e-4:
            assert float(step.qdot[0]) <= 1.0e-6
        q_meas[0] = min(float(q_meas[0]) + 0.002, hi)


def test_rail_task_alpha_parks_at_hard_stop() -> None:
    q = _safe_q()
    q[0] = 0.78
    inner = _inner(rail_task_alpha=0.07, q=q)
    twist = np.array([0.0, 0.08, 0.0, 0.0, 0.0, 0.0])
    rail = []
    for _ in range(80):
        step = inner.update(twist, q_meas=inner.q_cmd.copy(), vel_ff=twist)
        rail.append(float(inner.q_cmd[0]))
        assert float(step.qdot[0]) > -0.01
    rail_arr = np.asarray(rail, dtype=float)
    assert float(rail_arr.max()) <= float(inner.cfg.rail.hard_max_m) + 2.0e-4
    assert float(np.ptp(rail_arr[-20:])) < 5.0e-4


def test_healthy_sigma_still_escapes_when_margin_is_tight() -> None:
    kin = RobotKinematics()
    task = RailExtensionTask(
        kin,
        RailExtensionConfig(
            enabled=True,
            k_esc=1.2,
            k_ext=0.0,
            sigma_escape_enter=0.55,
            sigma_escape_exit=0.80,
            margin_escape_enter=0.12,
            margin_escape_exit=0.25,
            escape_enter_dwell_s=0.0,
            v_lpf_tau_s=0.0,
            v_lpf_tau_escape_s=0.0,
            escape_grad_floor=0.0,
            healthy_sigma_mute=0.08,
        ),
    )
    task.set_mode("reach")
    q = 0.5 * (kin.q_lower + kin.q_upper)
    task.capture_reference(q)
    v_ok, _ = task(
        q,
        sigma_scale=1.0,
        sigma_grad_rail=1.0,
        joint_margin_frac=1.0,
        sigma_raw=0.15,
        dt_s=0.005,
    )
    assert abs(v_ok) < 1e-6 or abs(task.last_v_escape) < 1e-9
    v_pin, _ = task(
        q,
        sigma_scale=1.0,
        sigma_grad_rail=1.0,
        joint_margin_frac=0.05,
        sigma_raw=0.15,
        dt_s=0.005,
    )
    assert abs(task.last_v_escape) > 1e-4 or abs(v_pin) > 1e-4
    assert task.last_margin_escape_active


def test_slew_vec_caps_per_tick_delta() -> None:
    prev = np.zeros(3)
    target = np.array([0.10, 0.0, 0.0])
    out = slew_vec(prev, target, 0.8, 0.005)
    assert float(np.linalg.norm(out - prev)) == pytest.approx(0.004, abs=1e-9)
    assert out[0] > 0.0


def test_gamepad_stick_step_is_rate_limited() -> None:
    pad = FakePad(axes=np.array([-1.0, 0.0, -1.0, 0.0, 0.0, -1.0]))
    cfg = GamepadTwistConfig(
        trans_m_s=0.10,
        deadzone=0.10,
        dt=0.005,
        trans_a_max_m_s2=0.8,
        rot_a_max_rad_s2=4.0,
        control_frame="base",
    )
    outer = GamepadTwistOuterLoop(pad, cfg)
    pose = np.array([0.4, 0.2, 0.3, 0.0, 0.0, 0.0])
    outer.set_origin(pose)
    prev = np.zeros(6)
    for _ in range(8):
        twist = outer.sample(0.0, pose, np.zeros(6))
        dv = float(np.linalg.norm(twist[:3] - prev[:3]))
        assert dv <= 0.8 * 0.005 + 1e-9
        prev = twist
    assert outer.last_twist_slewed


def test_hold_term_is_capped_on_large_error() -> None:
    pad = FakePad(axes=np.array([0.0, 0.0, -1.0, 0.0, 0.0, -1.0]))
    cfg = GamepadTwistConfig(
        hold_k_task=4.0,
        hold_v_max_m_s=0.03,
        hold_w_max_rad_s=0.20,
        hold_relatch_on_settle=False,
        trans_a_max_m_s2=100.0,
        rot_a_max_rad_s2=100.0,
        dt=0.005,
        control_frame="base",
    )
    outer = GamepadTwistOuterLoop(pad, cfg)
    pose0 = np.array([0.40, 0.20, 0.30, 0.0, 0.0, 0.0])
    outer.set_origin(pose0)
    far = pose0.copy()
    far[1] += 0.050
    twist = outer.sample(0.0, far, np.zeros(6))
    assert float(np.linalg.norm(twist[:3])) <= 0.03 + 1e-9


def test_release_coasts_then_relatches() -> None:
    pad = FakePad(axes=np.array([-1.0, 0.0, -1.0, 0.0, 0.0, -1.0]))
    cfg = GamepadTwistConfig(
        trans_m_s=0.10,
        deadzone=0.10,
        dt=0.005,
        trans_a_max_m_s2=100.0,
        hold_relatch_on_settle=True,
        hold_settle_v_m_s=0.005,
        control_frame="base",
    )
    outer = GamepadTwistOuterLoop(pad, cfg)
    pose = np.array([0.40, 0.20, 0.30, 0.0, 0.0, 0.0])
    outer.set_origin(pose)
    moving = pose.copy()
    moving[1] += 0.01
    outer.sample(0.0, moving, np.zeros(6))
    latch_before = outer.last_pose_d.copy()
    pad.axes = np.array([0.0, 0.0, -1.0, 0.0, 0.0, -1.0])
    coasting = moving.copy()
    coasting[1] += 0.002
    twist = outer.sample(0.0, coasting, np.zeros(6))
    assert outer._coast_until_settle
    assert float(np.linalg.norm(twist[:3])) < 0.10
    settled = coasting.copy()
    twist2 = outer.sample(0.0, settled, np.zeros(6))
    assert not outer._coast_until_settle
    np.testing.assert_allclose(outer.last_pose_d, settled, atol=1e-12)
    assert float(np.linalg.norm(twist2[:3])) <= 0.03 + 1e-9
    assert not np.allclose(outer.last_pose_d, latch_before)


def test_identify_rail_task_alpha_on_first_order_plant() -> None:
    import importlib.util

    path = (
        Path(__file__).resolve().parents[1]
        / "apps"
        / "joint_admittance_8dof"
        / "analyze_qpik_quality.py"
    )
    spec = importlib.util.spec_from_file_location("analyze_qpik_quality", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    identify_rail_task_alpha = mod.identify_rail_task_alpha

    dt = 1.0 / 60.0
    tau = 0.07
    a = 1.0 - math.exp(-dt / tau)
    u = np.zeros(240)
    u[40:] = 0.08
    y = np.zeros(240)
    for k in range(1, 240):
        y[k] = y[k - 1] + a * (u[k - 1] - y[k - 1])
    rows = [
        {"v_cmd_m_s": f"{uu:.8f}", "v_enc_m_s": f"{yy:.8f}"}
        for uu, yy in zip(u, y)
    ]
    report = identify_rail_task_alpha(rows, dt_s=dt)
    assert report["n_samples"] > 100
    assert report["tau_s"] == pytest.approx(tau, rel=0.25)
    assert report["alpha_5ms"] == pytest.approx(1.0 - math.exp(-0.005 / tau), rel=0.25)
