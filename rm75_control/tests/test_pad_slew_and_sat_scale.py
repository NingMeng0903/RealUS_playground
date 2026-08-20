"""Gates for pad slew / hold cap and continuous secondary sat scale."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from rm75_control.control.joint_admittance_8dof.config import build_joint_ik_config
from rm75_control.control.joint_admittance_8dof.loop import JointIkController
from rm75_control.control.joint_admittance_8dof.model import RobotKinematics
from rm75_control.control.joint_admittance_8dof.saturation_latch import (
    SaturationConfig,
    secondary_scale_from_slack,
)
from rm75_control.control.joint_admittance_8dof.tasks.nullspace_task import (
    JointCenteringTask,
    NullspaceTaskConfig,
)
from rm75_control.control.joint_admittance_8dof.tasks.secondary_composer import (
    SecondaryComposer,
)
from rm75_control.control.joint_admittance_8dof.teleop.gamepad_twist import (
    GamepadTwistConfig,
    GamepadTwistOuterLoop,
    slew_axes_jerk,
    slew_vec,
    slew_vec_jerk,
)
from rm75_control.control.joint_admittance_8dof.teleop.xbox_pad import FakePad


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "joint_admittance_8dof.yaml"


def _safe_q() -> np.ndarray:
    return np.array([0.40, 0.2, -0.5, 0.1, 1.6, -0.4, 0.6, 0.0], dtype=float)


def _inner(q: np.ndarray) -> JointIkController:
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    cfg = build_joint_ik_config(raw)
    cfg.backend = "python"
    cfg.collision.enabled = False
    cfg.qp.collision.enabled = False
    cfg.ird.enabled = False
    inner = JointIkController(RobotKinematics(), cfg)
    inner.reset(q)
    return inner


def test_secondary_scale_from_slack_is_smoothstep() -> None:
    cfg = SaturationConfig(slack_enter=0.03, slack_exit=0.015, secondary_scale=0.15)
    assert secondary_scale_from_slack(0.0, cfg) == pytest.approx(1.0)
    assert secondary_scale_from_slack(0.015, cfg) == pytest.approx(1.0)
    assert secondary_scale_from_slack(0.03, cfg) == pytest.approx(0.15)
    mid = secondary_scale_from_slack(0.0225, cfg)
    assert 0.15 < mid < 1.0


def test_shipped_slack_fade_has_midband_at_measured_slack() -> None:
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    cfg = build_joint_ik_config(raw).saturation
    assert cfg.slack_enter == pytest.approx(0.15)
    assert cfg.slack_exit == pytest.approx(0.03)
    mid = secondary_scale_from_slack(0.09, cfg)
    assert 0.15 < mid < 1.0
    assert secondary_scale_from_slack(0.142, cfg) < 1.0


def test_secondary_scale_lpf_does_not_jump() -> None:
    inner = _inner(_safe_q())
    inner.last_slack_norm = 0.20
    inner._secondary(_safe_q(), None, dt_s=0.005)
    s1 = float(inner.last_sat_scale)
    inner.last_slack_norm = 0.20
    inner._secondary(_safe_q(), None, dt_s=0.005)
    s2 = float(inner.last_sat_scale)
    assert 0.15 < s1 < 1.0
    assert s2 < s1
    assert s1 - s2 < 0.20


def test_secondary_soft_scale_fades_soft_tasks_not_ff() -> None:
    kin = RobotKinematics()
    centering = JointCenteringTask.from_kinematics(
        kin, NullspaceTaskConfig(k_center=1.0, k_limit=2.0, activation=0.75)
    )
    composer = SecondaryComposer(
        centering, None, v_max=kin.v_max, max_qdot_frac=0.2
    )
    q = np.array([0.40, 0.0, -0.4, 0.2, 1.2, 0.1, 0.8, 0.0])
    full = composer.compose(q, None, np.zeros(8), arm_suppressed=True, soft_scale=1.0)
    fade = composer.compose(q, None, np.zeros(8), arm_suppressed=True, soft_scale=0.15)
    assert np.linalg.norm(fade) == pytest.approx(0.15 * np.linalg.norm(full), rel=1e-6)
    ff = np.full(8, 0.05)
    with_ff = composer.compose(q, ff, np.zeros(8), arm_suppressed=True, soft_scale=0.15)
    assert np.linalg.norm(with_ff - ff) == pytest.approx(np.linalg.norm(fade), rel=1e-6)


def test_slew_vec_jerk_stop_does_not_ring_through_zero() -> None:
    vel = np.array([0.10, 0.0, 0.0])
    acc = np.array([0.8, 0.0, 0.0])
    target = np.zeros(3)
    lo = 0.0
    for _ in range(400):
        vel, acc = slew_vec_jerk(vel, acc, target, 0.8, 2.0, 0.005)
        lo = min(lo, float(vel[0]))
    assert float(vel[0]) == pytest.approx(0.0, abs=0.008)
    assert lo >= -0.005
    assert float(abs(acc[0])) < 0.05


def test_slew_vec_jerk_does_not_dump_accel_when_catching_target() -> None:
    vel = np.zeros(3)
    acc = np.zeros(3)
    target = np.array([0.10, 0.0, 0.0])
    dt = 0.005
    j_max = 4.0
    prev_a = acc.copy()
    peak_j = 0.0
    for _ in range(120):
        vel, acc = slew_vec_jerk(vel, acc, target, 0.8, j_max, dt)
        peak_j = max(peak_j, float(np.linalg.norm(acc - prev_a)) / dt)
        prev_a = acc.copy()
    assert peak_j <= j_max + 1.0e-6
    assert float(vel[0]) == pytest.approx(0.10, abs=0.012)


def test_deadzone_hysteresis_does_not_drop_idle_on_edge() -> None:
    pad = FakePad(axes=np.array([-1.0, 0.0, -1.0, 0.0, 0.0, -1.0]))
    cfg = GamepadTwistConfig(
        trans_m_s=0.10,
        deadzone=0.18,
        deadzone_release=0.12,
        dt=0.005,
        trans_a_max_m_s2=100.0,
        trans_j_max_m_s3=0.0,
        pad_lpf_hz=0.0,
        control_frame="base",
        hold_relatch_on_settle=False,
    )
    outer = GamepadTwistOuterLoop(pad, cfg)
    pose = np.array([0.4, 0.2, 0.3, 0.0, 0.0, 0.0])
    outer.set_origin(pose)
    outer.sample(0.0, pose, np.zeros(6))
    assert outer._pad_latched
    pad.axes = np.array([-0.15, 0.0, -1.0, 0.0, 0.0, -1.0])
    twist = outer.sample(0.0, pose, np.zeros(6))
    assert outer._pad_latched
    assert not outer._idle_hold_active
    assert float(np.linalg.norm(twist[:3])) >= 0.0


def test_slew_vec_jerk_first_step_is_smaller_than_accel_only() -> None:
    prev = np.zeros(3)
    acc = np.zeros(3)
    target = np.array([0.10, 0.0, 0.0])
    v_a = slew_vec(prev, target, 0.8, 0.005)
    v_j, _ = slew_vec_jerk(prev, acc, target, 0.8, 4.0, 0.005)
    assert float(v_j[0]) < float(v_a[0])
    assert float(v_j[0]) > 0.0


def test_slew_vec_caps_per_tick_delta() -> None:
    prev = np.zeros(3)
    target = np.array([0.10, 0.0, 0.0])
    out = slew_vec(prev, target, 0.8, 0.005)
    assert float(np.linalg.norm(out - prev)) == pytest.approx(0.004, abs=1e-9)
    assert out[0] > 0.0


def test_gamepad_button_first_step_is_jerk_limited() -> None:
    buttons = np.zeros(8, dtype=float)
    buttons[4] = 1.0
    pad = FakePad(axes=np.array([0.0, 0.0, -1.0, 0.0, 0.0, -1.0]), buttons=buttons)
    cfg = GamepadTwistConfig(
        trans_m_s=0.10,
        deadzone=0.10,
        dt=0.005,
        trans_a_max_m_s2=0.8,
        trans_j_max_m_s3=4.0,
        pad_lpf_hz=8.0,
        control_frame="base",
        hold_relatch_on_settle=False,
    )
    outer = GamepadTwistOuterLoop(pad, cfg)
    pose = np.array([0.4, 0.2, 0.3, 0.0, 0.0, 0.0])
    outer.set_origin(pose)
    twist = outer.sample(0.0, pose, np.zeros(6))
    a_only = 0.8 * 0.005
    assert float(abs(twist[2])) < a_only
    assert float(abs(twist[2])) > 0.0


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


def test_slew_axes_jerk_does_not_cross_couple() -> None:
    vel = np.array([0.0, 0.0, 0.0])
    acc = np.zeros(3)
    target = np.array([0.0, 0.0, -0.10])
    for _ in range(40):
        vel, acc = slew_axes_jerk(vel, acc, target, 0.8, 8.0, 0.005)
    assert abs(float(vel[0])) < 1.0e-12
    assert abs(float(vel[1])) < 1.0e-12
    assert float(vel[2]) < -0.02


def test_lt_does_not_slew_roll() -> None:
    buttons = np.zeros(8, dtype=float)
    pad = FakePad(axes=np.array([0.0, 0.0, 1.0, 0.0, 0.0, -1.0]), buttons=buttons)
    cfg = GamepadTwistConfig(
        trans_m_s=0.10,
        rot_rad_s=0.60,
        deadzone=0.10,
        trigger_deadzone=0.08,
        dt=0.005,
        pad_lpf_hz=0.0,
        trans_j_max_m_s3=8.0,
        rot_j_max_rad_s3=16.0,
        control_frame="base",
        hold_relatch_on_settle=False,
    )
    outer = GamepadTwistOuterLoop(pad, cfg)
    pose = np.array([0.4, 0.2, 0.3, 0.0, 0.0, 0.0])
    outer.set_origin(pose)
    twist = np.zeros(6)
    for _ in range(30):
        twist = outer.sample(0.0, pose, np.zeros(6))
    assert float(twist[2]) < -0.02
    assert float(np.linalg.norm(twist[3:6])) < 1.0e-9
    np.testing.assert_allclose(outer.last_twist_base[3:6], 0.0, atol=1e-12)


def test_roll_does_not_slew_translation() -> None:
    pad = FakePad(axes=np.array([0.0, 0.0, -1.0, 1.0, 0.0, -1.0]))
    cfg = GamepadTwistConfig(
        trans_m_s=0.10,
        rot_rad_s=0.60,
        deadzone=0.10,
        dt=0.005,
        pad_lpf_hz=0.0,
        control_frame="base",
        hold_relatch_on_settle=False,
    )
    outer = GamepadTwistOuterLoop(pad, cfg)
    pose = np.array([0.4, 0.2, 0.3, 0.0, 0.0, 0.0])
    outer.set_origin(pose)
    twist = np.zeros(6)
    for _ in range(30):
        twist = outer.sample(0.0, pose, np.zeros(6))
    assert float(twist[3]) > 0.05
    assert float(np.linalg.norm(twist[:3])) < 1.0e-9


def test_tool_frame_last_twist_base_is_world() -> None:
    pad = FakePad(axes=np.array([0.0, 0.0, 1.0, 0.0, 0.0, -1.0]))
    cfg = GamepadTwistConfig(
        trans_m_s=0.10,
        deadzone=0.10,
        trigger_deadzone=0.08,
        dt=0.005,
        pad_lpf_hz=0.0,
        trans_j_max_m_s3=0.0,
        trans_a_max_m_s2=100.0,
        control_frame="tool",
        hold_relatch_on_settle=False,
        euler_order="xyz",
    )
    outer = GamepadTwistOuterLoop(pad, cfg)
    pose = np.array([0.4, 0.2, 0.3, 0.0, 0.8, 0.0])
    outer.set_origin(pose)
    twist = outer.sample(0.0, pose, np.zeros(6))
    assert float(np.linalg.norm(twist[:3])) > 0.0
    assert abs(float(outer.last_twist_base[2]) + 0.10) < 1.0e-9
    assert abs(float(outer.last_twist_base[0])) < 1.0e-9
