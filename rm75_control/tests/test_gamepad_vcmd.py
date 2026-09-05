"""Gamepad → inner-loop v_cmd mapping and QPIK feed."""

from __future__ import annotations

from pathlib import Path

import csv

import numpy as np
import pytest
import yaml

from rm75_control.control.admittance_common.phase_ipc import SinToolYTaskParams
from rm75_control.control.joint_admittance_8dof.api import SecondaryPolicy
from rm75_control.control.joint_admittance_8dof.config import build_joint_ik_config
from rm75_control.control.joint_admittance_8dof.gamepad_vcmd_program import (
    build_gamepad_vcmd_program,
    close_built_pad,
)
from rm75_control.control.joint_admittance_8dof.loop import JointIkController
from rm75_control.control.joint_admittance_8dof.model import RobotKinematics
from rm75_control.control.joint_admittance_8dof.teleop.gamepad_twist import (
    GamepadTwistConfig,
    GamepadTwistOuterLoop,
    compose_inner_twist,
    map_pad_to_world_lin_tool_ang,
)
from rm75_control.control.joint_admittance_8dof.teleop.xbox_pad import FakePad, PadState
from rm75_control.control.joint_admittance_8dof.tasks.psi_retarget import (
    d_from_q,
    nearest_planar_psi,
)
from rm75_control.kinematics.srs_ik import psi_from_q


# Keep fixtures on the controller's calibrated design family.  The previous
# arbitrary joint seed is direction-infeasible for the direction-preserving
# HQP and makes tests exercise its pause path instead of the gamepad command.
_SEED_Q = np.array(
    [0.375, *np.deg2rad([-89.5, -94.5, 65.2, 96.0, 89.3, 61.0, 94.6])],
    dtype=float,
)
_CFG = Path(__file__).resolve().parents[1] / "configs" / "joint_admittance_8dof.yaml"


def _state(*, lx=0.0, ly=0.0, lt=-1.0, rx=0.0, ry=0.0, rt=-1.0, lb=0.0, rb=0.0) -> PadState:
    axes = np.array([lx, ly, lt, rx, ry, rt], dtype=float)
    buttons = np.zeros(8, dtype=float)
    buttons[4] = lb
    buttons[5] = rb
    return PadState(axes=axes, buttons=buttons)


def _ideal_update(inner: JointIkController, twist: np.ndarray, *args, **kwargs):
    """Advance the offline plant with explicit rail timing/feedback.

    The test loop is faster than wall time.  Treat the previous committed
    rail velocity as the measured rail velocity for the next tick, and use
    the configured period for wall-time integration.  This mirrors the
    timestamped hardware path without fabricating a stopped rail.
    """
    kwargs.setdefault("rail_exec_vel_m_s", float(inner.core.qdot_prev[0]))
    kwargs.setdefault("dt_wall_s", float(inner.cfg.dt))
    return inner.update(twist, *args, **kwargs)


def test_left_stick_left_is_world_plus_y() -> None:
    cfg = GamepadTwistConfig(trans_m_s=0.04, deadzone=0.10)
    v_world, w_tool = map_pad_to_world_lin_tool_ang(_state(lx=-1.0), cfg)
    assert v_world[1] > 0.03
    assert abs(v_world[0]) < 1e-12
    assert abs(v_world[2]) < 1e-12
    assert np.allclose(w_tool, 0.0)


def test_lb_up_lt_down_are_world_z() -> None:
    cfg = GamepadTwistConfig(trans_m_s=0.04, deadzone=0.10)
    v_up, _ = map_pad_to_world_lin_tool_ang(_state(lb=1.0), cfg)
    v_dn, _ = map_pad_to_world_lin_tool_ang(_state(lt=1.0), cfg)
    assert v_up[2] > 0.03
    assert v_dn[2] < -0.03


def test_right_stick_and_rb_rt_are_tool_rotation() -> None:
    cfg = GamepadTwistConfig(rot_rad_s=0.30, deadzone=0.10)
    _, w_up = map_pad_to_world_lin_tool_ang(_state(ry=-1.0), cfg)
    _, w_down = map_pad_to_world_lin_tool_ang(_state(ry=1.0), cfg)
    _, w_left = map_pad_to_world_lin_tool_ang(_state(rx=-1.0), cfg)
    _, w_right = map_pad_to_world_lin_tool_ang(_state(rx=1.0), cfg)
    _, w_rb = map_pad_to_world_lin_tool_ang(_state(rb=1.0), cfg)
    _, w_rt = map_pad_to_world_lin_tool_ang(_state(rt=1.0), cfg)
    assert w_up[1] < -0.2 and abs(w_up[0]) < 1e-12
    assert w_down[1] > 0.2 and abs(w_down[0]) < 1e-12
    assert w_left[0] < -0.2 and abs(w_left[1]) < 1e-12
    assert w_right[0] > 0.2 and abs(w_right[1]) < 1e-12
    assert w_rb[2] > 0.2
    assert w_rt[2] < -0.2


def test_tool_frame_inner_twist_roundtrips_world_y() -> None:
    pose = np.array([0.4, 0.2, 0.3, 0.0, 0.4, 0.1])
    v_world = np.array([0.0, 0.04, 0.0])
    w_tool = np.array([0.1, 0.0, 0.0])
    twist, twist_base = compose_inner_twist(
        v_world, w_tool, pose, euler_order="xyz", control_frame="tool"
    )
    assert abs(twist_base[1] - 0.04) < 1e-12
    from scipy.spatial.transform import Rotation as Rsc

    rotation = Rsc.from_euler("xyz", pose[3:6], degrees=False).as_matrix()
    recovered = np.zeros(6)
    recovered[:3] = rotation @ twist[:3]
    recovered[3:6] = rotation @ twist[3:6]
    np.testing.assert_allclose(recovered, twist_base, atol=1e-12)


def test_outer_loop_feeds_plus_y_into_qpik() -> None:
    raw = yaml.safe_load(_CFG.read_text(encoding="utf-8"))
    cfg = build_joint_ik_config(raw)
    cfg.backend = "python"
    cfg.collision.enabled = False
    cfg.qp.collision.enabled = False
    cfg.ird.enabled = False
    kin = RobotKinematics()
    inner = JointIkController(kin, cfg)
    q = np.deg2rad(np.array([0.0, -89.5, -94.5, 65.2, 96.0, 89.3, 61.0, 94.6]))
    q[0] = 0.375
    inner.reset(q)
    pad = FakePad(axes=np.array([-1.0, 0.0, -1.0, 0.0, 0.0, -1.0]))
    outer = GamepadTwistOuterLoop(
        pad,
        GamepadTwistConfig(
            trans_m_s=0.04,
            deadzone=0.10,
            dt=cfg.dt,
            euler_order=cfg.euler_order,
            control_frame=cfg.control_frame,
        ),
    )
    pose0 = kin.fk_pose(q)
    # Keep the original 8 mm displacement threshold while allowing the
    # direction-preserving task scaler enough time to accept the command.
    for _ in range(120):
        pose = kin.fk_pose(inner.q_cmd)
        twist = outer.sample(0.0, pose, np.zeros(6))
        _ideal_update(inner, twist, cfg.dt, q_meas=inner.q_cmd)
    pose1 = kin.fk_pose(inner.q_cmd)
    assert pose1[1] - pose0[1] > 0.008


def test_gamepad_can_reverse_off_plus_leave() -> None:
    """A leave command keeps its requested direction under the HQP."""
    raw = yaml.safe_load(_CFG.read_text(encoding="utf-8"))
    cfg = build_joint_ik_config(raw)
    cfg.backend = "python"
    cfg.collision.enabled = False
    cfg.qp.collision.enabled = False
    cfg.ird.enabled = False
    cfg.qp.joint_comfort.enabled = False
    kin = RobotKinematics()
    inner = JointIkController(kin, cfg)
    q = _SEED_Q.copy()
    q[0] = 0.75
    inner.reset(q)
    inner.begin_hybrid_episode(q, np.zeros(8))
    pose = kin.fk_pose(q)
    twist_away = np.zeros(6)
    twist_away[1] = -0.04
    step = _ideal_update(
        inner,
        twist_away,
        dt=cfg.dt,
        q_meas=q,
        pose_d=pose,
        vel_ff=twist_away,
        task_rotation_base=np.eye(3),
    )
    assert not step.rail_sat
    assert not step.task_paused
    assert float(step.task_progress) > 0.0
    assert float(step.v_cmd_received[1]) < -0.03
    assert float(step.v_cmd_feasible[1]) < 0.0
    assert float(step.protected_achieved[1]) < 0.0
    # Direction preserving HQP may carry this command with the arm.  The rail
    # contribution must still point away from the +soft_max wall when used.
    assert np.isfinite(step.rail_task_vel)
    assert float(step.rail_task_vel) <= 0.0
    assert np.isfinite(step.qdot).all()


def test_logger_records_pad_and_vcmd(tmp_path) -> None:
    from rm75_control.control.joint_admittance_8dof.loop import JointIkStep, _TickLogger

    pad = FakePad(axes=np.array([-1.0, 0.0, -1.0, 0.0, 0.0, -1.0]))
    outer = GamepadTwistOuterLoop(
        pad,
        GamepadTwistConfig(
            trans_m_s=0.08,
            deadzone=0.10,
            control_frame="base",
            pad_lpf_hz=0.0,
            trans_j_max_m_s3=0.0,
        ),
    )
    pose = np.array([0.4, 0.2, 0.3, 0.0, 0.0, 0.0])
    twist = outer.sample(0.0, pose, np.zeros(6))
    assert twist[1] > 0.0
    assert outer.last_twist_slewed
    path = tmp_path / "gamepad_vcmd.csv"
    logger = _TickLogger(str(path))
    step = JointIkStep(
        q_send=np.zeros(8),
        qdot=np.zeros(8),
        twist_base=twist,
        sigma_min=0.2,
        manip=0.1,
        slack_norm=0.0,
        n_cbf_active=0,
        follow_err_rad=0.0,
    )
    logger.write(
        0.0, "gamepad_vcmd", 0.0, step, np.zeros(8), pose, np.zeros(6), outer=outer
    )
    logger.close()
    with path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    header, values = rows[0], dict(zip(rows[0], rows[1], strict=True))
    assert "pad_lx" in header
    assert float(values["pad_lx"]) == pytest.approx(-1.0)
    assert float(values["pad_vy"]) > 0.0
    assert float(values["pad_vcmd_base_vy"]) > 0.0
    assert values["pad_connected"] == "1"


def test_idle_sample_latches_pose_d_and_rebases_on_stick() -> None:
    pad = FakePad(axes=np.array([0.0, 0.0, -1.0, 0.0, 0.0, -1.0]))
    outer = GamepadTwistOuterLoop(
        pad,
        GamepadTwistConfig(
            trans_m_s=0.10,
            deadzone=0.10,
            control_frame="base",
            trans_a_max_m_s2=100.0,
            rot_a_max_rad_s2=100.0,
            trans_j_max_m_s3=0.0,
            rot_j_max_rad_s3=0.0,
            pad_lpf_hz=0.0,
        ),
    )
    pose0 = np.array([0.40, 0.20, 0.30, 0.0, 0.0, 0.0])
    outer.set_origin(pose0)
    drifted = pose0.copy()
    drifted[1] += 0.012
    twist = outer.sample(0.0, drifted, np.zeros(6))
    np.testing.assert_allclose(outer.last_pose_d, pose0, atol=1e-12)
    assert float(np.linalg.norm(twist[:3])) > 0.0
    assert twist[1] < 0.0
    pad.axes = np.array([-1.0, 0.0, -1.0, 0.0, 0.0, -1.0])
    twist_go = outer.sample(0.0, drifted, np.zeros(6))
    assert twist_go[1] > 0.05
    assert outer.last_pose_d[1] == pytest.approx(
        drifted[1] + twist_go[1] * outer.cfg.dt
    )


def test_gamepad_trans_default_is_100_mm_s() -> None:
    assert SinToolYTaskParams(config_path="x").gamepad_trans_m_s == pytest.approx(0.10)
    assert GamepadTwistConfig().trans_m_s == pytest.approx(0.10)
    assert GamepadTwistConfig().pad_lpf_hz == pytest.approx(16.0)
    assert GamepadTwistConfig().trans_j_max_m_s3 == pytest.approx(8.0)
    assert GamepadTwistConfig().rot_j_max_rad_s3 == pytest.approx(16.0)
    params = SinToolYTaskParams(config_path="x")
    assert params.gamepad_pad_lpf_hz == pytest.approx(16.0)
    assert params.gamepad_trans_j_max_m_s3 == pytest.approx(8.0)
    assert params.gamepad_rot_j_max_rad_s3 == pytest.approx(16.0)


def test_lt_slew_does_not_move_roll() -> None:
    pad = FakePad(axes=np.array([0.0, 0.0, 1.0, 0.0, 0.0, -1.0]))
    outer = GamepadTwistOuterLoop(
        pad,
        GamepadTwistConfig(
            trans_m_s=0.10,
            rot_rad_s=0.60,
            deadzone=0.10,
            trigger_deadzone=0.08,
            dt=0.005,
            pad_lpf_hz=0.0,
            control_frame="base",
            hold_relatch_on_settle=False,
        ),
    )
    pose = np.array([0.4, 0.2, 0.3, 0.0, 0.0, 0.0])
    outer.set_origin(pose)
    twist = np.zeros(6)
    for _ in range(30):
        twist = outer.sample(0.0, pose, np.zeros(6))
    assert float(twist[2]) < -0.02
    assert float(np.linalg.norm(twist[3:6])) < 1.0e-9


def test_release_slew_does_not_cross_zero() -> None:
    pad = FakePad(axes=np.array([-1.0, 0.0, -1.0, 0.0, 0.0, -1.0]))
    outer = GamepadTwistOuterLoop(
        pad,
        GamepadTwistConfig(
            trans_m_s=0.10,
            deadzone=0.10,
            dt=0.005,
            trans_a_max_m_s2=0.8,
            trans_j_max_m_s3=8.0,
            pad_lpf_hz=0.0,
            control_frame="base",
            hold_relatch_on_settle=True,
        ),
    )
    pose = np.array([0.40, 0.20, 0.30, 0.0, 0.0, 0.0])
    outer.set_origin(pose)
    moving = pose.copy()
    for i in range(80):
        moving = pose.copy()
        moving[1] += 0.0004 * (i + 1)
        outer.sample(0.0, moving, np.zeros(6))
    assert outer.last_twist_base[1] > 0.02
    pad.axes = np.array([0.0, 0.0, -1.0, 0.0, 0.0, -1.0])
    lo = 0.0
    for _ in range(200):
        twist = outer.sample(0.0, moving, np.zeros(6))
        lo = min(lo, float(twist[1]))
    assert lo >= -0.005


def test_settle_relatch_pose_d_is_current() -> None:
    pad = FakePad(axes=np.array([-1.0, 0.0, -1.0, 0.0, 0.0, -1.0]))
    outer = GamepadTwistOuterLoop(
        pad,
        GamepadTwistConfig(
            trans_m_s=0.10,
            deadzone=0.10,
            dt=0.005,
            trans_a_max_m_s2=100.0,
            hold_relatch_on_settle=True,
            hold_settle_v_m_s=0.005,
            control_frame="base",
        ),
    )
    pose = np.array([0.40, 0.20, 0.30, 0.0, 0.0, 0.0])
    outer.set_origin(pose)
    flying = pose.copy()
    flying[1] += 0.020
    outer.sample(0.0, flying, np.zeros(6))
    latch_flight = outer.last_pose_d.copy()
    pad.axes = np.array([0.0, 0.0, -1.0, 0.0, 0.0, -1.0])
    coasting = flying.copy()
    coasting[1] += 0.002
    outer.sample(0.0, coasting, np.zeros(6))
    settled = coasting.copy()
    outer.sample(0.0, settled, np.zeros(6))
    assert not outer._coast_until_settle
    np.testing.assert_allclose(outer.last_pose_d, settled, atol=1e-12)
    assert not np.allclose(outer.last_pose_d, latch_flight)


def test_last_twist_base_stays_world_in_tool_frame() -> None:
    pad = FakePad(axes=np.array([0.0, 0.0, 1.0, 0.0, 0.0, -1.0]))
    outer = GamepadTwistOuterLoop(
        pad,
        GamepadTwistConfig(
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
        ),
    )
    pose = np.array([0.4, 0.2, 0.3, 0.0, 0.8, 0.0])
    outer.set_origin(pose)
    outer.sample(0.0, pose, np.zeros(6))
    assert abs(float(outer.last_twist_base[2]) + 0.10) < 1.0e-9
    assert abs(float(outer.last_twist_base[0])) < 1.0e-9
    pad.axes = np.array([0.0, 0.0, -1.0, 0.0, 0.0, -1.0])
    outer.sample(0.0, pose, np.zeros(6))
    assert abs(float(outer.last_twist_base[0])) < 1.0e-6
    assert abs(float(outer.last_twist_base[1])) < 1.0e-6


def test_xbox_pad_keeps_sigint_and_does_not_quit_pygame(monkeypatch) -> None:
    """Full pygame.init(); restore SIGINT so one Ctrl+C stops window A."""
    import signal

    from rm75_control.control.joint_admittance_8dof.teleop import xbox_pad

    calls = {"init": 0, "quit": 0}

    class _JoyMod:
        def init(self) -> None:
            return None

        def get_count(self) -> int:
            return 0

    class _Pygame:
        joystick = _JoyMod()

        def init(self) -> None:
            calls["init"] += 1
            signal.signal(signal.SIGINT, signal.SIG_DFL)

        def quit(self) -> None:
            calls["quit"] += 1

    fake = _Pygame()
    monkeypatch.setattr(xbox_pad, "_require_pygame", lambda: fake)

    def _handler(signum, _frame) -> None:
        return None

    prev = signal.getsignal(signal.SIGINT)
    try:
        signal.signal(signal.SIGINT, _handler)
        pad = xbox_pad.XboxPad(allow_missing=True, auto_select=False, device_index=0)
        assert calls["init"] == 1
        assert signal.getsignal(signal.SIGINT) is _handler
        pad.close()
        assert calls["quit"] == 0
    finally:
        signal.signal(signal.SIGINT, prev)


def test_build_gamepad_program_and_ipc_kind() -> None:
    params = SinToolYTaskParams(
        config_path=str(_CFG),
        task_kind="gamepad_vcmd",
        scan_duration=1.0,
        q0_rad=_SEED_Q.tolist(),
        q_target_rad=_SEED_Q.tolist(),
        pose_d=[0.0] * 6,
        tcp_offset_pose=[0.0] * 6,
    )
    text = params.to_json()
    decoded = SinToolYTaskParams.from_json(text)
    assert decoded.task_kind == "gamepad_vcmd"
    raw = yaml.safe_load(_CFG.read_text(encoding="utf-8"))
    raw.setdefault("inner", {})["backend"] = "python"
    built = build_gamepad_vcmd_program(params, raw=raw, pad=FakePad())
    try:
        assert built.phases[-1].label == "gamepad_vcmd"
        assert built.phases[-1].duration_s == 1.0
        assert built.phases[-1].governor_err_max_mm == 0.0
        built.phases[-1].on_enter()
        assert built.inner._rail_ext_active is True
        assert built.inner._arm_task_suppressed is False
        assert built.inner._centering_suppressed is False
        if built.inner.posture_retarget is not None:
            assert built.inner.posture_retarget.planned is False
        assert not hasattr(built.inner, "set_vcmd_owns_rail")
        assert not hasattr(built.inner, "set_rail_hold_when_idle")
    finally:
        close_built_pad(built)


def _yaml_inner_at_rail(q_rail_m: float) -> JointIkController:
    raw = yaml.safe_load(_CFG.read_text(encoding="utf-8"))
    cfg = build_joint_ik_config(raw)
    cfg.backend = "python"
    cfg.collision.enabled = False
    cfg.qp.collision.enabled = False
    cfg.ird.enabled = False
    inner = JointIkController(RobotKinematics(), cfg)
    q = _SEED_Q.copy()
    q[0] = float(q_rail_m)
    inner.reset(q)
    return inner


def _plus_y_step(inner: JointIkController, q_rail_m: float):
    q = _SEED_Q.copy()
    q[0] = float(q_rail_m)
    twist = np.array([0.0, 0.08, 0.0, 0.0, 0.0, 0.0])
    return _ideal_update(inner, twist, q_meas=q, vel_ff=twist)


def test_unplanned_inner_holds_taught_plane_not_q_nominal() -> None:
    inner = _yaml_inner_at_rail(0.375)
    SecondaryPolicy(preset="track", qdot_ff="off").apply(inner)
    q = np.array(
        [0.774, 0.0, np.deg2rad(-30.0), 0.0, np.pi / 2.0, 0.0, np.pi / 2.0, np.pi / 2.0]
    )
    inner.reset(q)
    inner.begin_hybrid_episode(q, np.zeros(8))
    d_yaml = d_from_q(inner.kin, inner.centering_task._q_target_default)
    d_live = d_from_q(inner.kin, q)
    psi_yaml = float(inner.cfg.psi_retarget.psi_attr_rad)
    psi_taught = nearest_planar_psi(psi_from_q(q))
    assert abs(d_live - d_yaml) > 1.0e-3
    assert abs(psi_taught - psi_yaml) > 1.0
    twist = np.zeros(6)
    last = None
    for _ in range(6):
        last = _ideal_update(inner, twist, q_meas=inner.q_cmd, vel_ff=twist)
    assert inner.posture_retarget is not None
    assert not inner.posture_retarget.planned
    assert inner.posture_retarget.d_star_m == pytest.approx(d_live, abs=0.08)
    assert abs(
        inner.posture_retarget.d_star_m - float(inner.cfg.psi_retarget.d_attr_m)
    ) > 0.05
    assert inner.posture_retarget.psi_star_rad == pytest.approx(
        float(inner.cfg.psi_retarget.psi_attr_rad), abs=1e-6
    )
    assert last is not None
    assert abs(abs(float(last.psi_ref_deg)) - 180.0) < 5.0
    assert float(last.psi_ref_deg) > 0.0


def test_gamepad_track_is_full_inner_loop() -> None:
    """Unplanned track (gamepad) keeps rail_ext / arm angle; no scan fade."""
    inner = _yaml_inner_at_rail(0.70)
    SecondaryPolicy(preset="track", qdot_ff="off").apply(inner)
    assert inner._rail_ext_active is True
    assert inner._arm_task_suppressed is False
    assert inner._centering_suppressed is False
    assert inner.posture_retarget is not None
    assert not inner.posture_retarget.planned
    step = _plus_y_step(inner, 0.70)
    assert float(step.v_cmd[1]) > 0.05
    assert np.isfinite(step.rail_task_vel)
    assert not bool(step.last_limit_saturated)
    assert not bool(inner.rail_ext_task.last_in_limit_band)


def test_unplanned_plus_leave_does_not_zero_rail_task() -> None:
    inner = _yaml_inner_at_rail(0.74)
    SecondaryPolicy(preset="track", qdot_ff="off").apply(inner)
    step = _plus_y_step(inner, 0.74)
    assert float(step.v_cmd[1]) > 0.05
    assert np.isfinite(step.rail_task_vel)
    assert float(step.rail_task_vel) != pytest.approx(0.0, abs=1e-4)


def test_planned_stroke_still_fades_and_holds_d_star() -> None:
    inner = _yaml_inner_at_rail(0.70)
    SecondaryPolicy(preset="track", qdot_ff="off").apply(inner)
    q = _SEED_Q.copy()
    q[0] = 0.70
    inner.reset(q)
    y_c = float(inner.kin.fk_placement(_SEED_Q).translation[1])
    d0, psi0 = inner.plan_scan_stroke(y_c, 0.04, q)
    assert inner.posture_retarget is not None
    assert inner.posture_retarget.planned
    step = _plus_y_step(inner, 0.70)
    assert float(step.v_cmd[1]) > 0.05
    assert not step.task_paused
    assert float(step.task_progress) > 0.0
    assert inner.posture_retarget.d_star_m == pytest.approx(d0, abs=1e-9)
    assert inner.posture_retarget.psi_star_rad == pytest.approx(psi0, abs=1e-9)
    assert bool(inner.rail_ext_task.last_in_limit_band)
    assert float(step.rail_task_vel) < 0.055

    # Evaluate the leave-band behavior from a fresh, physically consistent
    # state.  Teleporting q_cmd from 0.70 to 0.74 while retaining QP history
    # creates an artificial direction-infeasible tick.
    inner_leave = _yaml_inner_at_rail(0.74)
    SecondaryPolicy(preset="track", qdot_ff="off").apply(inner_leave)
    q74 = _SEED_Q.copy()
    q74[0] = 0.74
    inner_leave.reset(q74)
    d_leave, psi_leave = inner_leave.plan_scan_stroke(y_c, 0.04, q74)
    assert d_leave == pytest.approx(d0, abs=1e-9)
    assert psi_leave == pytest.approx(psi0, abs=1e-9)
    step_leave = _plus_y_step(inner_leave, 0.74)
    assert float(step_leave.v_cmd[1]) > 0.05
    assert not step_leave.task_paused
    assert float(step_leave.task_progress) > 0.0
    # Plus leave-band must not drive into the stop; mid-ranging may reverse.
    assert np.isfinite(step_leave.rail_task_vel)
    assert float(step_leave.rail_task_vel) <= 1.0e-3


def test_infeasible_press_direction_pauses_without_escape_and_healthy_pose_progresses() -> None:
    """A jerk-limited press returns a residual; a real design pose still moves."""
    inner = _yaml_inner_at_rail(0.40)
    SecondaryPolicy(preset="track", qdot_ff="off").apply(inner)
    q = np.array(
        [0.40, 0.0, np.deg2rad(-30.0), 0.0, np.pi / 2.0, 0.0, np.pi / 2.0, np.pi / 2.0]
    )
    q[4] = float(inner.limits.q_upper[4]) - 0.20
    inner.reset(q)
    inner.begin_hybrid_episode(q, np.zeros(8))
    twist = np.array([0.0, 0.0, -0.08, 0.0, 0.0, 0.0])
    step = _ideal_update(inner,
        twist, q_meas=q, vel_ff=twist, task_rotation_base=np.eye(3)
    )
    assert float(step.v_cmd_received[2]) < -0.05
    assert not bool(step.task_paused)
    assert step.qp1_status in ("solved", "max_iter")
    np.testing.assert_allclose(step.v_tcp_estimated, step.v_cmd_feasible, atol=1.0e-5)
    assert not bool(step.rail_escape_active)
    if np.isfinite(step.v_escape):
        assert abs(float(step.v_escape)) <= 1.0e-6

    # The pose used by the direction-preserving design is feasible for a
    # lateral command.  Keep this healthy case beside the explicit pause case
    # so a bad fixture cannot make the whole test look like a press failure.
    healthy = _yaml_inner_at_rail(0.40)
    SecondaryPolicy(preset="track", qdot_ff="off").apply(healthy)
    q_healthy = _SEED_Q.copy()
    q_healthy[0] = 0.40
    healthy.reset(q_healthy)
    healthy.begin_hybrid_episode(q_healthy, np.zeros(8))
    twist_healthy = np.array([0.0, 0.04, 0.0, 0.0, 0.0, 0.0])
    healthy_step = _ideal_update(
        healthy,
        twist_healthy,
        q_meas=healthy.q_cmd.copy(),
        vel_ff=twist_healthy,
        task_rotation_base=np.eye(3),
    )
    assert not bool(healthy_step.task_paused)
    assert float(healthy_step.task_progress) > 0.0
    assert float(healthy_step.v_cmd_feasible[1]) > 0.0
    assert float(healthy_step.protected_achieved[1]) > 0.0
    assert not bool(healthy_step.rail_escape_active)


def test_emid_uses_live_tcp_not_latched_pose_d() -> None:
    inner = _yaml_inner_at_rail(0.40)
    SecondaryPolicy(preset="track", qdot_ff="off").apply(inner)
    q = _SEED_Q.copy()
    q[0] = 0.40
    inner.reset(q)
    y_live = float(inner.kin.fk_pose(q)[1])
    d_live = y_live - float(q[0])
    d_star = d_live - 0.008
    assert inner.posture_retarget is not None
    inner.posture_retarget.nudge_d_star(
        d_star - d_live,
        y_des_m=y_live,
        rail_lo=float(inner.limits.q_lower[0]),
        rail_hi=float(inner.limits.q_upper[0]),
        dt_s=float(inner.cfg.dt),
    )
    pose_d = inner.kin.fk_pose(q).copy()
    pose_d[1] += 0.40
    twist = np.array([0.0, 0.05, 0.0, 0.0, 0.0, 0.0])
    _ideal_update(inner, twist, q_meas=q, vel_ff=twist, pose_d=pose_d)
    d_used = float(inner.rail_ext_task.d_pref_m)
    rail_ff = float(inner.rail_ext_task.last_rail_ff_m)
    assert rail_ff == pytest.approx(y_live - d_used, abs=1e-6)
    assert abs(rail_ff - (float(pose_d[1]) - d_used)) > 0.30
    e_mid = float(inner.rail_ext_task.last_e_mid_m)
    assert e_mid == pytest.approx((y_live - d_used) - float(q[0]), abs=0.02)


def test_bounded_d_star_reference_preserves_requested_direction_without_escape() -> None:
    inner = _yaml_inner_at_rail(0.40)
    SecondaryPolicy(preset="track", qdot_ff="off").apply(inner)
    q = _SEED_Q.copy()
    q[0] = 0.40
    inner.reset(q)
    inner.begin_hybrid_episode(q, np.zeros(8))
    zero = np.zeros(6)
    seed = _ideal_update(inner, zero, q_meas=inner.q_cmd.copy(), vel_ff=zero)
    assert inner.posture_retarget is not None
    d_ref0 = float(seed.rail_d_ref)
    y_live = float(inner.kin.fk_pose(inner.q_cmd)[1])
    # Keep the historical large requested-target jump.  The controller must
    # slew its live reference toward it rather than teleporting 80 mm.
    d_target = d_ref0 - 0.08
    assert d_target == pytest.approx(d_ref0 - 0.08)
    d_ref_before = float(inner.posture_retarget.d_star_m)
    inner.posture_retarget.nudge_d_star(
        d_target - d_ref0,
        y_des_m=y_live,
        rail_lo=float(inner.limits.q_lower[0]),
        rail_hi=float(inner.limits.q_upper[0]),
        dt_s=float(inner.cfg.dt),
    )

    twist = np.array([0.0, 0.04, 0.0, 0.0, 0.0, 0.0])
    step = _ideal_update(
        inner, twist, q_meas=inner.q_cmd.copy(), vel_ff=twist
    )
    # The large requested-target jump is continuous at the executed
    # reference: one tick moves at most rate*authority*dt.
    d_step_max = (
        float(inner.posture_retarget.cfg.d_center_rate_m_s)
        * float(np.clip(step.secondary_alpha, 0.0, 1.0))
        * float(inner.cfg.dt)
    )
    assert abs(float(step.rail_d_ref) - d_ref_before) <= d_step_max + 1.0e-6
    assert float(step.v_cmd_received[1]) > 0.03
    assert not bool(step.rail_escape_active)
    if step.task_paused:
        assert float(step.task_progress) == pytest.approx(0.0)
        assert "infeasible" in str(step.task_pause_reason)
    else:
        assert float(step.task_progress) > 0.0
        assert float(step.v_cmd_feasible[1]) > 0.0
        assert float(step.protected_achieved[1]) > 0.0
        assert np.isfinite(step.rail_task_vel)
    if np.isfinite(step.v_escape):
        assert abs(float(step.v_escape)) <= 1.0e-6


def test_coupled_command_uses_ideal_feedback_and_preserves_direction() -> None:
    inner = _yaml_inner_at_rail(0.666)
    SecondaryPolicy(preset="track", qdot_ff="off").apply(inner)
    q_cmd = _SEED_Q.copy()
    q_cmd[0] = 0.666
    inner.reset(q_cmd)
    inner.q_cmd = q_cmd.copy()
    twist = np.array([0.0, -0.085, 0.0, 0.0, 0.0, 0.0])
    step = _ideal_update(
        inner, twist, q_meas=inner.q_cmd.copy(), vel_ff=twist
    )
    assert not bool(step.task_paused)
    assert float(step.task_progress) > 0.0
    assert float(step.v_cmd_received[1]) < -0.05
    assert float(step.v_cmd_feasible[1]) < 0.0
    assert float(step.protected_achieved[1]) < 0.0
    assert np.isfinite(step.qdot).all()
    assert np.isfinite(step.rail_task_vel)
    assert float(step.rail_task_vel) <= 0.0


def test_qpik_rail_brakes_when_task_drops() -> None:
    """Generic QPIK: a dropped Y command must not coast near cruise speed."""
    inner = _yaml_inner_at_rail(0.40)
    SecondaryPolicy(preset="track", qdot_ff="off").apply(inner)
    q = _SEED_Q.copy()
    q[0] = 0.40
    inner.reset(q)
    cruise = np.zeros(8)
    # The previous 0.12 m/s seed is outside this tick's feasible acceleration
    # box once the slow rail reference has dropped near zero (a_max*dt is only
    # 8 mm/s).  Seed a physically consistent committed command so the test
    # exercises braking rather than an invalid QP interval.
    cruise[0] = 0.008
    inner.core.sync_applied(cruise)
    twist0 = np.zeros(6)
    last = None
    for _ in range(250):
        last = _ideal_update(inner, twist0, q_meas=inner.q_cmd, vel_ff=twist0)
    assert last is not None
    assert abs(float(last.qdot[0])) < 0.05
    plus_y = np.array([0.0, 0.08, 0.0, 0.0, 0.0, 0.0])
    moving = None
    # Quiescent latch on the zero-cmd stretch resets homotopy on resume
    # (s=0). d* slew can cancel u_task for ~0.4 s; wait past that.
    for _ in range(160):
        moving = _ideal_update(inner, plus_y, q_meas=inner.q_cmd, vel_ff=plus_y)
    assert moving is not None
    # Direction preserving HQP may carry most of +Y with the arm.  It must
    # report accepted progress and a finite, bounded rail command rather than
    # coasting at the seeded speed.
    assert np.isfinite(moving.qdot).all()
    assert abs(float(moving.qdot[0])) <= float(inner.limits.v_max[0]) + 1.0e-6
    assert not bool(moving.rail_escape_active)
    assert not bool(moving.task_paused)
    assert float(moving.task_progress) > 0.0
    assert float(moving.v_cmd_feasible[1]) > 0.0


def test_rail_task_vel_is_issued_when_weight_is_zero_but_ff_is_live() -> None:
    inner = _yaml_inner_at_rail(0.40)
    SecondaryPolicy(preset="track", qdot_ff="off").apply(inner)
    inner.rail_ext_task.cfg.w_max = 0.0
    inner.rail_ext_task.cfg.w_sigma_floor = 0.0
    step = _plus_y_step(inner, 0.40)
    assert inner.rail_ext_task.last_weight == pytest.approx(0.0, abs=1e-12)
    assert abs(float(inner.rail_ext_task.last_v_ff)) > 1.0e-4
    assert np.isfinite(step.rail_task_vel)
    assert abs(float(step.rail_task_vel)) > 1e-4


def test_rail_task_vel_stays_dropped_when_ff_is_zero() -> None:
    inner = _yaml_inner_at_rail(0.40)
    SecondaryPolicy(preset="track", qdot_ff="off").apply(inner)
    inner.rail_ext_task.cfg.w_max = 0.0
    inner.rail_ext_task.cfg.w_sigma_floor = 0.0
    q = _SEED_Q.copy()
    q[0] = 0.40
    twist = np.zeros(6)
    step = _ideal_update(inner, twist, q_meas=q, vel_ff=twist)
    assert inner.rail_ext_task.last_weight == pytest.approx(0.0, abs=1e-12)
    assert abs(float(inner.rail_ext_task.last_v_ff)) < 1.0e-4
    assert np.isfinite(step.rail_task_vel)
    assert abs(float(step.rail_task_vel)) < 0.03


def test_zero_v_cmd_d_star_reference_is_continuous_without_escape() -> None:
    """A bounded d* change slews continuously while zero twist stays safe."""
    inner = _yaml_inner_at_rail(0.40)
    SecondaryPolicy(preset="track", qdot_ff="off").apply(inner)
    q = _SEED_Q.copy()
    q[0] = 0.40
    inner.reset(q)
    twist = np.zeros(6)
    last = _ideal_update(inner, twist, q_meas=inner.q_cmd.copy(), vel_ff=twist)
    assert inner.posture_retarget is not None
    d_ref0 = float(last.rail_d_ref)
    y_live = float(inner.kin.fk_pose(inner.q_cmd)[1])
    d_target = d_ref0 - 0.08
    inner.posture_retarget.nudge_d_star(
        d_target - d_ref0,
        y_des_m=y_live,
        rail_lo=float(inner.limits.q_lower[0]),
        rail_hi=float(inner.limits.q_upper[0]),
        dt_s=float(inner.cfg.dt),
    )
    refs = [d_ref0]
    pose0 = inner.kin.fk_pose(inner.q_cmd)
    for _ in range(12):
        last = _ideal_update(inner, twist, q_meas=inner.q_cmd.copy(), vel_ff=twist)
        refs.append(float(last.rail_d_ref))
    assert last is not None
    d_ref_step_max = float(inner.posture_retarget.cfg.d_center_rate_m_s) * float(
        inner.cfg.dt
    )
    assert np.isfinite(refs).all()
    assert np.max(np.abs(np.diff(refs))) <= d_ref_step_max + 1.0e-6
    assert abs(refs[-1] - refs[0]) < 0.002
    assert not bool(last.task_paused)
    assert float(last.task_progress) > 0.0
    assert np.allclose(last.v_cmd_received, np.zeros(6), atol=1.0e-12)
    assert not bool(last.rail_escape_active)
    assert np.isfinite(last.rail_task_vel)
    pose1 = inner.kin.fk_pose(inner.q_cmd)
    assert float(np.linalg.norm(pose1[:3] - pose0[:3])) < 0.008


def test_leave_wall_v_cmd_not_cancelled_by_u_mid() -> None:
    from rm75_control.control.joint_admittance_8dof.tasks.rail_allocator import (
        wall_leave_only_sign,
    )

    inner = _yaml_inner_at_rail(0.755)
    SecondaryPolicy(preset="track", qdot_ff="off").apply(inner)
    q = _SEED_Q.copy()
    q[0] = 0.755
    inner.reset(q)
    leave = np.array([0.0, -0.08, 0.0, 0.0, 0.0, 0.0])
    step = _ideal_update(inner, leave, q_meas=q, vel_ff=leave, task_rotation_base=np.eye(3))
    assert float(step.v_cmd[1]) < -0.04
    assert float(step.rail_task_vel) <= 1.0e-4 or float(step.rail_task_vel) < 0.0
    leave_sign = wall_leave_only_sign(
        0.755,
        hard_min_m=float(inner.limits.q_lower[0]),
        hard_max_m=float(inner.limits.q_upper[0]),
        band_m=float(inner.cfg.qp.limit_damper_band_rail_m),
    )
    v_lpf = float(inner.rail_ref_model.last_v_lpf)
    if leave_sign > 0.0:
        assert v_lpf <= 1.0e-6
    elif leave_sign < 0.0:
        assert v_lpf >= -1.0e-6


def test_zero_v_cmd_tcp_drift_after_quiescent() -> None:
    inner = _yaml_inner_at_rail(0.40)
    SecondaryPolicy(preset="track", qdot_ff="off").apply(inner)
    q = _SEED_Q.copy()
    q[0] = 0.40
    inner.reset(q)
    pose0 = inner.kin.fk_pose(inner.q_cmd)
    last = None
    for _ in range(80):
        last = _ideal_update(inner, np.zeros(6), q_meas=inner.q_cmd.copy())
    assert last is not None
    assert bool(inner._quiescent)
    assert inner.posture_retarget is not None
    d_ref0 = float(last.rail_d_ref)
    y_live = float(inner.kin.fk_pose(inner.q_cmd)[1])
    d_target = d_ref0 - 0.08
    inner.posture_retarget.nudge_d_star(
        d_target - d_ref0,
        y_des_m=y_live,
        rail_lo=float(inner.limits.q_lower[0]),
        rail_hi=float(inner.limits.q_upper[0]),
        dt_s=float(inner.cfg.dt),
    )
    refs = [d_ref0]
    for _ in range(12):
        last = _ideal_update(inner, np.zeros(6), q_meas=inner.q_cmd.copy())
        refs.append(float(last.rail_d_ref))
    pose1 = inner.kin.fk_pose(inner.q_cmd)
    assert bool(inner._quiescent)
    d_ref_step_max = float(inner.posture_retarget.cfg.d_center_rate_m_s) * float(
        inner.cfg.dt
    )
    assert np.isfinite(refs).all()
    assert np.max(np.abs(np.diff(refs))) <= d_ref_step_max + 1.0e-6
    assert abs(refs[-1] - refs[0]) < 0.002
    assert not bool(last.task_paused)
    assert float(last.task_progress) > 0.0
    assert not bool(last.rail_escape_active)
    assert np.isfinite(last.rail_task_vel)
    assert float(np.linalg.norm(pose1[:3] - pose0[:3])) < 0.02
    assert abs(float(last.qdot[0])) < 0.08


def test_quiescent_latch_ignores_tcp_residual() -> None:
    inner = _yaml_inner_at_rail(0.40)
    SecondaryPolicy(preset="track", qdot_ff="off").apply(inner)
    q = _SEED_Q.copy()
    q[0] = 0.40
    inner.reset(q)
    last = None
    for _ in range(80):
        last = _ideal_update(inner, np.zeros(6), q_meas=inner.q_cmd.copy())
    assert last is not None
    assert bool(inner._quiescent)
    h0 = float(getattr(last, "homotopy_s", float("nan")))
    if inner.posture_retarget is not None:
        h0 = float(inner.posture_retarget.homotopy_s)
    for _ in range(40):
        inner._last_tcp_est = np.array([0.02, 0.0, 0.0, 0.0, 0.0, 0.0])
        last = _ideal_update(inner, np.zeros(6), q_meas=inner.q_cmd.copy())
    assert bool(inner._quiescent)
    h1 = float(getattr(last, "homotopy_s", float("nan")))
    if inner.posture_retarget is not None:
        h1 = float(inner.posture_retarget.homotopy_s)
    # Idle rail PI stays held; homotopy itself stays live (no ψ freeze).
    assert h1 >= h0 - 1.0e-15


def test_quiescent_exit_uses_command_hysteresis() -> None:
    inner = _yaml_inner_at_rail(0.40)
    SecondaryPolicy(preset="track", qdot_ff="off").apply(inner)
    q = _SEED_Q.copy()
    q[0] = 0.40
    inner.reset(q)
    for _ in range(80):
        _ideal_update(inner, np.zeros(6), q_meas=inner.q_cmd.copy())
    assert bool(inner._quiescent)
    keep = np.array([0.006, 0.0, 0.0, 0.0, 0.0, 0.0])
    for _ in range(10):
        _ideal_update(inner, keep, q_meas=inner.q_cmd.copy(), vel_ff=keep)
    assert bool(inner._quiescent)
    leave = np.array([0.010, 0.0, 0.0, 0.0, 0.0, 0.0])
    _ideal_update(inner, leave, q_meas=inner.q_cmd.copy(), vel_ff=leave)
    assert not bool(inner._quiescent)


def test_tracker_step_api_returns_status() -> None:
    inner = _yaml_inner_at_rail(0.40)
    SecondaryPolicy(preset="track", qdot_ff="off").apply(inner)
    q = _SEED_Q.copy()
    q[0] = 0.40
    inner.reset(q)
    inner.enable()
    status = inner.step(
        np.array([0.0, 0.04, 0.0, 0.0, 0.0, 0.0]),
        stamp=None,
        q_meas=q,
        task_rotation_base=np.eye(3),
        rail_exec_vel_m_s=float(inner.core.qdot_prev[0]),
        dt_wall_s=float(inner.cfg.dt),
    )
    assert status.v_cmd_received.shape == (6,)
    assert status.v_cmd_feasible.shape == (6,)
    assert status.v_tcp_estimated.shape == (6,)
    assert np.isfinite(status.slack_norm)
    inner.stop()
    stopped = inner.step(
        np.array([0.0, 0.08, 0.0, 0.0, 0.0, 0.0]),
        q_meas=inner.q_cmd,
        rail_exec_vel_m_s=float(inner.core.qdot_prev[0]),
        dt_wall_s=float(inner.cfg.dt),
    )
    assert bool(stopped.command_stale)
