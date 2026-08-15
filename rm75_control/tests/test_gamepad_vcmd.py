"""Gamepad → inner-loop v_cmd mapping and QPIK feed."""

from __future__ import annotations

from pathlib import Path

import csv

import numpy as np
import pytest
import yaml

from rm75_control.control.admittance_common.phase_ipc import SinToolYTaskParams
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


_SEED_Q = np.array([0.375, 0.194, -0.503, -0.069, 1.979, -0.776, 0.547, -4.370])
_CFG = Path(__file__).resolve().parents[1] / "configs" / "joint_admittance_8dof.yaml"


def _state(*, lx=0.0, ly=0.0, lt=-1.0, rx=0.0, ry=0.0, rt=-1.0, lb=0.0, rb=0.0) -> PadState:
    axes = np.array([lx, ly, lt, rx, ry, rt], dtype=float)
    buttons = np.zeros(8, dtype=float)
    buttons[4] = lb
    buttons[5] = rb
    return PadState(axes=axes, buttons=buttons)


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
    _, w_stick = map_pad_to_world_lin_tool_ang(_state(rx=1.0, ry=-1.0), cfg)
    _, w_rb = map_pad_to_world_lin_tool_ang(_state(rb=1.0), cfg)
    _, w_rt = map_pad_to_world_lin_tool_ang(_state(rt=1.0), cfg)
    assert w_stick[0] > 0.2
    assert w_stick[1] > 0.2
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
    cfg.collision.enabled = False
    cfg.qp.collision.enabled = False
    cfg.ird.enabled = False
    kin = RobotKinematics()
    inner = JointIkController(kin, cfg)
    q = _SEED_Q.copy()
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
    for _ in range(80):
        pose = kin.fk_pose(inner.q_cmd)
        twist = outer.sample(0.0, pose, np.zeros(6))
        inner.update(twist, cfg.dt, q_meas=inner.q_cmd)
    pose1 = kin.fk_pose(inner.q_cmd)
    assert pose1[1] - pose0[1] > 0.008


def test_gamepad_can_reverse_off_plus_leave() -> None:
    """A stick command away from +soft_max must move q0; no extra freeze wall."""
    raw = yaml.safe_load(_CFG.read_text(encoding="utf-8"))
    cfg = build_joint_ik_config(raw)
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
    step = inner.update(
        twist_away, dt=cfg.dt, q_meas=q, pose_d=pose, task_rotation_base=np.eye(3)
    )
    assert not step.rail_sat
    assert float(step.qdot[0]) < -1.0e-4


def test_logger_records_pad_and_vcmd(tmp_path) -> None:
    from rm75_control.control.joint_admittance_8dof.loop import JointIkStep, _TickLogger

    pad = FakePad(axes=np.array([-1.0, 0.0, -1.0, 0.0, 0.0, -1.0]))
    outer = GamepadTwistOuterLoop(
        pad,
        GamepadTwistConfig(trans_m_s=0.08, deadzone=0.10, control_frame="base"),
    )
    pose = np.array([0.4, 0.2, 0.3, 0.0, 0.0, 0.0])
    twist = outer.sample(0.0, pose, np.zeros(6))
    assert twist[1] > 0.05
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
    assert float(values["pad_vy"]) > 0.05
    assert float(values["pad_vcmd_base_vy"]) > 0.05
    assert values["pad_connected"] == "1"


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
    built = build_gamepad_vcmd_program(params, pad=FakePad())
    try:
        assert built.phases[-1].label == "gamepad_vcmd"
        assert built.phases[-1].duration_s == 1.0
        assert built.phases[-1].governor_err_max_mm == 0.0
    finally:
        close_built_pad(built)
