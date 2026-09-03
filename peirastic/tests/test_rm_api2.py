"""RM_API2 aliases, force isolation, and mode-switch ownership."""

from __future__ import annotations

import os

import numpy as np
import pytest
import yaml

from peirastic.api import OK, PeirasticArm
from peirastic.api.rm_api2 import rm_joint_to_si, rm_speed_scale, si_joint_to_rm
from peirastic.core.modes import Mode
from peirastic.core.session import pad_may_drive, stay_after_duration
from peirastic.realman8dof.session import compile_request
from peirastic.configs import DEFAULT_CONTROLLER_YAML
from rm75_control.control.joint_admittance_8dof.api import CompileContext
from rm75_control.control.joint_admittance_8dof.config import build_joint_ik_config
from rm75_control.control.joint_admittance_8dof.loop import JointIkController
from rm75_control.control.joint_admittance_8dof.model import RobotKinematics

_SEED = np.array([0.375, 0.194, -0.503, -0.069, 1.979, -0.776, 0.547, -4.370])


def _ctx():
    raw = yaml.safe_load(DEFAULT_CONTROLLER_YAML.read_text())
    cfg = build_joint_ik_config(raw)
    cfg.backend = "python"
    cfg.native_shm_prefix = f"rm75_wbc_rm_{os.getpid()}"
    kin = RobotKinematics()
    inner = JointIkController(kin, cfg)
    inner.reset(_SEED)
    ctx = CompileContext(
        kin=kin,
        inner=inner,
        euler_order=cfg.euler_order,
        control_frame=cfg.control_frame,
        v_scale=cfg.v_scale,
    )
    return raw, ctx


def _arm(ctx=None):
    return PeirasticArm(attach=False, ctx=ctx)


def test_rm_unit_converters() -> None:
    assert rm_speed_scale(40) == pytest.approx(0.4)
    assert rm_speed_scale(0.4) == pytest.approx(0.4)
    with pytest.raises(ValueError):
        rm_speed_scale(0.0)
    with pytest.raises(ValueError):
        rm_speed_scale(140)
    q = rm_joint_to_si([400.0, -33.366, -49.897, 69.078, 93.258, 14.974, 64.971, 132.895])
    assert q[0] == pytest.approx(0.4)
    assert q[1] == pytest.approx(np.deg2rad(-33.366))
    back = si_joint_to_rm(q)
    assert back[0] == pytest.approx(400.0)


def test_rm_movej_sends_si_and_fractional_v() -> None:
    raw, ctx = _ctx()
    del raw
    arm = _arm(ctx=ctx)
    joint = [400.0, -33.366, -49.897, 69.078, 93.258, 14.974, 64.971, 132.895]
    assert arm.rm_movej(joint, 40, 0, 0, 0) == OK
    req = arm.last_request
    assert req is not None
    assert req.mode == Mode.MOVEJ
    assert req.payload["v"] == pytest.approx(0.4)
    assert req.payload["q_target"][0] == pytest.approx(0.4)


def test_force_axes_do_not_leak_into_movej_or_servo() -> None:
    raw, ctx = _ctx()
    arm = _arm(ctx=ctx)
    assert arm.set_force_control(force_axes=[0, 0, 1, 0, 0, 0], desired_force=2.0) == OK
    assert arm.movej(_SEED.tolist(), v=0.2, block=0) == OK
    assert "force_axes" not in arm.last_request.payload
    assert "desired_z" not in arm.last_request.payload
    pose = ctx.kin.fk_pose(_SEED)
    assert arm.cartesian_velocity([0.0, 0.02, 0.0, 0.0, 0.0, 0.0], block=0) == OK
    assert "force_axes" not in arm.last_request.payload
    assert arm.hfpc([pose.tolist()], speed_m_s=0.02, law="tff", block=0) == OK
    assert arm.last_request.payload["force_axes"][2] == pytest.approx(1.0)
    assert arm.last_request.mode == Mode.TRACK_HYBRID
    compile_request(ctx, arm.last_request, raw=raw)


def test_commanded_servo_label_blocks_pad() -> None:
    assert pad_may_drive(Mode.SERVO_TWIST, label="cartesian_velocity") is False
    assert pad_may_drive(Mode.SERVO_TWIST, label="servo_twist") is True
    assert pad_may_drive(Mode.TRACK_CARTESIAN, label="cartesian_track") is False
    assert stay_after_duration(Mode.SERVO_TWIST) is True
    assert stay_after_duration(Mode.TRACK_CARTESIAN) is False


def test_rm_movev_follow_latches_filter() -> None:
    raw, ctx = _ctx()
    arm = _arm(ctx=ctx)
    assert arm.rm_set_movev_canfd_init(1, 0, 5) == OK
    req = arm.last_request
    assert req is not None
    assert req.payload.get("filter") is True
    phase = compile_request(ctx, req, raw=raw)
    assert np.all(phase.outer.filter_axes)
    assert arm.rm_movev_canfd([0.0, 0.02, 0.0, 0.0, 0.0, 0.0], follow=True) == OK
    req_hi = arm.last_request
    assert req_hi.payload.get("filter") is False
    phase_hi = compile_request(ctx, req_hi, raw=raw)
    assert not np.any(phase_hi.outer.filter_axes)


def test_swappable_track_then_servo_compile() -> None:
    raw, ctx = _ctx()
    arm = _arm(ctx=ctx)
    pose = ctx.kin.fk_pose(_SEED)
    assert arm.cartesian_track(reference="hold", block=0, label="cartesian_track") == OK
    track = compile_request(ctx, arm.last_request, raw=raw)
    assert "track" in str(track.label) or "hold" in str(track.label)
    assert arm.cartesian_velocity(duration_s=1.0, block=0) == OK
    servo = compile_request(ctx, arm.last_request, raw=raw)
    assert str(servo.label) == "cartesian_velocity"
    servo.outer.set_origin(pose, t_s=0.0)
    out = np.asarray(servo.outer.sample(0.0, pose, np.zeros(6)), dtype=float)
    assert out.shape == (6,)
