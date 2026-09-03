"""Facade payloads, compile dispatch, and error codes."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_PLAYGROUND = Path(__file__).resolve().parents[2]
if str(_PLAYGROUND) not in sys.path:
    sys.path.insert(0, str(_PLAYGROUND))

import numpy as np
import pytest
import yaml

from peirastic.api import (
    ERR_CONTROLLER,
    ERR_UNIMPLEMENTED,
    OK,
    PeirasticArm,
)
from peirastic.api.codes import ERR_NO_ACK, ERR_SEND, ERR_STOPPED, ERR_TIMEOUT
from peirastic.api.payloads import HfpcPayload, HfvcPayload, MoveJPayload
from peirastic.api.vel_filter import pack_vel_filter, resolve_filter_axes
from peirastic.core.modes import Mode
from peirastic.realman8dof.modes.servo import ServoTwistOuter
from peirastic.realman8dof.modes.track import HybridTffOuter
from peirastic.realman8dof.session import ModeEngine, compile_request
from rm75_control.control.joint_admittance_8dof.api import CompileContext
from rm75_control.control.joint_admittance_8dof.config import build_joint_ik_config
from rm75_control.control.joint_admittance_8dof.loop import JointIkController
from rm75_control.control.joint_admittance_8dof.model import RobotKinematics
from peirastic.configs import DEFAULT_CONTROLLER_YAML

_SEED = np.array([0.375, 0.194, -0.503, -0.069, 1.979, -0.776, 0.547, -4.370])


def _ctx():
    raw = yaml.safe_load(DEFAULT_CONTROLLER_YAML.read_text())
    cfg = build_joint_ik_config(raw)
    cfg.backend = "python"
    cfg.native_shm_prefix = f"rm75_wbc_api_{os.getpid()}"
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


def _arm(ctx=None, inner=None):
    return PeirasticArm(attach=False, ctx=ctx, inner=inner)


def test_error_code_table() -> None:
    assert OK == 0
    assert ERR_CONTROLLER == 1
    assert ERR_SEND == -1
    assert ERR_NO_ACK == -2
    assert ERR_TIMEOUT == -5
    assert ERR_STOPPED == -6
    assert ERR_UNIMPLEMENTED == -7


def test_connect_and_blend_are_unimplemented() -> None:
    arm = _arm()
    q = _SEED.tolist()
    pose = [0.4, 0.2, 0.3, 0.0, 0.0, 0.0]
    assert arm.movej(q, v=0.2, connect=1, block=0) == ERR_UNIMPLEMENTED
    assert arm.cartesian(pose, v=0.2, r=0.01, block=0) == ERR_UNIMPLEMENTED
    assert arm.moves([pose], v=0.2, connect=1, block=0) == ERR_UNIMPLEMENTED


def test_payloads_match_live_dict_keys() -> None:
    movej = MoveJPayload(q_target=_SEED.tolist(), v=0.2).to_json()
    assert movej["q_target"][0] == pytest.approx(_SEED[0])
    hfpc = HfpcPayload(poses=[[0.4, 0.2, 0.3, 0.0, 0.0, 0.0]], law="tff", speed_m_s=0.02).to_json()
    assert hfpc["reference"] == "polyline"
    assert hfpc["use_tff_split"] is True
    hfvc = HfvcPayload(reference="pad", force=2.0).to_json()
    assert hfvc["reference"] == "pad"
    assert hfvc["desired_z"] == pytest.approx(2.0)


def test_hfpc_ellipse_and_hfvc_shuttle_compile() -> None:
    raw, ctx = _ctx()
    pose = ctx.kin.fk_pose(_SEED)
    del pose
    arm = _arm(ctx=ctx)
    assert arm.hfpc_ellipse(
        amplitude_x_m=0.05,
        amplitude_y_m=0.15,
        max_vel_m_s=0.04,
        force=0.0,
        force_axes=[0, 0, 1, 0, 0, 0],
        duration_s=2.0,
        label="hfpc_ellipse",
    ) == OK
    req = arm.last_request
    assert req is not None
    assert req.mode == Mode.TRACK_HYBRID
    assert req.payload["reference"] == "ellipse"
    assert req.payload["use_tff_split"] is True
    assert req.payload["desired_z"] == pytest.approx(0.0)
    phase = compile_request(ctx, req, raw=raw)
    assert isinstance(phase.outer, HybridTffOuter)
    assert np.allclose(phase.outer.selection, [1, 1, 0, 1, 1, 1])
    assert arm.hfvc(
        source="twist",
        force=0.0,
        force_axes=[0, 0, 1, 0, 0, 0],
        duration_s=1.5,
        label="hfvc_shuttle",
    ) == OK
    req_v = arm.last_request
    assert req_v is not None
    assert req_v.mode == Mode.TRACK_HYBRID
    assert req_v.payload["reference"] == "twist"
    phase_v = compile_request(ctx, req_v, raw=raw)
    assert isinstance(phase_v.outer, HybridTffOuter)
    assert isinstance(phase_v.outer.position, ServoTwistOuter)
    assert np.array_equal(
        phase_v.outer.position.filter_axes,
        [True, True, False, True, True, True],
    )


def test_hfpc_compiles_to_pose_tff() -> None:
    raw, ctx = _ctx()
    pose = ctx.kin.fk_pose(_SEED)
    arm = _arm(ctx=ctx)
    assert arm.hfpc([pose], speed_m_s=0.02, law="tff", block=0, label="hfpc_test") == OK
    req = arm.last_request
    assert req is not None
    assert req.mode == Mode.TRACK_HYBRID
    assert req.payload["use_tff_split"] is True
    phase = compile_request(ctx, req, raw=raw)
    assert isinstance(phase.outer, HybridTffOuter)
    assert not isinstance(phase.outer.position, ServoTwistOuter)


def test_hfvc_compiles_to_twist_tff() -> None:
    raw, ctx = _ctx()
    arm = _arm(ctx=ctx)
    assert arm.hfvc([0.0, 0.01, 0.0, 0.0, 0.0, 0.0], source="pad") == OK
    req = arm.last_request
    assert req is not None
    assert req.mode == Mode.TRACK_HYBRID
    assert req.payload["reference"] == "pad"
    phase = compile_request(ctx, req, raw=raw)
    assert isinstance(phase.outer, HybridTffOuter)
    assert isinstance(phase.outer.position, ServoTwistOuter)
    assert np.array_equal(
        phase.outer.position.filter_axes,
        [True, True, False, True, True, True],
    )


def test_selection_passthrough_default_and_force_x() -> None:
    raw, ctx = _ctx()
    pose = ctx.kin.fk_pose(_SEED)
    arm = _arm(ctx=ctx)
    arm.hfpc([pose], speed_m_s=0.02, law="tff", force_axes=[0, 0, 1, 0, 0, 0], block=0)
    phase = compile_request(ctx, arm.last_request, raw=raw)
    assert np.allclose(phase.outer.selection, [1, 1, 0, 1, 1, 1])
    arm.hfpc([pose], speed_m_s=0.02, law="tff", force_axes=[1, 0, 0, 0, 0, 0], block=0)
    phase_x = compile_request(ctx, arm.last_request, raw=raw)
    assert np.allclose(phase_x.outer.selection, [0, 1, 1, 1, 1, 1])


def test_cartesian_unreachable_is_code_1() -> None:
    raw, ctx = _ctx()
    del raw
    arm = _arm(ctx=ctx)
    far = [10.0, 10.0, 10.0, 0.0, 0.0, 0.0]
    assert arm.cartesian(far, v=0.2, block=0) == ERR_CONTROLLER


def test_mode_engine_samples_all_modes() -> None:
    raw, ctx = _ctx()
    eng = ModeEngine(ctx, raw=raw)
    pose = ctx.kin.fk_pose(_SEED)
    f_ext = np.zeros(6)
    cases = [
        (Mode.SERVO_TWIST, {"v_cmd": [0.01, 0, 0, 0, 0, 0]}),
        (Mode.SERVO_TWIST_HOLD, {"v_cmd": [0.0] * 6}),
        (Mode.TRACK_CARTESIAN, {"reference": "hold"}),
        (
            Mode.TRACK_HYBRID,
            {"reference": "hold", "use_tff_split": True, "desired_z": 0.0},
        ),
        (
            Mode.TRACK_HYBRID,
            {"reference": "pad", "v_cmd": [0.0] * 6, "desired_z": 0.0},
        ),
        (Mode.MOVEJ, {"q_target": _SEED.tolist(), "duration_s": 0.8}),
    ]
    from peirastic.core.modes import ModeRequest

    def _origin() -> None:
        if eng.phase is None or not hasattr(eng.phase.outer, "set_origin"):
            return
        try:
            eng.phase.outer.set_origin(pose, t_s=0.0)
        except TypeError:
            eng.phase.outer.set_origin(pose)

    for mode, payload in cases:
        eng.set_mode(ModeRequest(mode, payload))
        _origin()
        out = eng.sample(0.0, pose, f_ext, q_meas=_SEED)
        assert np.asarray(out, dtype=float).shape == (6,)
        assert np.all(np.isfinite(out))

    near = pose.copy()
    near[1] += 0.01
    try:
        eng.set_mode(ModeRequest(Mode.CARTESIAN_PTP, {"pose": near.tolist(), "v": 0.2, "q_start": _SEED.tolist()}))
        out = eng.sample(0.0, pose, f_ext, q_meas=_SEED)
        assert np.asarray(out, dtype=float).shape == (6,)
        assert np.all(np.isfinite(out))
    except ValueError:
        pytest.skip("seed pose offset is unreachable for CARTESIAN_PTP on this model")
    poses = [pose.tolist(), near.tolist()]
    try:
        eng.set_mode(ModeRequest(Mode.MOVES, {"poses": poses, "speed_m_s": 0.05}))
        out = eng.sample(0.0, pose, f_ext, q_meas=_SEED)
        assert np.asarray(out, dtype=float).shape == (6,)
        assert np.all(np.isfinite(out))
    except ValueError:
        pytest.skip("polyline unreachable for MOVES on this model")


def test_resolve_pose_q_prefers_ns_d_star() -> None:
    from peirastic.realman8dof.modes.cartesian import reachable_rails, resolve_pose_q

    raw, ctx = _ctx()
    del raw
    pose = ctx.kin.fk_pose(_SEED)
    rails = reachable_rails(ctx.kin, pose, euler_order=ctx.euler_order)
    if len(rails) < 2:
        pytest.skip("need two reachable rails to rank by d*")
    far = max(rails, key=lambda r: abs(float(r) - float(_SEED[0])))
    if abs(float(far) - float(_SEED[0])) < 0.04:
        pytest.skip("reachable rails are too close to test d* preference")
    d_star = float(pose[1]) - float(far)
    pr = ctx.inner.posture_retarget
    if pr is None:
        pytest.skip("posture_retarget disabled")
    pr.d_star_m = d_star
    pr.psi_star_rad = float(ctx.inner.cfg.psi_retarget.psi_attr_rad)
    qt = resolve_pose_q(ctx, pose, q_seed=_SEED, require_path=False)
    assert abs(float(qt[0]) - float(far)) < abs(float(qt[0]) - float(_SEED[0]))


def test_cartesian_plan_is_joint_ptp() -> None:
    raw, ctx = _ctx()
    pose = ctx.kin.fk_pose(_SEED)
    arm = _arm(ctx=ctx)
    assert arm.cartesian(pose.tolist(), v=0.2, block=0) == OK
    phase = compile_request(ctx, arm.last_request, raw=raw)
    from rm75_control.control.joint_admittance_8dof.loop import JointTrackOuterLoop
    from rm75_control.control.joint_admittance_8dof.reference import JointSmoothMoveReference

    assert isinstance(phase.outer, JointTrackOuterLoop)
    assert isinstance(phase.outer.reference, JointSmoothMoveReference)


def test_idle_after_finite_is_hold_unless_gamepad() -> None:
    from peirastic.core.session import (
        idle_after_finite,
        pad_source_present,
        stay_after_duration,
    )

    assert idle_after_finite() == Mode.SERVO_TWIST_HOLD
    assert idle_after_finite(pad_source=True) == Mode.SERVO_TWIST
    assert stay_after_duration(Mode.SERVO_TWIST) is True
    assert stay_after_duration(Mode.SERVO_TWIST_HOLD) is False
    assert stay_after_duration(Mode.TRACK_CARTESIAN) is False
    assert pad_source_present(0.0, now_s=10.0) is False
    assert pad_source_present(9.9, now_s=10.0) is True
    assert pad_source_present(9.0, now_s=10.0) is False
    assert pad_source_present(9.9, now_s=10.0, hz=float("nan")) is False
    assert pad_source_present(9.9, now_s=10.0, hz=125.0) is True
    assert pad_source_present(9.9, now_s=10.0, hz=125.0, connected=False) is False


def test_pad_yields_to_command_modes() -> None:
    from peirastic.core.session import pad_may_drive

    assert pad_may_drive(Mode.SERVO_TWIST) is True
    assert pad_may_drive(Mode.SERVO_TWIST, label="servo_twist") is True
    assert pad_may_drive(Mode.SERVO_TWIST_HOLD) is True
    assert pad_may_drive(Mode.SERVO_TWIST, label="cartesian_velocity") is False
    assert pad_may_drive(Mode.SERVO_TWIST, label="movev_canfd") is False
    assert pad_may_drive(Mode.TRACK_HYBRID, label="track_hybrid_pad") is True
    assert pad_may_drive(Mode.TRACK_HYBRID, label="vessel_close") is False
    assert pad_may_drive(Mode.TRACK_HYBRID, program=True, label="track_hybrid_pad") is False
    assert pad_may_drive(Mode.CARTESIAN_PTP) is False
    assert pad_may_drive(Mode.MOVEJ) is False
    assert pad_may_drive(Mode.TRACK_CARTESIAN) is False
    assert pad_may_drive(0) is False
    assert pad_may_drive(99) is False
    assert pad_may_drive(None) is False


def test_qp_aux_hits_inner() -> None:
    raw, ctx = _ctx()
    del raw
    arm = _arm(ctx=ctx, inner=ctx.inner)
    assert arm.set_collision_avoidance(False) == OK
    assert ctx.inner.core.collision_cfg.enabled is False
    assert arm.set_nullspace(centering=False, arm_angle=False, manipulability=False) == OK
    assert ctx.inner._centering_suppressed is True
    assert arm.set_singularity_escape(False) == OK
    assert ctx.inner.core.cfg.sigma_setbased.enabled is False


def test_cartesian_track_ellipse_is_swappable_vcmd() -> None:
    raw, ctx = _ctx()
    arm = _arm(ctx=ctx)
    assert arm.cartesian_track(
        reference="ellipse",
        amplitude_x_m=0.05,
        amplitude_y_m=0.15,
        rot_amp_deg=[10.0, 8.0, 15.0],
        max_vel_m_s=0.04,
        duration_s=8.0,
        block=0,
    ) == OK
    req = arm.last_request
    assert req is not None
    assert req.mode == Mode.TRACK_CARTESIAN
    assert req.payload["reference"] == "ellipse"
    phase = compile_request(ctx, req, raw=raw)
    from rm75_control.control.joint_admittance_8dof.loop import CartesianTrackOuterLoop
    from rm75_control.control.joint_admittance_8dof.reference import EllipseToolXYReference
    from peirastic.core.session import is_swappable

    assert isinstance(phase.outer, CartesianTrackOuterLoop)
    assert isinstance(phase.outer.reference, EllipseToolXYReference)
    assert np.allclose(
        phase.outer.reference.rot_amp_rad, np.deg2rad([10.0, 8.0, 15.0])
    )
    assert phase.duration_s == pytest.approx(8.5)
    assert is_swappable(Mode.TRACK_CARTESIAN)
    pose = ctx.kin.fk_pose(_SEED)
    phase.outer.set_origin(pose, t_s=0.0)
    twist = np.asarray(phase.outer.sample(0.0, pose, np.zeros(6)), dtype=float)
    assert twist.shape == (6,)
    assert np.all(np.isfinite(twist))
    assert float(phase.outer.last_err_mm) == pytest.approx(0.0, abs=1e-6)
    assert float(phase.outer.last_rot_deg) == pytest.approx(0.0, abs=1e-6)
    later = np.asarray(phase.outer.sample(1.2, pose, np.zeros(6)), dtype=float)
    assert np.all(np.isfinite(later))
    assert float(phase.outer.last_rot_deg) > 1.0


def test_pack_vel_filter_canonical() -> None:
    assert pack_vel_filter() is None
    assert pack_vel_filter(filter=False) is False
    assert pack_vel_filter(filter=True) is True
    assert pack_vel_filter(follow=True) is False
    assert pack_vel_filter(follow=False) is True
    assert pack_vel_filter(filter=True, follow=True) is True
    assert pack_vel_filter(slew=False) is False
    assert pack_vel_filter(filter=[1, 1, 0, 1, 1, 1]) == [1.0, 1.0, 0.0, 1.0, 1.0, 1.0]
    assert not np.any(resolve_filter_axes(default=False))
    assert np.array_equal(
        resolve_filter_axes(filter=True, mask=[1, 1, 0, 1, 1, 1]),
        [True, True, False, True, True, True],
    )


def test_cartesian_velocity_is_swappable_passthrough() -> None:
    raw, ctx = _ctx()
    arm = _arm(ctx=ctx)
    assert arm.cartesian_velocity([0.0, 0.02, 0.0, 0.0, 0.0, 0.0], duration_s=1.5, block=0) == OK
    req = arm.last_request
    assert req is not None
    assert req.mode == Mode.SERVO_TWIST
    assert req.payload["v_cmd"][1] == pytest.approx(0.02)
    assert req.payload.get("label") == "cartesian_velocity"
    assert "filter" not in req.payload
    assert "slew" not in req.payload
    assert "follow" not in req.payload
    phase = compile_request(ctx, req, raw=raw)
    from peirastic.core.session import is_swappable
    from peirastic.realman8dof.modes.servo import ServoTwistOuter

    assert str(phase.label) == "cartesian_velocity"
    assert isinstance(phase.outer, ServoTwistOuter)
    assert not np.any(phase.outer.filter_axes)
    assert is_swappable(Mode.SERVO_TWIST)
    pose = ctx.kin.fk_pose(_SEED)
    phase.outer.set_origin(pose, t_s=0.0)
    out0 = np.asarray(phase.outer.sample(0.0, pose, np.zeros(6)), dtype=float)
    assert out0[1] == pytest.approx(0.02)
    phase.outer.source = lambda: np.array([0.0, 0.20, 0.0, 0.0, 0.0, 0.0])
    out1 = np.asarray(phase.outer.sample(0.005, pose, np.zeros(6)), dtype=float)
    assert out1[1] == pytest.approx(0.20)


def test_cartesian_velocity_filter_and_follow() -> None:
    raw, ctx = _ctx()
    arm = _arm(ctx=ctx)
    pose = ctx.kin.fk_pose(_SEED)
    assert arm.cartesian_velocity([0.0, 0.02, 0.0, 0.0, 0.0, 0.0], filter=True, block=0) == OK
    assert arm.last_request.payload.get("filter") is True
    phase = compile_request(ctx, arm.last_request, raw=raw)
    assert np.all(phase.outer.filter_axes)
    phase.outer.set_origin(pose, t_s=0.0)
    out0 = np.asarray(phase.outer.sample(0.0, pose, np.zeros(6)), dtype=float)
    assert out0[1] == pytest.approx(0.02)
    phase.outer.source = lambda: np.array([0.0, 0.20, 0.0, 0.0, 0.0, 0.0])
    out1 = np.asarray(phase.outer.sample(0.005, pose, np.zeros(6)), dtype=float)
    assert 0.02 < float(out1[1]) < 0.20
    assert arm.cartesian_velocity([0.0, 0.02, 0.0, 0.0, 0.0, 0.0], follow=True, block=0) == OK
    assert arm.last_request.payload.get("filter") is False
    phase_f = compile_request(ctx, arm.last_request, raw=raw)
    assert not np.any(phase_f.outer.filter_axes)


def test_hfvc_force_axis_skips_filter() -> None:
    raw, ctx = _ctx()
    arm = _arm(ctx=ctx)
    pose = ctx.kin.fk_pose(_SEED)
    assert arm.hfvc([0.0, 0.02, 0.02, 0.0, 0.0, 0.0], source="twist") == OK
    phase = compile_request(ctx, arm.last_request, raw=raw)
    pos = phase.outer.position
    assert np.array_equal(pos.filter_axes, [True, True, False, True, True, True])
    pos.set_origin(pose, t_s=0.0)
    out0 = np.asarray(pos.sample(0.0, pose, np.zeros(6)), dtype=float)
    assert out0[1] == pytest.approx(0.02)
    assert out0[2] == pytest.approx(0.02)
    pos.source = lambda: np.array([0.0, 0.20, 0.20, 0.0, 0.0, 0.0])
    out1 = np.asarray(pos.sample(0.005, pose, np.zeros(6)), dtype=float)
    assert 0.02 < float(out1[1]) < 0.20
    assert out1[2] == pytest.approx(0.20)
    assert arm.hfvc(source="twist", filter=False) == OK
    phase_off = compile_request(ctx, arm.last_request, raw=raw)
    assert not np.any(phase_off.outer.position.filter_axes)


def test_track_and_servo_compile() -> None:
    raw, ctx = _ctx()
    arm = _arm(ctx=ctx)
    pose = ctx.kin.fk_pose(_SEED)
    assert arm.track_pose(pose, block=0, label="hold_test") == OK
    phase = compile_request(ctx, arm.last_request, raw=raw)
    assert "hold" in str(phase.label) or "track" in str(phase.label)
    assert arm.movev_canfd([0.01, 0, 0, 0, 0, 0]) == OK
    phase_v = compile_request(ctx, arm.last_request, raw=raw)
    out = np.asarray(phase_v.outer.sample(0.0, pose, np.zeros(6)), dtype=float)
    assert out[0] == pytest.approx(0.01)


def test_window_a_ready_and_handoff_are_quiet() -> None:
    from peirastic.realman8dof.daemon import is_handoff_stop, ready_state_msg

    assert ready_state_msg(rail_m=0.6204, tcp="gripper2") == (
        "ready  rail=620.4 mm  tcp=gripper2"
    )
    assert ready_state_msg(rail_m=None, tcp=None) == "ready"
    assert is_handoff_stop("external_stop_before_send", pending_commanded=True)
    assert not is_handoff_stop("external_stop_before_send", pending_commanded=False)
    assert not is_handoff_stop("feedback_stale: age=1", pending_commanded=True)


def test_quiet_vendor_banner_drops_c_api_version(capsys) -> None:
    from rm75_control.core.session import _quiet_vendor_stdout

    with _quiet_vendor_stdout():
        print("current c api version:  1.1.3")
        print("keep this")
    assert capsys.readouterr().out == "keep this\n"


def test_rail_progress_is_silent_by_default(capsys, monkeypatch) -> None:
    from rm75_control.control.joint_admittance_8dof.hw.rail_servo import _rail_progress

    monkeypatch.delenv("LW100_RAIL_VERBOSE", raising=False)
    _rail_progress("lw100 rail: connecting hold @ +0.6204 m")
    assert capsys.readouterr().out == ""
    monkeypatch.setenv("LW100_RAIL_VERBOSE", "1")
    _rail_progress("lw100 rail: connecting hold @ +0.6204 m")
    assert "connecting hold" in capsys.readouterr().out


def test_gamepad_link_event_skips_initial_missing() -> None:
    from peirastic.apps.gamepad import pad_link_event

    assert pad_link_event(None, False) is None
    assert pad_link_event(None, True) == "[PAD] bluetooth live"
    assert pad_link_event(False, False) is None
    assert pad_link_event(False, True) == "[PAD] bluetooth live"
    assert pad_link_event(True, False) == "[PAD] bluetooth lost"
    assert pad_link_event(True, True) is None
