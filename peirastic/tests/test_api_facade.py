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
    assert arm.movel(pose, v=0.2, r=0.01, block=0) == ERR_UNIMPLEMENTED
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


def test_movel_unreachable_is_code_1() -> None:
    raw, ctx = _ctx()
    del raw
    arm = _arm(ctx=ctx)
    far = [10.0, 10.0, 10.0, 0.0, 0.0, 0.0]
    assert arm.movel(far, v=0.2, block=0) == ERR_CONTROLLER


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
        eng.set_mode(ModeRequest(Mode.MOVEL, {"pose": near.tolist(), "v": 0.2, "q_start": _SEED.tolist()}))
        out = eng.sample(0.0, pose, f_ext, q_meas=_SEED)
        assert np.asarray(out, dtype=float).shape == (6,)
        assert np.all(np.isfinite(out))
    except ValueError:
        pytest.skip("seed pose offset is unreachable for MOVEL on this model")
    poses = [pose.tolist(), near.tolist()]
    try:
        eng.set_mode(ModeRequest(Mode.MOVES, {"poses": poses, "speed_m_s": 0.05}))
        out = eng.sample(0.0, pose, f_ext, q_meas=_SEED)
        assert np.asarray(out, dtype=float).shape == (6,)
        assert np.all(np.isfinite(out))
    except ValueError:
        pytest.skip("polyline unreachable for MOVES on this model")


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


def test_idle_after_finite_holds_tcp() -> None:
    from peirastic.core.session import idle_after_finite

    assert idle_after_finite() == Mode.SERVO_TWIST_HOLD


def test_pad_yields_to_command_modes() -> None:
    from peirastic.core.session import pad_may_drive

    assert pad_may_drive(Mode.SERVO_TWIST) is True
    assert pad_may_drive(Mode.SERVO_TWIST_HOLD) is True
    assert pad_may_drive(Mode.TRACK_HYBRID, label="track_hybrid_pad") is True
    assert pad_may_drive(Mode.TRACK_HYBRID, label="vessel_close") is False
    assert pad_may_drive(Mode.TRACK_HYBRID, program=True, label="track_hybrid_pad") is False
    assert pad_may_drive(Mode.MOVEL) is False
    assert pad_may_drive(Mode.MOVEJ) is False
    assert pad_may_drive(Mode.TRACK_CARTESIAN) is False


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
