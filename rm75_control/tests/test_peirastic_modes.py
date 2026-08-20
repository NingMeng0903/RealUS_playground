"""Peirastic generic modes, TFF, pad source, IPC. Does not steal live A SHM."""

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

from rm75_control.control.joint_admittance_8dof.config import build_joint_ik_config
from rm75_control.control.joint_admittance_8dof.loop import CartesianTrackOuterLoop
from rm75_control.control.joint_admittance_8dof.model import RobotKinematics
from rm75_control.control.joint_admittance_8dof.reference import EllipseToolXYReference
from rm75_control.control.joint_admittance_8dof.teleop.gamepad_twist import GamepadTwistConfig
from rm75_control.control.joint_admittance_8dof.teleop.xbox_pad import FakePad
from peirastic.core.ipc import CommandHub, Cmd, Status, TwistBus
from peirastic.core.modes import Mode as ModeE, ModeRequest
from peirastic.realman8dof.force.tff import SELECTION_TOOL_Z_FORCE, compose_tff
from peirastic.realman8dof.modes.servo import ServoTwistHoldOuter, ServoTwistOuter
from peirastic.realman8dof.session import ModeEngine, compile_request
from peirastic.sources.gamepad import LOGICAL_L3, LOGICAL_R3, GamepadTwistSource
from rm75_control.control.joint_admittance_8dof.api import CompileContext
from rm75_control.control.joint_admittance_8dof.loop import JointIkController

from peirastic.configs import DEFAULT_CONTROLLER_YAML

_CFG = DEFAULT_CONTROLLER_YAML
_SEED = np.array([0.375, 0.194, -0.503, -0.069, 1.979, -0.776, 0.547, -4.370])


def _ctx():
    raw = yaml.safe_load(_CFG.read_text())
    cfg = build_joint_ik_config(raw)
    cfg.backend = "python"
    cfg.native_shm_prefix = f"rm75_wbc_peir_{os.getpid()}"
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


def test_tff_gives_force_axis_to_force_law() -> None:
    v_pos = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0])
    v_force = np.array([9.0, 9.0, -0.04, 0.0, 0.0, 0.0])
    out = compose_tff(v_pos, v_force, SELECTION_TOOL_Z_FORCE)
    assert out[0] == pytest.approx(0.1)
    assert out[1] == pytest.approx(0.2)
    assert out[2] == pytest.approx(-0.04)


def test_servo_twist_is_pure_passthrough() -> None:
    v = np.array([0.02, -0.01, 0.0, 0.0, 0.1, 0.0])
    outer = ServoTwistOuter(v)
    pose = np.array([0.4, 0.2, 0.3, 0.0, 0.0, 0.0])
    got = outer.sample(0.0, pose, np.zeros(6))
    assert np.allclose(got, v)
    assert outer.last_err_mm == 0.0


def test_servo_twist_hold_latches_pose() -> None:
    live = np.array([0.03, 0.0, 0.0, 0.0, 0.0, 0.0])
    box = {"v": live.copy()}

    def src():
        return box["v"].copy()

    outer = ServoTwistHoldOuter(src, dt=0.005)
    pose = np.array([0.4, 0.2, 0.3, 0.0, 0.0, 0.0])
    outer.set_origin(pose)
    moving = outer.sample(0.0, pose, np.zeros(6))
    assert float(np.linalg.norm(moving[:3])) > 0.01
    box["v"][:] = 0.0
    coast = pose.copy()
    for i in range(4):
        coast = coast.copy()
        outer.sample(0.005 * i, coast, np.zeros(6))
    hold_pose = pose + np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0])
    held = outer.sample(0.1, hold_pose, np.zeros(6))
    assert float(np.linalg.norm(held[:3])) > 0.0
    assert outer.last_pose_d is not None


def test_gamepad_source_is_not_a_mode() -> None:
    pad = FakePad()
    pad.axes[0] = -1.0
    src = GamepadTwistSource(pad=pad, cfg=GamepadTwistConfig(dt=0.005))
    src._tick()
    snap = src.snapshot()
    assert snap["twist"].shape == (6,)
    assert float(np.linalg.norm(snap["twist"][:3])) > 0.0
    assert LOGICAL_L3 == 6
    assert LOGICAL_R3 == 7
    assert "layout" in snap
    assert "armed" in snap


def test_compile_servo_and_ellipse() -> None:
    raw, ctx = _ctx()
    phase = compile_request(ctx, ModeRequest(ModeE.SERVO_TWIST, {"v_cmd": [0.01, 0, 0, 0, 0, 0]}))
    pose = ctx.kin.fk_pose(_SEED)
    v = phase.outer.sample(0.0, pose, np.zeros(6))
    assert v[0] == pytest.approx(0.01)
    ell = compile_request(
        ctx,
        ModeRequest(
            ModeE.TRACK_CARTESIAN,
            {"reference": "ellipse", "x_pp_cm": 10.0, "y_pp_cm": 30.0, "max_vel_cm_s": 4.0},
        ),
        raw=raw,
    )
    ell.outer.set_origin(pose)
    v2 = ell.outer.sample(0.5, pose, np.zeros(6))
    assert np.all(np.isfinite(v2))


def test_track_cartesian_matches_library_outer() -> None:
    raw, ctx = _ctx()
    pose0 = ctx.kin.fk_pose(_SEED)
    ref_a = EllipseToolXYReference(0.05, 0.15, max_vel_m_s=0.04, euler_order=ctx.euler_order)
    phase = compile_request(
        ctx,
        ModeRequest(
            ModeE.TRACK_CARTESIAN,
            {
                "reference": "ellipse",
                "amplitude_x_m": 0.05,
                "amplitude_y_m": 0.15,
                "max_vel_m_s": 0.04,
            },
        ),
        raw=raw,
    )
    lib = CartesianTrackOuterLoop(ref_a, phase.outer.cfg)
    lib.set_origin(pose0, t_s=0.0)
    phase.outer.set_origin(pose0, t_s=0.0)
    for t in (0.0, 0.2, 0.5, 1.0):
        a = np.asarray(lib.sample(t, pose0, np.zeros(6)), dtype=float)
        b = np.asarray(phase.outer.sample(t, pose0, np.zeros(6)), dtype=float)
        assert np.allclose(a, b, atol=1e-9)


def test_ipc_unique_prefix_roundtrip() -> None:
    prefix = f"peir_test_{os.getpid()}_"
    hub = CommandHub(prefix=prefix)
    twist = TwistBus(prefix=prefix, create=True)
    try:
        from peirastic.core.ipc import CommandClient

        client = CommandClient(prefix=prefix)
        seq = client.set_mode(ModeRequest(ModeE.SERVO_TWIST, {"v_cmd": [0.0] * 6}))
        polled = hub.poll()
        assert polled is not None
        cmd, got_seq, req = polled
        assert cmd == Cmd.SET_MODE
        assert got_seq == seq
        assert req is not None and req.mode == ModeE.SERVO_TWIST
        hub.ack(seq)
        hub.publish(status=Status.RUNNING, mode=ModeE.SERVO_TWIST, ticks=3)
        snap = client.snapshot()
        assert snap["ticks"] == 3
        twist.write(np.array([0.01, 0, 0, 0, 0, 0]), hz=125.0, r3=False)
        assert twist.read()["twist"][0] == pytest.approx(0.01)
        client.close()
    finally:
        twist.close()
        hub.close()


def test_velocity_modes_swap_in_place_joint_rebuilds() -> None:
    from peirastic.core.session import is_swappable

    assert is_swappable(ModeE.SERVO_TWIST)
    assert is_swappable(ModeE.SERVO_TWIST_HOLD)
    assert is_swappable(ModeE.TRACK_CARTESIAN)
    assert is_swappable(ModeE.TRACK_HYBRID)
    assert not is_swappable(ModeE.GOTO_JOINTS)
    assert not is_swappable(ModeE.MOVEJ)


def test_mode_engine_offline_sample() -> None:
    raw, ctx = _ctx()
    eng = ModeEngine(ctx, raw=raw)
    eng.set_mode(ModeRequest(ModeE.SERVO_TWIST, {"v_cmd": [0.0, 0.02, 0, 0, 0, 0]}))
    pose = ctx.kin.fk_pose(_SEED)
    v = eng.sample(0.0, pose, np.zeros(6), q_meas=_SEED)
    assert v[1] == pytest.approx(0.02)


def test_goto_and_movej_enable_direct_ptp() -> None:
    raw, ctx = _ctx()
    q_t = _SEED.copy()
    q_t[3] += 0.15
    for mode, label in (
        (ModeE.GOTO_JOINTS, "goto_joints"),
        (ModeE.MOVEJ, "movej"),
    ):
        phase = compile_request(
            ctx,
            ModeRequest(mode, {"q_target": q_t.tolist(), "duration_s": 1.2}),
            raw=raw,
        )
        assert phase.label == label
        assert phase.qdot_ff_provider is not None
        ctx.inner.set_direct_joint_ptp(False)
        phase.on_enter()
        assert ctx.inner._direct_joint_ptp
        phase.on_exit()
        assert not ctx.inner._direct_joint_ptp


def test_hybrid_tff_and_legacy_force_law() -> None:
    from peirastic.realman8dof.force.legacy import LegacyForceLaw
    from peirastic.realman8dof.force.protocol import ForceOutput
    from peirastic.realman8dof.modes.track import HybridTffOuter

    raw, ctx = _ctx()
    pose = ctx.kin.fk_pose(_SEED)
    legacy = compile_request(
        ctx,
        ModeRequest(ModeE.TRACK_HYBRID, {"reference": "hold", "desired_z": 0.0}),
        raw=raw,
    )
    legacy.outer.set_origin(pose, t_s=0.0)
    v_legacy = np.asarray(legacy.outer.sample(0.0, pose, np.zeros(6)), dtype=float)
    assert v_legacy.shape == (6,)
    assert np.all(np.isfinite(v_legacy))

    class _Law:
        def reset(self, *, pose, f_ext) -> None:
            del pose, f_ext

        def update(self, **kwargs) -> ForceOutput:
            del kwargs
            return ForceOutput(
                v_force=np.array([0.0, 0.0, -0.04, 0.0, 0.0, 0.0]),
                v_force_z=-0.04,
            )

    tff = compile_request(
        ctx,
        ModeRequest(
            ModeE.TRACK_HYBRID,
            {"reference": "hold", "desired_z": 1.0, "use_tff_split": True},
        ),
        raw=raw,
    )
    assert isinstance(tff.outer, HybridTffOuter)
    tff.outer.force_law = _Law()
    tff.outer.set_origin(pose, t_s=0.0)
    out = np.asarray(tff.outer.sample(0.1, pose, np.zeros(6)), dtype=float)
    assert out[2] == pytest.approx(-0.04)

    from rm75_control.control.admittance_common.controller import AdmittanceController
    from rm75_control.control.admittance_common.scaling import scale_admittance_for_desired_z

    ctrl = AdmittanceController(0.005, scale_admittance_for_desired_z(raw, 0.0))
    law = LegacyForceLaw(ctrl)
    law.reset(pose=pose, f_ext=np.zeros(6))
    fout = law.update(
        dt_s=0.005,
        pose=pose,
        f_ext=np.zeros(6),
        f_des=np.zeros(6),
        path_twist=np.zeros(6),
    )
    assert fout.v_force.shape == (6,)
    assert np.isfinite(fout.v_force_z)


def test_panel_ansi_refresh_and_event_tags() -> None:
    from peirastic.core.panel import Panel

    panel = Panel(enabled=False)
    panel.event("MODE", "SERVO_TWIST")
    panel.event("ESTOP", "pad R3")
    panel.update(mode="SERVO_TWIST", ticks=12, pad_hz=125.0, estop=True)
    joined = "\n".join(panel._events)
    assert "[MODE]" in joined
    assert "[ESTOP]" in joined
    assert "SERVO_TWIST" in joined


def test_daemon_dry_run_does_not_touch_live_shm() -> None:
    from peirastic.realman8dof.daemon import run_service

    assert run_service(_CFG, dry_run=True, panel=False) == 0


def test_resolve_log_csv_auto_writes_local_stamp(tmp_path) -> None:
    import time

    from peirastic.realman8dof.daemon import resolve_log_csv

    assert resolve_log_csv(None) is None
    now = 1_777_000_000.0
    path = resolve_log_csv("auto", now=now, log_dir=tmp_path)
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(now))
    assert path == str(tmp_path / f"run_{stamp}.csv")
    assert resolve_log_csv("/tmp/explicit.csv") == "/tmp/explicit.csv"


def test_hybrid_defaults_desired_z_from_force_yaml() -> None:
    from peirastic.realman8dof.force.config import desired_z_n, load_force_raw

    raw, ctx = _ctx()
    phase = compile_request(
        ctx, ModeRequest(ModeE.TRACK_HYBRID, {"reference": "hold"}), raw=raw
    )
    assert float(phase.outer.desired_force[2]) == pytest.approx(desired_z_n())
    force = load_force_raw()
    assert float(force["hybrid_motion"]["max_vz_tool_m_s"]) == pytest.approx(0.08)
    assert float(force["hybrid_motion"]["system_delay_s"]) == pytest.approx(0.055)
    assert float(force["hybrid_motion"]["force_barrier"]["v_seek_free_m_s"]) == pytest.approx(
        0.030
    )
    assert float(force["hybrid_motion"]["force_barrier"]["v_min_press_m_s"]) == pytest.approx(
        0.0
    )
    assert float(force["hybrid_motion"]["force_scale_fraction"]) == pytest.approx(0.0)
    assert force["hybrid_motion"]["cdyob"]["enabled"] is False


def test_pad_hybrid_keeps_pad_axes_force_owns_z() -> None:
    from peirastic.realman8dof.force.protocol import ForceOutput
    from peirastic.realman8dof.modes.track import HybridTffOuter

    raw, ctx = _ctx()
    v_cmd = np.array([0.02, -0.01, 0.05, 0.0, 0.0, 0.1])
    phase = compile_request(
        ctx,
        ModeRequest(ModeE.TRACK_HYBRID, {"reference": "pad", "desired_z": 1.0}),
        raw=raw,
        twist_read=lambda: v_cmd,
    )
    assert isinstance(phase.outer, HybridTffOuter)
    assert float(phase.outer.desired_force[2]) == pytest.approx(1.0)

    class _Law:
        def reset(self, **kwargs) -> None:
            del kwargs

        def update(self, **kwargs) -> ForceOutput:
            path = np.asarray(kwargs["path_twist"], dtype=float)
            assert abs(float(path[2])) < 1e-12
            return ForceOutput(
                v_force=np.array([0.0, 0.0, -0.04, 0.0, 0.0, 0.0]),
                v_force_z=-0.04,
            )

    phase.outer.force_law = _Law()
    pose = ctx.kin.fk_pose(_SEED)
    phase.outer.set_origin(pose)
    out = np.asarray(phase.outer.sample(0.0, pose, np.zeros(6)), dtype=float)
    assert out[0] == pytest.approx(0.02)
    assert out[1] == pytest.approx(-0.01)
    assert out[2] == pytest.approx(-0.04)
    assert out[5] == pytest.approx(0.1)


def test_force_yaml_payload_overrides_reload() -> None:
    from peirastic.realman8dof.force.config import build_force_controller, desired_z_n

    ctrl, raw, fz = build_force_controller(
        0.005, payload={"desired_z": 2.5, "max_vz_tool_m_s": 0.04, "v_seek_free_m_s": 0.01}
    )
    assert fz == pytest.approx(2.5)
    assert desired_z_n(raw) == pytest.approx(2.5)
    assert ctrl.cfg.max_vz_tool_m_s == pytest.approx(0.04)
    assert ctrl.cfg.max_velocity[2] == pytest.approx(0.04)
    assert ctrl.cfg.force_barrier.v_seek_free_m_s == pytest.approx(0.01)


def test_gamepad_l3_r3_edges() -> None:
    pad = FakePad()
    src = GamepadTwistSource(pad=pad, cfg=GamepadTwistConfig(dt=0.005))
    src._tick()
    assert not src.snapshot()["l3_edge"]
    pad.buttons[LOGICAL_L3] = 1.0
    src._tick()
    snap = src.snapshot()
    assert snap["l3"]
    assert snap["l3_edge"]
    pad.buttons[LOGICAL_R3] = 1.0
    src._tick()
    snap = src.snapshot()
    assert snap["r3_edge"]
