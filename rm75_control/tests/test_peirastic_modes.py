"""Peirastic generic modes, TFF, pad source, IPC. Does not steal live A SHM."""

from __future__ import annotations

import os
import sys
import time
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
from peirastic.sources.gamepad import (
    LOGICAL_B,
    LOGICAL_L3,
    LOGICAL_R3,
    LOGICAL_Y,
    GamepadTwistSource,
)
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


def test_servo_twist_refreshes_pose_d_from_live_tcp() -> None:
    v = np.array([0.0, 0.10, 0.0, 0.0, 0.0, 0.0])
    outer = ServoTwistOuter(v)
    origin = np.array([0.4, 0.18, 0.3, 0.0, 0.0, 0.0])
    live = origin + np.array([0.0, 0.40, 0.0, 0.0, 0.0, 0.0])
    outer.set_origin(origin)
    outer.sample(0.0, live, np.zeros(6))
    assert outer.last_pose_d is not None
    assert outer.last_pose_d[1] == pytest.approx(live[1])
    assert abs(float(outer.last_pose_d[1] - origin[1])) > 0.30


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
    assert LOGICAL_Y == 3
    assert LOGICAL_B == 1
    assert snap["y"] is False
    assert snap["y_edge"] is False
    assert snap["b"] is False
    assert snap["b_edge"] is False
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
        from peirastic.core.ipc import MotionBus

        reader = MotionBus(prefix=prefix, create=False)
        try:
            hub.motion.publish(
                v_tcp_z=0.02,
                a_tcp_z_plus=0.4,
                feedback_age_s=0.005,
                t_wall_s=1.0,
                valid=True,
            )
            row, why = reader.fresh(-1, max_age_s=0.020)
            assert why == ""
            assert row is not None
            assert row["v_tcp_z"] == pytest.approx(0.02)
            assert int(row["seq"]) % 2 == 0
            assert row["age_total_s"] <= 0.020
            stale, reason = reader.fresh(int(row["seq"]), max_age_s=0.020)
            assert stale is None
            assert reason == "seq_stale"
            hub.motion.publish(
                v_tcp_z=0.01,
                a_tcp_z_plus=0.0,
                feedback_age_s=0.014,
                t_wall_s=1.0,
                valid=True,
            )
            time.sleep(0.003)
            aged, age_why = reader.fresh(-1, max_age_s=0.015)
            assert aged is None
            assert age_why == "age_total"
            hub.motion._row[0]["seq"] = 3
            torn, torn_why = reader.fresh(-1, max_age_s=0.050)
            assert torn is None
            assert torn_why in ("torn", "empty")
        finally:
            reader.close()
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
    assert not is_swappable(ModeE.MOVEL)
    assert not is_swappable(ModeE.MOVES)


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
        0.020
    )
    assert float(force["hybrid_motion"]["press_envelope"]["first_touch_m_s"]) == pytest.approx(
        0.0
    )
    assert force["hybrid_motion"]["physical_contact"]["hold_until_reset"] is False
    assert force["hybrid_motion"]["safety_shield"]["mode"] == "observe"
    assert force["hybrid_motion"]["safety_shield"]["terminal_invariance_proven"] is False
    assert force["hybrid_motion"]["safety_shield"]["energy_sign_verified"] is False
    assert float(force["hybrid_motion"]["force_barrier"]["v_min_press_m_s"]) == pytest.approx(
        0.0
    )
    assert float(force["hybrid_motion"]["force_scale_fraction"]) == pytest.approx(0.12)
    assert force["hybrid_motion"]["cdyob"]["mode"] == "off"
    assert float(force["hybrid_motion"]["cdyob"]["t0_s"]) == pytest.approx(0.030)
    assert float(force["hybrid_motion"]["cdyob"]["tp_s"]) == pytest.approx(0.012)
    assert float(force["hybrid_motion"]["cdyob"]["omega_q_hz"]) == pytest.approx(
        0.75
    )
    assert float(
        force["hybrid_motion"]["cdyob"]["v_corr_max_m_s"]
    ) == pytest.approx(0.015)
    assert float(
        force["hybrid_motion"]["cdyob"]["active_press_max_m_s"]
    ) == pytest.approx(0.010)
    assert float(
        force["hybrid_motion"]["cdyob"]["active_retract_max_m_s"]
    ) == pytest.approx(0.015)
    assert force["hybrid_motion"]["cdyob"]["active_model_validated"] is False
    assert float(
        force["hybrid_motion"]["cdyob"]["active_settle_speed_m_s"]
    ) == pytest.approx(0.010)
    assert float(
        force["hybrid_motion"]["cdyob"]["active_settle_hold_s"]
    ) == pytest.approx(0.05)
    assert force["hybrid_motion"]["force_dob"]["enabled"] is True
    assert float(force["hybrid_motion"]["force_dob"]["ki"]) == pytest.approx(8.0)
    assert force["hybrid_motion"]["proactive_feedforward"] is True
    assert float(
        force["hybrid_motion"]["force_barrier"]["v_underforce_press_m_s"]
    ) == pytest.approx(0.010)


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


def test_gamepad_y_edge_without_motion() -> None:
    pad = FakePad()
    pad.transport = "usb"
    pad.link_transport = "usb"
    src = GamepadTwistSource(pad=pad, cfg=GamepadTwistConfig(dt=0.005))
    src._tick()
    assert src.snapshot()["y_edge"] is False
    pad.buttons[LOGICAL_Y] = 1.0
    src._tick()
    snap = src.snapshot()
    assert snap["connected"] is False
    assert snap["y"] is True
    assert snap["y_edge"] is True
    src._tick()
    assert src.snapshot()["y_edge"] is False


def test_gamepad_b_edge_without_motion() -> None:
    pad = FakePad()
    pad.transport = "usb"
    pad.link_transport = "usb"
    src = GamepadTwistSource(pad=pad, cfg=GamepadTwistConfig(dt=0.005))
    src._tick()
    assert src.snapshot()["b_edge"] is False
    pad.buttons[LOGICAL_B] = 1.0
    src._tick()
    snap = src.snapshot()
    assert snap["connected"] is False
    assert snap["b"] is True
    assert snap["b_edge"] is True
    src._tick()
    assert src.snapshot()["b_edge"] is False


def test_gamepad_usb_or_missing_cannot_command_motion() -> None:
    cfg = GamepadTwistConfig(dt=0.005)
    usb = FakePad(axes=np.array([-1.0, 0.0, -1.0, 0.0, 0.0, -1.0]))
    usb.transport = "usb"
    usb.link_transport = "usb"
    src = GamepadTwistSource(pad=usb, cfg=cfg)
    src._tick()
    snap = src.snapshot()
    assert snap["armed"] is False
    assert snap["connected"] is False
    assert snap["transport"] == "usb"
    assert np.allclose(snap["twist"], 0.0)
    usb.buttons[LOGICAL_L3] = 1.0
    usb.buttons[LOGICAL_R3] = 1.0
    src._tick()
    snap = src.snapshot()
    assert snap["l3"] is False
    assert snap["r3"] is False
    assert snap["l3_edge"] is False
    assert snap["r3_edge"] is False

    missing = FakePad(axes=np.array([-1.0, 0.0, -1.0, 0.0, 0.0, -1.0]))
    missing.transport = "none"
    missing.link_transport = "none"
    src2 = GamepadTwistSource(pad=missing, cfg=cfg)
    src2._tick()
    snap2 = src2.snapshot()
    assert snap2["connected"] is False
    assert np.allclose(snap2["twist"], 0.0)


def test_gamepad_bluetooth_live_arms_after_settle() -> None:
    pad = FakePad(axes=np.array([-1.0, 0.0, -1.0, 0.0, 0.0, -1.0]))
    pad.transport = "bluetooth"
    pad.link_transport = "bluetooth"
    src = GamepadTwistSource(pad=pad, cfg=GamepadTwistConfig(dt=0.005))
    src._tick()
    snap = src.snapshot()
    assert snap["connected"] is True
    assert snap["armed"] is False
    assert np.allclose(snap["twist"], 0.0)
    src._live_since_s = time.monotonic() - 1.0
    src._tick()
    snap = src.snapshot()
    assert snap["armed"] is True
    assert float(np.linalg.norm(snap["twist"][:3])) > 0.0


def test_world_polyline_stays_on_line_and_covers_10cm() -> None:
    from rm75_control.control.joint_admittance_8dof.reference import WorldPolylineReference

    pts = np.linspace([0.40, 0.10, 0.20], [0.40, 0.20, 0.20], 21)
    ref = WorldPolylineReference(pts, speed_m_s=0.05, soft_start=False, euler_order="xyz")
    ref.set_origin(np.zeros(6), t_s=0.0)
    samples = np.array([ref.sample(t).pose_d[:3] for t in np.linspace(0.0, 2.0, 41)])
    assert samples[0, 1] == pytest.approx(0.10, abs=1e-9)
    assert samples[-1, 1] == pytest.approx(0.20, abs=1e-9)
    assert float(np.linalg.norm(samples[-1] - samples[0])) == pytest.approx(0.10, abs=1e-6)
    assert np.allclose(samples[:, 0], 0.40, atol=1e-9)
    assert np.allclose(samples[:, 2], 0.20, atol=1e-9)
    assert ref.length_m == pytest.approx(0.10, abs=1e-9)


def test_compile_polyline_hybrid_is_path_not_pad() -> None:
    from peirastic.realman8dof.modes.track import HybridTffOuter

    raw, ctx = _ctx()
    pose = ctx.kin.fk_pose(_SEED)
    pts = np.linspace(pose[:3], pose[:3] + np.array([0.0, 0.10, 0.0]), 11)
    rpy = np.repeat(pose[3:6].reshape(1, 3), 11, axis=0)
    phase = compile_request(
        ctx,
        ModeRequest(
            ModeE.TRACK_HYBRID,
            {
                "reference": "polyline",
                "use_tff_split": True,
                "points": pts.tolist(),
                "rpy": rpy.tolist(),
                "speed_m_s": 0.02,
                "desired_z": 0.0,
            },
        ),
        raw=raw,
    )
    assert isinstance(phase.outer, HybridTffOuter)
    phase.outer.set_origin(pose, t_s=0.0)
    out = np.asarray(phase.outer.sample(0.2, pose, np.zeros(6)), dtype=float)
    assert out.shape == (6,)
    assert np.all(np.isfinite(out))


def test_vessel_b_ignored_without_plan(tmp_path, monkeypatch) -> None:
    from peirastic.apps.vessel_scan import vessel_b_refuse_reason

    out = tmp_path / "smplx_outputs"
    out.mkdir()
    monkeypatch.setenv("REALUS_SMPLX_OUTPUT_ROOT", str(out))
    assert vessel_b_refuse_reason(repo=tmp_path) == "no capture"


def test_approach_cartesian_payload_does_not_pick_rail() -> None:
    """B approach is a TCP hold. 8DOF QPIK owns the rail; no q_target."""
    from peirastic.apps.vessel_scan import approach_cartesian_payload, standoff_pose_from_contact

    start = np.array([0.40, 0.20, 0.35, 0.1, -0.2, 0.3], dtype=float)
    contact = np.array([0.40, 0.05, 0.30, 0.07, 0.19, 2.23], dtype=float)
    standoff = standoff_pose_from_contact(contact, approach_dz_m=0.05)
    payload = approach_cartesian_payload(start, standoff)
    assert payload["reference"] == "polyline"
    assert payload["label"] == "vessel_approach"
    assert "q_target" not in payload
    assert "y_rail_target" not in payload
    poses = np.asarray(payload["poses"], dtype=float).reshape(-1, 6)
    assert poses.shape[0] == 1
    assert np.allclose(poses[0, :3], standoff[:3])
    assert np.allclose(poses[0, 3:6], start[3:6])
    assert payload["duration_s"] is None


def test_close_and_scan_keep_live_rpy() -> None:
    from peirastic.apps.vessel_scan import poses_keep_rpy

    contact = np.array([0.40, 0.20, 0.30, 0.07, 0.19, 2.23], dtype=float)
    scan = np.array(
        [
            [0.40, 0.20, 0.30, 0.07, 0.19, 2.23],
            [0.40, 0.25, 0.30, 0.08, 0.20, 2.40],
        ],
        dtype=float,
    )
    live_rpy = np.array([3.14, 0.0, -3.14], dtype=float)
    close = poses_keep_rpy(contact, live_rpy)
    held = poses_keep_rpy(scan, live_rpy)
    assert np.allclose(close[0, :3], contact[:3])
    assert np.allclose(close[0, 3:6], live_rpy)
    assert np.allclose(held[:, :3], scan[:, :3])
    assert np.allclose(held[:, 3:6], live_rpy)


def test_wait_cartesian_arrival_uses_live_fk_not_path_err() -> None:
    from peirastic.apps.vessel_scan import wait_cartesian_arrival
    from peirastic.core.ipc import Status

    goal = np.array([0.40, 0.20, 0.30], dtype=float)
    poses = [
        np.array([0.10, 0.20, 0.55], dtype=float),
        np.array([0.25, 0.20, 0.42], dtype=float),
        np.array([0.401, 0.199, 0.301], dtype=float),
    ]
    stale = {
        "status": int(Status.RUNNING),
        "mode": int(ModeE.TRACK_CARTESIAN),
        "msg": "vessel_approach",
        "track_err_mm": 0.0,
    }

    class _Client:
        def snapshot(self):
            return dict(stale)

    got = wait_cartesian_arrival(
        _Client(),
        goal_xyz=goal,
        pose_fn=lambda: poses.pop(0) if poses else goal,
        arrive_mm=15.0,
        timeout_s=1.0,
    )
    assert float(got["goal_err_mm"]) <= 15.0


def test_wait_cartesian_arrival_ignores_zero_path_err_while_far() -> None:
    from peirastic.apps.vessel_scan import wait_cartesian_arrival
    from peirastic.core.ipc import Status

    calls = {"n": 0}

    class _Client:
        def snapshot(self):
            calls["n"] += 1
            return {
                "status": int(Status.RUNNING),
                "mode": int(ModeE.TRACK_CARTESIAN),
                "msg": "vessel_approach",
                "track_err_mm": 0.0,
            }

    with pytest.raises(TimeoutError):
        wait_cartesian_arrival(
            _Client(),
            goal_xyz=np.array([0.40, 0.20, 0.30], dtype=float),
            pose_fn=lambda: np.array([0.10, 0.20, 0.55], dtype=float),
            arrive_mm=15.0,
            timeout_s=0.2,
        )
    assert calls["n"] >= 2


def test_wait_status_closed_client_is_interrupted() -> None:
    from peirastic.apps.vessel_scan import _wait_status

    class _Client:
        def snapshot(self):
            raise TypeError("'NoneType' object is not subscriptable")

    with pytest.raises(RuntimeError, match="interrupted"):
        _wait_status(_Client(), want=set(), timeout_s=0.2)


def test_wait_contact_requires_air_then_fz() -> None:
    from peirastic.apps.vessel_scan import wait_contact
    from peirastic.core.ipc import Status

    ticks = [
        {"status": int(Status.RUNNING), "mode": int(ModeE.TRACK_CARTESIAN), "msg": "vessel_approach", "f_ext_z": 1.2},
        {"status": int(Status.RUNNING), "mode": int(ModeE.TRACK_HYBRID), "msg": "vessel_close", "f_ext_z": 1.2},
        {"status": int(Status.RUNNING), "mode": int(ModeE.TRACK_HYBRID), "msg": "vessel_close", "f_ext_z": 0.1},
        {"status": int(Status.RUNNING), "mode": int(ModeE.TRACK_HYBRID), "msg": "vessel_close", "f_ext_z": 1.1},
        {"status": int(Status.RUNNING), "mode": int(ModeE.TRACK_HYBRID), "msg": "vessel_close", "f_ext_z": 1.1},
    ]

    class _Client:
        def snapshot(self):
            return ticks.pop(0) if ticks else {
                "status": int(Status.RUNNING),
                "mode": int(ModeE.TRACK_HYBRID),
                "msg": "vessel_close",
                "f_ext_z": 1.1,
            }

    assert wait_contact(_Client(), enter_n=0.85, confirm_s=0.02, timeout_s=1.0) is True


def test_compile_approach_is_coupled_qpik_not_movej() -> None:
    from peirastic.apps.vessel_scan import approach_cartesian_payload, standoff_pose_from_contact
    from rm75_control.control.joint_admittance_8dof.tasks.rail_mode import RailMode

    raw, ctx = _ctx()
    start = ctx.kin.fk_pose(_SEED)
    contact = start.copy()
    contact[:3] = start[:3] + np.array([0.0, -0.10, -0.05])
    payload = approach_cartesian_payload(
        start, standoff_pose_from_contact(contact, approach_dz_m=0.05)
    )
    phase = compile_request(ctx, ModeRequest(ModeE.TRACK_CARTESIAN, payload), raw=raw)
    assert "approach" in str(phase.label)
    assert "q_target" not in payload
    ctx.inner.set_locked()
    phase.on_enter()
    assert ctx.inner.rail_mode == RailMode.COUPLED
    assert not ctx.inner.is_locked_hold
    phase.outer.set_origin(start, t_s=0.0)
    out = np.asarray(phase.outer.sample(0.1, start, np.zeros(6)), dtype=float)
    assert out.shape == (6,)
    assert np.all(np.isfinite(out))
    assert float(np.linalg.norm(out[:3])) > 1.0e-4
    assert float(np.linalg.norm(out[3:6])) < 0.05


def test_controller_T_is_stage2_T_world_railbase() -> None:
    from peirastic.apps.vessel_scan import (
        controller_T_world_from_rail_base,
        robot_world_yaml_path,
    )

    raw = yaml.safe_load(robot_world_yaml_path().read_text(encoding="utf-8"))
    T_yaml = np.asarray(raw["T_world_railbase"], dtype=float).reshape(4, 4)
    T = controller_T_world_from_rail_base()
    assert np.allclose(T, T_yaml, atol=1e-12)
    kin = RobotKinematics()
    M = kin.frame_placement(np.zeros(kin.nq), "base_link")
    p_w = T[:3, :3] @ np.asarray(M.translation, dtype=float) + T[:3, 3]
    p_yaml = np.asarray(raw["T_world_baselink_at_rail0"], dtype=float).reshape(4, 4)[:3, 3]
    assert np.allclose(p_w, p_yaml, atol=1e-6)


def test_vessel_world_pose_is_rotated_into_rail_base() -> None:
    from peirastic.apps.vessel_scan import (
        controller_T_world_from_rail_base,
        poses_world_to_rail_base,
    )

    T = controller_T_world_from_rail_base()
    contact_w = np.array([-0.15094593, 0.07237269, 0.46663934, 0.0, 0.0, 0.0])
    contact_r = poses_world_to_rail_base(contact_w, T)
    assert float(np.linalg.norm(contact_r[:3] - contact_w[:3])) > 0.20
    # Inverse: sending camera-world as rail_base lands beside the bed, not on it.
    fake_world = T[:3, :3] @ contact_w[:3] + T[:3, 3]
    assert float(fake_world[1]) > 0.50
    back = T[:3, :3] @ contact_r[:3] + T[:3, 3]
    assert np.allclose(back, contact_w[:3], atol=1e-6)


def test_standoff_is_5cm_along_minus_tool_z() -> None:
    from peirastic.apps.vessel_scan import standoff_pose_from_contact
    from scipy.spatial.transform import Rotation as Rsc

    contact = np.array([0.40, 0.20, 0.30, 0.0, 0.0, 0.0], dtype=float)
    planned = standoff_pose_from_contact(contact, approach_dz_m=0.05)
    R = Rsc.from_euler("xyz", contact[3:6]).as_matrix()
    assert float(np.linalg.norm(planned[:3] - contact[:3])) == pytest.approx(0.05, abs=1e-9)
    assert np.allclose(planned[:3], contact[:3] - 0.05 * R[:, 2])
