"""Cartesian TRACKING ellipse reference and QPIK program compile."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rm75_control.control.admittance_common.phase_ipc import SinToolYTaskParams
from rm75_control.control.joint_admittance_8dof.api import (
    CompileContext,
    compile_phase,
    phase_cartesian_track,
)
from rm75_control.control.joint_admittance_8dof.config import build_joint_ik_config
from rm75_control.control.joint_admittance_8dof.ellipse_track_program import (
    build_ellipse_track_program,
)
from rm75_control.control.joint_admittance_8dof.loop import (
    AdmittanceOuterLoop,
    CartesianTrackConfig,
    CartesianTrackOuterLoop,
    JointIkController,
)
from rm75_control.control.joint_admittance_8dof.model import RobotKinematics
from rm75_control.control.joint_admittance_8dof.reference import (
    DEFAULT_STOP_RAMP_S,
    EllipseToolXYReference,
    _soft_start_time_warp,
    ellipse_xy_motion,
)


_SEED_Q = np.array([0.375, 0.194, -0.503, -0.069, 1.979, -0.776, 0.547, -4.370])
_CFG = Path(__file__).resolve().parents[1] / "configs" / "joint_admittance_8dof.yaml"


def test_time_warp_soft_stop_reaches_rest_without_changing_start() -> None:
    tau0, d0 = _soft_start_time_warp(0.0, 0.4)
    assert tau0 == pytest.approx(0.0)
    assert d0 == pytest.approx(0.0)
    tau_c, d_c = _soft_start_time_warp(1.0, 0.4)
    assert d_c == pytest.approx(1.0)
    assert tau_c == pytest.approx(0.8)
    T = 2.0
    stop = 0.4
    tau_e, d_e = _soft_start_time_warp(T, 0.4, duration_s=T, stop_ramp_s=stop)
    assert d_e == pytest.approx(0.0)
    assert tau_e == pytest.approx(1.6)
    # Mid-stop is slower than cruise, not a step.
    _, d_mid = _soft_start_time_warp(T - 0.5 * stop, 0.4, duration_s=T, stop_ramp_s=stop)
    assert 0.0 < d_mid < 1.0
    # C1 at the stop-start seam.
    h = 1e-4
    _, d_a = _soft_start_time_warp(T - stop - h, 0.4, duration_s=T, stop_ramp_s=stop)
    _, d_b = _soft_start_time_warp(T - stop + h, 0.4, duration_s=T, stop_ramp_s=stop)
    assert abs(d_a - d_b) < 5e-3


def test_ellipse_soft_stop_zero_velocity_at_duration() -> None:
    ax, ay, omega = 0.02, 0.04, 1.0
    T = 3.0
    stop = 0.4
    dx, dy, vx, vy = ellipse_xy_motion(
        T, ax, ay, omega, soft_start=True, ramp_s=0.4, duration_s=T, stop_ramp_s=stop
    )
    assert abs(vx) < 1e-12
    assert abs(vy) < 1e-12
    ref = EllipseToolXYReference(
        ax, ay, period_s=2.0 * np.pi / omega, duration_s=2.5, stop_ramp_s=stop
    )
    assert ref.stop_ramp_s == pytest.approx(stop)
    assert ref.duration_s == pytest.approx(2.5 + stop)
    assert DEFAULT_STOP_RAMP_S == pytest.approx(0.5)


def test_ellipse_starts_at_origin_with_zero_soft_start_vel() -> None:
    dx, dy, vx, vy = ellipse_xy_motion(0.0, 0.02, 0.04, 1.0, soft_start=True, ramp_s=2.0)
    assert abs(dx) < 1e-12
    assert abs(dy) < 1e-12
    assert abs(vx) < 1e-12
    assert abs(vy) < 1e-12


def test_ellipse_is_offset_circle_and_pose_vel_consistent() -> None:
    ax, ay = 0.02, 0.04
    omega = 0.8
    ref = EllipseToolXYReference(ax, ay, period_s=2.0 * np.pi / omega, soft_start=False)
    origin = np.array([0.1, -0.2, 0.3, 0.0, 0.0, 0.0])
    ref.set_origin(origin)
    h = 1e-4
    for t in (0.0, 0.4, 1.1, 2.7):
        m = ref.sample(t)
        off = m.pose_d[:3] - origin[:3]
        assert abs((off[0] / ax) ** 2 + ((off[1] - ay) / ay) ** 2 - 1.0) < 1e-9
        assert abs(off[2]) < 1e-12
        m0 = ref.sample(t - h)
        m1 = ref.sample(t + h)
        v_fd = (m1.pose_d[:3] - m0.pose_d[:3]) / (2.0 * h)
        assert np.allclose(m.vel_ff[:3], v_fd, atol=2e-6)


def test_ellipse_peak_speed_respects_max_vel() -> None:
    ax, ay = 0.02, 0.04
    vmax = 0.03
    ref = EllipseToolXYReference(ax, ay, max_vel_m_s=vmax, soft_start=False)
    origin = np.zeros(6)
    ref.set_origin(origin)
    speeds = []
    for t in np.linspace(0.0, ref.period_s, 400):
        speeds.append(float(np.linalg.norm(ref.sample(t).vel_ff[:3])))
    assert max(speeds) <= vmax + 1e-9


def test_phase_cartesian_track_compiles_to_pd_outer() -> None:
    raw = _CFG.read_text(encoding="utf-8")
    import yaml

    cfg = build_joint_ik_config(yaml.safe_load(raw))
    cfg.backend = "python"
    kin = RobotKinematics()
    inner = JointIkController(kin, cfg)
    ctx = CompileContext(kin=kin, inner=inner, control_frame=cfg.control_frame)
    ref = EllipseToolXYReference(0.02, 0.04, max_vel_m_s=0.03)
    compiled = compile_phase(
        phase_cartesian_track(ref, label="ellipse_track", duration_s=5.0),
        ctx,
    )
    assert isinstance(compiled.outer, CartesianTrackOuterLoop)
    assert not isinstance(compiled.outer, AdmittanceOuterLoop)
    assert compiled.phase.label == "ellipse_track"
    assert compiled.phase.duration_s == 5.0
    assert cfg.cartesian_track.k_task_lin == pytest.approx(12.0)
    assert cfg.cartesian_track.fb_lpf_tau_s == pytest.approx(0.0)
    assert compiled.outer.cfg.fb_lpf_tau_s == pytest.approx(0.0)
    assert np.allclose(compiled.outer.cfg.k_task, [12.0, 12.0, 12.0, 2.0, 2.0, 2.0])
    overridden = compile_phase(
        phase_cartesian_track(ref, label="ellipse_kp", duration_s=5.0, move_kp=6.0),
        ctx,
    )
    assert np.allclose(overridden.outer.cfg.k_task, [6.0, 6.0, 6.0, 2.0, 2.0, 2.0])


def test_cartesian_p_lpf_leaves_ff_and_keeps_dc_gain() -> None:
    cfg = CartesianTrackConfig(
        k_task=np.array([16.0, 16.0, 16.0, 2.0, 2.0, 2.0]),
        fb_lpf_tau_s=0.02,
        path_feedforward=True,
        control_frame="base",
        max_lin_vel_m_s=1.0,
    )

    class _Ref:
        pose_d = np.zeros(6)
        vel_ff = np.array([0.04, 0.0, 0.0, 0.0, 0.0, 0.0])

        def sample(self, t_s: float):
            del t_s
            return self

    ref = _Ref()
    outer = CartesianTrackOuterLoop(ref, cfg)
    pose = np.zeros(6)
    out0 = outer.sample(0.0, pose, np.zeros(6))
    assert np.allclose(out0[:3], [0.04, 0.0, 0.0], atol=1e-12)
    ref.pose_d = np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0])
    out1 = outer.sample(0.005, pose, np.zeros(6))
    assert np.allclose(outer.last_path_twist[:3], [0.04, 0.0, 0.0], atol=1e-12)
    assert out1[0] < 0.20 - 1e-6
    assert out1[0] > 0.04
    for i in range(80):
        out = outer.sample(0.005 * (i + 2), pose, np.zeros(6))
    assert np.allclose(out[:3], [0.20, 0.0, 0.0], atol=1e-3)


def test_build_ellipse_program_from_live_pose_and_ipc() -> None:
    params = SinToolYTaskParams(
        config_path=str(_CFG),
        task_kind="ellipse_track",
        x_pp_cm=4.0,
        y_pp_cm=8.0,
        max_vel_cm_s=3.0,
        scan_duration=12.0,
        q0_rad=_SEED_Q.tolist(),
        q_target_rad=_SEED_Q.tolist(),
        pose_d=[0.0] * 6,
        tcp_offset_pose=[0.0] * 6,
    )
    decoded = SinToolYTaskParams.from_json(params.to_json())
    assert decoded.task_kind == "ellipse_track"
    assert decoded.x_pp_cm == 4.0
    import yaml

    raw = yaml.safe_load(_CFG.read_text(encoding="utf-8"))
    raw.setdefault("inner", {})["backend"] = "python"
    built = build_ellipse_track_program(params, raw=raw)
    assert [p.label for p in built.phases] == ["ellipse_track"]
    assert isinstance(built.compiled[0].outer, CartesianTrackOuterLoop)
    assert built.reference.amplitude_x_m == 0.02
    assert built.reference.amplitude_y_m == 0.04
    built.inner.reset(_SEED_Q)
    built.phases[0].on_enter()
    assert built.inner._centering_suppressed is False
    assert built.inner._arm_task_suppressed is False
