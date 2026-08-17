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
    CartesianTrackOuterLoop,
    JointIkController,
)
from rm75_control.control.joint_admittance_8dof.model import RobotKinematics
from rm75_control.control.joint_admittance_8dof.reference import (
    EllipseToolXYReference,
    ellipse_xy_motion,
)


_SEED_Q = np.array([0.375, 0.194, -0.503, -0.069, 1.979, -0.776, 0.547, -4.370])
_CFG = Path(__file__).resolve().parents[1] / "configs" / "joint_admittance_8dof.yaml"


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
    assert cfg.cartesian_track.k_task_lin == pytest.approx(10.0)
    assert np.allclose(compiled.outer.cfg.k_task, [10.0, 10.0, 10.0, 2.0, 2.0, 2.0])
    overridden = compile_phase(
        phase_cartesian_track(ref, label="ellipse_kp", duration_s=5.0, move_kp=6.0),
        ctx,
    )
    assert np.allclose(overridden.outer.cfg.k_task, [6.0, 6.0, 6.0, 2.0, 2.0, 2.0])


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
    built = build_ellipse_track_program(params)
    assert [p.label for p in built.phases] == ["ellipse_track"]
    assert isinstance(built.compiled[0].outer, CartesianTrackOuterLoop)
    assert built.reference.amplitude_x_m == 0.02
    assert built.reference.amplitude_y_m == 0.04
    built.phases[0].on_enter()
    assert built.inner._centering_suppressed is False
    assert built.inner._arm_task_suppressed is False
