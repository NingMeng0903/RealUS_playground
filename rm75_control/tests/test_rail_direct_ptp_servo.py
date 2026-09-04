"""direct_ptp rail servo: rebase skip, arrival, slow creep, mixer continuity."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from pathlib import Path

from rm75_control.control.joint_admittance_8dof.api import make_move_arrived
from rm75_control.control.joint_admittance_8dof.config import build_joint_ik_config
from rm75_control.control.joint_admittance_8dof.hw.rail_servo import RailCommandMode
from rm75_control.control.joint_admittance_8dof.loop import (
    JointIkController,
    _rail_settled_for_arrival,
)
from rm75_control.control.joint_admittance_8dof.model import RobotKinematics
from rm75_control.control.joint_admittance_8dof.tasks.rail_allocator import (
    RailReferenceModel,
)
from rm75_control.control.joint_admittance_8dof.tasks.rail_command import (
    RailCommandMixer,
)
from rm75_control.control.joint_admittance_8dof.tasks.rail_mode import LockedStyle, RailMode


_CFG = Path(__file__).resolve().parents[1] / "configs" / "joint_admittance_8dof.yaml"
_SEED_Q = np.array([0.375, 0.194, -0.503, -0.069, 1.979, -0.776, 0.547, -4.370])


def _python_inner() -> JointIkController:
    raw = yaml.safe_load(_CFG.read_text(encoding="utf-8"))
    cfg = build_joint_ik_config(raw)
    cfg.backend = "python"
    cfg.collision.enabled = False
    cfg.qp.collision.enabled = False
    cfg.ird.enabled = False
    inner = JointIkController(RobotKinematics(), cfg)
    inner.reset(_SEED_Q.copy())
    return inner


def test_direct_ptp_locked_hold_pins_rail() -> None:
    inner = _python_inner()
    ref = float(inner.q_cmd[0])
    inner.set_locked(LockedStyle.HOLD, q_ref_m=ref)
    inner.set_direct_joint_ptp(True)
    inner.set_plan_drives_rail(True)
    q = inner.q_cmd.copy()
    qdot = np.zeros(8)
    qdot[0] = 0.04
    step = inner.update(np.zeros(6), q_meas=q, qdot_ff=qdot)
    assert step.controller_mode == "direct_joint_ptp"
    assert float(step.q_send[0]) == pytest.approx(ref, abs=1e-12)
    assert float(step.qdot[0]) == pytest.approx(0.0, abs=1e-12)
    assert float(inner.q_cmd[0]) == pytest.approx(ref, abs=1e-12)
    assert float(step.rail_qdot_ff) == pytest.approx(0.0, abs=1e-12)
    assert step.plan_drives_rail is False


def test_direct_ptp_skips_rail_rebase_and_integrates_qdot_ff() -> None:
    inner = _python_inner()
    obs = inner.rail_observer
    obs._initialized = True
    obs._last_sample_t = 1.0
    obs.q_hat = float(inner.q_cmd[0]) + 0.003
    inner.set_direct_joint_ptp(True)
    inner.set_plan_drives_rail(True)
    q = inner.q_cmd.copy()
    qdot = np.zeros(8)
    qdot[0] = 0.04
    step = inner.update(np.zeros(6), q_meas=q, qdot_ff=qdot)
    assert step.controller_mode == "direct_joint_ptp"
    assert step.q_send[0] == pytest.approx(q[0] + 0.04 * float(inner.cfg.dt))
    assert abs(float(step.q_send[0]) - float(obs.q_hat)) > 0.002


def test_qp_modes_still_rebase_rail_to_observer() -> None:
    inner = _python_inner()
    obs = inner.rail_observer
    obs._initialized = True
    obs._last_sample_t = 1.0
    hat = float(inner.q_cmd[0]) + 0.0025
    obs.q_hat = hat
    q = inner.q_cmd.copy()
    inner.set_direct_joint_ptp(False)
    inner.set_plan_drives_rail(False)
    step = inner.update(np.zeros(6), q_meas=q)
    assert step.controller_mode == "qpik"
    assert float(inner.q_cmd[0]) == pytest.approx(hat)


def test_moves_plan_drives_rail_without_direct_ptp_still_rebases() -> None:
    inner = _python_inner()
    obs = inner.rail_observer
    obs._initialized = True
    obs._last_sample_t = 1.0
    hat = float(inner.q_cmd[0]) + 0.002
    obs.q_hat = hat
    q = inner.q_cmd.copy()
    inner.set_plan_drives_rail(True)
    inner.set_direct_joint_ptp(False)
    inner.update(np.zeros(6), q_meas=q)
    assert float(inner.q_cmd[0]) == pytest.approx(hat)


def test_make_move_arrived_rejects_mm_rail_residual() -> None:
    q_goal = _SEED_Q.copy()
    pose = np.zeros(6)
    gate = make_move_arrived(pose, q_goal, joint_only=True)
    q_ok = q_goal.copy()
    q_ok[0] += 0.0004
    assert gate(pose, q_ok)
    q_far = q_goal.copy()
    q_far[0] += 0.002894
    assert not gate(pose, q_far)


def test_rail_settled_rejects_position_residual() -> None:
    bridge = SimpleNamespace(
        enabled=True,
        servo_sample=SimpleNamespace(
            sample_mono_s=10.0,
            v_cmd_m_s=0.0,
            v_meas_m_s=0.0,
            x_goal_m=0.403,
            x_meas_m=0.400,
        ),
    )
    assert (
        _rail_settled_for_arrival(
            bridge, speed_limit_m_s=0.003, now_s=10.01, freshness_s=0.05
        )
        is False
    )
    assert (
        _rail_settled_for_arrival(
            bridge,
            speed_limit_m_s=0.003,
            now_s=10.01,
            freshness_s=0.05,
            pos_err_m=0.0003,
        )
        is True
    )


def test_reference_model_can_output_slow_creep() -> None:
    model = RailReferenceModel(f_c_hz=4.0, a_max=0.60, j_max=60.0, v_max=0.15)
    model.reset(0.0)
    v = 0.0
    for _ in range(80):
        v = model.step(5.17e-4, 0.005, x_m=0.40, apply_wall=False)
    assert 1.0e-4 < abs(v) < 8.0e-4


def test_quiescent_entry_keeps_u_post_continuous() -> None:
    mix = RailCommandMixer(kp=1.2, ki=0.8, u_mid_max=0.12, kaw=8.0)
    mix.d_star.init_from_live(0.22)
    prev = None
    for i in range(8):
        tel = mix.step(
            d_live=0.25,
            d_star_target=0.22,
            u_task_raw=0.02,
            u_escape_raw=0.0,
            escape_explicit=False,
            dt=0.005,
            u_max=0.12,
            quiescent=i >= 4,
        )
        if prev is not None:
            assert abs(tel.u_post_raw - prev) < 0.02
        prev = float(tel.u_post_raw)


def test_python_mixer_matches_itself_across_quiescent_flag() -> None:
    kwargs = dict(
        d_live=0.26,
        d_star_target=0.24,
        u_task_raw=0.03,
        u_escape_raw=0.0,
        escape_explicit=False,
        dt=0.005,
        u_max=0.12,
    )
    a = RailCommandMixer(kp=1.2, ki=0.8, u_mid_max=0.12, kaw=8.0)
    b = RailCommandMixer(kp=1.2, ki=0.8, u_mid_max=0.12, kaw=8.0)
    a.d_star.init_from_live(0.24)
    b.d_star.init_from_live(0.24)
    ta = a.step(**kwargs, quiescent=False)
    tb = b.step(**kwargs, quiescent=True)
    assert ta.u_feasible == pytest.approx(tb.u_feasible)
    assert ta.u_post_raw == pytest.approx(tb.u_post_raw)
    assert a.xi == pytest.approx(b.xi)


def test_rail_command_mode_tracked_position_aliases() -> None:
    assert RailCommandMode.coerce("tracked") is RailCommandMode.TRACKED_POSITION
    assert RailCommandMode.coerce("tracked-position") is RailCommandMode.TRACKED_POSITION
    assert RailMode.COUPLED.name == "COUPLED"


def test_tracked_position_recovers_injected_velocity_deficit() -> None:
    """Open-loop FF leaves ~2.9 mm; the PTP PD law drives residual to 0.5 mm."""

    x_goal = 0.400
    x = 0.400
    dt = 0.005
    kp, catch = 14.0, 0.02
    for k in range(80):
        v_ff = 0.04 if k < 40 else 0.0
        x_goal += v_ff * dt
        x += 0.70 * v_ff * dt
        err_x = x_goal - x
        v_p = max(-0.05, min(0.05, kp * err_x))
        v_corr = max(-catch, min(catch, v_p))
        x += v_corr * dt
    assert abs(x_goal - x) <= 5.0e-4


def test_aj_box_holds_through_slow_creep() -> None:
    model = RailReferenceModel(f_c_hz=4.0, a_max=0.60, j_max=60.0, v_max=0.15)
    model.reset(0.0)
    dt = 0.005
    prev_v = 0.0
    prev_a = 0.0
    for k in range(200):
        u = 5.17e-4 if k < 120 else 0.0
        v = model.step(u, dt, x_m=0.40, apply_wall=False)
        a = (v - prev_v) / dt
        j = (a - prev_a) / dt
        assert abs(a) <= 0.60 + 1.0e-9
        if k > 0:
            assert abs(j) <= 60.0 + 1.0e-6
        prev_v, prev_a = v, a


def test_track_applied_does_not_rewrite_integrator() -> None:
    mix = RailCommandMixer(kp=1.2, ki=0.8, u_mid_max=0.12, kaw=8.0)
    mix.d_star.init_from_live(0.22)
    mix.xi = 0.037
    mix.step(
        d_live=0.25,
        d_star_target=0.22,
        u_task_raw=0.01,
        u_escape_raw=0.0,
        escape_explicit=False,
        dt=0.005,
        u_max=0.12,
    )
    xi_before = float(mix.xi)
    mix.track_applied(
        d_live=0.26,
        d_star_target=0.22,
        applied_rail_vel=0.03,
        dt=0.005,
    )
    assert mix.xi == pytest.approx(xi_before)


def test_publish_uses_explicit_tracked_position_mode() -> None:
    from rm75_control.control.joint_admittance_8dof.loop import (
        _publish_rail_target_before_arm,
    )

    seen: dict[str, object] = {}

    class _Bridge:
        enabled = True
        calibrated = True
        panicked = False
        armed = True

        def set_target_m(self, target_m, v_ff_m_s=None, mode=None):
            seen["target"] = float(target_m)
            seen["v_ff"] = v_ff_m_s
            seen["mode"] = mode
            return True

    ok, reason = _publish_rail_target_before_arm(
        _Bridge(),
        0.412,
        lambda _r: None,
        v_ff_m_s=0.04,
        mode=RailCommandMode.TRACKED_POSITION,
    )
    assert ok and reason == ""
    assert seen["mode"] is RailCommandMode.TRACKED_POSITION
    assert seen["v_ff"] == pytest.approx(0.04)


def test_python_ptp_reports_rail_goal_error() -> None:
    inner = _python_inner()
    q = inner.q_cmd.copy()
    q_meas = q.copy()
    q_meas[0] += 0.002894
    inner.set_direct_joint_ptp(True)
    inner.set_plan_drives_rail(True)
    step = inner.update(np.zeros(6), q_meas=q_meas, qdot_ff=np.zeros(8))
    assert step.plan_drives_rail is True
    assert step.rail_qdot_ff == pytest.approx(0.0)
    assert step.rail_goal_err_m == pytest.approx(q[0] - q_meas[0], abs=1e-9)


def test_python_and_native_ptp_integrate_the_same_rail_cmd() -> None:
    from rm75_control.control.joint_admittance_8dof.wbc_rt.client import (
        find_wbc_rt_binary,
    )

    if find_wbc_rt_binary() is None:
        pytest.skip("wbc_rt binary not built")

    import os

    raw = yaml.safe_load(_CFG.read_text(encoding="utf-8"))
    qdot = np.zeros(8)
    qdot[0] = 0.001
    py_q = []
    py = _python_inner()
    py.set_direct_joint_ptp(True)
    py.set_plan_drives_rail(True)
    q = py.q_cmd.copy()
    for _ in range(8):
        step = py.update(np.zeros(6), q_meas=q, qdot_ff=qdot)
        q = np.asarray(step.q_send, dtype=float).copy()
        py_q.append(float(q[0]))

    cfg = build_joint_ik_config(raw)
    cfg.backend = "native"
    cfg.native_shm_prefix = f"rm75_wbc_ptp_{os.getpid()}"
    cfg.collision.enabled = False
    cfg.qp.collision.enabled = False
    cfg.ird.enabled = False
    nt = JointIkController(RobotKinematics(), cfg)
    try:
        nt.reset(_SEED_Q.copy())
        nt.enable()
        nt.set_coupled()
        nt.set_direct_joint_ptp(True)
        nt.set_plan_drives_rail(True)
        q = nt.q_cmd.copy()
        nt_q = []
        for _ in range(8):
            step = nt.update(np.zeros(6), q_meas=q, qdot_ff=qdot)
            q = np.asarray(step.q_send, dtype=float).copy()
            nt_q.append(float(q[0]))
    finally:
        if nt._native is not None:
            nt._native.shutdown()
    assert nt_q[-1] == pytest.approx(py_q[-1], abs=5.0e-5)


def test_smooth_move_qdot0_matches_live_and_rests_at_end() -> None:
    from rm75_control.control.joint_admittance_8dof.reference import (
        JointSmoothMoveReference,
    )
    from rm75_control.kinematics.srs_ik import psi_from_q

    kin = RobotKinematics()
    q0 = _SEED_Q.copy()
    qt = q0.copy()
    qt[0] -= 0.11
    v0 = np.zeros(8)
    v0[0] = -0.008
    ref = JointSmoothMoveReference(kin, q0, qt, 2.0, qdot0_rad_s=v0)
    q, qdot = ref.sample_q(0.0)
    np.testing.assert_allclose(q, q0, atol=1e-12)
    np.testing.assert_allclose(qdot, v0, atol=1e-12)
    qe, ve = ref.sample_q(2.0)
    np.testing.assert_allclose(qe, qt, atol=1e-9)
    np.testing.assert_allclose(ve, 0.0, atol=1e-9)
    psi0 = float(psi_from_q(q[1:]))
    dt = 0.005
    a_max = 1.6
    prev_v = float(qdot[0])
    for i in range(1, 11):
        qi, vi = ref.sample_q(i * dt)
        assert abs(float(psi_from_q(qi[1:])) - psi0) < np.deg2rad(0.5)
        a = (float(vi[0]) - prev_v) / dt
        assert abs(a) <= a_max + 1.0e-6
        prev_v = float(vi[0])


def test_ptp_on_enter_reseeds_hold_end_q_and_qdot() -> None:
    from rm75_control.control.joint_admittance_8dof.api import attach_joint_move_rail
    from rm75_control.control.joint_admittance_8dof.reference import (
        JointSmoothMoveReference,
    )
    from rm75_control.kinematics.srs_ik import psi_from_q

    inner = _python_inner()
    applied = np.zeros(8)
    applied[0] = -0.008
    inner.last_applied_qdot = applied.copy()
    inner.last_v_r_ref = 0.0
    inner.core.qdot_prev = np.zeros(8)
    q_live = inner.q_cmd.copy()
    q_stale = q_live.copy()
    q_stale[0] += 0.012
    q_stale[4] += np.deg2rad(3.2)
    qt = q_live.copy()
    qt[0] -= 0.10
    ref = JointSmoothMoveReference(inner.kin, q_stale, qt, 2.2)
    q_before, _ = ref.sample_q(0.0)
    assert np.linalg.norm(q_before - q_live) > 1.0e-3
    phase = SimpleNamespace(on_enter=None, on_exit=None, outer=SimpleNamespace(reference=ref))
    attach_joint_move_rail(phase, inner, move_ref=ref)
    phase.on_enter()
    q, qdot = ref.sample_q(0.0)
    np.testing.assert_allclose(q, q_live, atol=1e-12)
    assert qdot[0] == pytest.approx(-0.008, abs=1e-12)
    psi0 = float(psi_from_q(q[1:]))
    dt = 0.005
    prev_v = float(qdot[0])
    for i in range(1, 11):
        qi, vi = ref.sample_q(i * dt)
        assert abs(float(psi_from_q(qi[1:])) - psi0) < np.deg2rad(0.5)
        assert abs((float(vi[0]) - prev_v) / dt) <= 1.6 + 1.0e-6
        prev_v = float(vi[0])
    assert inner._direct_joint_ptp is True
    assert inner._plan_drives_rail is True


def test_ptp_on_enter_freezes_rail_when_locked_hold() -> None:
    from rm75_control.control.joint_admittance_8dof.api import attach_joint_move_rail
    from rm75_control.control.joint_admittance_8dof.reference import (
        JointSmoothMoveReference,
    )

    inner = _python_inner()
    rail_ref = float(inner.q_cmd[0])
    inner.set_locked(LockedStyle.HOLD, q_ref_m=rail_ref)
    applied = np.zeros(8)
    applied[0] = -0.008
    inner.last_applied_qdot = applied.copy()
    q_live = inner.q_cmd.copy()
    qt = q_live.copy()
    qt[0] -= 0.10
    qt[4] += np.deg2rad(4.0)
    ref = JointSmoothMoveReference(inner.kin, q_live, qt, 2.2)
    phase = SimpleNamespace(on_enter=None, on_exit=None, outer=SimpleNamespace(reference=ref))
    attach_joint_move_rail(phase, inner, move_ref=ref)
    phase.on_enter()
    q0, v0 = ref.sample_q(0.0)
    qe, ve = ref.sample_q(2.2)
    assert float(q0[0]) == pytest.approx(rail_ref, abs=1e-12)
    assert float(qe[0]) == pytest.approx(rail_ref, abs=1e-12)
    assert float(v0[0]) == pytest.approx(0.0, abs=1e-12)
    assert float(ve[0]) == pytest.approx(0.0, abs=1e-12)
    assert float(ref.q_target[0]) == pytest.approx(rail_ref, abs=1e-12)
    assert inner._direct_joint_ptp is True
    assert inner._plan_drives_rail is False


def test_build_movej_payload_id_freezes_rail_interpolator() -> None:
    from peirastic.realman8dof.modes.joint import build_movej_phase
    from rm75_control.control.joint_admittance_8dof.api import (
        CompileContext,
        SecondaryPolicy,
    )

    inner = _python_inner()
    rail0 = float(inner.q_cmd[0])
    SecondaryPolicy(preset="payload_id").apply(inner)
    ctx = CompileContext(
        kin=inner.kin,
        inner=inner,
        euler_order="xyz",
        control_frame="tool",
        v_scale=0.8,
    )
    qt = inner.q_cmd.copy()
    qt[0] = rail0 + 0.08
    qt[4] += np.deg2rad(4.0)
    phase = build_movej_phase(
        ctx, qt, v=0.25, secondary="payload_id", label="payload_id_WX+15"
    )
    phase.on_enter()
    ref = phase.outer.reference
    assert inner.is_locked_hold
    assert inner._plan_drives_rail is False
    assert float(ref.q_target[0]) == pytest.approx(rail0, abs=1e-12)
    qe, ve = ref.sample_q(float(ref.duration_s))
    assert float(qe[0]) == pytest.approx(rail0, abs=1e-12)
    assert float(ve[0]) == pytest.approx(0.0, abs=1e-12)


def test_build_movej_default_unlocks_leftover_lock() -> None:
    from peirastic.realman8dof.modes.joint import build_movej_phase
    from rm75_control.control.joint_admittance_8dof.api import (
        CompileContext,
        SecondaryPolicy,
    )

    inner = _python_inner()
    rail0 = float(inner.q_cmd[0])
    SecondaryPolicy(preset="payload_id").apply(inner)
    ctx = CompileContext(
        kin=inner.kin,
        inner=inner,
        euler_order="xyz",
        control_frame="tool",
        v_scale=0.8,
    )
    qt = inner.q_cmd.copy()
    qt[0] = rail0 + 0.08
    phase = build_movej_phase(ctx, qt, v=0.4, label="payload_id_movej_mid")
    phase.on_enter()
    ref = phase.outer.reference
    assert not inner.is_locked_hold
    assert inner._plan_drives_rail is True
    assert float(ref.q_target[0]) == pytest.approx(qt[0], abs=1e-12)


def test_ptp_on_enter_move_policy_unlocks_rail_for_8dof() -> None:
    from rm75_control.control.joint_admittance_8dof.api import (
        SecondaryPolicy,
        attach_joint_move_rail,
    )
    from rm75_control.control.joint_admittance_8dof.reference import (
        JointSmoothMoveReference,
    )

    inner = _python_inner()
    rail0 = float(inner.q_cmd[0])
    inner.set_locked(LockedStyle.HOLD, q_ref_m=rail0)
    q_live = inner.q_cmd.copy()
    qt = q_live.copy()
    qt[0] = rail0 + 0.08
    ref = JointSmoothMoveReference(inner.kin, q_live, qt, 2.2)
    phase = SimpleNamespace(
        on_enter=lambda: SecondaryPolicy(preset="move").apply(inner),
        on_exit=None,
        outer=SimpleNamespace(reference=ref),
    )
    attach_joint_move_rail(phase, inner, move_ref=ref)
    phase.on_enter()
    assert not inner.is_locked_hold
    assert inner._plan_drives_rail is True
    assert float(ref.q_target[0]) == pytest.approx(qt[0], abs=1e-12)
    qe, _ = ref.sample_q(2.2)
    assert float(qe[0]) == pytest.approx(qt[0], abs=1e-9)


def test_ptp_on_enter_reseeds_full_applied_qdot() -> None:
    from rm75_control.control.joint_admittance_8dof.api import attach_joint_move_rail
    from rm75_control.control.joint_admittance_8dof.reference import (
        JointSmoothMoveReference,
    )
    from rm75_control.kinematics.srs_ik import psi_from_q

    inner = _python_inner()
    applied = np.array([0.0, 0.06, -0.07, 0.04, 0.12, -0.08, 0.03, -0.05])
    inner.last_applied_qdot = applied.copy()
    inner.last_v_r_ref = -0.008
    inner.core.qdot_prev = np.zeros(8)
    q_live = inner.q_cmd.copy()
    qt = q_live.copy()
    qt[0] -= 0.10
    qt[4] += np.deg2rad(4.0)
    ref = JointSmoothMoveReference(inner.kin, q_live, qt, 2.2)
    phase = SimpleNamespace(on_enter=None, on_exit=None, outer=SimpleNamespace(reference=ref))
    attach_joint_move_rail(phase, inner, move_ref=ref)
    phase.on_enter()
    q, qdot = ref.sample_q(0.0)
    np.testing.assert_allclose(q, q_live, atol=1e-12)
    np.testing.assert_allclose(qdot, applied, atol=1e-12)
    psi0 = float(psi_from_q(q[1:]))
    dt = 0.005
    prev = qdot.copy()
    a_max = np.asarray(inner.limits.a_max, dtype=float)
    for i in range(1, 11):
        qi, vi = ref.sample_q(i * dt)
        assert abs(float(psi_from_q(qi[1:])) - psi0) < np.deg2rad(0.5)
        a = (vi - prev) / dt
        assert np.all(np.abs(a) <= a_max + 1.0e-6)
        prev = vi


def test_live_qdot_is_last_committed_command() -> None:
    inner = _python_inner()
    inner.set_direct_joint_ptp(True)
    inner.set_plan_drives_rail(True)
    q = inner.q_cmd.copy()
    qdot_ff = np.zeros(8)
    qdot_ff[0] = 0.04
    qdot_ff[4] = 0.05
    step = inner.update(np.zeros(6), q_meas=q, qdot_ff=qdot_ff)
    np.testing.assert_allclose(inner.live_qdot(), step.qdot, atol=1e-9)
    assert abs(float(inner.live_qdot()[4])) > 0.01
    inner.last_v_r_ref = -0.008
    np.testing.assert_allclose(inner.live_qdot(), step.qdot, atol=1e-9)


def test_native_observer_updates_during_direct_ptp() -> None:
    from rm75_control.control.joint_admittance_8dof.wbc_rt.client import (
        find_wbc_rt_binary,
    )

    if find_wbc_rt_binary() is None:
        pytest.skip("wbc_rt binary not built")

    import os

    raw = yaml.safe_load(_CFG.read_text(encoding="utf-8"))
    cfg = build_joint_ik_config(raw)
    cfg.backend = "native"
    cfg.native_shm_prefix = f"rm75_wbc_obs_{os.getpid()}"
    cfg.collision.enabled = False
    cfg.qp.collision.enabled = False
    cfg.ird.enabled = False
    inner = JointIkController(RobotKinematics(), cfg)
    try:
        inner.reset(_SEED_Q.copy())
        inner.enable()
        inner.set_coupled()
        q0 = float(_SEED_Q[0])
        q = _SEED_Q.copy()
        inner.update(np.zeros(6), q_meas=q)
        inner.set_direct_joint_ptp(True)
        inner.set_plan_drives_rail(True)
        qdot = np.zeros(8)
        qdot[0] = 0.04
        x_meas = q0
        plant_v = 0.028
        for _ in range(24):
            x_meas += plant_v * float(inner.cfg.dt)
            q_meas = q.copy()
            q_meas[0] = x_meas
            step = inner.update(
                np.zeros(6),
                q_meas=q_meas,
                qdot_ff=qdot,
                rail_exec_vel_m_s=plant_v,
            )
            q = np.asarray(step.q_send, dtype=float).copy()
        inner.set_direct_joint_ptp(False)
        inner.set_plan_drives_rail(False)
        q_meas = q.copy()
        q_meas[0] = x_meas
        step = inner.update(np.zeros(6), q_meas=q_meas, rail_exec_vel_m_s=plant_v)
        assert abs(float(step.q_send[0]) - x_meas) < 1.5e-3
        assert abs(float(step.q_send[0]) - q0) > 1.5e-3
    finally:
        if inner._native is not None:
            inner._native.shutdown()
