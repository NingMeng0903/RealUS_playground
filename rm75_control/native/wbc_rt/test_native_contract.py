"""Offline checks for the native v7 HQP/rail contract.

The dynamic checks use the named-SHM client with an offline kinematic input
stream; they do not connect to hardware.  They cover the ABI and the
source-level invariants that are easy to regress when the Python builder is
changed in parallel: residual QP1, no preview/hold/alpha, QP2 locks r1*,
and final acceleration history follows the command actually committed.
"""

from __future__ import annotations

import subprocess
import os
import sys
import uuid
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parent
SRC = (ROOT / "src/inner.cpp").read_text(encoding="utf-8")
PROTO = (ROOT / "include/wbc_rt/protocol.hpp").read_text(encoding="utf-8")
BIN = ROOT / "build/wbc_rt"


def test_native_v7_protocol_sizes() -> None:
    assert "kVersion = 7" in PROTO
    assert "static_assert(sizeof(WbcIn) == 616" in PROTO
    assert "static_assert(sizeof(WbcOut) == 1440" in PROTO
    for field in (
        "rail_refresh_dt",
        "task_progress_alpha",
        "rail_preview_arm",
        "rail_base_shaped",
        "rail_pi_xi",
        "rail_d_ref",
        "rail_ref_acceleration",
    ):
        assert field in PROTO
    if not BIN.exists():
        pytest.skip("native binary not built")
    sizes = subprocess.check_output([str(BIN), "--sizes"], text=True).split()
    assert sizes == ["616", "1440"]
    info = subprocess.check_output([str(BIN), "--protocol-info"], text=True)
    assert "version 7 in 616 out 1440" in info


def test_residual_qp1_layout_contract() -> None:
    assert "constexpr int kNVar = kNv + kNTaskSlack + kNPref;" in SRC
    assert "constexpr int kNEq1 = kTask;" in SRC
    assert "constexpr int kNEq2 = kTask;" in SRC
    assert "kNPreviewArm" not in SRC
    assert "kAlphaCol" not in SRC
    assert "A1.leftCols(kNv) = J_task;" in SRC
    assert "Vec6 b_task = v_cmd - rail_actual_contrib;" in SRC
    assert "A2.leftCols(kNv) = J_task;" in SRC
    assert "VecX b2 = t1;" in SRC
    assert "handle_pending_flags" in SRC
    assert "kInAutoCommit" in SRC
    assert "collapse_interval(&lo_box, &hi_box, &qdot_prev_, &a_max_, h1)" in SRC
    assert "inbox_brake(qdot_prev_, lo_box, hi_box, a_max_, h1)" in SRC
    ptp = SRC[SRC.index("if (direct_ptp_ && (in.flags & kInHasQdotFf))") :]
    ptp = ptp[: ptp.index("double rail_exec")]
    assert "pending_ = capture_history()" in ptp
    assert "restore_history(committed_snap_)" in ptp
    assert "out.qp1_status = kQpSolved" in ptp
    assert "out.status = kStatusOk" in ptp


def test_slow_rail_owner_uses_reference_derivative_box() -> None:
    # The slow reference remains a QP2 preference.  QP1 receives only the
    # reference-model derivative envelope from final committed history; an
    # amplitude interval between zero/brake and the nominal target would
    # prevent the redundant rail from carrying a valid low-frequency task.
    assert "const double brake_rail" not in SRC
    assert "const double nominal_rail" not in SRC
    assert "rail_prev_committed_ref_" in SRC
    assert "rail_prev_committed_a_" in SRC
    assert "const double a_ref" in SRC
    assert "const double j_ref" in SRC
    assert "rail_prev_committed_ref_ - a_ref * h" in SRC
    assert "rail_prev_committed_ref_ + (a_prev - j_ref * h) * h" in SRC
    assert "rail_prev_committed_ref_ + a_ref * h" in SRC
    assert "rail_prev_committed_ref_ + (a_prev + j_ref * h) * h" in SRC


def test_proxqp_infeasibility_tolerances_are_tightened() -> None:
    assert "settings.eps_primal_inf = eps_inf;" in SRC
    assert "settings.eps_dual_inf = eps_inf;" in SRC


def test_rail_is_not_reseeded_after_qp1() -> None:
    solve = SRC[SRC.index("bool InnerLoop::solve_hqp") :]
    solve = solve[: solve.index("TickOut InnerLoop::step")]
    assert "qdot1[0] = clip(seed" not in solve
    assert "qdot1[0] = rail_exec" not in solve
    assert "J_task.col(0).setZero()" in solve
    assert "b_task = v_cmd - rail_actual_contrib" in solve


def test_final_rail_acceleration_uses_preclip_history() -> None:
    assert "const double previous = *ref;" in SRC
    assert "*acc = (shaped - previous) / h;" in SRC
    assert "v_r_a_ = (v - (v_r_ref_ - a * dt)) / dt;" not in SRC


def test_qp_status_alone_is_not_a_certificate() -> None:
    assert "qp_eq_violation(A1, b1, x)" in SRC
    assert "qp_ineq_violation(C, lo_use, hi_use, x)" in SRC
    assert "qp_eq_violation(A2, b2, x2)" in SRC
    assert "qp_ineq_violation(C, lo, hi, x2)" in SRC


def test_mixer_final_components_sum_exactly() -> None:
    # Same identity used by the native commit path.  This catches accidental
    # PI feedback from the unfiltered full total when a task tail is present.
    base_shaped = 0.017
    rail_final = 0.011
    task = 0.0105
    escape = base_shaped - task
    post = rail_final - base_shaped
    assert post + task + escape == pytest.approx(rail_final, abs=1e-15)
    assert "u_post_committed_ = beta * (rail_total_ref - u_base_);" in SRC
    assert "u_base_committed_ = rail_y - u_post_committed_;" in SRC
    assert "u_mid_committed_ - alpha * u_pi_raw_" in SRC


def test_brake_and_shadow_history_do_not_cross_or_rewind() -> None:
    # The fallback brake is a one-sided command.  It must reach zero and stop
    # there when one acceleration step is larger than the stored speed.
    types = (ROOT / "include/wbc_rt/types.hpp").read_text(encoding="utf-8")
    assert "std::max(0.0, qdot_prev[i] - step)" in types
    assert "std::min(0.0, qdot_prev[i] + step)" in types
    assert "v_r_base_ref_ = u_base_committed_;" in SRC
    assert "v_r_ref_ = u_total_committed_;" in SRC
    assert "const bool attribution_override" in SRC


def _runtime_controller(*, native: bool, psi_enabled: bool = False):
    """Build the matched controller used by the Python dynamics regressions."""
    try:
        from rm75_control.control.joint_admittance_8dof.collision_model import (
            CollisionConfig,
        )
        from rm75_control.control.joint_admittance_8dof.loop import (
            JointIkConfig,
            JointIkController,
        )
        from rm75_control.control.joint_admittance_8dof.model import RobotKinematics
        from rm75_control.control.joint_admittance_8dof.solver.qp_builder import QpConfig
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"runtime kinematics dependencies unavailable: {exc}")

    collision = CollisionConfig(enabled=False)
    qp = QpConfig(
        backend="proxqp",
        collision=collision,
        smoothness_weight=np.r_[0.0, np.full(7, 0.15)],
    )
    qp.j4_design_comfort.enabled = False
    cfg = JointIkConfig(control_frame="base", qp=qp, collision=collision)
    cfg.backend = "native" if native else "python"
    cfg.psi_retarget.enabled = bool(psi_enabled)
    cfg.rail_extension.enabled = False
    if native:
        if not BIN.exists():
            pytest.skip("native wbc_rt binary not built")
        cfg.native_bin = str(BIN)
        cfg.native_shm_prefix = f"wbc_rt_contract_{os.getpid()}_{uuid.uuid4().hex}"
    return JointIkController(RobotKinematics(), cfg)


def _safe_q():
    from rm75_control.control.joint_admittance_8dof.model import full_q_from_arm

    return full_q_from_arm(
        np.deg2rad([5.0, -30.0, 10.0, 60.0, -5.0, 45.0, 0.0]),
        0.40,
    )


def _close_runtime_controller(ctrl) -> None:
    try:
        ctrl.stop()
    finally:
        native = getattr(ctrl, "_native", None)
        if native is not None:
            native.shutdown()


def _run_pure_z(ctrl, *, mode_name: str, ticks: int = 100):
    from rm75_control.control.joint_admittance_8dof.tasks.rail_mode import RailMode

    q0 = _safe_q()
    mode = RailMode.COUPLED if mode_name == "coupled" else RailMode.LOCKED
    ctrl.reset(q0)
    ctrl.enable()
    ctrl.set_rail_mode(mode, q_ref_m=float(q0[0]) if mode == RailMode.LOCKED else None)
    ctrl.begin_hybrid_episode(q0, np.zeros(8))
    q = q0.copy()
    steps = []
    for _ in range(ticks):
        step = ctrl.update(
            np.array([0.0, 0.0, 0.008, 0.0, 0.0, 0.0]),
            0.005,
            q_meas=q,
            # Match the offline fixture's ideal encoder feedback.  Omitting
            # this flag exercises the observer startup path instead of the
            # rail/arm contract under test.
            rail_exec_vel_m_s=float(ctrl.core.qdot_prev[0]),
            rail_refresh_dt_s=0.020,
        )
        assert not step.task_paused, step.task_pause_reason
        assert step.qp1_status in ("solved", "max_iter"), step.qp1_status
        assert step.qp2_status in ("solved", "max_iter", "failed"), step.qp2_status
        assert np.isfinite(step.task_progress)
        assert abs(float(step.u_alloc)) < 1.0e-9
        np.testing.assert_allclose(step.v_tcp_estimated, step.v_cmd_feasible, atol=1.0e-5)
        assert float(step.qpik_hard_residual_max) <= 2.0e-5
        assert np.isfinite(step.qpik_total_ms)
        # This is a regression guard against a stuck child or an accidental
        # unbounded solve, not a hardware real-time certification.
        assert float(step.qpik_total_ms) < 100.0
        steps.append(step)
        q = step.q_send.copy()
    return steps


@pytest.mark.parametrize("mode_name", ["coupled", "locked"])
def test_native_pure_z_progress_matches_python_without_false_pause(mode_name: str) -> None:
    """The same ideal q_meas/rail feedback reaches full progress in 7/8 DoF."""
    py = _runtime_controller(native=False)
    native = _runtime_controller(native=True)
    try:
        py_steps = _run_pure_z(py, mode_name=mode_name)
        native_steps = _run_pure_z(native, mode_name=mode_name)
        assert py_steps[-1].task_progress > 0.99
        assert native_steps[-1].task_progress > 0.99

        # Compare physical outputs and completed residual progress.  Native
        # and Python may pick different QP2 preference points in the same
        # locked QP1 task.
        py_vy = max(abs(float(s.v_tcp_estimated[1])) for s in py_steps[-20:])
        native_vy = max(abs(float(s.v_tcp_estimated[1])) for s in native_steps[-20:])
        assert py_vy < 2.0e-4
        assert native_vy < 2.0e-4
        py_alpha = np.asarray([s.task_progress for s in py_steps])
        native_alpha = np.asarray([s.task_progress for s in native_steps])
        np.testing.assert_allclose(native_alpha[-1], py_alpha[-1], atol=3.0e-3, rtol=0.0)
        assert np.all(native_alpha >= -1.0e-9)
        assert np.max(native_alpha) > 0.99
        py_rail = np.asarray([s.qdot[0] for s in py_steps])
        native_rail = np.asarray([s.qdot[0] for s in native_steps])
        assert float(np.max(np.abs(py_rail))) < 2.0e-3
        assert float(np.max(np.abs(native_rail))) < 2.0e-3
        native_times = np.asarray([s.qpik_total_ms for s in native_steps], dtype=float)
        p95, p99 = np.percentile(native_times, [95.0, 99.0])
        # Report measured percentiles in a failure message so a CI log keeps
        # the observed budget evidence without pretending this is a hard RT
        # certification.  The max guard catches a stuck native child.
        assert float(p99) < 20.0, f"native QPIK timing p95={p95:.3f}ms p99={p99:.3f}ms"
        assert float(np.max(native_times)) < 100.0
    finally:
        _close_runtime_controller(native)
        _close_runtime_controller(py)


def test_native_dstar_motion_preserves_zero_task_and_preview_certificate() -> None:
    """A slow d* change may move the rail while the requested TCP stays zero."""
    ctrl = _runtime_controller(native=True, psi_enabled=True)
    try:
        q0 = _safe_q()
        ctrl.reset(q0)
        ctrl.enable()
        from rm75_control.control.joint_admittance_8dof.tasks.rail_mode import RailMode

        ctrl.set_rail_mode(RailMode.COUPLED)
        ctrl.begin_hybrid_episode(q0, np.zeros(8))
        d_live = float(ctrl.kin.fk_pose(q0)[1] - q0[0])
        psi = float(ctrl.posture_retarget.psi_star_rad)
        # This uses the same native client command as a planned stroke.  It is
        # an offline setpoint change; no rail or arm is physically commanded.
        ctrl._native.set_stroke(d_live + 0.010, psi)
        q = q0.copy()
        travel = 0.0
        steps = []
        for _ in range(100):
            step = ctrl.update(
                np.zeros(6),
                0.005,
                q_meas=q,
                rail_exec_vel_m_s=float(ctrl.core.qdot_prev[0]),
                rail_refresh_dt_s=0.020,
            )
            assert not step.task_paused, step.task_pause_reason
            np.testing.assert_allclose(step.v_tcp_estimated, 0.0, atol=2.0e-5)
            assert float(step.qpik_hard_residual_max) <= 2.0e-5
            assert float(step.qpik_total_ms) < 100.0
            travel += abs(float(step.qdot[0])) * 0.005
            steps.append(step)
            q = step.q_send.copy()
        assert travel > 1.0e-4
        assert max(abs(float(s.v_tcp_estimated[1])) for s in steps) < 2.0e-5
        assert float(np.median([s.qpik_total_ms for s in steps])) < 20.0
    finally:
        _close_runtime_controller(ctrl)


def test_native_ptp_prepare_then_commit_advances_q_cmd() -> None:
    """Direct PTP with auto_commit=False must not freeze Python q_cmd."""
    ctrl = _runtime_controller(native=True)
    try:
        q0 = _safe_q()
        ctrl.reset(q0)
        ctrl.enable()
        from rm75_control.control.joint_admittance_8dof.tasks.rail_mode import RailMode

        ctrl.set_rail_mode(RailMode.COUPLED)
        ctrl.set_direct_joint_ptp(True)
        ctrl.set_plan_drives_rail(True)
        qdot_ff = np.zeros(8)
        qdot_ff[0] = 0.02
        qdot_ff[4] = 0.05
        q_before = ctrl.q_cmd.copy()
        step = ctrl.update(
            np.zeros(6),
            0.005,
            q_meas=q0,
            qdot_ff=qdot_ff,
            auto_commit=False,
            commit_history=False,
        )
        assert step.controller_mode == "direct_joint_ptp"
        assert step.qp1_status in ("solved", "max_iter")
        assert not step.task_paused
        np.testing.assert_allclose(ctrl.q_cmd, q_before, atol=1e-12)
        assert float(np.max(np.abs(step.qdot))) > 1.0e-6
        ctrl.commit_publication(step.qdot)
        np.testing.assert_allclose(ctrl.q_cmd, step.q_send, atol=1e-9)
        np.testing.assert_allclose(ctrl.live_qdot(), step.qdot, atol=1e-9)
        assert float(ctrl._last_q_meas[0]) == pytest.approx(float(q0[0]))
    finally:
        _close_runtime_controller(ctrl)


def test_native_fast_twist_does_not_false_pause() -> None:
    """A large SERVO_TWIST request must publish a residual, not latch ESTOP."""
    ctrl = _runtime_controller(native=True)
    try:
        q0 = _safe_q()
        ctrl.reset(q0)
        ctrl.enable()
        from rm75_control.control.joint_admittance_8dof.tasks.rail_mode import RailMode

        ctrl.set_rail_mode(RailMode.COUPLED)
        ctrl.begin_hybrid_episode(q0, np.zeros(8))
        q = q0.copy()
        for _ in range(8):
            step = ctrl.update(
                np.array([0.12, 0.08, 0.05, 0.25, 0.25, 0.25]),
                0.005,
                q_meas=q,
                rail_exec_vel_m_s=float(ctrl.core.qdot_prev[0]),
                rail_refresh_dt_s=0.020,
            )
            assert not step.solver_fault_latched, step.fallback_reason
            assert step.fallback_level != "stop", step.fallback_reason
            assert not step.task_paused, step.task_pause_reason
            assert step.qp1_status in ("solved", "max_iter"), step.qp1_status
            q = step.q_send.copy()
    finally:
        _close_runtime_controller(ctrl)
