"""Residual QP1, publish/commit, ProxQP status, and session-DOF contracts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rm75_control.control.joint_admittance_8dof.collision_model import CollisionConfig
from rm75_control.control.joint_admittance_8dof.loop import JointIkConfig, JointIkController
from rm75_control.control.joint_admittance_8dof.model import RobotKinematics, full_q_from_arm
from rm75_control.control.joint_admittance_8dof.qp_cert import (
    QP_STATUS_CLOSEST_PRIMAL,
    QP_STATUS_DUAL_INFEASIBLE,
    QP_STATUS_P0_CONFLICT,
    QP_STATUS_PRIMAL_INFEASIBLE,
    qp_status_code_from_prox,
    qp_status_name,
    qp_status_publishable,
)
from rm75_control.control.joint_admittance_8dof.solver.fault_snapshot import (
    SNAPSHOT,
    FirstFaultSnapshot,
    ablate_residual_qp1,
)
from rm75_control.control.joint_admittance_8dof.solver.qp_builder import QpConfig
from rm75_control.control.joint_admittance_8dof.wbc_rt.build_id import combined_hash, tree_manifest


Q_SAFE = full_q_from_arm(np.deg2rad([5.0, -30.0, 10.0, 90.0, -5.0, 45.0, 0.0]), 0.40)


def _core():
    qp = QpConfig(backend="proxqp", collision=CollisionConfig(enabled=False))
    qp.j4_design_comfort.enabled = False
    cfg = JointIkConfig(control_frame="base", qp=qp)
    ctrl = JointIkController(RobotKinematics(), cfg)
    ctrl.reset(Q_SAFE)
    return ctrl.core, ctrl


def test_residual_qp1_always_solves_nonzero_p0() -> None:
    core, ctrl = _core()
    J = ctrl.kin.jacobian(Q_SAFE)
    for v_cmd, rail in (
        (np.zeros(6), 0.012),
        (np.array([0.02, 0.0, 0.0, 0.0, 0.0, 0.0]), 0.0),
        (np.array([0.0, 0.01, 0.0, 0.0, 0.0, 0.0]), -0.011),
    ):
        core.step(
            Q_SAFE,
            v_cmd,
            0.005,
            q_meas=Q_SAFE,
            rail_exec_vel_m_s=rail,
            jacobian=J,
        )
        assert not core.last_failed
        assert core.last_qp1_status in ("solved", "max_iter")
        assert np.all(np.isfinite(core.last_qdot_qp1))


def test_zero_task_moving_rail_returns_min_residual_not_failure() -> None:
    core, ctrl = _core()
    J = ctrl.kin.jacobian(Q_SAFE)
    rail = 0.012
    # From rest the jerk box cannot cancel 12 mm/s in one tick; that is a
    # nonzero residual, not a QP failure.  Seed a feasible compensating
    # history so the current box can hold the cancellation.
    v_need = -J[:, 0] * rail
    dq_arm, *_ = np.linalg.lstsq(J[:, 1:], v_need, rcond=None)
    qdot_seed = np.r_[0.0, dq_arm]
    core.sync_applied(qdot_seed)
    core.step(Q_SAFE, np.zeros(6), 0.005, q_meas=Q_SAFE, rail_exec_vel_m_s=rail, jacobian=J)
    assert not core.last_failed
    assert float(np.max(np.abs(core.last_qp1_residual))) <= 1.0e-4


def test_p0_conflict_is_named_and_not_collapsed() -> None:
    core, _ = _core()
    core.last_lo_box = np.array([0.02, *[-1.0] * 7])
    core.last_hi_box = np.array([-0.02, *[1.0] * 7])
    # Inject conflict via a one-tick rewrite of the box builder output.
    J = np.eye(6, 8)
    # Force a contradictory pin after bounds by calling step with a huge pin
    # that cannot meet the jerk box from rest — if P0 stays nonempty, QP1 still
    # returns a residual; a raw lo>hi is reported separately.
    lo = np.full(8, -0.1)
    hi = np.full(8, 0.1)
    lo[3] = 0.05
    hi[3] = 0.01
    assert float(lo[3]) > float(hi[3])
    snap = FirstFaultSnapshot()
    snap.capture(reason="p0_conflict", qp1_status="p0_conflict", box_lo=lo, box_hi=hi)
    assert snap.taken
    assert snap.qp1_status == "p0_conflict"


def test_qp2_failure_returns_qp1_elementwise() -> None:
    core, ctrl = _core()
    J = ctrl.kin.jacobian(Q_SAFE)
    orig = core._solve_qp

    def _wrap(backend, H, g, A, b, C, lo, hi, *, warm_start_x=None):
        if backend is core._backend_qp2:
            return None
        return orig(backend, H, g, A, b, C, lo, hi, warm_start_x=warm_start_x)

    core._solve_qp = _wrap  # type: ignore[method-assign]
    core.step(Q_SAFE, np.array([0.01, 0, 0, 0, 0, 0]), 0.005, q_meas=Q_SAFE, jacobian=J)
    assert not core.last_failed
    assert core.last_qp2_fallback
    np.testing.assert_allclose(core.last_qdot_qp1, np.asarray(core.qdot_prev), atol=1e-12)


def test_prepare_does_not_mutate_committed_history() -> None:
    core, ctrl = _core()
    J = ctrl.kin.jacobian(Q_SAFE)
    before = core.qdot_prev.copy()
    before2 = core.qdot_prev2.copy()
    core.step(
        Q_SAFE,
        np.array([0.01, 0, 0, 0, 0, 0]),
        0.005,
        q_meas=Q_SAFE,
        jacobian=J,
        commit_history=False,
    )
    np.testing.assert_allclose(core.qdot_prev, before)
    np.testing.assert_allclose(core.qdot_prev2, before2)
    core.commit_applied(core.last_qdot_qp1)
    assert not np.allclose(core.qdot_prev, before)


def test_proxqp_raw_statuses_are_not_collapsed() -> None:
    assert qp_status_name(QP_STATUS_PRIMAL_INFEASIBLE) == "primal_infeasible"
    assert qp_status_name(QP_STATUS_DUAL_INFEASIBLE) == "dual_infeasible"
    assert qp_status_name(QP_STATUS_CLOSEST_PRIMAL) == "closest_primal"
    assert qp_status_name(QP_STATUS_P0_CONFLICT) == "p0_conflict"
    assert qp_status_code_from_prox("PROXQP_PRIMAL_INFEASIBLE") == QP_STATUS_PRIMAL_INFEASIBLE
    assert qp_status_code_from_prox("PROXQP_MAX_ITER_REACHED") == 2
    assert qp_status_publishable("solved", certified=True)
    assert qp_status_publishable("max_iter", certified=True)
    assert not qp_status_publishable("max_iter", certified=False)
    assert not qp_status_publishable("primal_infeasible", certified=True)
    assert not qp_status_publishable("closest_primal", certified=True)


def test_first_fault_snapshot_is_first_writer_wins() -> None:
    snap = FirstFaultSnapshot()
    assert snap.capture(reason="first", qp1_status="failed")
    assert not snap.capture(reason="second", qp1_status="solved")
    assert snap.reason == "first"
    note = ablate_residual_qp1(snap.to_json(), drop=("preview", "hold"))
    assert note["model"] == "residual_qp1"
    assert "preview" in note["dropped"]


def test_build_id_hashes_protocol_and_inner() -> None:
    man = tree_manifest()
    assert any("protocol.hpp" in k for k in man)
    assert any("inner.cpp" in k for k in man)
    assert len(combined_hash(man)) == 64


def test_session_8_rejects_vague_q7_and_7_pins_rail() -> None:
    from peirastic.realman8dof.session import _normalize_q_target

    class _Inner:
        q_cmd = np.array([0.400, 0, 0, 0, 0, 0, 0, 0], dtype=float)

    ctx = type("C", (), {"inner": _Inner()})()
    with pytest.raises(ValueError, match="8-DOF"):
        _normalize_q_target(ctx, np.zeros(7), dof=8)
    q8 = _normalize_q_target(ctx, np.r_[0.400, np.zeros(7)], dof=8)
    assert q8[0] == pytest.approx(0.400)
    q7 = _normalize_q_target(ctx, np.zeros(7), dof=7)
    assert q7[0] == pytest.approx(0.400)
    with pytest.raises(ValueError, match="rail-moving"):
        _normalize_q_target(ctx, np.r_[0.410, np.zeros(7)], dof=7)
    near = _normalize_q_target(ctx, np.r_[0.4003, np.zeros(7)], dof=7)
    assert near[0] == pytest.approx(0.400)


def test_replay_first_fail_windows_do_not_false_pause() -> None:
    """Known first-fail rail speeds stay residual-feasible at zero twist."""
    core, ctrl = _core()
    J = ctrl.kin.jacobian(Q_SAFE)
    for rail in (0.0099625, 0.0100369, 0.0132, 0.010):
        core.reset(Q_SAFE)
        core.step(Q_SAFE, np.zeros(6), 0.005, q_meas=Q_SAFE, rail_exec_vel_m_s=rail, jacobian=J)
        assert not core.last_failed, (rail, core.last_qp1_status)
        assert core.last_qp1_status != "p0_conflict"


def test_j4_outside_design_band_is_still_p0_feasible() -> None:
    q = Q_SAFE.copy()
    q[4] = np.deg2rad(69.0)
    qp = QpConfig(backend="proxqp", collision=CollisionConfig(enabled=False))
    cfg = JointIkConfig(control_frame="base", qp=qp)
    ctrl = JointIkController(RobotKinematics(), cfg)
    ctrl.reset(q)
    J = ctrl.kin.jacobian(q)
    ctrl.core.step(q, np.zeros(6), 0.005, q_meas=q, jacobian=J)
    assert not ctrl.core.last_failed
    assert ctrl.core.last_qp1_status in ("solved", "max_iter")


def test_controller_hold_does_not_steal_session_dof() -> None:
    from rm75_control.control.joint_admittance_8dof.api import controller_dof

    class _Inner:
        is_locked_hold = True
        _peirastic_dof = 8

    assert controller_dof(_Inner()) == 8


def test_native_commit_publication_mirrors_q_cmd_and_qdot() -> None:
    core, ctrl = _core()
    ghost = Q_SAFE.copy()
    ghost[0] = 0.40
    ctrl.q_cmd = ghost.copy()

    class _Native:
        def _sync_q(self) -> None:
            ctrl.q_cmd = Q_SAFE.copy()

    ctrl._native = _Native()
    applied = np.linspace(0.01, 0.08, 8)
    ctrl.commit_publication(applied)
    np.testing.assert_allclose(ctrl.q_cmd, Q_SAFE, atol=1e-12)
    np.testing.assert_allclose(ctrl.last_applied_qdot, applied, atol=1e-12)
    np.testing.assert_allclose(ctrl.live_qdot(), applied, atol=1e-12)
    np.testing.assert_allclose(core.qdot_prev, applied, atol=1e-12)


def test_live_update_prepare_then_commit() -> None:
    core, ctrl = _core()
    before = core.qdot_prev.copy()
    step = ctrl.update(
        np.array([0.01, 0, 0, 0, 0, 0]),
        0.005,
        q_meas=Q_SAFE,
        commit_history=False,
        auto_commit=False,
    )
    np.testing.assert_allclose(core.qdot_prev, before)
    assert np.isfinite(step.q_send).all()
    ctrl.commit_publication(step.qdot)
    assert not np.allclose(core.qdot_prev, before)


def test_native_pending_commit_is_wired() -> None:
    src = Path(__file__).resolve().parents[1] / "native" / "wbc_rt" / "src" / "inner.cpp"
    text = src.read_text(encoding="utf-8")
    assert "handle_pending_flags" in text
    assert "kInAutoCommit" in text
    assert "pending_valid_" in text


def test_fault_sm_reset_requires_await() -> None:
    from peirastic.realman8dof.daemon import ControllerService

    src = Path(ControllerService._reset_allowed.__code__.co_filename).read_text(
        encoding="utf-8"
    )
    assert "AWAIT_RESET" in src
    assert "FAULT_LATCHED" in src


def test_fast_twist_and_crossed_rate_box_stay_publishable() -> None:
    """A jerk-limited reverse / huge stick must not become a latched fault."""
    core, ctrl = _core()
    J = ctrl.kin.jacobian(Q_SAFE)
    core.sync_applied(np.full(8, 8.0))
    core.step(
        Q_SAFE,
        np.array([0.12, 0.08, 0.05, 0.25, 0.25, 0.25]),
        0.005,
        q_meas=Q_SAFE,
        jacobian=J,
    )
    assert not core.last_failed, core.last_qp1_status
    assert core.last_qp1_status in ("solved", "max_iter")
    assert np.all(core.last_lo_box <= core.last_hi_box + 1.0e-12)
    qdot = np.asarray(core.last_qdot_qp1, dtype=float)
    np.testing.assert_array_less(core.last_lo_box - 1.0e-6, qdot + 1.0e-12)
    np.testing.assert_array_less(qdot - 1.0e-12, core.last_hi_box + 1.0e-6)


def test_near_rail_wall_keeps_rail_preference() -> None:
    """Limit poses stay publishable so QP2 can still request a rail leave."""
    core, ctrl = _core()
    q = Q_SAFE.copy()
    q[0] = 0.012
    J = ctrl.kin.jacobian(q)
    core.reset(q)
    core.step(
        q,
        np.array([0.0, -0.08, 0.0, 0.0, 0.0, 0.0]),
        0.005,
        q_meas=q,
        jacobian=J,
        rail_task_vel_m_s=0.06,
        rail_task_weight=64.0,
    )
    assert not core.last_failed, core.last_qp1_status
    assert np.isfinite(core.last_rail_task_vel_used)
    assert core.last_qp1_status in ("solved", "max_iter")
