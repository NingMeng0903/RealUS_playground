"""Fixed two-level ProxQP call budget and retained motion regressions."""

from __future__ import annotations

import numpy as np
import pytest

from rm75_control.control.joint_admittance_8dof.collision_model import CollisionConfig
from rm75_control.control.joint_admittance_8dof.generic_runtime import (
    GenericQpikRuntimeConfig,
)
from rm75_control.control.joint_admittance_8dof.loop import JointIkConfig, JointIkController
from rm75_control.control.joint_admittance_8dof.model import RobotKinematics, full_q_from_arm
from rm75_control.control.joint_admittance_8dof.reference import SrsSmoothMoveReference
from rm75_control.control.joint_admittance_8dof.solver.two_level_qpik import (
    TwoLevelQpikConfig,
)


Q_HOME = full_q_from_arm(
    np.deg2rad([5.0, -30.0, 10.0, 60.0, -5.0, 45.0, 0.0]), 0.4
)


def _controller(backend: str = "proxqp") -> JointIkController:
    cfg = JointIkConfig(
        generic_qpik=GenericQpikRuntimeConfig(
            solver=TwoLevelQpikConfig(
                backend=backend,
                max_iter=200,
                max_rows=96,
                max_scalable_groups=4,
            )
        ),
        collision=CollisionConfig(enabled=False),
    )
    ctrl = JointIkController(RobotKinematics(), cfg)
    ctrl.reset(Q_HOME)
    return ctrl


def test_proxqp_has_exactly_one_call_per_level_and_no_retry() -> None:
    ctrl = _controller("proxqp")
    qp1 = ctrl.core.solver.backend_qp1
    qp2 = ctrl.core.solver.backend_qp2
    before = (qp1.solve_count, qp2.solve_count)
    step = ctrl.update(
        np.array([0.006, -0.003, 0.002, 0.0, 0.0, 0.0]),
        q_meas=Q_HOME,
    )
    # QP1 once; QP2 (max α) + QP3 (posture) share backend_qp2 → two level-2 calls.
    assert (qp1.solve_count - before[0], qp2.solve_count - before[1]) == (1, 2)
    assert step.qp1_status == "solved"
    # QP2 may soft-fail and fall back to same-tick QP1 — no ProxQP retry storm.
    assert not step.solver_fault_latched
    assert step.qp2_status == "solved" or step.fallback_level == "qp1"


def test_qp1_failure_is_p0_safe_not_previous_velocity_decay() -> None:
    ctrl = _controller("scipy")
    previous = np.full(8, 0.05)
    ctrl.core.sync_applied(previous)
    backend = ctrl.core.solver.backend_qp1
    real_solve = backend.solve
    backend.solve = lambda *args, **kwargs: None  # type: ignore[method-assign]
    try:
        step = ctrl.update(np.zeros(6), q_meas=Q_HOME)
    finally:
        backend.solve = real_solve  # type: ignore[method-assign]
    assert not step.solver_fault_latched
    assert step.fallback_level == "p0_safe"
    assert step.fallback_reason.startswith("qp1_failed_p0_")
    assert not np.allclose(step.qdot, previous)


def test_srs_midpath_ik_none_raises_after_streak(monkeypatch):
    """Consecutive srs_ik=None must raise instead of silently holding."""
    kin = RobotKinematics()
    q0 = Q_HOME.copy()
    pose_tgt = kin.fk_pose(q0)
    pose_tgt[0] += 0.05
    ref = SrsSmoothMoveReference(
        kin,
        q0,
        pose_tgt,
        y_rail_target_m=float(q0[0]),
        psi_target_rad=0.0,
        duration_s=2.0,
        max_ik_fail_streak=3,
    )

    import rm75_control.kinematics.srs_ik as srs_mod

    monkeypatch.setattr(srs_mod, "srs_ik", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="srs_ik returned None"):
        for _ in range(5):
            ref._q_at(0.5)


def test_direct_joint_ptp_skips_qpik() -> None:
    ctrl = _controller("scipy")
    ctrl.set_direct_joint_ptp(True)
    ctrl.set_plan_drives_rail(True)
    qdot_ff = np.array([0.05, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    called = {"n": 0}
    real_solve = ctrl.core.solve

    def _boom(*args, **kwargs):
        called["n"] += 1
        return real_solve(*args, **kwargs)

    ctrl.core.solve = _boom  # type: ignore[method-assign]
    try:
        result = ctrl.update(np.zeros(6), q_meas=Q_HOME, qdot_ff=qdot_ff)
    finally:
        ctrl.core.solve = real_solve  # type: ignore[method-assign]
    assert called["n"] == 0
    assert np.isfinite(result.qdot).all()
    assert abs(float(result.qdot[0]) - 0.05) < 1e-6 or abs(float(result.rail_vel_pin) - 0.05) < 1e-6


def test_plan_joint_wraps_revolute_delta() -> None:
    from rm75_control.control.joint_admittance_8dof.api import SecondaryPolicy
    from rm75_control.control.joint_admittance_8dof.reference import JointSmoothMoveReference

    kin = RobotKinematics()
    q0 = Q_HOME.copy()
    q_tgt = q0.copy()
    q_tgt[1] = q0[1] + np.deg2rad(170.0)
    ref = JointSmoothMoveReference(kin, q0, q_tgt, 4.0)

    class _Inner:
        def __init__(self) -> None:
            self.q_cmd = q0.copy()

    ff = SecondaryPolicy(qdot_ff="plan_joint").make_qdot_ff_provider(_Inner(), ref)
    assert ff is not None
    assert ff(2.0)[1] > 0.0
