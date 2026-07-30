"""ProxQP retry + one-policy low-σ behaviour (8-DOF)."""

from __future__ import annotations

import numpy as np
import pytest

from rm75_control.control.joint_admittance_8dof.collision_model import CollisionConfig
from rm75_control.control.joint_admittance_8dof.loop import JointIkConfig, JointIkController
from rm75_control.control.joint_admittance_8dof.model import RobotKinematics, full_q_from_arm
from rm75_control.control.joint_admittance_8dof.reference import (
    SrsSmoothMoveReference,
)
from rm75_control.control.joint_admittance_8dof.solver.qp_builder import (
    QpConfig,
    QpIkController,
    _ProxQpWbcBackend,
)
from rm75_control.control.joint_admittance_8dof.utils.safety import SafetyLimits

Q_HOME = full_q_from_arm(
    np.deg2rad([5.0, -30.0, 10.0, 60.0, -5.0, 45.0, 0.0]), 0.4
)

def test_proxqp_retry_uses_stored_max_iter_not_cfg():
    kin = RobotKinematics()
    limits = SafetyLimits.from_kinematics(kin, v_scale=0.5, a_max=20.0)
    ctrl = QpIkController(
        kin,
        limits,
        QpConfig(
            task_weight=np.array([100.0, 100.0, 100.0, 50.0, 50.0, 50.0]),
            collision=CollisionConfig(enabled=False),
            max_iter=3000,
            max_iter_cap=400,
            warn_on_fail=False,
        ),
    )
    if ctrl.backend_name != "proxqpwbc":
        pytest.skip("proxsuite not available")
    backend = ctrl.backend
    assert isinstance(backend, _ProxQpWbcBackend)
    assert hasattr(backend, "_max_iter")
    assert int(backend._max_iter) == 400  # clamped by max_iter_cap
    assert not hasattr(backend, "cfg")

    # Priming solve so _initialized is True (update path hits retry).
    ctrl.step(Q_HOME, np.zeros(6), 0.005)
    assert backend._initialized

    n_solved = {"n": 0}
    real_solved = backend._solved

    def _flaky_solved():
        n_solved["n"] += 1
        # First check after the warm solve → False triggers retry path.
        if n_solved["n"] == 1:
            return False
        return real_solved()

    backend._solved = _flaky_solved  # type: ignore[method-assign]
    try:
        r = ctrl.step(
            Q_HOME, np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0]), 0.005
        )
    finally:
        backend._solved = real_solved  # type: ignore[method-assign]

    assert np.isfinite(r.qdot).all()
    assert int(backend.qp.settings.max_iter) == int(backend._max_iter)
    assert n_solved["n"] >= 2  # first fail + retry check


def test_srs_midpath_ik_none_raises_after_streak(monkeypatch):
    """Consecutive srs_ik=None must raise (no silent hold → governor freeze)."""
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


def test_fail_qdot_decay_uses_cfg_alpha():
    """Solver failure must apply fail_qdot_decay, not a hard 0.5 chop."""
    kin = RobotKinematics()
    limits = SafetyLimits.from_kinematics(kin, v_scale=0.5, a_max=20.0)
    cfg = QpConfig(
        task_weight=np.array([100.0, 100.0, 100.0, 50.0, 50.0, 50.0]),
        collision=CollisionConfig(enabled=False),
        warn_on_fail=False,
        fail_qdot_decay=0.85,
    )
    ctrl = QpIkController(kin, limits, cfg)
    if ctrl.backend_name != "proxqpwbc":
        pytest.skip("proxsuite not available")
    ctrl.qdot_prev = np.full(kin.nv, 0.2)
    ctrl.backend.solve = lambda *a, **k: None  # type: ignore[method-assign]
    r = ctrl.step(Q_HOME, np.zeros(6), 0.005)
    assert np.allclose(r.qdot, 0.85 * 0.2)


def test_direct_joint_ptp_skips_proxqp():
    """MoveJ path must integrate qdot_ff without calling ProxQP."""
    kin = RobotKinematics()
    cfg = JointIkConfig(
        qp=QpConfig(
            collision=CollisionConfig(enabled=False),
            warn_on_fail=False,
        ),
    )
    ctrl = JointIkController(kin, cfg)
    ctrl.reset(Q_HOME)
    ctrl.set_direct_joint_ptp(True)
    ctrl.set_plan_drives_rail(True)
    qdot_ff = np.array([0.05, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    called = {"n": 0}
    real_step = ctrl.core.step

    def _boom(*a, **k):
        called["n"] += 1
        return real_step(*a, **k)

    ctrl.core.step = _boom  # type: ignore[method-assign]
    try:
        r = ctrl.update(np.zeros(6), q_meas=Q_HOME, qdot_ff=qdot_ff)
    finally:
        ctrl.core.step = real_step  # type: ignore[method-assign]
    assert called["n"] == 0
    assert np.isfinite(r.qdot).all()
    assert abs(float(r.qdot[0]) - 0.05) < 1e-6 or abs(float(r.rail_vel_pin) - 0.05) < 1e-6


def test_plan_joint_wraps_revolute_delta():
    """plan_joint must use wrap_joint_delta (J1 ~180° must not take long way)."""
    from rm75_control.control.joint_admittance_8dof.api import SecondaryPolicy
    from rm75_control.control.joint_admittance_8dof.reference import (
        JointSmoothMoveReference,
    )

    kin = RobotKinematics()
    q0 = Q_HOME.copy()
    q_tgt = q0.copy()
    q_tgt[1] = q0[1] + np.deg2rad(170.0)
    ref = JointSmoothMoveReference(kin, q0, q_tgt, 4.0)

    class _Inner:
        def __init__(self) -> None:
            self.q_cmd = q0.copy()

    inner = _Inner()
    ff = SecondaryPolicy(qdot_ff="plan_joint").make_qdot_ff_provider(inner, ref)
    assert ff is not None
    qdot = ff(2.0)
    assert qdot[1] > 0.0