"""ProxQP retry + one-policy low-σ behaviour (8-DOF)."""

from __future__ import annotations

import numpy as np
import pytest

from rm75_control.control.joint_admittance_8dof.collision_model import CollisionConfig
from rm75_control.control.joint_admittance_8dof.loop import JointIkConfig, JointIkController
from rm75_control.control.joint_admittance_8dof.model import RobotKinematics, full_q_from_arm
from rm75_control.control.joint_admittance_8dof.reference import (
    SrsSmoothMoveReference,
    _SRS_IK_FAIL_STREAK_LIMIT,
)
from rm75_control.control.joint_admittance_8dof.solver.qp_builder import (
    QpConfig,
    QpIkController,
    _ProxQpWbcBackend,
)
from rm75_control.control.joint_admittance_8dof.tasks.manipulability_task import (
    ManipulabilityTaskConfig,
)
from rm75_control.control.joint_admittance_8dof.utils.safety import SafetyLimits

Q_HOME = full_q_from_arm(
    np.deg2rad([5.0, -30.0, 10.0, 60.0, -5.0, 45.0, 0.0]), 0.4
)
Q_ARM_SINGULAR = np.array(
    [0.0, np.pi / 2 - 0.1, 0.0, 0.15, 0.0, 0.15, 0.0], dtype=float
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
            warn_on_fail=False,
        ),
    )
    if ctrl.backend_name != "proxqpwbc":
        pytest.skip("proxsuite not available")
    backend = ctrl.backend
    assert isinstance(backend, _ProxQpWbcBackend)
    assert hasattr(backend, "_max_iter")
    assert int(backend._max_iter) == 3000
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


def test_low_sigma_scan_enables_manipulability():
    """COUPLED scan (centering on): σ < sigma_ref forces manipulability_active."""
    kin = RobotKinematics()
    cfg = JointIkConfig(
        qp=QpConfig(
            task_weight=np.array([100.0, 100.0, 100.0, 50.0, 50.0, 50.0]),
            collision=CollisionConfig(enabled=False),
            warn_on_fail=False,
        ),
        manipulability=ManipulabilityTaskConfig(k_mu=0.8),
    )
    ctrl = JointIkController(kin, cfg)
    q_sing = full_q_from_arm(Q_ARM_SINGULAR, 0.4)
    sigma = float(kin.singular_values(kin.jacobian(q_sing)).min())
    assert sigma < float(cfg.qp.sr_damping.sigma_ref)

    ctrl.reset(q_sing)
    ctrl.set_coupled()
    ctrl.set_centering_suppressed(False)
    ctrl.set_manipulability_active(False)
    ctrl.set_rail_extension_mode("reach")
    ctrl.capture_rail_extension_ref()
    ctrl.set_rail_extension_active(True)

    captured: dict = {}
    real_secondary = ctrl._secondary

    def _spy(q, qdot_ff=None, **kwargs):
        captured.update(kwargs)
        return real_secondary(q, qdot_ff, **kwargs)

    ctrl._secondary = _spy  # type: ignore[method-assign]
    try:
        ctrl.update(np.zeros(6), q_meas=q_sing)
    finally:
        ctrl._secondary = real_secondary  # type: ignore[method-assign]

    assert captured.get("manipulability_active") is True


def test_centering_survives_when_manip_armed():
    """Low-σ manip must not replace the joint attractor (J1 unwind)."""
    from rm75_control.control.joint_admittance_8dof.tasks.nullspace_task import (
        JointCenteringTask,
        NullspaceTaskConfig,
    )
    from rm75_control.control.joint_admittance_8dof.tasks.manipulability_task import (
        ManipulabilityTask,
    )
    from rm75_control.control.joint_admittance_8dof.tasks.secondary_composer import (
        SecondaryComposer,
    )

    kin = RobotKinematics()
    q_nom = np.zeros(kin.nv)
    q_nom[1] = 0.0
    cfg = NullspaceTaskConfig(
        k_center=1.0,
        k_limit=0.0,
        q_nominal_rad=q_nom,
        weights=np.ones(kin.nv),
    )
    centering = JointCenteringTask.from_kinematics(kin, cfg)
    manip = ManipulabilityTask(kin, ManipulabilityTaskConfig(k_mu=0.8))
    composer = SecondaryComposer(centering, None, manipulability=manip, max_qdot_frac=0.0)
    q = q_nom.copy()
    q[1] = np.deg2rad(170.0)  # twisted J1
    qdot = composer.compose(
        q,
        None,
        None,
        arm_suppressed=True,
        centering_suppressed=False,
        manipulability_active=True,
        centering_sigma_fade=False,
        sigma_min=0.05,
        sigma_ref=0.08,
    )
    # Centering alone wants −J1; combined stack must still unwind J1.
    assert qdot[1] < -0.05


def test_srs_midpath_ik_none_fails_loud_without_q_target(monkeypatch):
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
    )

    import rm75_control.kinematics.srs_ik as srs_mod

    monkeypatch.setattr(srs_mod, "srs_ik", lambda *a, **k: None)

    with pytest.raises(RuntimeError, match="SRS mid-path IK failed"):
        for i in range(_SRS_IK_FAIL_STREAK_LIMIT):
            ref.sample_q(0.1 * i)


def test_srs_midpath_ik_none_uses_joint_geodesic_ff(monkeypatch):
    """With q_target, analytic holes seed qdot_ff via joint wrap — no abort."""
    kin = RobotKinematics()
    q0 = Q_HOME.copy()
    q_tgt = q0.copy()
    q_tgt[1] += np.deg2rad(30.0)
    pose_tgt = kin.fk_pose(q_tgt)
    ref = SrsSmoothMoveReference(
        kin,
        q0,
        pose_tgt,
        y_rail_target_m=float(q_tgt[0]),
        psi_target_rad=0.0,
        duration_s=2.0,
        q_target_rad=q_tgt,
    )

    import rm75_control.kinematics.srs_ik as srs_mod

    monkeypatch.setattr(srs_mod, "srs_ik", lambda *a, **k: None)

    # At t=duration, s=1 → joint geodesic reaches q_target.
    q_end, _ = ref.sample_q(2.0)
    np.testing.assert_allclose(q_end, q_tgt, atol=1e-9)
    # Mid-path progresses toward target (not frozen at start).
    q_mid, _ = ref.sample_q(1.0)
    assert float(np.abs(q_mid[1] - q0[1])) > np.deg2rad(5.0)
    # Cartesian primary still comes from line-slerp, not FK(joint).
    pose_ref = ref.sample(1.0).pose_d
    assert pose_ref.shape == (6,)
    # No raise after many samples through the hole.
    for i in range(_SRS_IK_FAIL_STREAK_LIMIT + 3):
        ref.sample_q(0.05 * i)
