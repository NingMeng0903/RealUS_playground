from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from rm75_control.control.joint_admittance_8dof.collision_model import CollisionConfig
from rm75_control.control.joint_admittance_8dof.loop import (
    JointIkConfig,
    JointIkController,
)
from rm75_control.control.joint_admittance_8dof.model import (
    RobotKinematics,
    full_q_from_arm,
)
from rm75_control.control.joint_admittance_8dof.solver.qp_builder import QpConfig
from rm75_control.control.joint_admittance_8dof.solver.cbf_constraints import CbfRows
from rm75_control.control.joint_admittance_8dof.solver import qp_builder as qp_builder_module


Q_SAFE = full_q_from_arm(
    np.deg2rad([5.0, -30.0, 10.0, 60.0, -5.0, 45.0, 0.0]), 0.40
)


def _controller() -> JointIkController:
    collision = CollisionConfig(enabled=False)
    qp = QpConfig(
        backend="proxqp",
        collision=collision,
        smoothness_weight=np.r_[0.0, np.full(7, 0.15)],
    )
    qp.j4_design_comfort.enabled = False
    cfg = JointIkConfig(control_frame="base", qp=qp, collision=collision)
    controller = JointIkController(RobotKinematics(), cfg)
    controller.reset(Q_SAFE)
    return controller


def test_rail_jerk_box_moves_all_feasible_task_residual_to_arm() -> None:
    controller = _controller()
    core = controller.core
    jacobian = controller.kin.jacobian(Q_SAFE)
    task = jacobian[:, 1] * 0.004
    resync = np.r_[0.02, np.full(7, 0.10)]
    lo, hi = core.constraints.bounds(
        Q_SAFE,
        0.005,
        core.qdot_prev,
        q_meas=Q_SAFE,
        q_cmd=Q_SAFE,
        resync_err=resync,
        qdot_prev2=core.qdot_prev2,
        j_max=core._j_max,
    )

    result = core.step(
        Q_SAFE,
        task,
        0.005,
        q_meas=Q_SAFE,
        resync_err=resync,
        rail_exec_vel_m_s=0.0,
        rail_task_vel_m_s=0.08,
        rail_task_weight=1.0e5,
        zero_secondary_rail=True,
        jacobian=jacobian,
        sigma=controller.kin.singular_values(jacobian),
    )

    assert result.qdot[0] <= hi[0] + 2.0e-6
    assert not core.last_failed
    np.testing.assert_allclose(
        core.last_task_achieved,
        task,
        atol=1.0e-4,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        core.last_qp2_residual,
        core.last_qp1_residual,
        atol=2.0e-3,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        core.last_arm_contrib,
        task,
        atol=1.0e-3,
        rtol=0.0,
    )


def test_rail_task_vel_is_clipped_into_velocity_box() -> None:
    controller = _controller()
    core = controller.core
    jacobian = controller.kin.jacobian(Q_SAFE)
    task = jacobian[:, 1] * 0.004
    resync = np.r_[0.02, np.full(7, 0.10)]
    core.step(
        Q_SAFE,
        task,
        0.005,
        q_meas=Q_SAFE,
        resync_err=resync,
        rail_exec_vel_m_s=0.0,
        rail_task_vel_m_s=0.08,
        rail_task_weight=1.0e5,
        zero_secondary_rail=True,
        jacobian=jacobian,
        sigma=controller.kin.singular_values(jacobian),
    )
    lo = float(core.last_rail_box_lo)
    hi = float(core.last_rail_box_hi)
    used = float(core.last_rail_task_vel_used)
    assert np.isfinite(lo) and np.isfinite(hi) and np.isfinite(used)
    assert lo - 1.0e-9 <= used <= hi + 1.0e-9
    assert used != pytest.approx(0.08, abs=1.0e-4)


def test_measured_rail_recenter_is_cancelled_by_arm() -> None:
    controller = _controller()
    core = controller.core
    jacobian = controller.kin.jacobian(Q_SAFE)
    rail_velocity = 0.02
    arm_seed = np.linalg.lstsq(
        jacobian[:, 1:], -jacobian[:, 0] * rail_velocity, rcond=None
    )[0]
    core.sync_applied(np.r_[rail_velocity, arm_seed])

    core.step(
        Q_SAFE,
        np.zeros(6),
        0.005,
        q_meas=Q_SAFE,
        rail_exec_vel_m_s=rail_velocity,
        rail_task_vel_m_s=rail_velocity,
        rail_task_weight=1.0e3,
        zero_secondary_rail=True,
        jacobian=jacobian,
        sigma=controller.kin.singular_values(jacobian),
    )

    np.testing.assert_allclose(
        core.last_rail_exec_contrib + core.last_arm_contrib,
        np.zeros(6),
        atol=1.0e-4,
        rtol=0.0,
    )
    assert np.max(np.abs(core.last_task_residual)) <= 1.0e-4


def test_qp2_failure_uses_same_tick_qp1_not_previous_velocity() -> None:
    controller = _controller()
    core = controller.core
    jacobian = controller.kin.jacobian(Q_SAFE)
    task = jacobian[:, 1] * 0.004
    previous = np.r_[0.001, np.zeros(7)]
    core.sync_applied(previous)
    orig_solve = core._solve_qp

    def _fail_qp2(backend, *args, **kwargs):
        if backend is core._backend_qp2:
            return None
        return orig_solve(backend, *args, **kwargs)

    core._solve_qp = _fail_qp2
    try:
        result = core.step(
            Q_SAFE,
            task,
            0.005,
            q_meas=Q_SAFE,
            rail_exec_vel_m_s=0.0,
            jacobian=jacobian,
            sigma=controller.kin.singular_values(jacobian),
        )
    finally:
        core._solve_qp = orig_solve

    assert core.last_qp2_fallback
    assert np.linalg.norm(result.qdot) > 1.0e-4
    assert not np.allclose(result.qdot, 0.85 * previous)
    assert result.qdot[0] == pytest.approx(core.last_qdot_qp1[0], abs=2.0e-7)
    np.testing.assert_allclose(
        core.last_qp2_residual,
        core.last_qp1_residual,
        atol=1.0e-9,
        rtol=0.0,
    )


def test_qp2_failure_follows_nonzero_rail_macro() -> None:
    controller = _controller()
    core = controller.core
    jacobian = controller.kin.jacobian(Q_SAFE)
    task = jacobian[:, 1] * 0.004
    orig_solve = core._solve_qp

    def _fail_qp2(backend, *args, **kwargs):
        if backend is core._backend_qp2:
            return None
        return orig_solve(backend, *args, **kwargs)

    core._solve_qp = _fail_qp2
    result = core.step(
        Q_SAFE,
        task,
        0.005,
        q_meas=Q_SAFE,
        rail_exec_vel_m_s=0.0,
        rail_task_vel_m_s=-0.06,
        rail_task_weight=1.0e3,
        zero_secondary_rail=True,
        jacobian=jacobian,
        sigma=controller.kin.singular_values(jacobian),
    )
    assert core.last_qp2_fallback
    np.testing.assert_allclose(result.qdot, core.last_qdot_qp1, atol=1e-10)
    assert core.validate_final_qdot(result.qdot)[0] <= 1e-5


def test_final_publication_certificate_detects_post_qp_rewrite() -> None:
    controller = _controller()
    core = controller.core
    jacobian = controller.kin.jacobian(Q_SAFE)
    task = jacobian[:, 1] * 0.004
    result = core.step(
        Q_SAFE,
        task,
        0.005,
        q_meas=Q_SAFE,
        rail_exec_vel_m_s=0.0,
        jacobian=jacobian,
        sigma=controller.kin.singular_values(jacobian),
    )

    hard, task_lock = core.validate_final_qdot(result.qdot)
    assert hard <= 1.0e-5
    assert task_lock <= 1.0e-5

    changed_arm = result.qdot.copy()
    changed_arm[1] += 0.01
    _hard, changed_lock = core.validate_final_qdot(changed_arm)
    assert changed_lock > 1.0e-5

    outside_box = result.qdot.copy()
    outside_box[0] = float(core.last_hi_box[0]) + 0.01
    changed_hard, _lock = core.validate_final_qdot(outside_box)
    assert changed_hard >= 0.009


def test_qp_config_rejects_nonpositive_task_weights() -> None:
    controller = _controller()
    limits = controller.limits
    bad = QpConfig(
        backend="proxqp",
        collision=CollisionConfig(enabled=False),
        task_weight=np.array([100.0, 100.0, 0.0, 50.0, 50.0, 50.0]),
    )
    with np.testing.assert_raises_regex(ValueError, "strictly positive"):
        qp_builder_module.QpIkController(
            controller.kin, limits, bad, collision=None
        )


def test_truly_infeasible_task_reports_true_six_axis_residual() -> None:
    controller = _controller()
    core = controller.core
    jacobian = controller.kin.jacobian(Q_SAFE)
    task = np.array([0.8, -0.6, 0.5, 1.0, -0.8, 0.7])

    core.step(
        Q_SAFE,
        task,
        0.005,
        q_meas=Q_SAFE,
        rail_exec_vel_m_s=0.0,
        jacobian=jacobian,
        sigma=controller.kin.singular_values(jacobian),
    )

    achieved = core.last_rail_exec_contrib + core.last_arm_contrib
    np.testing.assert_allclose(
        core.last_task_residual,
        task - achieved,
        atol=2.0e-6,
        rtol=0.0,
    )
    assert np.max(np.abs(core.last_task_residual)) > 1.0e-4


def test_secondary_macro_preferences_and_cbf_cannot_change_qp1(monkeypatch) -> None:
    def fake_cbf(*_args, **_kwargs) -> CbfRows:
        row = np.zeros((1, 8), dtype=float)
        row[0, 1] = 1.0
        return CbfRows(
            jacobian=row,
            lower=np.array([0.001]),
            names=("self_collision:test_a:test_b",),
        )

    monkeypatch.setattr(qp_builder_module, "build_cbf_rows", fake_cbf)
    plain = _controller()
    stressed = _controller()
    for core in (plain.core, stressed.core):
        core.collision_cfg.enabled = True
        core.collision = SimpleNamespace(closest_pair=lambda: None)
    jacobian = plain.kin.jacobian(Q_SAFE)
    task = jacobian[:, 2] * 0.004

    plain.core.step(
        Q_SAFE,
        task,
        0.005,
        q_meas=Q_SAFE,
        secondary_qdot=np.zeros(8),
        rail_exec_vel_m_s=0.0,
        jacobian=jacobian,
        sigma=plain.kin.singular_values(jacobian),
    )
    stressed.core.step(
        Q_SAFE,
        task,
        0.005,
        q_meas=Q_SAFE,
        secondary_qdot=np.full(8, 0.2),
        rail_exec_vel_m_s=0.0,
        rail_task_vel_m_s=0.08,
        rail_task_weight=1.0e8,
        jacobian=jacobian,
        sigma=stressed.kin.singular_values(jacobian),
    )

    np.testing.assert_allclose(
        plain.core.last_qp1_residual,
        stressed.core.last_qp1_residual,
        atol=2.0e-7,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        jacobian[:, 1:] @ plain.core.last_qdot_qp1[1:],
        jacobian[:, 1:] @ stressed.core.last_qdot_qp1[1:],
        atol=2.0e-7,
        rtol=0.0,
    )


def test_qp2_rail_outside_box_publishes_qp1_xstar() -> None:
    controller = _controller()
    core = controller.core
    jacobian = controller.kin.jacobian(Q_SAFE)
    task = jacobian[:, 1] * 0.004
    orig = core._solve_qp

    def _leak_rail(backend, *args, **kwargs):
        x = orig(backend, *args, **kwargs)
        if backend is core._backend_qp2 and x is not None:
            x = np.asarray(x, dtype=float).copy()
            x[0] = 0.08194
            return x
        return x

    core._solve_qp = _leak_rail
    try:
        result = core.step(
            Q_SAFE,
            task,
            0.005,
            q_meas=Q_SAFE,
            rail_exec_vel_m_s=0.0,
            rail_task_vel_m_s=0.08,
            rail_task_weight=1.0e5,
            zero_secondary_rail=True,
            jacobian=jacobian,
            sigma=controller.kin.singular_values(jacobian),
        )
    finally:
        core._solve_qp = orig

    lo = float(core.last_rail_box_lo)
    hi = float(core.last_rail_box_hi)
    used = float(core.last_rail_task_vel_used)
    assert result.qdot[0] != pytest.approx(0.08194, abs=1.0e-4)
    assert lo - 1.0e-9 <= result.qdot[0] <= hi + 1.0e-9
    assert result.qdot[0] == pytest.approx(core.last_qdot_qp1[0], abs=2.0e-7)
    np.testing.assert_allclose(result.qdot, core.last_qdot_qp1, atol=0.0, rtol=0.0)
    assert core.last_qp2_status == "uncertified"
    assert core.last_qp2_fallback


def test_l1_jerk_is_independent_of_qp_j_max_rail() -> None:
    controller = _controller()
    assert controller.cfg.qp.j_max_rail_m_s3 == pytest.approx(120.0)
    assert float(controller.rail_ref_model.j_max) == pytest.approx(60.0)


def test_j4_design_box_stays_hard_in_c() -> None:
    from pathlib import Path

    import yaml

    from rm75_control.control.joint_admittance_8dof.solver.joint_comfort import (
        J4DesignComfortBuilder,
        j4_design_qdot_bounds,
    )

    raw = yaml.safe_load(
        (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "joint_admittance_8dof.yaml"
        ).read_text()
    )
    jd = raw["inner"]["qp"]["j4_design_comfort"]
    assert bool(jd["enabled"])
    assert float(jd["lower_deg"]) == pytest.approx(70.0)
    assert float(jd["upper_deg"]) == pytest.approx(115.0)
    controller = _controller()
    controller.cfg.qp.j4_design_comfort.enabled = True
    controller.core.constraints.j4_design_enabled = True
    q = Q_SAFE.copy()
    q[4] = np.deg2rad(70.0)
    lo, hi = controller.core.constraints.bounds(q, 0.005)
    want_lo, _want_hi = j4_design_qdot_bounds(
        float(q[4]),
        lower_rad=np.deg2rad(70.0),
        upper_rad=np.deg2rad(115.0),
        gamma=float(controller.cfg.qp.j4_design_comfort.gamma),
    )
    assert lo[4] == pytest.approx(want_lo)
    assert lo[4] == pytest.approx(0.0, abs=1.0e-12)
    rows = J4DesignComfortBuilder().build_rows(q)
    assert rows.jacobian.shape[0] == 0


def test_cbf_uses_measured_not_commanded_rail_velocity(monkeypatch) -> None:
    def fake_cbf(*_args, **_kwargs) -> CbfRows:
        # Actual collision separation speed is qdot_rail + qdot_arm_1.
        row = np.zeros((1, 8), dtype=float)
        row[0, 0] = 1.0
        row[0, 1] = 1.0
        return CbfRows(
            jacobian=row,
            lower=np.array([0.03]),
            names=("self_collision:test_a:test_b",),
        )

    monkeypatch.setattr(qp_builder_module, "build_cbf_rows", fake_cbf)
    controller = _controller()
    core = controller.core
    core.collision_cfg.enabled = True
    core.collision = SimpleNamespace(closest_pair=lambda: None)
    jacobian = np.zeros((6, 8), dtype=float)
    jacobian[:, 1:7] = np.eye(6)
    jacobian[0, 0] = 1.0

    result = core.step(
        Q_SAFE,
        np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0]),
        0.1,
        q_meas=Q_SAFE,
        rail_exec_vel_m_s=-0.02,
        jacobian=jacobian,
        sigma=np.ones(6),
    )

    #  -0.02 + qdot_arm_1 >= 0.03  ->  qdot_arm_1 >= 0.05.
    assert result.qdot[1] >= 0.05 - 5.0e-6
    assert not core.last_failed
    assert core.validate_final_qdot(result.qdot)[0] <= 1e-5
