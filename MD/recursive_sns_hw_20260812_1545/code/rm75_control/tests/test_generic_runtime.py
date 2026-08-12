from types import SimpleNamespace

import numpy as np
import pytest

from rm75_control.control.joint_admittance_8dof.collision_model import CollisionConfig
from rm75_control.control.joint_admittance_8dof.generic_runtime import (
    GenericQpikRuntime,
    GenericQpikRuntimeConfig,
)
from rm75_control.control.joint_admittance_8dof.generic_tasks import (
    ProtectedTask,
    RobotState,
    ScalableTask,
)
from rm75_control.control.joint_admittance_8dof.solver.two_level_qpik import (
    TwoLevelQpikConfig,
)
from rm75_control.control.joint_admittance_8dof.utils.safety import SafetyLimits


class _Kinematics:
    nv = 7

    @staticmethod
    def jacobian(q):
        del q
        return np.eye(6, 7)


def _runtime() -> GenericQpikRuntime:
    limits = SafetyLimits(
        q_lower=np.full(7, -2.0),
        q_upper=np.full(7, 2.0),
        v_max=np.ones(7),
        a_max=np.full(7, 20.0),
        position_margin=np.zeros(7),
    )
    config = GenericQpikRuntimeConfig(
        solver=TwoLevelQpikConfig(
            backend="scipy",
            max_rows=64,
            max_scalable_groups=4,
            max_solve_ms=500.0,
        ),
        rail_indices=(),
        wrist_indices=(4, 5, 6),
    )
    return GenericQpikRuntime(
        _Kinematics(),
        limits,
        config,
        collision_config=CollisionConfig(enabled=False),
        damper_band=0.0,
    )


def _state() -> RobotState:
    return RobotState(
        q_meas=np.zeros(7),
        q_cmd=np.zeros(7),
        qdot_applied_prev=np.zeros(7),
        dt=0.005,
        contact_active=False,
    )


def test_direct_generic_rows_do_not_require_cartesian_axis_semantics() -> None:
    runtime = _runtime()
    protected = ProtectedTask(
        np.array([[0.2, -0.4, 0.3, 0.0, 0.1, 0.0, 0.5]]),
        np.array([0.08]),
        row_scales=np.array([0.1]),
        name="application_row",
    )
    scalable = ScalableTask(
        np.array([[0.0, 0.1, 0.0, 0.7, -0.2, 0.4, 0.0]]),
        np.array([0.12]),
        "arbitrary_group",
        slack_limits=np.array([0.02]),
    )
    result = runtime.solve_tasks(
        _state(), protected=protected, scalable=(scalable,)
    )
    assert result.solver.qp1.success
    assert result.solver.qp2.success
    np.testing.assert_allclose(
        protected.A @ result.qdot,
        result.solver.protected_locked_output,
        atol=7e-6,
    )
    assert "arbitrary_group" in result.solver.group_alphas


def test_runtime_reuses_one_measured_state_jacobian_snapshot() -> None:
    kin = _Kinematics()
    calls = 0

    def jacobian(q):
        nonlocal calls
        calls += 1
        return np.eye(6, 7)

    kin.jacobian = jacobian
    runtime = _runtime()
    runtime.kin = kin
    runtime.p0_builder.kin = kin
    protected = ProtectedTask(np.eye(7)[:1], [0.0])
    runtime.solve_tasks(_state(), protected=protected)
    assert calls == 1


def test_low_arm_health_keeps_alpha_feasible_without_fault() -> None:
    runtime = _runtime()
    for _ in range(3):
        runtime.health_monitor.update(arm_rho=0.01, dt=0.01)
    protected = ProtectedTask(np.array([[1.0, 0, 0, 0, 0, 0, 0]]), [0.05])
    scalable = ScalableTask(np.array([[0, 1.0, 0, 0, 0, 0, 0]]), [0.5], "motion")

    def _force_recovery_health(state, jacobian_base):
        del state, jacobian_base
        return runtime.health_monitor.update(arm_rho=0.01, dt=0.01)

    runtime._update_health = _force_recovery_health  # type: ignore[method-assign]
    result = runtime.solve_tasks(
        _state(), protected=protected, scalable=(scalable,)
    )
    assert result.health.state.name in {"RECOVERY", "SETTLING"}
    # α is decided by SNS feasibility (here accel-limited ≈0.2), not ρ→0.
    assert result.solver.group_alphas["motion"] == pytest.approx(0.2, abs=0.05)
    assert not result.solver.fault_latched
    assert result.solver.qp1.success


def test_dexterity_cbf_row_built_when_d_arm_low(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime()
    health = runtime.health_monitor.update(arm_rho=0.01, dt=0.01)
    monkeypatch.setattr(
        "rm75_control.control.joint_admittance_8dof.health_metrics.arm_dexterity_gradient",
        lambda *args, **kwargs: np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )
    rows = runtime._dexterity_and_branch_rows(
        _state(), health, np.eye(6, 7)
    )
    assert any(row.name == "arm_dexterity_cbf" for row in rows)


def test_arm_health_reg_scale_is_continuous_alpha_uncapped() -> None:
    runtime = _runtime()
    danger = float(runtime.config.health.arm_danger)
    warn = float(runtime.config.health.arm_warn)
    mid = 0.5 * (danger + warn)
    a0, r0 = runtime._authority_from_arm_health(danger)
    a1, r1 = runtime._authority_from_arm_health(mid)
    a2, r2 = runtime._authority_from_arm_health(warn)
    a_typ, r_typ = runtime._authority_from_arm_health(0.09)
    # α_cap is always 1.0; only reg_scale grows as ρ drops.
    assert a0 == pytest.approx(1.0)
    assert a1 == pytest.approx(1.0)
    assert a2 == pytest.approx(1.0)
    assert a_typ == pytest.approx(1.0)
    assert r0 > r1 > r2
    assert r_typ == pytest.approx(1.0)
    assert runtime._authority_from_arm_health(None) == (1.0, 1.0)


def test_joint_near_limit_keeps_alpha_and_leaves_limit() -> None:
    """Joint margin danger must not freeze XY; P3 near-limit soft cost leaves."""
    limits = SafetyLimits(
        q_lower=np.full(7, -2.0),
        q_upper=np.full(7, 2.0),
        v_max=np.ones(7),
        a_max=np.full(7, 20.0),
        position_margin=np.zeros(7),
    )
    config = GenericQpikRuntimeConfig(
        solver=TwoLevelQpikConfig(
            backend="scipy",
            max_rows=64,
            max_scalable_groups=4,
            max_solve_ms=500.0,
            margin_weight=5.0e-2,
            margin_weight_gain=40.0,
        ),
        rail_indices=(),
        wrist_indices=(4, 5, 6),
        working_arm_margin_rad=0.30,
        working_gamma=8.0,
    )
    runtime = GenericQpikRuntime(
        _Kinematics(),
        limits,
        config,
        collision_config=CollisionConfig(enabled=False),
        damper_band=0.15,
    )
    # Push joint 1 near its upper soft limit (limits are ±2.0).
    q = np.array([0.0, 1.92, 0.0, 0.0, 0.0, 0.0, 0.0])
    state = RobotState(
        q_meas=q,
        q_cmd=q.copy(),
        qdot_applied_prev=np.zeros(7),
        dt=0.005,
        contact_active=False,
    )
    protected = ProtectedTask(np.zeros((0, 7)), np.zeros(0), name="none")
    scalable = ScalableTask(
        np.array([[0, 0, 1.0, 0, 0, 0, 0]]), [0.0], "motion"
    )

    def _healthy(state_in, jacobian_base):
        del jacobian_base
        return runtime.health_monitor.update(
            arm_rho=0.2,
            joint_margin_rad=0.08,  # below 15° danger, above zero
            dt=state_in.dt,
        )

    runtime._update_health = _healthy  # type: ignore[method-assign]
    result = runtime.solve_tasks(state, protected=protected, scalable=(scalable,))
    assert result.health.state.name == "NORMAL"
    assert result.solver.group_alphas["motion"] == pytest.approx(1.0, abs=1e-6)
    # Prefer leaving the upper limit on joint 1.
    assert result.qdot[1] < -1e-4
    assert not result.solver.fault_latched
    assert any(
        cid.startswith("joint_working_cbf:1")
        or cid == "joint_working_cbf:1"
        for cid in result.solver.active_constraint_ids
    )


def test_joint_working_cbf_rows_include_j4_name() -> None:
    runtime = _runtime()
    runtime.config.working_arm_margin_rad = 0.30
    runtime.config.working_gamma = 8.0
    # 8-DOF naming: index 4 is J4 when rail is present; here n=7 with no rail.
    q = np.zeros(7)
    q[3] = 1.85  # near +limit 2.0 → inside hard, outside working (margin 0.30)
    state = RobotState(
        q_meas=q,
        q_cmd=q.copy(),
        qdot_applied_prev=np.zeros(7),
        dt=0.005,
        contact_active=False,
    )
    rows = runtime._joint_working_cbf_rows(state)
    names = [r.name for r in rows]
    assert "joint_working_cbf:3" in names
    row = next(r for r in rows if r.name == "joint_working_cbf:3")
    # Outside working+: block further intrusion but keep q̇=0 feasible.
    assert row.upper is not None and float(row.upper) == pytest.approx(0.0)
    assert row.lower is None or float(row.lower) <= 0.0


def test_joint_working_cbf_keeps_zero_feasible_outside_envelope() -> None:
    runtime = _runtime()
    runtime.config.working_arm_margin_rad = 0.30
    runtime.config.working_gamma = 8.0
    q = np.array([0.0, 1.92, 0.0, 0.0, 0.0, 0.0, 0.0])
    state = RobotState(
        q_meas=q,
        q_cmd=q.copy(),
        qdot_applied_prev=np.zeros(7),
        dt=0.005,
        contact_active=False,
    )
    rows = runtime._joint_working_cbf_rows(state)
    for row in rows:
        lo = -np.inf if row.lower is None else float(row.lower)
        hi = np.inf if row.upper is None else float(row.upper)
        assert lo <= 0.0 <= hi
