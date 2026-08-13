from __future__ import annotations

import numpy as np

from rm75_control.control.joint_admittance_8dof.collision_model import CollisionConfig
from rm75_control.control.joint_admittance_8dof.generic_runtime import (
    GenericQpikRuntime,
    GenericQpikRuntimeConfig,
)
from rm75_control.control.joint_admittance_8dof.generic_tasks import RobotState
from rm75_control.control.joint_admittance_8dof.solver.single_qpik import (
    SingleQpikConfig,
)
from rm75_control.control.joint_admittance_8dof.utils.safety import SafetyLimits


class _Kinematics:
    nv = 8

    @staticmethod
    def jacobian(q):
        del q
        jacobian = np.zeros((6, 8))
        jacobian[:, :6] = np.eye(6)
        jacobian[1, 6] = 0.35
        jacobian[2, 7] = -0.20
        return jacobian


def _limits() -> SafetyLimits:
    return SafetyLimits(
        q_lower=np.full(8, -2.0),
        q_upper=np.full(8, 2.0),
        v_max=np.ones(8),
        a_max=np.full(8, 20.0),
        position_margin=np.zeros(8),
    )


def _runtime(kin=None) -> GenericQpikRuntime:
    return GenericQpikRuntime(
        kin or _Kinematics(),
        _limits(),
        GenericQpikRuntimeConfig(
            solver=SingleQpikConfig(
                backend="scipy",
                max_iter=200,
                max_solve_ms=500.0,
                authority_rise_per_s=1000.0,
            ),
            rail_indices=(0,),
            wrist_indices=(5, 6, 7),
            rail_center_k=0.0,
        ),
        collision_config=CollisionConfig(enabled=False),
        damper_band=0.0,
    )


def _state(q: np.ndarray | None = None) -> RobotState:
    measured = np.zeros(8) if q is None else np.asarray(q, dtype=float)
    measured = measured.copy()
    measured[4] = measured[4] or 0.8
    measured[6] = measured[6] or 0.7
    return RobotState(
        q_meas=measured,
        q_cmd=measured.copy(),
        qdot_applied_prev=np.zeros(8),
        dt=0.005,
        contact_active=False,
    )


def _solve(runtime: GenericQpikRuntime, state: RobotState | None = None, **twists):
    return runtime.solve(
        state or _state(),
        protected_twist_task=twists.get("protected", np.zeros(6)),
        path_twist_task=twists.get("path", np.zeros(6)),
        feedback_twist_task=twists.get("feedback", np.zeros(6)),
        rotation_base_task=np.eye(3),
    )


def test_runtime_reuses_one_measured_state_jacobian_snapshot_with_cached_gradient() -> None:
    kin = _Kinematics()
    calls = 0

    def jacobian(q):
        nonlocal calls
        calls += 1
        return _Kinematics.jacobian(q)

    kin.jacobian = jacobian
    runtime = _runtime(kin)
    runtime._arm_gradient[:] = 1.0
    runtime._arm_gradient_valid = True
    _solve(runtime)
    assert calls == 1


def test_hybrid_episode_reset_preserves_applied_velocity_and_healthy_branches() -> None:
    runtime = _runtime()
    q = np.zeros(8)
    q[4] = -0.9
    q[6] = 0.8
    applied = np.linspace(-0.04, 0.04, 8)

    runtime.begin_hybrid_episode(q, applied)

    np.testing.assert_allclose(runtime.qdot_prev, applied)
    assert runtime._elbow_branch_sign == -1.0
    assert runtime._wrist_branch_sign == 1.0
    assert runtime._arm_nominal is None
    np.testing.assert_allclose(runtime._feedback_xy, 0.0)


def test_hybrid_episode_keeps_previous_wrist_branch_inside_warning_band() -> None:
    runtime = _runtime()
    q_healthy = np.zeros(8)
    q_healthy[4] = 0.8
    q_healthy[6] = 0.9
    runtime.begin_hybrid_episode(q_healthy, np.zeros(8))

    q_near_zero = q_healthy.copy()
    q_near_zero[6] = -1.0e-6
    runtime.begin_hybrid_episode(q_near_zero, np.zeros(8))

    assert runtime._wrist_branch_sign == 1.0


def test_fixed_command_has_z_orientation_protected_and_xy_without_residual() -> None:
    runtime = _runtime()
    protected = np.array([9.0, -7.0, 0.01, 0.02, -0.03, 0.04])
    path = np.array([0.03, -0.02, 8.0, 7.0, 6.0, 5.0])
    feedback = np.array([-0.01, 0.015, 4.0, 3.0, 2.0, 1.0])
    result = _solve(
        runtime, protected=protected, path=path, feedback=feedback
    )

    np.testing.assert_allclose(result.command.protected_velocity, protected[[2, 3, 4, 5]])
    np.testing.assert_allclose(result.command.path_velocity, path[:2])
    assert np.all(np.sign(result.command.feedback_velocity) == np.sign(feedback[:2]))
    assert np.all(np.abs(result.command.feedback_velocity) < np.abs(feedback[:2]))
    assert result.command.protected_jacobian.shape == (4, 8)
    assert result.command.scan_jacobian.shape == (2, 8)
    assert result.solver.diagnostics.call_count == 1


def test_rail_macro_depends_only_on_xy_path_feedforward() -> None:
    path = np.array([0.04, -0.03, 0.0, 0.0, 0.0, 0.0])
    first = _solve(
        _runtime(),
        path=path,
        feedback=np.array([0.4, -0.5, 0.0, 0.0, 0.0, 0.0]),
        protected=np.array([0.0, 0.0, 0.08, 0.4, -0.3, 0.2]),
    )
    second = _solve(
        _runtime(),
        path=path,
        feedback=np.array([-0.7, 0.8, 0.0, 0.0, 0.0, 0.0]),
        protected=np.array([0.0, 0.0, -0.09, -0.2, 0.5, -0.4]),
    )
    assert first.rail_macro_preference == second.rail_macro_preference


def test_nominal_posture_is_arm_only_and_never_attracts_rail() -> None:
    runtime = _runtime()
    initial = _state()
    runtime._healthy_dwell_s = runtime.config.risk_exit_dwell_s
    runtime._whole_body_preference(initial, np.zeros(8))
    moved_rail = initial.q_meas.copy()
    moved_rail[0] = 0.6
    preference, _ = runtime._whole_body_preference(
        _state(moved_rail), np.zeros(8)
    )

    assert runtime._arm_nominal is not None
    assert runtime._arm_nominal.shape == (7,)
    assert preference[0] == 0.0
    np.testing.assert_allclose(preference[1:], np.zeros(7))


def test_wrist_branch_direction_stays_latched_across_zero() -> None:
    runtime = _runtime()
    q_positive = _state().q_meas.copy()
    q_positive[6] = np.deg2rad(30.0)
    positive = _state(q_positive)
    runtime._risk_preference(
        positive,
        runtime._update_health(positive, _Kinematics.jacobian(positive.q_meas)),
        np.zeros((4, 8)),
        np.full(4, -np.inf),
    )
    assert runtime._wrist_branch_sign == 1.0

    q_crossed = _state().q_meas.copy()
    q_crossed[6] = np.deg2rad(-2.0)
    crossed = _state(q_crossed)
    risk, _, _, branch_jacobian, branch_lower, _ = runtime._risk_preference(
        crossed,
        runtime._update_health(crossed, _Kinematics.jacobian(crossed.q_meas)),
        np.zeros((4, 8)),
        np.full(4, -np.inf),
    )
    assert runtime._wrist_branch_sign == 1.0
    assert risk[6] > 0.0
    assert branch_jacobian[1, 6] == 1.0
    assert branch_lower[1] > 0.0


def test_fresh_danger_entry_latches_both_branch_signs_and_rows() -> None:
    runtime = _runtime()
    q = _state().q_meas.copy()
    q[4] = -1.0e-3
    q[6] = 1.0e-3
    runtime.begin_hybrid_episode(q, np.zeros(8))
    state = _state(q)
    _, _, _, branch_jacobian, branch_lower, _ = runtime._risk_preference(
        state,
        runtime._update_health(state, _Kinematics.jacobian(state.q_meas)),
        np.zeros((4, 8)),
        np.full(4, -np.inf),
    )
    assert runtime._elbow_branch_sign == -1.0
    assert runtime._wrist_branch_sign == 1.0
    np.testing.assert_allclose(branch_jacobian[0], np.array([0, 0, 0, 0, -1, 0, 0, 0]))
    np.testing.assert_allclose(branch_jacobian[1], np.array([0, 0, 0, 0, 0, 0, 1, 0]))
    assert np.all(np.isfinite(branch_lower))


def test_simultaneous_j4_j6_branch_rows_are_kept_in_qp() -> None:
    runtime = _runtime()
    q = _state().q_meas.copy()
    q[4] = 1.0e-3
    q[6] = 1.0e-3
    runtime.begin_hybrid_episode(q, np.zeros(8))
    state = _state(q)
    result = _solve(runtime, state=state)
    command = result.command
    assert command.branch_jacobian.shape == (2, 8)
    assert np.isfinite(command.branch_lower).all()
    assembled = runtime.solver._assemble(
        command, result.p0.C, result.p0.lower, result.p0.upper, result.solver.anchor
    )
    _, _, _, _, _, lower, _, _, _, _, names, _, _ = assembled
    assert np.isfinite(lower[55]) and np.isfinite(lower[56])
    assert names[55] == "branch_recovery:0"
    assert names[56] == "branch_recovery:1"


def test_gradient_lpf_uses_actual_gradient_update_period(monkeypatch) -> None:
    import rm75_control.control.joint_admittance_8dof.generic_runtime as runtime_module

    runtime = _runtime()
    runtime.config.gradient_period_ticks = 10
    runtime.config.gradient_lpf_tau_s = 0.10
    gradients = iter((np.ones(8), np.full(8, 2.0)))
    monkeypatch.setattr(
        runtime_module,
        "arm_dexterity_gradient",
        lambda *args, **kwargs: next(gradients),
    )
    state = _state()
    health = runtime._update_health(state, _Kinematics.jacobian(state.q_meas))
    runtime._risk_preference(
        state, health, np.zeros((4, 8)), np.full(4, -np.inf)
    )
    assert runtime._arm_gradient_valid
    np.testing.assert_allclose(runtime._arm_gradient, 1.0)
    for _ in range(9):
        runtime._risk_preference(
            state, health, np.zeros((4, 8)), np.full(4, -np.inf)
        )
    np.testing.assert_allclose(runtime._arm_gradient, np.full(8, 4.0 / 3.0))


def test_fresh_danger_entry_initializes_dexterity_row_without_fallback(
    monkeypatch,
) -> None:
    import rm75_control.control.joint_admittance_8dof.generic_runtime as runtime_module

    runtime = _runtime()
    runtime.config.gradient_period_ticks = 10
    monkeypatch.setattr(
        runtime_module,
        "arm_dexterity_gradient",
        lambda *args, **kwargs: np.ones(8),
    )

    result = _solve(runtime)

    assert runtime._tick == 1
    assert runtime._arm_gradient_valid
    assert np.linalg.norm(result.command.dexterity_gradient) > 0.0
    assert np.isfinite(result.command.dexterity_lower)
    assert result.solver.diagnostics.call_count == 1
    assert not result.solver.fallback


def test_scan_target_matches_exact_alpha_beta_anchor_equality() -> None:
    path = np.array([0.50, -0.40, 0.0, 0.0, 0.0, 0.0])
    feedback = np.array([0.30, 0.20, 0.0, 0.0, 0.0, 0.0])
    result = _solve(_runtime(), path=path, feedback=feedback)
    expected = (
        (1.0 - result.solver.beta)
        * (result.scan_jacobian @ result.solver.anchor)
        + result.solver.beta * result.command.feedback_velocity
        + result.solver.alpha * result.command.path_velocity
    )

    np.testing.assert_allclose(result.scan_target, expected, atol=1.0e-12)
    assert not np.allclose(
        result.scan_target,
        result.command.feedback_velocity + result.command.path_velocity,
    )


def test_rail_center_is_disabled_until_healthy_exit_dwell() -> None:
    runtime = _runtime()
    runtime.config.rail_center_k = 0.04
    q = _state().q_meas.copy()
    q[0] = 0.6
    state = _state(q)
    _, center = runtime._rail_preferences(
        _Kinematics.jacobian(state.q_meas)[:2], np.zeros(2), state
    )
    assert center == 0.0
    runtime._healthy_dwell_s = runtime.config.risk_exit_dwell_s + runtime.config.risk_release_s
    _, center = runtime._rail_preferences(
        _Kinematics.jacobian(state.q_meas)[:2], np.zeros(2), state
    )
    assert center < 0.0


def test_raw_arm_risk_preference_is_not_projected_before_qp() -> None:
    runtime = _runtime()
    state = _state()
    runtime._whole_body_preference(state, np.zeros(8))
    runtime._risk_level = 1.0
    risk = np.array([0.9, 0.02, -0.03, 0.04, -0.05, 0.06, -0.07, 0.08])
    preference, applied_risk = runtime._whole_body_preference(state, risk)

    assert preference[0] == 0.0
    np.testing.assert_allclose(preference[1:], risk[1:])
    np.testing.assert_allclose(applied_risk[1:], risk[1:])


def test_working_bounds_push_a_near_limit_joint_back_without_becoming_hard() -> None:
    runtime = _runtime()
    q = _state().q_meas.copy()
    q[4] = 1.90
    lower, upper = runtime._working_bounds(q)
    assert upper[4] < 0.0
    assert lower[4] < upper[4]
