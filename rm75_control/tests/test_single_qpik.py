from types import SimpleNamespace

import numpy as np
import pytest

from rm75_control.control.joint_admittance_8dof.solver.single_qpik import (
    ALPHA,
    BETA,
    N_DOF,
    N_EQ,
    N_INEQ,
    N_VAR,
    PROTECTED_SLACK,
    QDOT,
    CartesianQpCommand,
    SingleQpikConfig,
    SingleQpikController,
    build_hard_anchor,
)


def _command() -> CartesianQpCommand:
    protected = np.zeros((4, N_DOF))
    protected[:, 2:6] = np.eye(4)
    scan = np.zeros((2, N_DOF))
    scan[0, 0] = 1.0
    scan[1, 1] = 1.0
    return CartesianQpCommand(
        protected_jacobian=protected,
        protected_velocity=np.array([0.01, 0.02, -0.03, 0.04]),
        scan_jacobian=scan,
        path_velocity=np.array([0.20, -0.10]),
        feedback_velocity=np.array([0.04, 0.03]),
        qdot_preference=np.zeros(N_DOF),
        working_lower=np.full(N_DOF, -0.8),
        working_upper=np.full(N_DOF, 0.8),
        rail_macro_velocity=0.20,
    )


def _hard_box(lo: np.ndarray | None = None, hi: np.ndarray | None = None):
    return (
        np.eye(N_DOF),
        np.full(N_DOF, -1.0) if lo is None else np.asarray(lo, dtype=float),
        np.full(N_DOF, 1.0) if hi is None else np.asarray(hi, dtype=float),
    )


def test_fixed_native_equality_layout_and_anchor_seed() -> None:
    controller = SingleQpikController(
        np.ones(N_DOF), SingleQpikConfig(backend="scipy", max_solve_ms=500.0)
    )
    command = _command()
    hard_C, hard_lo, hard_hi = _hard_box()
    anchor = np.array([0.06, -0.02, 0.01, 0.02, -0.03, 0.04, 0.0, 0.0])
    assembled = controller._assemble(command, hard_C, hard_lo, hard_hi, anchor)
    _, _, A, b, C, lower, upper, A_phys, b_phys, C_phys, _, _, _ = assembled

    assert A.shape == (N_EQ, N_VAR)
    assert C.shape == (N_INEQ, N_VAR)
    np.testing.assert_allclose(A_phys[:4, QDOT], command.protected_jacobian)
    np.testing.assert_allclose(A_phys[:4, PROTECTED_SLACK], -np.eye(4))
    np.testing.assert_allclose(A_phys[4:6, QDOT], command.scan_jacobian)
    np.testing.assert_allclose(A_phys[4:6, ALPHA], -command.path_velocity)
    np.testing.assert_allclose(
        A_phys[4:6, BETA],
        -(command.feedback_velocity - command.scan_jacobian @ anchor),
    )
    np.testing.assert_allclose(b_phys[:4], command.protected_velocity)
    np.testing.assert_allclose(b_phys[4:6], command.scan_jacobian @ anchor)

    x_h = np.zeros(N_VAR)
    x_h[QDOT] = anchor
    x_h[PROTECTED_SLACK] = (
        command.protected_jacobian @ anchor - command.protected_velocity
    )
    np.testing.assert_allclose(A_phys @ x_h, b_phys, atol=1.0e-14)
    assert lower.shape == (N_INEQ,)
    assert upper.shape == (N_INEQ,)

    row_scale = controller.inequality_row_scale
    np.testing.assert_allclose(
        C,
        (C_phys * controller.variable_scale[None, :]) * row_scale[:, None],
    )
    finite_lower = np.isfinite(lower)
    finite_upper = np.isfinite(upper)
    x_test = x_h / controller.variable_scale
    np.testing.assert_allclose(
        (C @ x_test)[finite_lower] - lower[finite_lower],
        (
            C_phys @ x_h
            - np.divide(
                lower,
                row_scale,
                out=np.full_like(lower, -np.inf),
                where=finite_lower,
            )
        )[finite_lower]
        * row_scale[finite_lower],
    )
    np.testing.assert_allclose(
        upper[finite_upper] - (C @ x_test)[finite_upper],
        (
            np.divide(
                upper,
                row_scale,
                out=np.full_like(upper, np.inf),
                where=finite_upper,
            )
            - C_phys @ x_h
        )[finite_upper]
        * row_scale[finite_upper],
    )


def test_backend_failure_calls_once_and_returns_certified_nonzero_anchor() -> None:
    class FailedBackend:
        name = "failed_spy"

        def __init__(self) -> None:
            self.calls = 0
            self.cold_starts = 0

        def solve(self, *args):
            self.calls += 1
            return SimpleNamespace(
                x=None,
                success=False,
                status="forced_failure",
                iterations=0,
                elapsed_s=0.001,
                message="forced",
            )

        def cold_start(self):
            self.cold_starts += 1

    backend = FailedBackend()
    controller = SingleQpikController(
        np.ones(N_DOF), SingleQpikConfig(backend=backend)
    )
    lower = np.full(N_DOF, -0.2)
    upper = np.full(N_DOF, 0.2)
    lower[0] = 0.05  # zero is not hard-feasible this tick.
    controller.sync_applied(np.array([0.08, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    result = controller.solve(_command(), *_hard_box(lower, upper), dt=0.005)

    assert backend.calls == 1
    assert result.fallback
    assert not result.hard_failure
    assert result.authority == 0.0
    assert result.qdot[0] >= 0.05 - 1.0e-12
    assert result.hard_residual_max <= 1.0e-5


def test_backend_exception_calls_once_and_cold_starts() -> None:
    class RaisingBackend:
        name = "raising_spy"

        def __init__(self) -> None:
            self.calls = 0
            self.cold_starts = 0

        def solve(self, *args):
            self.calls += 1
            raise TypeError("backend signature failure")

        def cold_start(self):
            self.cold_starts += 1

    backend = RaisingBackend()
    controller = SingleQpikController(
        np.ones(N_DOF), SingleQpikConfig(backend=backend)
    )
    result = controller.solve(_command(), *_hard_box(), dt=0.005)

    assert backend.calls == 1
    assert backend.cold_starts == 1
    assert result.diagnostics.call_count == 1
    assert result.diagnostics.status == "backend_exception"
    assert result.fallback
    assert result.anchor_valid
    assert not result.hard_failure
    assert result.authority == 0.0


def test_anchor_failure_is_not_reported_as_proven_hard_infeasible() -> None:
    C = np.vstack((np.eye(N_DOF), np.eye(N_DOF)[0]))
    lower = np.concatenate((np.full(N_DOF, -1.0), [2.0]))
    upper = np.concatenate((np.full(N_DOF, 1.0), [np.inf]))
    _, valid, violation = build_hard_anchor(
        np.zeros(N_DOF),
        0.005,
        C,
        lower,
        upper,
        decay_tau_s=0.08,
        sweeps=4,
        tolerance=1.0e-5,
    )
    assert not valid
    assert violation > 0.0


def test_anchor_certifies_known_feasible_coupled_halfspaces_with_fixed_budget() -> None:
    """Regression for a feasible set that four cyclic sweeps did not certify."""

    C2 = np.array(
        [
            [0.1257302210933933, -0.1321048632913019],
            [0.6404226504432821, 0.1049001171530397],
            [-0.535669373161111, 0.3615950549094847],
            [1.3040000451301372, 0.9470809631292422],
        ]
    )
    C = np.vstack((np.eye(N_DOF), np.pad(C2, ((0, 0), (0, N_DOF - 2)))))
    lower = np.concatenate(
        (
            np.full(N_DOF, -1.0),
            np.array(
                [
                    -0.2153450605143384,
                    0.072032619351386,
                    -0.0376719859042275,
                    0.4612522715058192,
                ]
            ),
        )
    )
    upper = np.concatenate(
        (
            np.full(N_DOF, 1.0),
            np.array(
                [
                    0.1113646540260314,
                    0.075122542419105,
                    0.3055749161776251,
                    0.476619330477394,
                ]
            ),
        )
    )
    known_feasible = np.array(
        [0.0436249914654229, 0.4350724237877682, 0, 0, 0, 0, 0, 0]
    )
    assert np.max(lower - C @ known_feasible) <= 0.0
    assert np.max(C @ known_feasible - upper) <= 0.0
    previous = np.array(
        [0.4593108928598881, -0.648688758794882, 0, 0, 0, 0, 0, 0]
    )

    anchor, valid, violation = build_hard_anchor(
        previous,
        0.005,
        C,
        lower,
        upper,
        decay_tau_s=0.08,
        sweeps=64,
        tolerance=1.0e-5,
    )

    assert valid
    assert violation <= 1.0e-5
    assert np.all(np.isfinite(anchor))


def test_scipy_solution_has_exact_direction_and_authority_order() -> None:
    config = SingleQpikConfig(
        backend="scipy",
        max_iter=200,
        max_solve_ms=500.0,
        authority_rise_per_s=1000.0,
    )
    controller = SingleQpikController(np.ones(N_DOF), config)
    result = controller.solve(_command(), *_hard_box(), dt=0.005)

    assert result.diagnostics.call_count == 1
    assert result.diagnostics.success
    assert not result.fallback
    assert 0.0 <= result.alpha <= result.beta <= 1.0
    assert result.authority <= result.alpha + 1.0e-12
    assert np.max(np.abs(result.scan_residual)) <= config.equality_tolerance
    assert result.hard_residual_max <= config.feasibility_tolerance


def test_recovery_slack_is_shared_by_each_working_bound_pair() -> None:
    controller = SingleQpikController(
        np.ones(N_DOF), SingleQpikConfig(backend="scipy", max_solve_ms=500.0)
    )
    command = _command()
    hard_C, hard_lo, hard_hi = _hard_box()
    assembled = controller._assemble(
        command, hard_C, hard_lo, hard_hi, np.zeros(N_DOF)
    )
    C_phys = assembled[9]
    for index in range(N_DOF):
        lower_row = 38 + 2 * index
        upper_row = lower_row + 1
        slack = 14 + index
        assert C_phys[lower_row, slack] == pytest.approx(1.0)
        assert C_phys[upper_row, slack] == pytest.approx(1.0)


class _PresetBackend:
    name = "preset"

    def __init__(self, physical_solutions: list[np.ndarray], scale: np.ndarray):
        self._solutions = [np.asarray(value, dtype=float) for value in physical_solutions]
        self._scale = np.asarray(scale, dtype=float)
        self.calls = 0
        self.cold_starts = 0

    def solve(self, *args):
        del args
        index = min(self.calls, len(self._solutions) - 1)
        self.calls += 1
        return SimpleNamespace(
            x=self._solutions[index] / self._scale,
            success=True,
            status="preset_solved",
            iterations=1,
            elapsed_s=0.0,
            message="",
        )

    def cold_start(self):
        self.cold_starts += 1


def _authority_command(
    *, path=(0.2, 0.0), feedback=(0.0, 0.1), working_upper=0.8
) -> CartesianQpCommand:
    protected = np.zeros((4, N_DOF))
    protected[:, 2:6] = np.eye(4)
    scan = np.zeros((2, N_DOF))
    scan[0, 0] = 1.0
    scan[1, 1] = 1.0
    return CartesianQpCommand(
        protected_jacobian=protected,
        protected_velocity=np.zeros(4),
        scan_jacobian=scan,
        path_velocity=np.asarray(path, dtype=float),
        feedback_velocity=np.asarray(feedback, dtype=float),
        qdot_preference=np.zeros(N_DOF),
        working_lower=np.full(N_DOF, -0.8),
        working_upper=np.full(N_DOF, working_upper),
    )


def _physical_solution(
    command: CartesianQpCommand,
    *,
    alpha: float,
    beta: float,
    anchor: np.ndarray | None = None,
    work_slack: np.ndarray | None = None,
) -> np.ndarray:
    x = np.zeros(N_VAR)
    x[ALPHA] = alpha
    x[BETA] = beta
    y_h = np.zeros(2) if anchor is None else command.scan_jacobian @ anchor
    x[QDOT.start : QDOT.start + 2] = (
        (1.0 - beta) * y_h
        + beta * command.feedback_velocity
        + alpha * command.path_velocity
    )
    x[PROTECTED_SLACK] = (
        command.protected_jacobian @ x[QDOT] - command.protected_velocity
    )
    if work_slack is not None:
        x[14:22] = np.asarray(work_slack, dtype=float)
    return x


def test_authority_handles_non_collinear_path_feedback_and_zero_path_speed() -> None:
    command = _authority_command(path=(0.2, 0.0), feedback=(0.0, 0.1))
    zero_path = _authority_command(path=(0.0, 0.0), feedback=(0.02, -0.03))
    config = SingleQpikConfig(authority_rise_per_s=1000.0)
    temporary = SingleQpikController(np.ones(N_DOF), SingleQpikConfig(backend="scipy"))
    backend = _PresetBackend(
        [_physical_solution(command, alpha=0.5, beta=0.8)],
        temporary.variable_scale,
    )
    config.backend = backend
    controller = SingleQpikController(np.ones(N_DOF), config)

    first = controller.solve(command, *_hard_box(), dt=0.005)
    second_anchor, valid, _ = build_hard_anchor(
        first.qdot, 0.005, *_hard_box(), decay_tau_s=config.anchor_decay_tau_s,
        sweeps=config.anchor_projection_sweeps,
        tolerance=config.feasibility_tolerance,
    )
    assert valid
    backend._solutions.append(
        _physical_solution(
            zero_path, alpha=0.7, beta=0.9, anchor=second_anchor
        )
    )
    second = controller.solve(zero_path, *_hard_box(), dt=0.005)

    assert first.authority == pytest.approx(0.5, abs=1.0e-12)
    assert first.alpha == pytest.approx(0.5)
    assert first.beta == pytest.approx(0.8)
    assert second.authority == pytest.approx(0.7, abs=1.0e-12)
    assert np.max(np.abs(first.scan_residual)) <= config.equality_tolerance
    assert np.max(np.abs(second.scan_residual)) <= config.equality_tolerance


def test_authority_drops_immediately_and_rises_at_configured_rate() -> None:
    command = _authority_command(path=(0.1, 0.0), feedback=(0.0, 0.0))
    temporary = SingleQpikController(np.ones(N_DOF), SingleQpikConfig(backend="scipy"))
    backend = _PresetBackend(
        [_physical_solution(command, alpha=1.0, beta=1.0)],
        temporary.variable_scale,
    )
    controller = SingleQpikController(
        np.ones(N_DOF),
        SingleQpikConfig(backend=backend, authority_rise_per_s=2.0),
    )
    controller._authority = 1.0

    first = controller.solve(command, *_hard_box(), dt=0.005)
    assert first.authority == pytest.approx(1.0)
    second_anchor, valid, _ = build_hard_anchor(
        first.qdot, 0.005, *_hard_box(), decay_tau_s=controller.config.anchor_decay_tau_s,
        sweeps=controller.config.anchor_projection_sweeps,
        tolerance=controller.config.feasibility_tolerance,
    )
    assert valid
    backend._solutions.append(
        _physical_solution(command, alpha=0.2, beta=0.9, anchor=second_anchor)
    )
    dropped = controller.solve(command, *_hard_box(), dt=0.005)
    assert dropped.authority == pytest.approx(0.2)
    third_anchor, valid, _ = build_hard_anchor(
        dropped.qdot, 0.005, *_hard_box(), decay_tau_s=controller.config.anchor_decay_tau_s,
        sweeps=controller.config.anchor_projection_sweeps,
        tolerance=controller.config.feasibility_tolerance,
    )
    assert valid
    backend._solutions.append(
        _physical_solution(command, alpha=0.21, beta=0.21, anchor=third_anchor)
    )
    rising = controller.solve(command, *_hard_box(), dt=0.005)
    assert rising.authority == pytest.approx(0.21)


def test_recovery_path_velocity_cannot_outrun_reference_authority() -> None:
    command = _authority_command(path=(0.1, 0.0), feedback=(0.0, 0.0))
    controller = SingleQpikController(
        np.ones(N_DOF),
        SingleQpikConfig(
            backend="proxqp",
            max_iter=200,
            max_iter_in=100,
            max_solve_ms=500.0,
            authority_rise_per_s=2.0,
        ),
    )

    first = controller.solve(command, *_hard_box(), dt=0.005)
    second = controller.solve(command, *_hard_box(), dt=0.005)

    assert first.alpha == pytest.approx(0.01, abs=1.0e-5)
    assert first.authority == pytest.approx(first.alpha, abs=1.0e-5)
    assert second.alpha == pytest.approx(0.02, abs=1.0e-5)
    assert second.authority == pytest.approx(second.alpha, abs=1.0e-5)
    np.testing.assert_allclose(
        command.scan_jacobian @ first.qdot,
        first.alpha * command.path_velocity,
        atol=1.0e-5,
    )


def test_weighted_policy_reduces_path_alpha_before_using_working_slack() -> None:
    controller = SingleQpikController(
        np.ones(N_DOF),
        SingleQpikConfig(
            backend="proxqp", max_iter=200, max_iter_in=100,
            max_solve_ms=500.0,
            authority_rise_per_s=1000.0,
        ),
    )
    command = _authority_command(
        path=(0.5, 0.0), feedback=(0.0, 0.0), working_upper=0.1
    )
    hard_lo = np.full(N_DOF, -1.0)
    hard_hi = np.full(N_DOF, 1.0)
    hard_lo[2] = hard_hi[2] = 0.0
    result = controller.solve(command, *_hard_box(hard_lo, hard_hi), dt=0.005)

    assert result.diagnostics.success
    assert result.beta > 0.95
    assert result.alpha < 0.8
    full_path_slack = 0.5 - 0.1
    assert result.working_slack[0] <= controller.config.feasibility_tolerance


def test_weighted_policy_uses_recovery_before_reducing_feedback_beta() -> None:
    controller = SingleQpikController(
        np.ones(N_DOF),
        SingleQpikConfig(
            backend="scipy", max_iter=400, max_solve_ms=500.0,
            authority_rise_per_s=1000.0,
        ),
    )
    command = _authority_command(
        path=(0.0, 0.0), feedback=(0.5, 0.0), working_upper=0.1
    )
    hard_lo = np.full(N_DOF, -1.0)
    hard_hi = np.full(N_DOF, 1.0)
    hard_lo[2] = hard_hi[2] = 0.0
    result = controller.solve(command, *_hard_box(hard_lo, hard_hi), dt=0.005)

    assert result.diagnostics.success
    assert result.beta > 0.9
    assert result.working_slack[0] > 0.3
    assert 0.0 <= result.alpha <= result.beta <= 1.0


def test_protected_cap_conflict_falls_back_to_certified_hard_anchor() -> None:
    controller = SingleQpikController(
        np.ones(N_DOF),
        SingleQpikConfig(
            backend="scipy", max_iter=400,
            max_solve_ms=500.0, authority_rise_per_s=1000.0,
        ),
    )
    command = _command()
    command = CartesianQpCommand(
        **{
            **command.__dict__,
            "protected_velocity": np.array([0.08, 0.02, -0.03, 0.04]),
        }
    )
    hard_lo = np.full(N_DOF, -1.0)
    hard_hi = np.full(N_DOF, 1.0)
    hard_lo[2] = hard_hi[2] = 0.0
    result = controller.solve(command, *_hard_box(hard_lo, hard_hi), dt=0.005)

    assert result.diagnostics.call_count == 1
    assert not result.diagnostics.success
    assert result.fallback
    assert result.anchor_valid
    assert not result.hard_failure
    assert result.authority == 0.0
    assert result.hard_residual_max <= controller.config.feasibility_tolerance
    assert abs(result.protected_residual[0]) > controller.protected_limits[0]
    assert result.protected_nominal_overflow[0] > 0.0
    assert np.all(result.protected_nominal_overflow[1:] <= 1.0e-9)
