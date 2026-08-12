"""Invariants for the generic fixed two-level velocity QPIK."""

from __future__ import annotations

import numpy as np
import pytest

from rm75_control.control.joint_admittance_8dof.generic_tasks import (
    HardConstraintRow,
    LinearConstraintSet,
    PostureGuide,
    ProtectedTask,
    RobotState,
    ScalableTask,
)
from rm75_control.control.joint_admittance_8dof.solver.two_level_qpik import (
    TwoLevelQpikConfig,
    TwoLevelQpikController,
)


def _state(n: int, *, prev: np.ndarray | None = None, timestamp: float = 0.0) -> RobotState:
    q = np.zeros(n)
    return RobotState(q, q.copy(), np.zeros(n) if prev is None else prev, 0.005, False, timestamp)


def _controller(n: int, **kwargs) -> TwoLevelQpikController:
    cfg = TwoLevelQpikConfig(
        backend="scipy",
        qdot_lower=-np.ones(n),
        qdot_upper=np.ones(n),
        max_rows=96,
        max_scalable_groups=8,
        **kwargs,
    )
    return TwoLevelQpikController(n, cfg)


def test_protected_rows_are_locked_and_posture_cannot_change_them() -> None:
    c = _controller(8)
    state = _state(8)
    protected = ProtectedTask(np.eye(8)[:3], [0.2, -0.1, 0.05], name="tcp")
    posture = PostureGuide(
        np.full(8, 10.0),
        np.array([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0]),
        2.0,
        1.0,
        "valid",
    )
    result = c.solve(state, protected, posture_guide=posture)
    assert result.fallback_level == "none"
    np.testing.assert_allclose(result.protected_target, protected.b)
    np.testing.assert_allclose(result.protected_achieved, protected.A @ result.qdot, atol=1e-7)
    np.testing.assert_allclose(
        result.protected_achieved,
        result.protected_locked_output,
        atol=2.0e-6,
    )


def test_multiple_rows_share_one_alpha_per_group() -> None:
    c = _controller(7)
    protected = ProtectedTask(np.array([[1.0, 0, 0, 0, 0, 0, 0]]), [0.1])
    tasks = [
        ScalableTask(np.array([[0, 1.0, 0, 0, 0, 0, 0], [0, 0, 1.0, 0, 0, 0, 0]]), [0.3, -0.2], "arm"),
        ScalableTask(np.array([[0, 0, 0, 1.0, 0, 0, 0]]), [0.25], "rail"),
    ]
    result = c.solve(_state(7), protected, tasks)
    assert set(result.group_alphas) == {"arm", "rail"}
    assert all(0.0 <= float(value) <= 1.0 for value in result.group_alphas.values())
    assert result.qp1.success and result.qp2.success


def test_alpha_group_ids_cannot_collide_in_telemetry_encoding() -> None:
    c = _controller(7)
    protected = ProtectedTask(np.zeros((0, 7)), np.zeros(0), name="none")
    tasks = [
        ScalableTask(
            np.array([[1.0, 0, 0, 0, 0, 0, 0]]),
            [0.1],
            1,
        ),
        ScalableTask(
            np.array([[0, 1.0, 0, 0, 0, 0, 0]]),
            [0.1],
            "1",
        ),
    ]
    with pytest.raises(ValueError, match="telemetry encoding"):
        c.solve(_state(7), protected, tasks)


def test_one_sided_named_rows_and_continuous_rail() -> None:
    c = _controller(8)
    protected = ProtectedTask(
        np.array([[0.0, 1, 0, 0, 0, 0, 0, 0]]),
        [0.2],
        one_sided_constraints=(HardConstraintRow([0.0, 0, 0, 1, 0, 0, 0, 0], lower=0.03, name="joint_floor"),),
    )
    hard = LinearConstraintSet(
        np.array([[1.0, 0, 0, 0, 0, 0, 0, 0]]),
        [-0.4],
        [0.4],
        ["rail_velocity"],
    )
    result = c.solve(_state(8), protected, hard_constraints=hard)
    assert result.qp1.success and result.qp2.success
    assert result.qdot[3] >= 0.03 - 5e-6
    assert abs(result.qdot[0]) <= 0.4 + 5e-6
    assert "joint_floor" in result.active_constraint_ids or result.qdot[3] > 0.03


def test_qp2_failure_falls_back_to_same_tick_qp1() -> None:
    c = _controller(7)
    protected = ProtectedTask(np.array([[1.0, 0, 0, 0, 0, 0, 0]]), [0.2])
    q1_backend = c.backend_qp1
    q2_backend = c.backend_qp2
    real = q2_backend.solve
    q2_backend.solve = lambda *args, **kwargs: None  # type: ignore[method-assign]
    try:
        result = c.solve(_state(7), protected)
    finally:
        q2_backend.solve = real  # type: ignore[method-assign]
    assert result.fallback_level == "qp1"
    np.testing.assert_allclose(result.qdot, result.protected_locked_output[0] * protected.A[0], atol=2e-5)
    assert q1_backend.solve_count == 1


def test_qp1_failure_is_zero_stop_not_previous_velocity_decay() -> None:
    c = _controller(7)
    protected = ProtectedTask(np.array([[1.0, 0, 0, 0, 0, 0, 0]]), [0.3])
    c.sync_applied(np.full(7, 0.8))
    backend = c.backend_qp1
    backend.solve = lambda *args, **kwargs: None  # type: ignore[method-assign]
    result = c.solve(_state(7, prev=np.full(7, 0.8)), protected)
    assert result.fallback_level == "zero_stop"
    np.testing.assert_allclose(result.qdot, np.zeros(7))
    assert result.fault_latched

    calls = getattr(backend, "solve_count", None)
    repeated = c.solve(_state(7), protected)
    assert repeated.fallback_level == "fault"
    assert repeated.qp1.status == "fault_latched"
    np.testing.assert_allclose(repeated.qdot, np.zeros(7))
    if calls is not None:
        assert backend.solve_count == calls


def test_qp1_failure_fault_when_zero_is_not_p0_feasible() -> None:
    c = _controller(7)
    protected = ProtectedTask(np.array([[1.0, 0, 0, 0, 0, 0, 0]]), [0.1])
    hard = HardConstraintRow(np.array([0.0, 1, 0, 0, 0, 0, 0]), lower=0.1, name="must_move")
    c.backend_qp1.solve = lambda *args, **kwargs: None  # type: ignore[method-assign]
    result = c.solve(_state(7), protected, hard_constraints=[hard])
    assert result.fallback_level == "fault"
    assert result.fault_latched
    np.testing.assert_allclose(result.qdot, np.zeros(7))


def test_explicit_backend_does_not_silently_use_osqp() -> None:
    with pytest.raises(ValueError, match="osqp"):
        TwoLevelQpikController(7, TwoLevelQpikConfig(backend="osqp"))


def test_sync_applied_validates_shape_and_finite() -> None:
    c = _controller(7)
    c.sync_applied(np.arange(7, dtype=float))
    np.testing.assert_allclose(c.qdot_prev, np.arange(7, dtype=float))
    with pytest.raises(ValueError, match="n_dof"):
        c.sync_applied(np.zeros(8))
    with pytest.raises(ValueError, match="finite"):
        c.sync_applied(np.full(7, np.nan))


def test_backend_internal_typeerror_is_not_retried() -> None:
    class BrokenBackend:
        name = "broken"

        def __init__(self) -> None:
            self.calls = 0

        def clone(self):
            return BrokenBackend()

        def solve(self, H, g, C, lower, upper, x0=None):
            del H, g, C, lower, upper, x0
            self.calls += 1
            raise TypeError("internal backend defect")

    backend = BrokenBackend()
    c = TwoLevelQpikController(
        2,
        TwoLevelQpikConfig(
            backend=backend,
            max_rows=16,
            max_scalable_groups=1,
        ),
    )
    result = c.solve(_state(2), ProtectedTask([[1.0, 0.0]], [0.1]))
    assert backend.calls == 1
    assert result.qp1.status == "backend_exception"
    assert result.fault_latched


def test_scaled_hard_row_has_scale_invariant_feasibility() -> None:
    class ZeroBackendResult:
        def __init__(self, n: int) -> None:
            self.x = np.zeros(n)
            self.success = True
            self.status = "claimed_solved"
            self.nit = 1
            self.elapsed_s = 0.0
            self.message = ""

    class ZeroBackend:
        name = "zero"

        def __init__(self, n: int) -> None:
            self.n = n

        def clone(self):
            return ZeroBackend(self.n)

        def solve(self, H, g, C, lower, upper, x0=None):
            del H, g, C, lower, upper, x0
            return ZeroBackendResult(self.n)

    backend = ZeroBackend(3)  # 2 qdot + one fixed alpha variable
    c = TwoLevelQpikController(
        2,
        TwoLevelQpikConfig(
            backend=backend,
            max_rows=16,
            max_scalable_groups=1,
            feasibility_tolerance=1e-7,
        ),
    )
    tiny_but_active = HardConstraintRow(
        [0.0, 1.0e-9], lower=1.0e-9, name="scaled_must_move"
    )
    result = c.solve(
        _state(2),
        ProtectedTask([[1.0, 0.0]], [0.0]),
        hard_constraints=[tiny_but_active],
    )
    assert not result.qp1.success
    assert result.fallback_level == "fault"
    assert "scaled_must_move" in result.active_constraint_ids


@pytest.mark.parametrize("coefficient", [1.0e-300, 1.0e300])
def test_finite_extreme_hard_row_cannot_disappear(coefficient: float) -> None:
    c = _controller(2)
    result = c.solve(
        _state(2),
        ProtectedTask([[1.0, 0.0]], [1.0], name="request_positive"),
        hard_constraints=[
            HardConstraintRow(
                [coefficient, 0.0],
                upper=0.0,
                name="do_not_move_positive",
            )
        ],
    )
    assert result.qp1.success
    assert result.qp2.success
    assert result.qdot[0] <= 2.0e-6


def test_scipy_max_iteration_is_not_promoted_to_solved() -> None:
    c = _controller(2, max_iter=1)
    result = c.solve(
        _state(2),
        ProtectedTask(np.eye(2), [0.8, -0.7], name="nontrivial"),
    )
    assert not result.qp1.success
    assert result.fault_latched
    assert result.fallback_level in {"zero_stop", "fault"}
