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
    # SciPy SLSQP is not a hard-real-time backend; keep a generous wall clock
    # so unit tests exercise formulation, not production ProxQP budgets.
    kwargs.setdefault("max_solve_ms", 500.0)
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


def test_qp1_failure_falls_back_to_p0_safe_without_latch() -> None:
    c = _controller(7)
    protected = ProtectedTask(np.array([[1.0, 0, 0, 0, 0, 0, 0]]), [0.3])
    c.sync_applied(np.full(7, 0.8))
    backend = c.backend_qp1
    backend.solve = lambda *args, **kwargs: None  # type: ignore[method-assign]
    result = c.solve(_state(7, prev=np.full(7, 0.8)), protected)
    assert result.fallback_level == "p0_safe"
    assert not result.fault_latched
    # Soft singularity/numeric failure must remain sendable next tick.
    repeated = c.solve(_state(7), protected)
    assert repeated.fallback_level == "p0_safe"
    assert not repeated.fault_latched
    assert repeated.qp1.status != "fault_latched"


def test_qp1_failure_fault_when_zero_is_not_p0_feasible() -> None:
    c = _controller(7)
    protected = ProtectedTask(np.array([[1.0, 0, 0, 0, 0, 0, 0]]), [0.1])
    hard = HardConstraintRow(np.array([0.0, 1, 0, 0, 0, 0, 0]), lower=0.1, name="must_move")
    c.backend_qp1.solve = lambda *args, **kwargs: None  # type: ignore[method-assign]
    result = c.solve(_state(7), protected, hard_constraints=[hard])
    assert result.fallback_level == "fault"
    assert result.fault_latched
    np.testing.assert_allclose(result.qdot, np.zeros(7))


def test_explicit_alpha_cap_zero_still_respected() -> None:
    """Caller-supplied alpha_cap=0 remains a hard upper bound (not health)."""
    c = _controller(7)
    protected = ProtectedTask(np.array([[1.0, 0, 0, 0, 0, 0, 0]]), [0.05])
    tasks = [
        ScalableTask(np.array([[0, 1.0, 0, 0, 0, 0, 0]]), [0.4], "motion"),
    ]
    result = c.solve(_state(7), protected, tasks, alpha_cap=0.0, reg_scale=10.0)
    assert result.group_alphas["motion"] == pytest.approx(0.0)
    assert not result.fault_latched
    assert result.qp1.success


def test_qp2_maximizes_alpha_without_posture_competition() -> None:
    c = _controller(3, alpha_weight=10.0, posture_weight=1.0)
    protected = ProtectedTask(np.zeros((0, 3)), np.zeros(0))
    tasks = [ScalableTask(np.array([[1.0, 0.0, 0.0]]), [0.3], "motion")]
    # Conflicting posture that would pull qdot[0] opposite if it competed with α.
    guide = {"qdot_guide": np.array([-1.0, 0.0, 0.0]), "quality": 1.0}
    result = c.solve(_state(3), protected, tasks, posture_guide=guide)
    assert result.qp2.success
    assert result.group_alphas["motion"] == pytest.approx(1.0, abs=0.05)
    assert result.qdot[0] > 0.0


def test_joint_margin_cost_repels_from_nearest_limit() -> None:
    from rm75_control.control.joint_admittance_8dof.generic_tasks import RobotState

    c = TwoLevelQpikController(
        3,
        TwoLevelQpikConfig(
            backend="scipy",
            max_solve_ms=500.0,
            max_rows=32,
            max_scalable_groups=1,
            margin_weight=5.0e-2,
            margin_weight_gain=40.0,
        ),
    )
    st = RobotState(
        q_meas=np.array([0.95, 0.0, 0.0]),
        q_cmd=np.array([0.95, 0.0, 0.0]),
        qdot_applied_prev=np.zeros(3),
        dt=0.005,
        contact_active=False,
    )
    protected = ProtectedTask(np.zeros((0, 3)), np.zeros(0))
    result = c.solve(
        st,
        protected,
        q_lower=np.array([-1.0, -1.0, -1.0]),
        q_upper=np.array([1.0, 1.0, 1.0]),
        margin_band=np.array([0.2, 0.2, 0.2]),
    )
    assert result.qp1.success
    assert result.qdot[0] < -1e-3
    assert not result.fault_latched


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
    # Intentional degrade: QP1 → reg-retry → analytic P0 projection (no 3rd QP).
    assert backend.calls == 2
    assert result.qp1.status == "backend_exception"
    assert result.fallback_level == "p0_safe"
    assert not result.fault_latched


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
    # Numeric failure degrades to a P0-feasible command; it must not latch.
    assert not result.fault_latched
    assert result.fallback_level == "p0_safe"


def test_near_soft_rail_p0_damper_hands_xy_to_arm_without_handoff_cost() -> None:
    """P0 qdot clamp + SNS max-α hand XY to the arm; no rail_handoff soft H."""

    n = 3
    c = TwoLevelQpikController(
        n,
        TwoLevelQpikConfig(
            backend="scipy",
            max_solve_ms=500.0,
            max_rows=32,
            max_scalable_groups=1,
            margin_weight=1.0e-2,
            margin_weight_gain=40.0,
            qdot_lower=np.array([0.0, -1.0, -1.0]),  # toward-limit rail clamped
            qdot_upper=np.array([0.02, 1.0, 1.0]),
        ),
    )
    protected = ProtectedTask(np.zeros((0, n)), np.zeros(0))
    tasks = [
        ScalableTask(
            np.array([[1.0, 0.25, 0.0]]),
            np.array([0.2]),
            "motion",
        )
    ]
    st = RobotState(
        q_meas=np.array([0.97, 0.0, 0.0]),
        q_cmd=np.array([0.97, 0.0, 0.0]),
        qdot_applied_prev=np.zeros(n),
        dt=0.005,
        contact_active=False,
    )
    result = c.solve(
        st,
        protected,
        tasks,
        q_lower=np.array([0.0, -1.0, -1.0]),
        q_upper=np.array([1.0, 1.0, 1.0]),
        margin_band=np.array([0.05, 0.2, 0.2]),
        alpha_cap=1.0,
    )
    assert result.qp1.success
    assert result.group_alphas.get("motion", 0.0) > 0.2
    assert result.qdot[0] <= 0.05 + 1e-6
    assert abs(result.qdot[1]) > abs(result.qdot[0]) * 0.5


def test_joint_margin_cost_zero_outside_working_band() -> None:
    c = TwoLevelQpikController(
        3,
        TwoLevelQpikConfig(
            backend="scipy",
            max_solve_ms=500.0,
            max_rows=32,
            max_scalable_groups=1,
            margin_weight=5.0,
            margin_weight_gain=40.0,
            posture_weight=0.0,
            alpha_weight=0.0,
        ),
    )
    H = np.zeros((3, 3))
    g = np.zeros(3)
    c._add_joint_margin_cost(
        H,
        g,
        np.array([0.0, 0.0, 0.0]),
        q_lower=np.array([-1.0, -1.0, -1.0]),
        q_upper=np.array([1.0, 1.0, 1.0]),
        margin_band=np.array([0.2, 0.2, 0.2]),
        n=3,
    )
    np.testing.assert_allclose(H, 0.0)
    np.testing.assert_allclose(g, 0.0)


def test_psi_soft_cost_pulls_along_gradient() -> None:
    c = _controller(3, psi_weight=1.0, psi_k=2.0)
    protected = ProtectedTask(np.zeros((0, 3)), np.zeros(0))
    grad = np.array([0.0, 1.0, 0.0])
    result = c.solve(
        _state(3),
        protected,
        psi_soft={"grad": grad, "err": 0.4, "weight": 1.0, "k": 2.0},
    )
    assert result.qp1.success
    # Soft cost: J qdot ≈ k err → qdot[1] ≈ 0.8
    assert result.qdot[1] > 0.2


def test_heartbeat_pulses_around_backend_solve() -> None:
    beats = {"n": 0}

    def _beat() -> bool:
        beats["n"] += 1
        return True

    c = _controller(2)
    result = c.solve(
        _state(2),
        ProtectedTask([[1.0, 0.0]], [0.1]),
        heartbeat=_beat,
    )
    assert result.qp1.success
    assert beats["n"] >= 2


def test_recursive_sns_alpha_zero_recovers_qp1() -> None:
    """α=0 must keep A_s q̇ = y0 = A_s q1 (q1 always P2-feasible)."""

    c = _controller(3, alpha_weight=0.0, scalable_weight=1.0)
    protected = ProtectedTask(np.array([[1.0, 0.0, 0.0]]), [0.2])
    # Conflicting P0: block the scalable DOF so α cannot rise, but α=0 stays OK.
    hard = HardConstraintRow(np.array([0.0, 1.0, 0.0]), upper=0.0, name="block_y")
    tasks = [ScalableTask(np.array([[0.0, 1.0, 0.0]]), [0.5], "motion")]
    result = c.solve(_state(3), protected, tasks, hard_constraints=[hard])
    assert result.qp1.success
    assert result.qp2.success
    # With Y blocked, q1_y≈0 ⇒ y0≈0; SNS must not force b=0.5.
    assert abs(float((tasks[0].A @ result.qdot).item())) <= 0.05 + 1e-6
    assert result.group_alphas["motion"] == pytest.approx(0.0, abs=0.05)


def test_recursive_sns_b_near_zero_drives_as_q_to_y0() -> None:
    """When b≈0, α→1 means cancel QP1 tangential drift (A_s q → 0)."""

    c = _controller(3, alpha_weight=50.0, scalable_weight=10.0, posture_weight=0.0)
    # Protected pulls X; scalable Y target is ~0 (sin reversal).
    protected = ProtectedTask(np.array([[1.0, 0.2, 0.0]]), [0.3])
    tasks = [ScalableTask(np.array([[0.0, 1.0, 0.0]]), [0.0], "motion")]
    result = c.solve(_state(3), protected, tasks)
    assert result.qp1.success and result.qp2.success
    assert result.group_alphas["motion"] == pytest.approx(1.0, abs=0.05)
    assert abs(float((tasks[0].A @ result.qdot).item())) <= 5e-3


def test_legacy_slack_limits_are_ignored_for_hard_sns() -> None:
    c = _controller(2, alpha_weight=10.0)
    protected = ProtectedTask(np.zeros((0, 2)), np.zeros(0))
    tasks = [
        ScalableTask(
            np.array([[1.0, 0.0]]),
            [0.4],
            "motion",
            slack_limits=np.array([1.0]),  # would have allowed huge residual
        )
    ]
    result = c.solve(_state(2), protected, tasks)
    assert result.qp2.success
    assert result.group_alphas["motion"] == pytest.approx(1.0, abs=0.05)
    np.testing.assert_allclose(tasks[0].A @ result.qdot, [0.4], atol=5e-3)


def test_qp_time_budget_falls_back_without_long_stall() -> None:
    import time

    class SlowBackend:
        name = "slow"

        def __init__(self) -> None:
            self.calls = 0

        def clone(self):
            return SlowBackend()

        def solve(self, H, g, C, lower, upper, x0=None):
            del H, g, C, lower, upper
            from rm75_control.control.joint_admittance_8dof.solver.two_level_qpik import (
                _BackendResult,
            )

            self.calls += 1
            n = 2
            if x0 is None:
                # Fast QP1
                x = np.zeros(n + 1)
                x[0] = 0.1
                return _BackendResult(x, True, "ok", 1, 0.001, "")
            time.sleep(0.02)
            return _BackendResult(
                None, False, "proxqp_time_budget", 0, 0.02, "budget"
            )

    t0 = time.perf_counter()
    c = TwoLevelQpikController(
        2,
        TwoLevelQpikConfig(
            backend=SlowBackend(),
            max_rows=16,
            max_scalable_groups=1,
            max_solve_ms=3.0,
        ),
    )
    result = c.solve(
        _state(2),
        ProtectedTask([[1.0, 0.0]], [0.1]),
        [ScalableTask([[0.0, 1.0]], [0.3], "motion")],
    )
    elapsed_ms = (time.perf_counter() - t0) * 1e3
    assert result.fallback_level == "qp1"
    assert "time_budget" in result.fallback_reason or result.qp2.status == "proxqp_time_budget"
    assert elapsed_ms < 80.0


def test_qp3_diagnostics_populated_on_success() -> None:
    c = _controller(3, posture_weight=1.0e-2)
    protected = ProtectedTask(np.array([[1.0, 0.0, 0.0]]), [0.1])
    tasks = [ScalableTask(np.array([[0.0, 1.0, 0.0]]), [0.2], "motion")]
    guide = {"qdot_guide": np.array([0.0, 0.0, 0.5]), "quality": 1.0}
    result = c.solve(_state(3), protected, tasks, posture_guide=guide)
    assert result.qp3.status not in {"", "not_run"}
    assert result.qp3.solve_time_ms >= 0.0
    assert result.qp3.iterations >= 0


def test_qp2_fallback_reports_projected_or_zero_alpha() -> None:
    class FailQp2Backend:
        name = "fail2"

        def __init__(self) -> None:
            self.calls = 0

        def clone(self):
            return FailQp2Backend()

        def solve(self, H, g, C, lower, upper, x0=None):
            del H, g, C, lower, upper
            self.calls += 1
            from rm75_control.control.joint_admittance_8dof.solver.two_level_qpik import (
                _BackendResult,
            )

            n = 2
            # QP1 succeeds with protected-only motion; QP2 (warm-started) fails.
            if x0 is not None:
                return _BackendResult(None, False, "forced_fail", 0, 0.0, "fail")
            x = np.zeros(n + 1)
            x[0] = 0.1
            return _BackendResult(x, True, "ok", 1, 0.0, "")

    backend = FailQp2Backend()
    c = TwoLevelQpikController(
        2,
        TwoLevelQpikConfig(
            backend=backend,
            max_rows=16,
            max_scalable_groups=1,
        ),
    )
    result = c.solve(
        _state(2),
        ProtectedTask([[1.0, 0.0]], [0.1]),
        [ScalableTask([[0.0, 1.0]], [0.3], "motion")],
        alpha_cap=1.0,
    )
    assert result.fallback_level == "qp1"
    # q1 has no Y motion → projected α for motion is 0 (no 0.25 forgery).
    assert result.group_alphas["motion"] == pytest.approx(0.0)
