"""Bounded numerical retries keep the original problem and admission tolerances.

T7 fixtures are open-loop reconstructions, not exact physical/closed-loop replay.
The rejected sample's unlogged wrench/energy use its last logged successful row.
"""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import proxsuite
import pytest

from peirastic.contact_qp import qp as module
from peirastic.contact_qp.qp import ContactQp, QpConfig, _violation
from peirastic.contact_qp.types import ContactStatus
from peirastic.tests.test_contact_qp_solver import datum


FIXTURE = Path(__file__).parent / "fixtures" / "contact_qp_t7_numeric.npz"
BOUNDED = QpConfig(solver_policy="bounded_retry_v1")


def problem(prefix):
    with np.load(FIXTURE) as saved:
        return tuple(saved[prefix + key].copy() for key in ("H", "g", "C", "l", "u"))


def assert_accepted(solver, matrices, result):
    x, status, solved, _ = result
    assert solved and status.endswith("PROXQP_SOLVED"), solver._numeric_attempts
    assert np.isfinite(x).all()
    assert _violation(*matrices[2:], x) <= BOUNDED.feasibility_tolerance
    accepted = solver._numeric_attempts[-1]
    assert accepted["accepted"]
    assert accepted["primal_residual"] <= BOUNDED.solver_tolerance
    assert accepted["dual_residual"] <= BOUNDED.solver_tolerance
    assert abs(accepted["duality_gap"]) <= BOUNDED.solver_tolerance
    assert len(solver._numeric_attempts) <= 2


@pytest.mark.parametrize("case", range(8))
def test_eight_recorded_failures_solve_without_changing_problem(case):
    matrices = problem(f"case_{case:02d}_")
    originals = tuple(value.copy() for value in matrices)
    solver = ContactQp(BOUNDED)
    assert_accepted(solver, matrices, solver._solve_numeric(*matrices, energy_active=True))
    for current, original in zip(matrices, originals):
        np.testing.assert_array_equal(current, original)


def test_twenty_sample_history_preserves_preconditioning_failure_regression():
    legacy, bounded = ContactQp(), ContactQp(BOUNDED)
    for index in range(21):
        matrices = problem(f"history_{index}_")
        old_result = legacy._solve_numeric(*matrices, energy_active=True)
        new_result = bounded._solve_numeric(*matrices, energy_active=True)
        assert_accepted(bounded, matrices, new_result)
    # Backend revisions may repair the old numerical instability; assert the
    # captured rejection only on the library version used for this fixture.
    if proxsuite.__version__ == "0.7.3":
        assert not old_result[2]
        assert old_result[3] == 19303
    assert bounded._numeric_attempts[-1]["max_inner_iterations"] == 10


@pytest.mark.parametrize("available", [.543, .0001, 0.])
def test_previous_003_energy_regression_still_passes_with_bounded_policy(monkeypatch, available):
    from peirastic.tests import test_contact_qp_recorded_energy_failure as recorded

    def bounded_solver(config):
        return ContactQp(replace(config, solver_policy="bounded_retry_v1"))

    monkeypatch.setattr(recorded, "ContactQp", bounded_solver)
    recorded.test_recorded_failure_keeps_energy_row_and_solves_or_refuses(available)


def fake_backend(monkeypatch, outcomes, clock=None, expire_at=None):
    """Inject native outcomes without weakening the production acceptance gate."""
    calls = []

    class Fake:
        def __init__(self, *shape):
            self.settings = SimpleNamespace()
            self.results = SimpleNamespace()
            calls.append(self)
            self.solved = False

        def init(self, H, g, A, b, C, l, u, **kwargs):
            self.problem = tuple(np.array(value, copy=True) for value in (H, g, C, l, u))
            self.init_kwargs = kwargs
            if expire_at == "init":
                clock[0] = 1.

        def update(self, **kwargs):
            self.problem = tuple(np.array(kwargs[key], copy=True) for key in ("H", "g", "C", "l", "u"))

        def solve(self):
            self.solved = True
            outcome = outcomes[min(len(calls)-1, len(outcomes)-1)]
            self.results.x = np.asarray(outcome.get("x", [0.]), dtype=float)
            self.results.z = np.asarray(outcome.get("z", [0.]), dtype=float)
            values = dict(status=proxsuite.proxqp.QPSolverOutput.PROXQP_SOLVED,
                          pri_res=0., dua_res=0., duality_gap=0., iter=3, iter_ext=2)
            values.update({key: value for key, value in outcome.items() if key not in ("x", "z")})
            self.results.info = SimpleNamespace(**values)
            if expire_at == "solve":
                clock[0] = 1.

    monkeypatch.setattr(proxsuite.proxqp.dense, "QP", Fake)
    return calls


def tiny_problem():
    return np.eye(1), np.zeros(1), np.eye(1), -np.ones(1), np.ones(1)


@pytest.mark.parametrize("invalid", [
    {"x": [np.nan]}, {"z": [np.nan]}, {"x": [2.]},
    {"pri_res": 1e-7}, {"dua_res": 1e-7}, {"duality_gap": -1e-7},
    {"duality_gap": np.nan},
    {"status": proxsuite.proxqp.QPSolverOutput.PROXQP_MAX_ITER_REACHED},
])
def test_native_solved_flag_cannot_bypass_strict_acceptance(monkeypatch, invalid):
    calls = fake_backend(monkeypatch, [invalid, invalid])
    solver = ContactQp(BOUNDED)
    result = solver._solve_numeric(*tiny_problem(), energy_active=True)
    assert not result[2]
    assert len(calls) == 2 and all(call.solved for call in calls)
    assert solver._numeric_deferred_reason == "solver_attempts_exhausted"
    assert solver._solver is None and solver._solver_shape is None
    assert not any(attempt["accepted"] for attempt in solver._numeric_attempts)


def test_retry_is_fresh_has_fixed_caps_and_keeps_every_problem_entry(monkeypatch):
    calls = fake_backend(monkeypatch, [{"x": [np.nan]}, {}])
    solver = ContactQp(BOUNDED)
    matrices = tiny_problem()
    assert solver._solve_numeric(*matrices, energy_active=True)[2]
    assert len(calls) == 2
    assert [call.settings.max_iter for call in calls] == [30, 200]
    assert [call.settings.max_iter_in for call in calls] == [10, 10]
    assert calls[1].init_kwargs == dict(compute_preconditioner=True, rho=1e-3)
    for call in calls:
        assert call.settings.check_duality_gap
        for got, expected in zip(call.problem, matrices):
            np.testing.assert_array_equal(got, expected)
    # A successful backup is not the next proposal's changed numerical policy.
    assert solver._solver is None


@pytest.mark.parametrize("expire_at", ["before", "init", "solve"])
def test_absolute_deadline_fences_initialization_and_native_result(monkeypatch, expire_at):
    clock = [1. if expire_at == "before" else 0.]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    calls = fake_backend(monkeypatch, [{}, {}], clock, expire_at)
    solver = ContactQp(BOUNDED)
    assert not solver._solve_numeric(*tiny_problem(), energy_active=True, deadline_s=1.)[2]
    assert solver._numeric_deferred_reason == "solver_deadline_exceeded"
    assert len(calls) == (0 if expire_at == "before" else 1)
    if expire_at == "init":
        assert not calls[0].solved
    assert solver._solver is None


def test_online_refusal_contains_replay_input_without_lp_or_projection(monkeypatch):
    solver = ContactQp(BOUNDED)
    monkeypatch.setattr(solver, "_solve_numeric", lambda *a, **k: (np.zeros(5), "injected", False, 1))
    forbidden = lambda *a, **k: pytest.fail("online failure ran offline feasibility diagnostics")
    monkeypatch.setattr(module, "_linear_feasible", forbidden)
    monkeypatch.setattr(module, "_omega_interval", forbidden)
    data = datum((.1, .1, .1))
    result = solver.solve(data, online=True, interval_diagnostics=True)
    assert result.status == ContactStatus.DEFERRED and result.qp_twist is None
    assert not result.success and not result.final_velocity_admissible(np.zeros(6))
    assert result.diagnostics["retryable"]
    assert result.diagnostics["reason"] == "solver_attempts_exhausted"
    np.testing.assert_array_equal(result.diagnostics["qp_input"]["nominal_twist"], data.nominal_twist)
    assert set(result.diagnostics["numeric_problem"]) == {"H", "g", "C", "l", "u"}
    assert not result.diagnostics["interval_diagnostics_computed"]


def test_transparent_candidate_cannot_escape_deadline(monkeypatch):
    monkeypatch.setattr(module.time, "monotonic", lambda: 1.)
    result = ContactQp(BOUNDED).solve(datum(), online=True, deadline_s=1.)
    assert result.status == ContactStatus.DEFERRED
    assert result.diagnostics["reason"] == "solver_deadline_exceeded"


def test_deadline_is_checked_after_transparent_result_assembly(monkeypatch):
    clock = iter((0., 1.))
    monkeypatch.setattr(module.time, "monotonic", lambda: next(clock))
    result = ContactQp(BOUNDED).solve(datum(), online=True, deadline_s=1.)
    assert result.status == ContactStatus.DEFERRED and not result.success
    assert result.diagnostics["transparent"]


def test_infeasible_numeric_problem_is_never_promoted_to_command():
    solver = ContactQp(BOUNDED)
    H, g = np.eye(1), np.zeros(1)
    C, lo, hi = np.ones((2, 1)), np.array([1., -np.inf]), np.array([np.inf, -1.])
    result = solver._solve_numeric(H, g, C, lo, hi, energy_active=True)
    assert not result[2]
    assert solver._numeric_deferred_reason == "solver_attempts_exhausted"
    assert len(solver._numeric_attempts) == 2
    assert not any(attempt["accepted"] for attempt in solver._numeric_attempts)


@pytest.mark.parametrize("deadline", [np.nan, np.inf, -1., True])
def test_deadline_requires_valid_absolute_time(deadline):
    with pytest.raises(ValueError):
        ContactQp(BOUNDED).solve(datum(), deadline_s=deadline)


def test_solver_policy_is_explicit_and_legacy_remains_default():
    assert QpConfig().solver_policy == "legacy_v1"
    with pytest.raises(ValueError, match="solver policy"):
        QpConfig(solver_policy="unbounded_retry")
