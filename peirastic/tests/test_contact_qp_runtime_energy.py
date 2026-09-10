"""Actual port evidence, command-model liabilities, and one-tank isolation."""
import numpy as np
import pytest

from peirastic.contact_qp.energy import EnergyLedger, ExposureSegment, PortBounds, PortInterval
from peirastic.contact_qp.runtime_energy import RuntimeEnergy


X = np.array([1.0, 0, 0, 0, 0, 0])


def runtime(*, enabled=True, initial=1.0, capacity=2.0, reserve=0.1, **kwargs):
    bounds = PortBounds(reference="tcp_tool", calibration_version="physical-v1", **kwargs)
    ledger = EnergyLedger(initial, capacity, reserve, bounds)
    return RuntimeEnergy(ledger, command_budget_enforced=enabled)


def observe(rt, stamp, velocity=X, *, wrench=-X, **kwargs):
    args = dict(source_t_s=stamp, source_id=("actual-pair", stamp), now_s=stamp,
                valid=True, time_aligned=True, physical_w_checked=True,
                reference="tcp_tool", calibration_version="physical-v1")
    args.update(kwargs)
    return rt.observe_port(wrench, velocity, **args)


def snapshot(rt, now=1.0, *, wrench=-X, hold=0.01, **kwargs):
    args = dict(now_s=now, source_t_s=now, wrench_environment=wrench, hold_s=hold)
    args.update(kwargs)
    return rt.snapshot(**args)


def test_full_six_dimensional_net_work_cancels_before_charging():
    rt = runtime()
    w = np.array([-2, 0, 0, 0, 2, 0.0])
    v = np.array([1, 0, 0, 0, 1, 0.0])
    assert not observe(rt, 1.0, v, wrench=w)
    assert observe(rt, 1.01, v, wrench=w)
    assert rt.ledger.balance_j == 1.0
    assert rt.events[-1]["work_estimate_j"] == 0.0
    assert rt.events[-1]["wrench_start"] == w.tolist()
    assert rt.events[-1]["velocity_end"] == v.tolist()
    assert rt.events[-1]["calibration_version"] == "physical-v1"
    assert not rt.facts["certified"]
    assert rt.facts["monitor_status"] == "monitor_estimated"


def test_measured_recovery_uses_same_ledger_only_after_completed_interval():
    rt = runtime()
    constraint = snapshot(rt, wrench=X)
    before = rt.ledger.snapshot()
    assert rt.reserve(1, constraint, X, now_s=1.0)
    assert rt.ledger.balance_j == before["balance_j"]
    assert rt.events[-1]["predicted_recovery_credited_j"] == 0.0
    assert not observe(rt, 1.0, wrench=X)
    assert rt.ledger.balance_j == 1.0
    assert observe(rt, 1.01, wrench=X)
    assert rt.ledger.balance_j == pytest.approx(1.01)
    assert rt.events[-1]["work_estimate_j"] == pytest.approx(-0.01)
    assert rt.events[-1]["balance_change_j"] == pytest.approx(0.01)


def test_capacity_dissipation_and_overdraw_keep_truth():
    rt = runtime(initial=0.01, capacity=0.02, reserve=0.0)
    observe(rt, 1.0, 10 * X, wrench=X)
    assert observe(rt, 1.01, 10 * X, wrench=X)
    assert rt.ledger.balance_j == 0.02
    # A conservative max-endpoint estimate first charges the adverse endpoint.
    assert observe(rt, 1.02, 10 * X, wrench=-X)
    assert rt.ledger.balance_j == pytest.approx(-0.08)
    with pytest.raises(ValueError, match="available_j"):
        snapshot(rt, now=1.02)


def test_duplicate_poll_is_ignored_without_losing_consecutive_evidence():
    rt = runtime()
    observe(rt, 1.0)
    assert not observe(rt, 1.0, now_s=1.003)
    assert observe(rt, 1.01)
    balance = rt.ledger.balance_j
    assert not observe(rt, 1.01, now_s=1.012)
    assert rt.ledger.balance_j == balance
    assert observe(rt, 1.02)
    assert rt.ledger.balance_j == pytest.approx(0.98)


@pytest.mark.parametrize("invalid", [
    {"valid": False}, {"time_aligned": False}, {"physical_w_checked": False},
    {"reference": "wrong-point"}, {"calibration_version": "wrong-version"},
    {"source_id": None}, {"valid": "true"},
])
def test_bad_measurement_fences_interpolation_and_retains_old_liability(invalid):
    rt = runtime()
    rt.ledger.reserve("old", [ExposureSegment(1.0, 1.03, 1.0, "old_or_mixed")])
    observe(rt, 1.0)
    assert not observe(rt, 1.01, wrench=X, **invalid)
    assert rt.facts["monitor_status"] == "monitor_unavailable"
    assert rt.ledger.balance_j == 1.0
    assert rt.ledger.liability("old") == pytest.approx(0.03)
    assert not observe(rt, 1.02)
    assert observe(rt, 1.03)
    assert rt.ledger.balance_j == pytest.approx(0.99)
    # Only measured [1.02,1.03] releases its overlapping earmark.
    assert rt.ledger.liability("old") == pytest.approx(0.02)


def test_missing_interval_never_releases_old_work_or_interpolates_across_gap():
    rt = runtime()
    rt.ledger.reserve("old", [ExposureSegment(1.0, 1.10, 1.0, "old_or_mixed")])
    observe(rt, 1.0)
    assert not observe(rt, 1.04, wrench=X)
    assert rt.ledger.balance_j == 1.0
    assert rt.ledger.liability("old") == pytest.approx(0.1)
    assert observe(rt, 1.05, wrench=X)
    assert rt.ledger.balance_j == pytest.approx(1.01)
    assert rt.ledger.liability("old") == pytest.approx(0.09)


def test_delayed_actual_old_command_work_settles_once_not_predicted_velocity():
    rt = runtime()
    rt.ledger.reserve("old", [ExposureSegment(1.0, 1.02, 3.0, "old_or_mixed")])
    candidate = snapshot(rt, 1.02, wrench=X)
    assert rt.reserve("new", candidate, 10 * X, now_s=1.02)
    assert rt.reject_new_only("new", definitely_not_sent=True)
    assert not observe(rt, 1.0, 2 * X, now_s=1.03)
    assert observe(rt, 1.01, 2 * X, now_s=1.031)
    assert rt.ledger.balance_j == pytest.approx(0.98)
    assert rt.ledger.liability("old") == pytest.approx(0.03)
    assert not rt.reject_new_only("old", definitely_not_sent=True)


@pytest.mark.parametrize("bad_sample", [
    {"source_t_s": float("nan")}, {"now_s": float("inf")},
    {"source_t_s": 1.02, "now_s": 1.01},
    {"source_id": ("actual-pair", 1.0)},
    {"now_s": 1.2},
])
def test_anomalous_source_fails_closed(bad_sample):
    rt = runtime()
    observe(rt, 1.0)
    assert not observe(rt, 1.01, wrench=X, **bad_sample)
    assert rt.ledger.balance_j == 1.0
    assert rt.facts["monitor_status"] == "monitor_unavailable"


def test_regressing_now_and_out_of_order_samples_cannot_create_recovery():
    rt = runtime()
    observe(rt, 1.0, wrench=X)
    assert not observe(rt, 0.99, wrench=X, now_s=1.01)
    assert not observe(rt, 1.01, wrench=X, now_s=1.005)
    assert not observe(rt, 1.02, wrench=X)
    assert rt.ledger.balance_j == 1.0
    assert observe(rt, 1.03, wrench=X)
    assert rt.ledger.balance_j == pytest.approx(1.01)


def test_previously_settled_overlapping_physical_interval_cannot_settle_again():
    rt = runtime()
    interval = PortInterval(1.0, 1.01, X, X, X, X,
                            reference="tcp_tool", calibration_version="physical-v1",
                            time_aligned=True)
    assert rt.ledger.settle(interval, now_s=1.01)
    balance = rt.ledger.balance_j
    observe(rt, 1.0, wrench=X, now_s=1.01)
    assert not observe(rt, 1.01, wrench=X)
    assert rt.ledger.balance_j == balance
    assert rt.facts["monitor_status"] == "monitor_unavailable"


def test_port_overflow_never_credits_or_releases_reservations():
    rt = runtime()
    rt.ledger.reserve("old", [ExposureSegment(1, 1.02, 1)])
    observe(rt, 1.0)
    assert not observe(rt, 1.01, 1e308 * X, wrench=1e308 * X)
    assert rt.ledger.balance_j == 1.0
    assert rt.ledger.liability("old") == 0.020000000000000018


def test_snapshot_is_read_only_source_age_is_counted_and_final_age_is_rechecked():
    rt = runtime(initial=0.2, reserve=0.0)
    before = rt.ledger.snapshot()
    candidate = snapshot(rt, 1.02, source_t_s=1.0,
                         wrench_error=0.1 * X, wrench_rate=10 * X)
    assert candidate.wrench_error[0] == pytest.approx(0.3)
    assert rt.ledger.snapshot() == before
    assert rt.reserve(1, candidate, X, now_s=1.03)
    # W=-1, publication error=.4, integrated rate uncertainty=.05.
    assert rt.ledger.liability(1) == pytest.approx(0.0145)
    assert rt.ledger.balance_j == 0.2
    assert rt.events[-1]["wrench_error"][0] == pytest.approx(0.4)


def test_final_payload_velocity_and_new_balance_control_admission():
    rt = runtime(initial=0.02, reserve=0.0)
    candidate = snapshot(rt)
    assert candidate.admissible(X)
    assert not rt.reserve(1, candidate, 3 * X, now_s=1.0)
    assert rt.ledger.reserved_j == 0.0
    candidate = snapshot(rt)
    rt.ledger.reserve("other", [ExposureSegment(1, 1.01, 1.5)])
    assert not rt.reserve(2, candidate, X, now_s=1.0)
    assert rt.ledger.balance_j == 0.02


def test_final_source_age_can_make_previously_feasible_command_infeasible():
    rt = runtime(initial=0.012, reserve=0.0)
    candidate = snapshot(rt, wrench_rate=10 * X)
    assert candidate.admissible(X)
    assert not rt.reserve(1, candidate, X, now_s=1.03)
    assert rt.ledger.reserved_j == 0.0


def test_final_velocity_respects_damping_contact_speed_declaration():
    rt = runtime()
    candidate = snapshot(rt, wrench=X, contact_map=X.reshape(1, 6),
                         damping=np.array([1.0]), contact_speed_bound=np.array([0.5]))
    assert not rt.reserve(1, candidate, X, now_s=1.0)


def test_single_use_snapshot_and_unique_command_reservation():
    rt = runtime()
    old = snapshot(rt)
    candidate = snapshot(rt)
    with pytest.raises(ValueError, match="retired"):
        rt.reserve(1, old, X, now_s=1.0)
    assert rt.reserve(1, candidate, X, now_s=1.0)
    with pytest.raises(ValueError, match="consumed"):
        rt.reserve(2, candidate, X, now_s=1.0)
    assert not rt.reserve(1, snapshot(rt), X, now_s=1.0)
    assert rt.ledger.reserved_j == pytest.approx(0.01)


def test_no_send_refunds_only_new_proven_unexposed_candidate():
    rt = runtime()
    rt.ledger.reserve("old", [ExposureSegment(1, 1.02, 1, "old_or_mixed")])
    assert rt.reserve(1, snapshot(rt), X, now_s=1.0)
    assert not rt.reject_new_only(1, definitely_not_sent=False)
    assert rt.reject_new_only(1, definitely_not_sent=True)
    assert rt.ledger.liability("old") == pytest.approx(0.02)
    assert rt.ledger.balance_j == 1.0
    assert rt.reserve(2, snapshot(rt), X, now_s=1.0)
    rt.publication_started(2)
    assert not rt.reject_new_only(2, definitely_not_sent=True)
    assert rt.ledger.liability(2) == pytest.approx(0.01)


def test_monitor_only_does_not_add_energy_row_or_require_command_inputs():
    rt = runtime(enabled=False)
    assert snapshot(rt, now=float("nan"), wrench=None) is None
    assert rt.facts["energy_constraint_enabled"] is False
    observe(rt, 1)
    assert observe(rt, 1.01)
    assert rt.ledger.balance_j == pytest.approx(0.99)
    assert any(e.get("reason") == "execution_exceeded_reserved_power_envelope"
               for e in rt.ledger.events)
    assert rt.facts["energy_constraint_enabled"] is False
    assert rt.facts["physical_assurance"] == "unverified"
    with pytest.raises(ValueError, match="pure monitoring"):
        rt.reserve(1, None, X, now_s=1.01)


def test_validity_and_command_assurance_never_promote_physical_certification():
    rt = runtime()
    constraint = snapshot(rt, assurance="monitor")
    assert constraint is not None  # monitor assurance still has an enabled row
    assert rt.reserve(1, constraint, X, now_s=1)
    assert not rt.facts["certified"]
    assert rt.facts["physical_assurance"] == "unverified"
    assert rt.facts["monitor_status"] == "monitor_unavailable"
    with pytest.raises(ValueError, match="cannot certify"):
        snapshot(rt, assurance="declared_bound", bounds_version="claimed")
    with pytest.raises(ValueError, match="unverified"):
        RuntimeEnergy(EnergyLedger(1, 1, 0, PortBounds(verified=True)))


def test_snapshot_invalid_bounds_expiry_and_overflow_fail_closed():
    rt = runtime()
    for kwargs in ({"wrench_rate": -X}, {"wrench_error": np.full(6, np.inf)},
                   {"source_t_s": 2.0}, {"source_t_s": 0.0}):
        with pytest.raises(ValueError):
            snapshot(rt, **kwargs)
    with pytest.raises(ValueError):
        snapshot(rt, tracking_error=np.full(6, 1e308), wrench=np.full(6, 1e308))
    assert not rt.reserve(1, snapshot(rt), X, now_s=1.1)
    assert not rt.reserve(2, snapshot(rt), X, now_s=0.9)
    assert rt.ledger.balance_j == 1.0 and rt.ledger.reserved_j == 0.0


def test_estimator_records_declared_error_terms_without_certifying_them():
    rt = runtime(wrench_error=0.1 * X, velocity_error=0.2 * X, power_rate_bound_w_s=2)
    observe(rt, 1.0)
    assert observe(rt, 1.01)
    # -W.V=1; component errors=.2+.1+.02; rate interval term=.01.
    assert rt.events[-1]["work_estimate_j"] == pytest.approx(0.0133)
    assert rt.events[-1]["wrench_error"] == (0.1 * X).tolist()
    assert rt.events[-1]["velocity_error"] == (0.2 * X).tolist()
    assert rt.events[-1]["power_rate_bound_w_s"] == 2
    assert rt.events[-1]["estimated"] and not rt.events[-1]["certified"]


def test_event_drain_keeps_unsettled_liabilities_and_following_measurement_evidence():
    rt = runtime()
    rt.ledger.reserve("old", [ExposureSegment(1, 1.04, 1)])
    observe(rt, 1.0)
    assert observe(rt, 1.01)
    remaining = rt.ledger.liability("old")
    events = rt.drain_events()
    assert events["runtime"][-1]["event"] == "measured_port_estimate"
    assert events["ledger"][-1]["event"] == "settled"
    assert rt.events == [] and rt.ledger.events == []
    assert rt.ledger.liability("old") == remaining
    assert observe(rt, 1.02)
    assert rt.events[-1]["work_estimate_j"] == pytest.approx(0.01)
