from dataclasses import replace

import numpy as np
import pytest

from peirastic.contact_qp.energy import EnergyLedger, PortBounds
from peirastic.contact_qp.execution import DeviceState, ExecutionBounds, ExecutionCoordinator, FinalCommand
from peirastic.contact_qp.geometry import motion_basis
from peirastic.contact_qp.types import TwistConstraints


def setup(initial=10.):
    ledger = EnergyLedger(initial, initial, initial*.1, PortBounds(verified=True, calibration_version="test"))
    coordinator = ExecutionCoordinator(ledger, ExecutionBounds(verified=True))
    command = FinalCommand(1, 0, np.zeros(7), .2, [.005, 0, .001, 0, 0, 0],
                           [.005, 0, .001, 0, 0, 0], [.004, 0, .001, 0, 0, 0], [.001, 0, 0, 0, 0, 0], True)
    cert = TwistConstraints(np.eye(6), np.full(6, -.1), np.full(6, .1), valid_until_s=1.01, sequence=1)
    return coordinator, command, cert


def review(c, command, cert, now=1.):
    return c.review(command, cert, now_s=now, h=motion_basis([.01, 0, 0, 0, 0, 0]),
                    alpha=.5, dt_s=.005, wrench_environment=[0, 0, -4, 0, 0, 0])


def publish(c, p, device, now=1.001):
    c.before_send(p, device, now_s=now)
    c.device_result(p, device, DeviceState.SENT, now_s=now+.0001)


def test_success_commits_bounded_progress_once_and_does_not_claim_motion():
    c, cmd, cert = setup(); p = review(c, cmd, cert)
    publish(c, p, "arm"); publish(c, p, "rail", 1.002)
    assert c.commit(p, now_s=1.003) == pytest.approx(.0025)
    assert c.commit(p, now_s=1.004) == 0
    assert c.reference_s == pytest.approx(.0025)
    assert all(v == DeviceState.SENT for v in p.states.values())
    assert c.ledger.reserved_j > 0 and c.ledger.balance_j == 10
    with pytest.raises(ValueError, match="rolled back"):
        c.abort(p, "cannot undo sends")


def test_rejected_new_command_keeps_previous_motion_and_budget():
    c, cmd, cert = setup(); p = review(c, cmd, cert)
    publish(c, p, "arm"); publish(c, p, "rail", 1.002); c.commit(p, now_s=1.003)
    next_cmd = replace(cmd, sequence=2)
    p2 = review(c, next_cmd, replace(cert, sequence=2, valid_until_s=1.02), now=1.005)
    reserve = c.ledger.reserved_j
    c.device_result(p2, "arm", DeviceState.REJECTED, now_s=1.006)
    c.abort(p2, "arm refused before I/O")
    assert c.ledger.reserved_j == reserve and c.reference_s == pytest.approx(.0025)
    assert np.linalg.norm(c.held["arm"]+c.held["rail"]) > 0


def test_partial_arm_publish_is_not_rolled_back_or_awarded_progress():
    c, cmd, cert = setup(); p = review(c, cmd, cert)
    publish(c, p, "arm")
    c.device_result(p, "rail", DeviceState.REJECTED, now_s=1.002)
    assert c.commit(p, now_s=1.003) == 0 and not c.execution_known
    assert p.states["arm"] == DeviceState.SENT and p.states["rail"] == DeviceState.REJECTED
    assert c.ledger.liability(1) > 0
    with pytest.raises(ValueError, match="reconciled"):
        review(c, replace(cmd, sequence=2), replace(cert, sequence=2), now=1.004)


def test_timeout_late_execution_old_epoch_and_duplicate_ack_are_facts_only():
    c, cmd, cert = setup(); p = review(c, cmd, cert)
    c.before_send(p, "arm", now_s=1.001)
    c.device_result(p, "arm", DeviceState.UNKNOWN, now_s=1.002)
    c.abort(p, "SDK timeout may still execute")
    epoch = c.stop(now_s=1.003, wrench_environment=[0, 0, -4, 0, 0, 0])
    assert epoch == 1 and c.ledger.reserved_j > 0
    assert c.device_result(p, "arm", DeviceState.ACKNOWLEDGED, now_s=1.5)
    assert not c.device_result(p, "arm", DeviceState.ACKNOWLEDGED, now_s=1.501)
    assert not c.device_result(p, "arm", DeviceState.SENT, now_s=1.502)
    assert not c.execution_known and not c.ledger.certification_valid and c.reference_s == 0
    with pytest.raises(ValueError, match="stale"):
        c.before_send(p, "rail", now_s=1.005)


@pytest.mark.parametrize("change", ["expiry", "epoch", "payload", "hard", "mechanical", "long_tick"])
def test_last_modification_and_last_instant_review_fail_closed(change):
    c, cmd, cert = setup()
    if change == "hard":
        with pytest.raises(ValueError, match="hard"):
            review(c, cmd, replace(cert, upper=np.zeros(6)))
        return
    if change == "mechanical":
        with pytest.raises(ValueError, match="hard"):
            review(c, replace(cmd, mechanical_valid=False), cert)
        return
    if change == "long_tick":
        with pytest.raises(ValueError, match="step"):
            c.review(cmd, cert, now_s=1, h=motion_basis([.01, 0, 0, 0, 0, 0]), alpha=.5,
                     dt_s=3, wrench_environment=np.zeros(6))
        return
    p = review(c, cmd, cert)
    if change == "epoch":
        c.stop(now_s=1.001, wrench_environment=np.zeros(6))
    with pytest.raises(ValueError, match="stale|modified"):
        c.before_send(p, "arm", now_s=1.02 if change == "expiry" else 1.002,
                      command=replace(cmd, rail_target_m=.2001) if change == "payload" else cmd)
    assert c.reference_s == 0 and c.ledger.liability(1) > 0


def test_budget_exhaustion_does_not_block_stop_or_erase_unfunded_debt():
    c, cmd, cert = setup(.001)
    with pytest.raises(ValueError, match="budget"):
        review(c, cmd, cert)
    c.stop(now_s=1., wrench_environment=[0, 0, -4, 0, 0, 0])
    assert c.stop_epoch == 1 and c.ledger.available_j < 0 and not c.ledger.certification_valid
    assert not c.reconcile_stopped(now_s=1.5, measured_tool=np.zeros(6), feedback_valid=True,
                                   queues_fenced=True, no_inflight=False)
    assert c.reconcile_stopped(now_s=1.5, measured_tool=np.zeros(6), feedback_valid=True,
                               queues_fenced=True, no_inflight=True, arm_stopped=True, rail_stopped=True)
    assert c.ledger.reserved_j > 0  # Known stop is not measured energy settlement.


def test_review_snapshot_is_immutable_and_only_one_candidate_can_be_outstanding():
    from dataclasses import FrozenInstanceError
    c, cmd, cert = setup(); p = review(c, cmd, cert)
    with pytest.raises(FrozenInstanceError):
        p.command = replace(cmd, arm_payload=np.ones(7))
    with pytest.raises(FrozenInstanceError):
        p.progress_s = 50.
    with pytest.raises(TypeError):
        p.states["arm"] = DeviceState.SENT
    with pytest.raises(ValueError, match="outstanding"):
        review(c, replace(cmd, sequence=2), replace(cert, sequence=2), now=1.001)


def test_final_commit_rechecks_budget_and_event_time():
    for fault in ("energy", "clock"):
        c, cmd, cert = setup(); p = review(c, cmd, cert)
        publish(c, p, "arm"); publish(c, p, "rail", 1.002)
        if fault == "energy":
            c.ledger.invalidate("injected port bound violation")
        assert c.commit(p, now_s=1.001 if fault == "clock" else 1.003) == 0
        assert c.reference_s == 0 and c.ledger.reserved_j > 0


def test_publish_delay_is_independent_of_certificate_expiry_and_cancellation_is_not_stop():
    c, cmd, cert = setup(); p = review(c, cmd, replace(cert, valid_until_s=2.))
    with pytest.raises(ValueError, match="stale"):
        c.before_send(p, "arm", now_s=1.051)
    assert not c.reconcile_stopped(now_s=1.2, measured_tool=np.zeros(6), feedback_valid=True,
                                   queues_fenced=True, no_inflight=True)


def test_stop_retires_pending_without_refund_and_allows_evidenced_new_epoch():
    c, cmd, cert = setup(); p = review(c, cmd, cert)
    c.stop(now_s=1.001, wrench_environment=np.zeros(6))
    assert p.aborted and c.ledger.liability(1) > 0
    assert c.reconcile_stopped(now_s=1.002, measured_tool=np.zeros(6), feedback_valid=True,
                               queues_fenced=True, no_inflight=True, arm_stopped=True, rail_stopped=True)
    next_command = replace(cmd, sequence=2, stop_epoch=1)
    next_cert = replace(cert, sequence=2, stop_epoch=1)
    assert review(c, next_command, next_cert, now=1.003).command.sequence == 2
