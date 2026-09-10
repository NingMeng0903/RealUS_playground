from types import SimpleNamespace
from dataclasses import replace

import numpy as np
import pytest

from peirastic.contact_qp.energy import EnergyLedger, PortBounds
from peirastic.contact_qp.execution import ExecutionBounds, ExecutionCoordinator, DeviceState
from peirastic.contact_qp.geometry import motion_basis
from peirastic.contact_qp.types import TwistConstraints
from rm75_control.control.joint_admittance_8dof.hw.contact_publication import publish_contact_candidate
from rm75_control.control.joint_admittance_8dof.hw.rail_servo import RailServoBridge, RailServoConfig


def fixture(monkeypatch):
    clock = [1.]
    monkeypatch.setattr("rm75_control.control.joint_admittance_8dof.hw.rail_servo.time.monotonic", lambda: clock[0])
    rail = RailServoBridge(RailServoConfig(enabled=True))
    rail._armed = rail._calibrated = True
    rail._measured_m = rail._target_m = .4
    ledger = EnergyLedger(10, 10, 1, PortBounds(verified=True, calibration_version="test"))
    execution = ExecutionCoordinator(ledger, ExecutionBounds(verified=True))
    cert = TwistConstraints(np.eye(6), np.full(6, -.1), np.full(6, .1), valid_until_s=1.01, sequence=1)
    outer = SimpleNamespace(execution=execution, pending_result=SimpleNamespace(hard_constraints=cert, alpha=.5),
                            pending_basis=motion_basis([.01, 0, 0, 0, 0, 0]), pending_dt=.005,
                            pending_wrench_environment=np.array([0, 0, -4, 0, 0, 0]),
                            pending_rotation_base_tcp=np.eye(3), nominal_committed=False, aborted=False)
    def commit(p, *, now_s):
        execution.commit(p, now_s=now_s)
        outer.nominal_committed = p.committed
    outer.publication_commit = commit
    outer.publication_abort = lambda reason: setattr(outer, "aborted", True)
    inner = SimpleNamespace(commits=[], aborts=[])
    inner.commit_publication = lambda q: (inner.commits.append(q.copy()) or True)
    inner.abort_publication = lambda: inner.aborts.append(True)
    step = SimpleNamespace(q_send=np.array([.4, 0, 0, 0, 0, 0, 0, 0]), qdot=np.array([.001, 0, 0, 0, 0, 0, 0, 0]),
                           cartesian_constraints_valid=True, cartesian_constraints_sequence=1, cartesian_constraints_stop_epoch=0,
                           v_tcp_commanded=np.array([.005, 0, .001, 0, 0, 0]), v_tcp_estimated=np.array([.005, 0, .001, 0, 0, 0]),
                           arm_model_twist=np.array([.004, 0, .001, 0, 0, 0]), rail_model_twist=np.array([.001, 0, 0, 0, 0, 0]))
    return clock, rail, outer, inner, step


@pytest.mark.parametrize("fault", ["none", "arm_timeout", "rail_failed", "stop_after_arm", "expired_after_arm",
                                   "rail_clamp", "final_mutation", "wrong_certificate"])
def test_final_sender_facts_and_commit_are_causal(monkeypatch, fault):
    clock, rail, outer, inner, step = fixture(monkeypatch)
    native_q, native_v = step.q_send.copy(), step.qdot.copy()
    sends = []
    def send(payload):
        sends.append(payload.copy()); clock[0] += .001
        if fault == "arm_timeout":
            raise TimeoutError("may execute late")
        if fault == "rail_failed":
            rail._armed = False
        if fault == "expired_after_arm":
            clock[0] = 1.02
    def gate():
        if fault == "stop_after_arm" and sends:
            outer.execution.stop(now_s=clock[0], wrench_environment=outer.pending_wrench_environment)
            rail.fence_publications(outer.execution.stop_epoch)
            raise ValueError("external stop")
    if fault == "final_mutation":
        step.q_send[0] += .0001
    if fault == "wrong_certificate":
        step.cartesian_constraints_sequence = 99
    if fault == "rail_clamp":
        native_q[0] = step.q_send[0] = rail._soft_lo_hi()[1]+.001
    def run():
        return publish_contact_candidate(outer, inner, step, rail, native_q_send=native_q,
                                          native_qdot=native_v, send_arm=send, publication_gate=gate, now=lambda: clock[0])
    if fault == "none":
        command = run()
        assert outer.nominal_committed and len(inner.commits) == 1
        assert outer.execution.reference_s == pytest.approx(.0025)
        assert np.array_equal(sends[0], command.arm_payload)
        assert rail._candidate_sequence == 1 and rail._follow_enabled
    else:
        with pytest.raises((ValueError, TimeoutError)):
            run()
        assert outer.execution.reference_s == 0 and not outer.nominal_committed and not inner.commits
        if sends:
            p = outer.execution.publications[1]
            assert p.states["arm"] in (DeviceState.SENT, DeviceState.UNKNOWN)
            assert outer.execution.ledger.reserved_j > 0 and outer.execution.ledger.balance_j == 10
        else:
            assert not rail._follow_enabled
        assert rail.reserved_publication is None
