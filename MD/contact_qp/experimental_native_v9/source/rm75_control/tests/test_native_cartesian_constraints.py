"""Detached v9 certificate tests: private SHM processes, never device I/O."""
from dataclasses import replace
from pathlib import Path
import os
import runpy
import signal
import subprocess
import time
from types import SimpleNamespace

import numpy as np
import pytest

from rm75_control.control.joint_admittance_8dof.loop import JointIkController
from rm75_control.control.joint_admittance_8dof.wbc_rt import protocol as P
from rm75_control.control.joint_admittance_8dof.wbc_rt.cartesian import validate_cartesian_constraints
_transport_fixtures = runpy.run_path(str(Path(__file__).with_name("test_wbc_rt_notifications.py")))
client, peer_binary = _transport_fixtures["client"], _transport_fixtures["peer_binary"]


def certificate(A=None, lower=None, upper=None, *, sequence=1, epoch=0, expiry=None):
    A = np.eye(6) if A is None else np.asarray(A, dtype=float)
    return SimpleNamespace(A=A, lower=np.full(len(A), -.01) if lower is None else np.asarray(lower),
                           upper=np.full(len(A), .01) if upper is None else np.asarray(upper),
                           frame="tcp_base", sequence=sequence, stop_epoch=epoch,
                           valid_until_s=time.monotonic()+1. if expiry is None else expiry)


def test_v9_protocol_layout_and_preserved_legacy_offsets():
    assert P.WBC_VERSION == 9
    assert P.WBC_IN_SIZE == 6796 and P.WBC_OUT_SIZE == 1556
    assert P.WBC_IN_DTYPE.fields["commit_seq"][1] == 616
    assert P.WBC_OUT_DTYPE.fields["cartesian_sequence"][1] == 1472
    binary = Path(__file__).resolve().parents[1]/"native/wbc_rt/build/wbc_rt"
    assert subprocess.check_output([str(binary), "--sizes"], text=True).split() == ["6796", "1556"]


@pytest.mark.parametrize("change", [
    {"A": np.zeros((97, 6)), "lower": np.zeros(97), "upper": np.ones(97)},
    {"A": np.zeros((2, 5))}, {"A": np.full((6, 6), np.nan)},
    {"lower": np.ones(6), "upper": -np.ones(6)}, {"upper": np.full(6, np.nan)},
    {"lower": np.full(6, np.inf)}, {"frame": "tcp_tool"}, {"sequence": -1},
    {"sequence": 2**64}, {"stop_epoch": True}, {"valid_until_s": np.inf}, {"valid_until_s": 0.},
])
def test_malformed_or_expired_certificate_rejected_before_request(change, client):
    cert = certificate()
    vars(cert).update(change)
    seq = client._seq
    with pytest.raises(ValueError):
        client.update(np.zeros(6), q_meas=client.ctrl.q_cmd, cartesian_constraints=cert,
                      auto_commit=False, rail_exec_vel_m_s=0.)
    assert client._seq == seq


def test_zero_rows_still_enable_finite_identity_certificate_and_copy_inputs():
    cert = certificate(np.empty((0, 6)), sequence=2**40)
    parsed = validate_cartesian_constraints(cert, now_s=time.monotonic())
    assert parsed.sequence == 2**40 and parsed.A.shape == (0, 6)
    assert not parsed.A.flags.writeable
    with pytest.raises(ValueError):
        validate_cartesian_constraints(certificate(expiry=np.inf), now_s=time.monotonic())


@pytest.mark.parametrize("kwargs", [{}, {"auto_commit": False},
                                    {"auto_commit": False, "rail_exec_vel_m_s": np.nan}])
def test_certification_requires_deferred_commit_and_explicit_finite_rail_estimate(client, kwargs):
    with pytest.raises(ValueError):
        client.update(np.zeros(6), q_meas=client.ctrl.q_cmd, cartesian_constraints=certificate(), **kwargs)


@pytest.mark.parametrize("field,value", [("cartesian_sequence", 9), ("cartesian_stop_epoch", 9),
                                         ("cartesian_valid", 0), ("cartesian_command_residual", np.nan),
                                         ("cartesian_predicted_residual", 1e-5)])
def test_reply_identity_or_residual_mismatch_never_becomes_a_pending_command(client, field, value):
    parsed = validate_cartesian_constraints(certificate(sequence=3, epoch=4), now_s=time.monotonic())
    client._inflight_seq = 17
    client._inflight_certificate = parsed
    output = client._out[0]
    output["seq"] = 17
    output["cartesian_sequence"], output["cartesian_stop_epoch"] = 3, 4
    output["cartesian_valid"] = 1
    output["cartesian_command_residual"], output["cartesian_predicted_residual"] = 0., 0.
    output[field] = value
    step = client._accept_ok_step(np.zeros(6), 17, q_meas=client.ctrl.q_cmd, auto_commit=False)
    assert not step.cartesian_constraints_valid and client._pending_commit_seq == 0
    assert client._published_q_cmd is None and client._abort_next


def test_expired_original_reply_is_rejected_even_if_wire_identity_matches(client):
    parsed = validate_cartesian_constraints(certificate(sequence=3, epoch=4), now_s=time.monotonic())
    client._inflight_certificate = replace(parsed, valid_until_s=time.monotonic()-1.)
    client._inflight_seq = 17
    output = client._out[0]
    output["seq"] = 17
    output["cartesian_sequence"], output["cartesian_stop_epoch"] = 3, 4
    output["cartesian_valid"] = 1
    output["cartesian_command_residual"], output["cartesian_predicted_residual"] = 0., 0.
    step = client._accept_ok_step(np.zeros(6), 17, q_meas=client.ctrl.q_cmd, auto_commit=False)
    assert not step.cartesian_constraints_valid and client._pending_commit_seq == 0


def test_python_backend_cannot_silently_ignore_certificate():
    controller = JointIkController.__new__(JointIkController)
    controller._native = None
    with pytest.raises(RuntimeError, match="native backend"):
        controller.update(np.zeros(6), q_meas=np.zeros(8), cartesian_constraints=certificate())


def test_certified_reply_after_soft_miss_is_discarded_and_never_bound_to_new_candidate(client):
    os.kill(client._proc.pid, signal.SIGSTOP)
    os.waitpid(client._proc.pid, os.WUNTRACED)
    first = client.update(np.ones(6), q_meas=client.ctrl.q_cmd, auto_commit=False,
                          rail_exec_vel_m_s=0., cartesian_constraints=certificate(sequence=41))
    assert not first.cartesian_constraints_valid and first.fallback_reason == "native_timeout_coast"
    old_seq = client._seq
    os.kill(client._proc.pid, signal.SIGCONT)
    assert client._wait_seq(old_seq, timeout_s=.5)
    late = client.update(np.zeros(6), q_meas=client.ctrl.q_cmd, auto_commit=False,
                         rail_exec_vel_m_s=0., cartesian_constraints=certificate(sequence=42))
    assert not late.cartesian_constraints_valid
    assert late.fallback_reason == "native_timeout_coast"
    assert client._seq == old_seq and client._pending_commit_seq == 0
    assert client._inflight_seq == 0 and client._abort_next


@pytest.fixture
def runtime():
    source = Path(__file__).resolve().parents[1]/"native/wbc_rt/test_native_contract.py"
    helper = runpy.run_path(str(source))
    controller = helper["_runtime_controller"](native=True)
    q = helper["_safe_q"]()
    controller.reset(q)
    controller.enable()
    controller.begin_hybrid_episode(q, np.zeros(8))
    try:
        yield controller, q
    finally:
        helper["_close_runtime_controller"](controller)


def test_native_enforces_command_and_rail_compensated_mapping(runtime):
    ctrl, q = runtime
    cert = certificate(lower=np.full(6, -.0005), upper=np.full(6, .0005), sequence=2**40, epoch=7)
    rail_execution = .0003
    step = ctrl.update(np.array([0., .005, .002, 0., 0., 0.]), q_meas=q,
                       cartesian_constraints=cert, auto_commit=False, rail_exec_vel_m_s=rail_execution)
    assert step.cartesian_constraints_valid, (step.fallback_reason, step.qp1_status)
    assert step.cartesian_constraints_sequence == 2**40 and step.cartesian_constraints_stop_epoch == 7
    J = ctrl.kin.jacobian(q)
    sent = J @ step.qdot
    predicted = J[:, 1:] @ step.qdot[1:]+J[:, 0]*rail_execution
    np.testing.assert_allclose(step.v_tcp_commanded, sent, atol=1e-10)
    np.testing.assert_allclose(step.v_tcp_estimated, predicted, atol=1e-10)
    assert np.max(np.abs(cert.A@sent)) <= .0005+1e-8
    assert np.max(np.abs(cert.A@predicted)) <= .0005+1e-8
    assert step.cartesian_command_residual <= 1e-8 and step.cartesian_predicted_residual <= 1e-8


def test_native_infeasible_rows_fail_without_committing_motion(runtime):
    ctrl, q = runtime
    original = ctrl.q_cmd.copy()
    cert = certificate(np.zeros((1, 6)), lower=[1.], upper=[2.])
    step = ctrl.update(np.ones(6), q_meas=q, cartesian_constraints=cert,
                       auto_commit=False, rail_exec_vel_m_s=0.)
    assert not step.cartesian_constraints_valid
    assert ctrl._native._pending_commit_seq == 0
    np.testing.assert_array_equal(ctrl.q_cmd, original)


def test_native_rejects_duplicate_candidate_and_previous_epoch(runtime):
    ctrl, q = runtime
    cert = certificate(sequence=100, epoch=9)
    first = ctrl.update(np.zeros(6), q_meas=q, cartesian_constraints=cert,
                        auto_commit=False, rail_exec_vel_m_s=0.)
    assert first.cartesian_constraints_valid
    ctrl.abort_publication()
    for stale in (cert, certificate(sequence=999, epoch=8)):
        step = ctrl.update(np.zeros(6), q_meas=q, cartesian_constraints=stale,
                           auto_commit=False, rail_exec_vel_m_s=0.)
        assert not step.cartesian_constraints_valid
    recovered = ctrl.update(np.zeros(6), q_meas=q, cartesian_constraints=certificate(sequence=1, epoch=10),
                            auto_commit=False, rail_exec_vel_m_s=0.)
    assert recovered.cartesian_constraints_valid


def test_native_pending_commit_requires_exact_request_sequence(runtime):
    ctrl, q = runtime
    first = ctrl.update(np.array([0., 0., .0005, 0., 0., 0.]), q_meas=q,
                        cartesian_constraints=certificate(sequence=10), auto_commit=False,
                        rail_exec_vel_m_s=0.)
    assert first.cartesian_constraints_valid
    ctrl.commit_publication(first.qdot)
    ctrl._native._pending_commit_seq += 1  # wrong ACK cannot commit the pending history
    rejected = ctrl.update(np.zeros(6), q_meas=q, cartesian_constraints=certificate(sequence=11),
                            auto_commit=False, rail_exec_vel_m_s=0.)
    assert not rejected.cartesian_constraints_valid
    np.testing.assert_array_equal(ctrl.q_cmd, first.q_send)


def test_native_matching_commit_retains_final_previous_command(runtime):
    ctrl, q = runtime
    ctrl._native._seq = 2**40  # wire pending identity must not truncate to old cmd_u[0]
    first = ctrl.update(np.array([0., 0., .0005, 0., 0., 0.]), q_meas=q,
                        cartesian_constraints=certificate(sequence=10), auto_commit=False,
                        rail_exec_vel_m_s=0.)
    assert first.cartesian_constraints_valid
    ctrl.commit_publication(first.qdot)
    second = ctrl.update(np.zeros(6), q_meas=first.q_send,
                         cartesian_constraints=certificate(sequence=11), auto_commit=False,
                         rail_exec_vel_m_s=float(first.qdot[0]))
    assert second.cartesian_constraints_valid
    assert int(ctrl._native._in["commit_seq"][0]) > 2**40
    np.testing.assert_allclose(second.qdot_prev_used, first.qdot, atol=1e-12)


def test_unpublished_certified_candidate_never_advances_native_history(runtime):
    ctrl, q = runtime
    first = ctrl.update(np.array([0., 0., .0005, 0., 0., 0.]), q_meas=q,
                        cartesian_constraints=certificate(sequence=10), auto_commit=False,
                        rail_exec_vel_m_s=0.)
    assert first.cartesian_constraints_valid and np.linalg.norm(first.qdot) > 1e-4
    assert ctrl._native._pending_commit_seq == 0
    second = ctrl.update(np.zeros(6), q_meas=q, cartesian_constraints=certificate(sequence=11),
                         auto_commit=False, rail_exec_vel_m_s=0.)
    assert second.cartesian_constraints_valid
    np.testing.assert_array_equal(second.qdot_prev_used, np.zeros(8))


def test_confirmation_requires_exact_final_qdot_and_is_consumed_once(runtime):
    ctrl, q = runtime
    first = ctrl.update(np.array([0., 0., .0005, 0., 0., 0.]), q_meas=q,
                        cartesian_constraints=certificate(sequence=10), auto_commit=False,
                        rail_exec_vel_m_s=0.)
    assert first.cartesian_constraints_valid
    assert ctrl._native.confirm_publication(first.qdot)
    assert not ctrl._native.confirm_publication(first.qdot)
    second = ctrl.update(np.zeros(6), q_meas=q, cartesian_constraints=certificate(sequence=11),
                         auto_commit=False, rail_exec_vel_m_s=0.)
    assert second.cartesian_constraints_valid
    assert not ctrl._native.confirm_publication(second.qdot+1e-4)
    assert ctrl._native._pending_commit_seq == 0


def test_confirmed_previous_history_survives_rejected_next_wire_certificate(runtime, monkeypatch):
    ctrl, q = runtime
    first = ctrl.update(np.array([0., 0., .0005, 0., 0., 0.]), q_meas=q,
                        cartesian_constraints=certificate(sequence=10), auto_commit=False,
                        rail_exec_vel_m_s=0.)
    assert first.cartesian_constraints_valid
    ctrl.commit_publication(first.qdot)
    notify = ctrl._native._notify_request

    def expire_next():
        ctrl._native._in["cartesian_valid_until"][0] = time.monotonic()-1.
        notify()

    monkeypatch.setattr(ctrl._native, "_notify_request", expire_next)
    rejected = ctrl.update(np.zeros(6), q_meas=first.q_send, cartesian_constraints=certificate(sequence=11),
                           auto_commit=False, rail_exec_vel_m_s=float(first.qdot[0]))
    assert not rejected.cartesian_constraints_valid
    monkeypatch.setattr(ctrl._native, "_notify_request", notify)
    third = ctrl.update(np.zeros(6), q_meas=first.q_send, cartesian_constraints=certificate(sequence=12),
                        auto_commit=False, rail_exec_vel_m_s=float(first.qdot[0]))
    assert third.cartesian_constraints_valid
    np.testing.assert_allclose(third.qdot_prev_used, first.qdot, atol=1e-12)


def test_native_final_rail_override_cannot_bypass_cartesian_rows(runtime):
    ctrl, q = runtime
    ctrl.set_plan_drives_rail(True)
    cert = certificate(np.array([[0., 1., 0., 0., 0., 0.]]), lower=[-.004], upper=[.004])
    ff = np.zeros(8); ff[0] = .02
    step = ctrl.update(np.zeros(6), q_meas=q, qdot_ff=ff, cartesian_constraints=cert,
                       auto_commit=False, rail_exec_vel_m_s=0.)
    assert not step.cartesian_constraints_valid
    output = ctrl._native._out[0]
    assert int(output["qp1_status"]) in (P.QP_SOLVED, P.QP_MAX_ITER)
    assert float(output["cartesian_command_residual"]) > 1e-8
    np.testing.assert_array_equal(ctrl.q_cmd, q)


def test_native_accepts_maximum_row_count_without_truncation(runtime):
    ctrl, q = runtime
    A = np.tile(np.eye(6), (16, 1))
    cert = certificate(A, lower=np.full(96, -.01), upper=np.full(96, .01))
    step = ctrl.update(np.zeros(6), q_meas=q, cartesian_constraints=cert,
                       auto_commit=False, rail_exec_vel_m_s=0.)
    assert step.cartesian_constraints_valid
    assert int(ctrl._native._in["cartesian_count"][0]) == 96


@pytest.mark.parametrize("probe_stale", [False, True])
def test_native_stop_retires_epoch_even_for_a_higher_candidate_id(runtime, probe_stale):
    ctrl, q = runtime
    step = ctrl.update(np.zeros(6), q_meas=q, cartesian_constraints=certificate(sequence=1, epoch=8),
                       auto_commit=False, rail_exec_vel_m_s=0.)
    assert step.cartesian_constraints_valid
    ctrl.stop()
    ctrl.enable()
    if probe_stale:
        stale = ctrl.update(np.zeros(6), q_meas=q, cartesian_constraints=certificate(sequence=999, epoch=8),
                            auto_commit=False, rail_exec_vel_m_s=0.)
        assert not stale.cartesian_constraints_valid
    current = ctrl.update(np.zeros(6), q_meas=q, cartesian_constraints=certificate(sequence=1, epoch=9),
                          auto_commit=False, rail_exec_vel_m_s=0.)
    assert current.cartesian_constraints_valid


@pytest.mark.parametrize("fault", ["overflow", "nan_row", "reversed_bounds", "expired"])
def test_native_rejects_malformed_wire_even_after_python_validation(runtime, monkeypatch, fault):
    ctrl, q = runtime
    notify = ctrl._native._notify_request

    def tampered_notify():
        record = ctrl._native._in[0]
        if int(record["cmd"]) == P.CMD_STEP:
            if fault == "overflow":
                record["cartesian_count"] = 97
            elif fault == "nan_row":
                record["cartesian_A"][0, 0] = np.nan
            elif fault == "reversed_bounds":
                record["cartesian_lower"][0], record["cartesian_upper"][0] = 1., -1.
            else:
                record["cartesian_valid_until"] = time.monotonic()-1.
        notify()

    monkeypatch.setattr(ctrl._native, "_notify_request", tampered_notify)
    step = ctrl.update(np.zeros(6), q_meas=q, cartesian_constraints=certificate(),
                       auto_commit=False, rail_exec_vel_m_s=0.)
    assert not step.cartesian_constraints_valid
    assert int(ctrl._native._out["status"][0]) == P.STATUS_FAIL
    assert ctrl._native._pending_commit_seq == 0
    np.testing.assert_array_equal(ctrl.q_cmd, q)
