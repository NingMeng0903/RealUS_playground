"""An aborted, late native proposal cannot become a fresh outer candidate."""
from types import SimpleNamespace
import time

import numpy as np
import pytest

from rm75_control.control.joint_admittance_8dof.wbc_rt import protocol as P
from rm75_control.control.joint_admittance_8dof.wbc_rt.client import NativeWbcClient


@pytest.fixture
def detached():
    class Owner:
        cfg = SimpleNamespace(dt=.005, rail_refresh_dt_s=.02)
        q_cmd = np.arange(8, dtype=float)
    owner = Owner()
    client = NativeWbcClient(owner)
    client._in = P.view_in(bytearray(P.WBC_IN_SIZE))
    client._out = P.view_out(bytearray(P.WBC_OUT_SIZE))
    client._seq = client._inflight_seq = 7
    client._inflight_t0 = time.monotonic()
    client._in['seq'] = 7
    client._out['seq'] = 7
    client._out['q_cmd'] = 999.
    client._pending_commit_seq = 7
    client._published_q_cmd = np.full(8, 999.)
    client._published_qdot = np.full(8, 999.)
    client._notify_request = lambda: None
    client._sync_q = lambda: pytest.fail('aborted reply must not synchronize q')
    yield client, owner


@pytest.mark.parametrize('explicit_abort', [False, True])
def test_aborted_ready_reply_is_discarded_before_fresh_input(detached, explicit_abort):
    client, owner = detached
    q_before = owner.q_cmd.copy()
    if not explicit_abort:
        client.abort_pending()
    client._inflight_t0 = time.monotonic() - .001
    seen = []
    def wait_new(seq, **kwargs):
        assert seq == 8
        record = client._in[0].copy()
        assert int(record['flags']) & P.IN_ABORT_PREV
        assert not int(record['flags']) & P.IN_COMMIT_PREV
        assert client._pending_commit_seq == 0
        assert client._published_q_cmd is None
        assert client._published_qdot is None
        seen.append(record)
        return True
    client._wait_seq = wait_new
    client._accept_ok_step = lambda twist, seq, **kw: seq
    bounds = np.array([-.2, .2, -.02, .02, -.002, .002])
    result = client.update(np.arange(6), q_meas=owner.q_cmd + .1,
                           auto_commit=False, abort_prev=explicit_abort,
                           rocking_axis_base=[1., 0., 0.], rocking_bounds=bounds)
    assert result == 8
    assert len(seen) == 1
    np.testing.assert_array_equal(seen[0]['v_cmd'], np.arange(6))
    np.testing.assert_array_equal(seen[0]['q_meas'], q_before + .1)
    np.testing.assert_array_equal(seen[0]['rocking_axis_base'], [1., 0., 0.])
    np.testing.assert_array_equal(seen[0]['rocking_bounds'], bounds)
    np.testing.assert_array_equal(owner.q_cmd, q_before)


def test_aborted_unfinished_reply_keeps_original_age_and_slot(detached):
    client, owner = detached
    client._out['seq'] = 6
    client.abort_pending()
    started = client._inflight_t0
    first = client.update(np.zeros(6), q_meas=owner.q_cmd, auto_commit=False)
    assert first.fallback_reason == 'native_timeout_coast'
    assert client._seq == client._inflight_seq == 7
    assert client._inflight_t0 == started
    assert client._abort_next
    client._inflight_t0 = time.monotonic() - client._inflight_limit_s - .001
    expired = client.update(np.zeros(6), q_meas=owner.q_cmd, auto_commit=False)
    assert expired.fallback_reason == 'native_timeout'
    assert expired.solver_fault_latched
    assert client._seq == 7


@pytest.mark.parametrize('explicit_abort', [False, True])
def test_expired_aborted_ready_reply_keeps_original_deadline(detached, explicit_abort):
    client, owner = detached
    if not explicit_abort:
        client.abort_pending()
    client._inflight_t0 = time.monotonic() - client._inflight_limit_s - .001
    original = client._in.copy()
    original_q = owner.q_cmd.copy()
    client._accept_ok_step = lambda *a, **kw: pytest.fail('expired reply cannot be accepted')
    client._notify_request = lambda: pytest.fail('expired request cannot be replaced')
    expired = client.update(np.ones(6), q_meas=owner.q_cmd + .1,
                            auto_commit=False, abort_prev=explicit_abort,
                            rocking_axis_base=[1., 0., 0.],
                            rocking_bounds=[-.2, .2, -.02, .02, -.002, .002])
    assert expired.fallback_reason == 'native_timeout'
    assert expired.solver_fault_latched
    assert client._last_wait_reason == 'request_age'
    assert client._seq == client._inflight_seq == 7
    np.testing.assert_array_equal(client._in, original)
    np.testing.assert_array_equal(owner.q_cmd, original_q)
    assert client._pending_commit_seq == 0
