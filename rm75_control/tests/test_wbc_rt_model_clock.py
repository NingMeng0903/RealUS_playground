"""A consumer may snapshot a request at seq publication, before notification."""
from types import SimpleNamespace
import time

import numpy as np
import pytest

from rm75_control.control.joint_admittance_8dof.wbc_rt import protocol as P
from rm75_control.control.joint_admittance_8dof.wbc_rt.client import NativeWbcClient


class PublicationArray:
    """Snapshot the exact payload visible at each nonzero seq write."""
    def __init__(self, array):
        self.array = array
        self.published = []

    def __getitem__(self, key):
        if key != 0:
            return self.array[key]
        parent = self

        class Record:
            def __getitem__(self, field):
                return parent.array[0][field]

            def __setitem__(self, field, value):
                parent.array[0][field] = value
                if field == "seq" and int(value):
                    parent.published.append(parent.array[0].copy())

        return Record()


@pytest.fixture
def detached():
    class Owner:
        cfg = SimpleNamespace(dt=.005, rail_refresh_dt_s=.02)
        q_cmd = np.arange(8, dtype=float) / 10

    owner = Owner()
    client = NativeWbcClient(owner)
    client._in = PublicationArray(P.view_in(bytearray(P.WBC_IN_SIZE)))
    client._out = P.view_out(bytearray(P.WBC_OUT_SIZE))
    client._notify_request = lambda: None
    client._wait_seq = lambda seq: True
    client._accept_ok_step = lambda twist, seq, **kw: seq
    yield client, owner


def test_model_timestamp_is_final_at_seq_publication(detached):
    client, owner = detached
    wall_before = time.monotonic()
    client.update(np.zeros(6), q_meas=owner.q_cmd, model_time_s=1000.005)
    assert len(client._in.published) == 1
    record = client._in.published[0]
    assert float(record["t_mono"]) == 1000.005
    assert wall_before <= client._inflight_t0 <= time.monotonic()
    assert client._inflight_t0 != float(record["t_mono"])
    np.testing.assert_array_equal(record["q_meas"], owner.q_cmd)


def test_default_timestamp_and_deadline_share_wall_clock(detached):
    client, owner = detached
    client.update(np.zeros(6), q_meas=owner.q_cmd)
    assert float(client._in.published[0]["t_mono"]) == client._inflight_t0


def test_notification_time_rewrite_is_too_late_for_a_polling_peer(detached):
    client, owner = detached
    client._notify_request = lambda: client._in.array["t_mono"].fill(1000.005)
    client.update(np.zeros(6), q_meas=owner.q_cmd)
    # The old replay saw the changed SHM after the call, but a valid consumer
    # could already have copied the original wall timestamp with the same seq.
    assert float(client._in.array["t_mono"][0]) == 1000.005
    assert float(client._in.published[0]["t_mono"]) == client._inflight_t0
    assert float(client._in.published[0]["t_mono"]) != 1000.005


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 0.0, -1.0])
def test_invalid_model_time_never_publishes(detached, bad):
    client, owner = detached
    with pytest.raises(ValueError, match="model_time_s"):
        client.update(np.zeros(6), q_meas=owner.q_cmd, model_time_s=bad)
    assert not client._in.published
    assert client._seq == client._inflight_seq == 0
