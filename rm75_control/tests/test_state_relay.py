"""Tests for shared-memory state relay (no robot required)."""

from __future__ import annotations

import threading
import time
import uuid

import numpy as np
import pytest

from rm75_control.control.admittance_common.async_state import AsyncStateSnapshot
from rm75_control.control.admittance_common.state_bus import RobotStateBus, expand_q_meas_8dof
from rm75_control.control.admittance_common.state_relay import (
    RelayStateBus,
    StateRelayPublisher,
    parse_state_relay_config,
)


class _FakeObserver:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._seq = 0
        self._q_deg = np.array([10.0, 20.0, -10.0, 30.0, 0.0, 15.0, 0.0], dtype=float)
        self.push_period_ms = 5.0
        self.config = type("C", (), {"port": 8098})()
        self._target_ip = "127.0.0.1"
        self._listeners = []

    def add_listener(self, fn) -> None:
        self._listeners.append(fn)

    def remove_listener(self, fn) -> None:
        try:
            self._listeners.remove(fn)
        except ValueError:
            pass

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def read(self):
        with self._lock:
            self._seq += 1
            snap = AsyncStateSnapshot(
                pose=np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0]),
                q_deg=self._q_deg.copy(),
                force_raw=np.array([0.0, 0.0, 1.5, 0.0, 0.0, 0.0]),
                t_s=time.monotonic(),
                ok=True,
                seq=self._seq,
            )
        for fn in self._listeners:
            try:
                fn(snap)
            except Exception:
                pass
        return snap

    def wait_first_pose(self, timeout_s: float = 5.0):
        return np.zeros(6)


@pytest.fixture
def relay_name():
    name = f"rm75_test_{uuid.uuid4().hex[:8]}"
    yield name


def test_parse_state_relay_config():
    cfg = parse_state_relay_config({"state_relay": {"enabled": True, "name": "foo", "hz": 60}})
    assert cfg.enabled is True
    assert cfg.name == "foo"
    assert cfg.hz == 60.0


def test_relay_pub_sub_roundtrip(relay_name):
    obs = _FakeObserver()
    bus = RobotStateBus(None, observer=obs)
    pub = StateRelayPublisher(bus, name=relay_name, hz=200.0, rail_m_fn=lambda: 0.05)
    pub.start()
    try:
        sub = RelayStateBus(relay_name)
        deadline = time.monotonic() + 2.0
        snap = AsyncStateSnapshot()
        while time.monotonic() < deadline:
            snap = sub.read()
            if snap.ok and snap.seq > 0:
                break
            time.sleep(0.005)
        assert snap.ok
        assert snap.seq > 0
        assert snap.q_deg is not None
        assert snap.q_deg[0] == pytest.approx(10.0)
        q8 = sub.q_meas_8dof()
        assert q8 is not None
        assert q8[0] == pytest.approx(0.05)
        assert q8[2] == pytest.approx(np.deg2rad(20.0))
        sub.stop()
    finally:
        pub.stop()


def test_relay_concurrent_read_while_publish(relay_name):
    obs = _FakeObserver()
    bus = RobotStateBus(None, observer=obs)
    pub = StateRelayPublisher(bus, name=relay_name, hz=500.0, rail_m_fn=lambda: 0.0)
    pub.start()
    sub = RelayStateBus(relay_name)
    seen_seq = []
    stop = threading.Event()

    def reader():
        while not stop.is_set():
            snap = sub.read()
            if snap.ok:
                seen_seq.append(int(snap.seq))

    t = threading.Thread(target=reader)
    t.start()
    time.sleep(0.2)
    stop.set()
    t.join()
    pub.stop()
    sub.stop()
    assert len(seen_seq) > 10
    assert max(seen_seq) >= min(seen_seq)


def test_relay_reconnect_after_publisher_restart(relay_name):
    obs = _FakeObserver()
    bus = RobotStateBus(None, observer=obs)
    pub = StateRelayPublisher(bus, name=relay_name, hz=200.0, rail_m_fn=lambda: 0.0)
    pub.start()
    sub = RelayStateBus(relay_name)
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if sub.read().ok:
            break
        time.sleep(0.01)
    assert sub.read().ok
    sid1 = sub.session_id
    pub.stop()
    time.sleep(0.05)
    assert not sub.is_live()
    pub2 = StateRelayPublisher(bus, name=relay_name, hz=200.0, rail_m_fn=lambda: 0.0)
    pub2.start()
    try:
        deadline = time.monotonic() + 2.0
        reconnected = False
        while time.monotonic() < deadline:
            if sub.is_live():
                reconnected = True
                break
            time.sleep(0.02)
        assert reconnected
        assert sub.session_id != sid1
        sub.stop()
    finally:
        pub2.stop()


def test_expand_q_matches_relay_rail():
    q7 = np.array([0.0, 10.0, 20.0, -10.0, 30.0, 0.0, 15.0])
    direct = expand_q_meas_8dof(q7, 0.07)
    assert direct[0] == pytest.approx(0.07)
