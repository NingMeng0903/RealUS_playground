"""Tests for shared RobotStateBus (no robot required)."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from rm75_control.control.admittance_common.state_bus import RobotStateBus, expand_q_meas_8dof


class _FakeObserver:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._seq = 0
        self._q_deg = np.array([10.0, 20.0, -10.0, 30.0, 0.0, 15.0, 0.0], dtype=float)
        self.push_period_ms = 5.0
        self.config = type("C", (), {"port": 8098})()
        self._target_ip = "127.0.0.1"

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def read(self):
        from rm75_control.control.admittance_common.async_state import AsyncStateSnapshot

        with self._lock:
            self._seq += 1
            return AsyncStateSnapshot(
                q_deg=self._q_deg.copy(),
                ok=True,
                seq=self._seq,
            )

    def wait_first_pose(self, timeout_s: float = 5.0):
        return np.zeros(6)


def test_expand_q_meas_8dof_from_7_deg():
    q7 = np.array([0.0, 10.0, 20.0, -10.0, 30.0, 0.0, 15.0])
    q8 = expand_q_meas_8dof(q7, 0.05)
    assert q8.size == 8
    assert q8[0] == pytest.approx(0.05)
    assert q8[1] == pytest.approx(0.0)
    assert q8[2] == pytest.approx(np.deg2rad(10.0))


def test_bus_shared_read_seq():
    obs = _FakeObserver()
    bus = RobotStateBus(None, observer=obs)
    a = bus.read()
    b = bus.read()
    assert a.seq < b.seq
    q8 = bus.q_meas_8dof(0.0)
    assert q8 is not None
    assert q8.size == 8


def test_bus_concurrent_readers():
    obs = _FakeObserver()
    bus = RobotStateBus(None, observer=obs)
    seen = []

    def reader():
        for _ in range(20):
            snap = bus.read()
            seen.append(int(snap.seq))
            time.sleep(0.001)

    t1 = threading.Thread(target=reader)
    t2 = threading.Thread(target=reader)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert len(seen) == 40
