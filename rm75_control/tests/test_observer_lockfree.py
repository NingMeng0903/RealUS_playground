"""Tests for lock-free UDP snapshot reads (no robot)."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from rm75_control.control.admittance_common.async_state import AsyncStateSnapshot, RealtimeStateObserver


def test_observer_concurrent_store_and_read():
    obs = RealtimeStateObserver(None)
    stop = threading.Event()
    max_seq = [0]

    def writer():
        i = 0
        while not stop.is_set():
            obs._store_snap(
                AsyncStateSnapshot(
                    pose=np.array([float(i), 0.0, 0.0, 0.0, 0.0, 0.0]),
                    q_deg=np.full(7, float(i)),
                    qdot_deg_s=np.full(7, 0.5 * float(i)),
                    ok=True,
                    t_s=time.monotonic(),
                )
            )
            i += 1

    def reader():
        for _ in range(500):
            snap = obs.read()
            if snap.ok and snap.pose is not None:
                max_seq[0] = max(max_seq[0], int(snap.seq))
            time.sleep(0.0005)

    t_w = threading.Thread(target=writer)
    t_r = threading.Thread(target=reader)
    t_w.start()
    t_r.start()
    t_r.join()
    stop.set()
    t_w.join(timeout=1.0)
    assert max_seq[0] > 0


def test_observer_read_copies_sdk_qdot():
    obs = RealtimeStateObserver(None)
    qdot = np.array([1.0, -2.0, 3.0, 0.0, 0.5, 0.0, 4.0])
    obs._store_snap(
        AsyncStateSnapshot(
            pose=np.zeros(6),
            q_deg=np.zeros(7),
            qdot_deg_s=qdot,
            ok=True,
            t_s=time.monotonic(),
        )
    )
    snap = obs.read()
    assert snap.qdot_deg_s is not None
    np.testing.assert_allclose(snap.qdot_deg_s, qdot)
    snap.qdot_deg_s[0] = 99.0
    again = obs.read()
    assert again.qdot_deg_s is not None
    assert again.qdot_deg_s[0] == pytest.approx(1.0)
