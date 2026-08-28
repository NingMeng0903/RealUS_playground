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
            force = getattr(
                self,
                "_force",
                np.array([0.0, 0.0, 1.5, 0.0, 0.0, 0.0]),
            )
            snap = AsyncStateSnapshot(
                pose=np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0]),
                q_deg=self._q_deg.copy(),
                force_raw=np.asarray(force, dtype=float).copy(),
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
        assert int(getattr(snap, "wall_time_ns", 0) or 0) > 0
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


class _FakeForceObs:
    def __init__(self, wrench: np.ndarray) -> None:
        self.wrench = np.asarray(wrench, dtype=float)
        self.n = 0

    def update(self, t_s, pose_l7, force_raw):
        del t_s, pose_l7, force_raw
        self.n += 1
        return np.zeros(6), self.wrench.copy()

    def ready_causal(self) -> bool:
        return self.n >= 2


class _FakeKin:
    def frame_pose(self, q8, frame: str):
        del q8, frame
        return np.zeros(6)

    def wrench_link7_to_tcp(self, wrench):
        return np.asarray(wrench, dtype=float).copy()

    def fk_pose(self, q8):
        del q8
        return np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0])


def test_idle_f_ext_uses_compensator_not_raw(relay_name):
    """Idle viz must not publish raw sensor Z (tool-weight jump when C stops)."""
    from rm75_control.control.admittance_common.state_relay import (
        ForceExtBus,
        f_ext_name_for_relay,
    )

    obs = _FakeObserver()
    # Raw sensor looks like ~3.5 N tool weight on Z.
    obs._force = np.array([0.0, 0.0, 3.5, 0.0, 0.0, 0.0])
    bus = RobotStateBus(None, observer=obs)
    pub = StateRelayPublisher(
        bus, name=relay_name, hz=200.0, rail_m_fn=lambda: 0.0, kin=_FakeKin()
    )
    pub.set_force_observer(_FakeForceObs(np.array([0.0, 0.0, 0.4, 0.0, 0.0, 0.0])))
    pub.start()
    try:
        # Drive a few UDP-style publishes with advancing seq.
        for _ in range(4):
            snap = obs.read()
            # Simulate pre-fix UDP snaps that always carried seq=0.
            snap.seq = 0
            pub._publish_snap(snap, source="udp")
        fbus = ForceExtBus(name=f_ext_name_for_relay(relay_name))
        ok, seq, _t, f_ext = fbus.read()
        assert ok
        assert seq >= 4  # must keep publishing even when snap.seq stays 0
        assert f_ext[2] == pytest.approx(0.4)
        assert f_ext[2] != pytest.approx(3.5)
        fbus.stop()
    finally:
        pub.stop()


def test_f_ext_on_separate_shm_does_not_shift_rail(relay_name):

    from rm75_control.control.admittance_common.state_relay import (
        ForceExtBus,
        f_ext_name_for_relay,
    )

    obs = _FakeObserver()
    bus = RobotStateBus(None, observer=obs)
    pub = StateRelayPublisher(bus, name=relay_name, hz=200.0, rail_m_fn=lambda: 0.123)
    pub.start()
    try:
        sub = RelayStateBus(relay_name)
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if sub.read().ok:
                break
            time.sleep(0.005)
        assert sub.read().ok
        assert sub.last_rail_m == pytest.approx(0.123)

        wrench = np.array([0.1, -0.2, 1.7, 0.0, 0.0, 0.0])
        pub.set_f_ext(wrench)
        fbus = ForceExtBus(name=f_ext_name_for_relay(relay_name))
        deadline = time.monotonic() + 2.0
        ok = False
        while time.monotonic() < deadline:
            ok, _seq, _t, f_ext = fbus.read()
            if ok and np.isfinite(f_ext[2]):
                break
            time.sleep(0.005)
        assert ok
        assert f_ext[2] == pytest.approx(1.7)
        # Rail still correct after f_ext publish (layout not poisoned).
        assert sub.read().ok
        assert sub.last_rail_m == pytest.approx(0.123)
        fbus.stop()
        sub.stop()
    finally:
        pub.stop()


def test_load_joint_zero_offsets_missing_is_zero(tmp_path):
    from rm75_control.control.admittance_common.state_relay import load_joint_zero_offsets_deg

    q = load_joint_zero_offsets_deg(
        {"joint_zero_offsets": str(tmp_path / "nope.yaml")},
        urdf_path=None,
    )
    np.testing.assert_allclose(q, np.zeros(7))


def test_load_joint_zero_offsets_sha1_mismatch_is_zero(tmp_path):
    from rm75_control.control.admittance_common.state_relay import load_joint_zero_offsets_deg

    p = tmp_path / "off.yaml"
    p.write_text(
        "joint_zero_offsets_deg: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]\n"
        "wbc_urdf_sha1: deadbeef\n",
        encoding="utf-8",
    )
    urdf = tmp_path / "dummy.urdf"
    urdf.write_text("<robot/>", encoding="utf-8")
    q = load_joint_zero_offsets_deg({"joint_zero_offsets": str(p)}, urdf_path=urdf)
    np.testing.assert_allclose(q, np.zeros(7))


def test_pose_from_kin_applies_offsets_before_fk():
    class _RecKin:
        def __init__(self):
            self.last_q = None

        def fk_pose(self, q8):
            self.last_q = np.asarray(q8, dtype=float).copy()
            return np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0])

    obs = _FakeObserver()
    bus = RobotStateBus(None, observer=obs)
    pub = StateRelayPublisher(bus, name="unused", hz=10.0, rail_m_fn=lambda: 0.05, kin=_RecKin())
    pub.set_joint_zero_offsets_deg(np.array([1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 9.0]))
    snap = obs.read()
    pose = pub._pose_from_kin(snap, 0.05)
    assert pose is not None
    assert pub._kin.last_q is not None
    q8 = pub._kin.last_q
    assert q8[0] == pytest.approx(0.05)
    assert np.degrees(q8[1]) == pytest.approx(11.0)
    assert np.degrees(q8[6]) == pytest.approx(17.0)
    assert np.degrees(q8[7]) == pytest.approx(0.0)

