"""Protocol and multiprocess tests for the automatic shared clock; no hardware."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

import pytest

from realus_clock import SharedClock, HeaderPublisher, HeaderReader, get_clock, stamp_from_ns, stamp_to_ns


@pytest.fixture(autouse=True)
def isolated_clock(monkeypatch, tmp_path):
    monkeypatch.setenv("REALUS_CLOCK_DIR", str(tmp_path))
    monkeypatch.setenv("REALUS_CLOCK_NAMESPACE", "test")
    monkeypatch.delenv("REALUS_CLOCK_MODE", raising=False)


def test_ros2_time_fields_and_negative_time():
    for value in (0, 1, 1_700_000_000_123_456_789, -1_700_000_000):
        stamp = stamp_from_ns(value)
        assert 0 <= stamp["nanosec"] < 10**9
        assert stamp_to_ns(stamp) == value
    assert stamp_from_ns(-1_700_000_000) == {"sec": -2, "nanosec": 300_000_000}
    with pytest.raises(ValueError):
        stamp_to_ns({"sec": 1, "nanosec": 10**9})
    with pytest.raises(OverflowError):
        stamp_from_ns(2**31 * 10**9)


def test_first_creates_later_attaches_and_clock_has_no_owner_dependency(monkeypatch):
    first, second = SharedClock(), SharedClock()
    assert first.created and not second.created
    assert first.clock_id == second.clock_id
    mono = time.monotonic_ns()
    assert first.from_monotonic_ns(mono) == second.from_monotonic_ns(mono)
    # Tick conversion must not touch a file lock or the adjustable wall clock.
    monkeypatch.setattr("fcntl.flock", lambda *a: (_ for _ in ()).throw(AssertionError("tick took lock")))
    monkeypatch.setattr("time.time_ns", lambda: (_ for _ in ()).throw(AssertionError("tick sampled wall")))
    assert first.now_ns() <= second.now_ns()
    assert set(first.header("tcp")) == {"stamp", "frame_id"}


def test_concurrent_launches_elect_one_clock_and_survive_creator_exit():
    code = "from realus_clock import SharedClock; import json; c=SharedClock(); print(json.dumps(dict(c.description, created=c.created)))"
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]))
    def launch(_):
        return json.loads(subprocess.check_output([sys.executable, "-c", code], env=env, text=True))
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(launch, range(6)))
    assert len({r["clock_id"] for r in results}) == 1
    assert sum(r["created"] for r in results) == 1
    assert launch(7)["clock_id"] == results[0]["clock_id"]


def test_elapsed_and_mismatched_modes():
    clock = SharedClock(mode="elapsed")
    assert clock.from_monotonic_ns(clock.anchor_monotonic_ns) == 0
    assert SharedClock().clock_id == clock.clock_id
    with pytest.raises(RuntimeError, match="already uses elapsed"):
        SharedClock(mode="epoch")
    assert SharedClock(namespace="separate", mode="epoch").clock_id != clock.clock_id


def test_corrupt_anchor_is_not_silently_reset():
    clock = SharedClock()
    clock.path.write_text("not json")
    with pytest.raises(RuntimeError, match="Invalid shared clock"):
        SharedClock()


def test_header_companion_matches_sample_and_rejects_other_clock():
    clock = get_clock()
    name = "clock_header_test_" + uuid.uuid4().hex
    publisher = HeaderPublisher(name, "tcp", clock)
    reader = HeaderReader(name, clock)
    wrong = HeaderReader(name, SharedClock(namespace="other"))
    try:
        assert reader.read() is None
        mono = time.monotonic_ns()
        publisher.publish(mono, 20)
        assert reader.read(source_seq=19) is None
        assert reader.read(monotonic_ns=mono + 2) is None
        result = reader.read(source_seq=20, monotonic_ns=mono)
        assert result["header"] == clock.header("tcp", monotonic_ns=mono)
        assert result["timestamp_ns"] == clock.from_monotonic_ns(mono)
        with pytest.raises(RuntimeError, match="Clock domain mismatch"):
            wrong.read()
        reader.close()
        assert publisher.path.exists()  # A reader never unlinks a publisher.
    finally:
        wrong.close()
        reader.close()
        publisher.close()


def test_realsense_device_clock_is_not_mistaken_for_epoch(monkeypatch):
    from perception.realsense_timestamps import shared_frame_metadata
    class Frame:
        def get_frame_timestamp_domain(self):
            return "hardware_clock"
    clock = get_clock()
    mono, wall = time.monotonic_ns(), time.time_ns()
    monkeypatch.setattr("realus_clock.clock_pair", lambda: (mono, wall, 1))
    meta = shared_frame_metadata(Frame(), "cam1", 100_000, wall)
    assert meta["timestamp_ns"] == clock.from_monotonic_ns(mono)
    assert meta["timestamp_source"] == "host_frame_receipt"
    Frame.get_frame_timestamp_domain = lambda _: "global_time"
    meta = shared_frame_metadata(Frame(), "cam1", wall - 10_000_000, wall)
    assert meta["timestamp_ns"] == clock.from_monotonic_ns(mono - 10_000_000)


def test_control_and_force_publishers_write_matching_headers():
    from peirastic.core.ipc import CommandHub, TwistBus, MotionBus, Status
    from peirastic.core.modes import Mode
    from rm75_control.control.admittance_common.state_relay import _ForceExtShm
    prefix = "clock_integrated_" + uuid.uuid4().hex + "_"
    hub = CommandHub(prefix=prefix)
    twist = TwistBus(prefix=prefix, create=True)
    force = _ForceExtShm(prefix + "force")
    force.start_publisher()
    readers = [HeaderReader(hub.ctl_name), HeaderReader(twist.name),
               HeaderReader(hub.motion.name), HeaderReader(prefix + "force")]
    try:
        hub.publish(status=Status.RUNNING, mode=Mode.TRACK_HYBRID)
        twist.write(twist=[0] * 6)
        hub.motion.publish(v_tcp_z=0.01, valid=True)
        force.publish([0, 0, 4, 0, 0, 0], t_s=time.monotonic())
        for reader in readers:
            sample = reader.read()
            assert sample["clock_id"] == hub.clock.clock_id
            assert stamp_to_ns(sample["header"]["stamp"]) == sample["timestamp_ns"]
    finally:
        for reader in readers:
            reader.close()
        force.stop_publisher()
        twist.close()
        hub.close()
