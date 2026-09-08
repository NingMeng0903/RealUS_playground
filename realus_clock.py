"""One automatically created, same-host clock domain. Standard library; no ROS.

The first participant creates an immutable anchor in /dev/shm under flock.
Later participants attach to that anchor. Linux CLOCK_MONOTONIC supplies the
ticks, so no daemon, network round trips, or per-sample file locks are needed.
The anchor outlives individual processes and is scoped to the current boot.
ROS 2 Header field semantics: {stamp: {sec: int32, nanosec: uint32}, frame_id}.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import mmap
import os
from pathlib import Path
import re
import struct
import tempfile
import threading
import time
import uuid

NSEC = 1_000_000_000
_CLOCKS = {}
_CACHE_LOCK = threading.Lock()


def stamp_from_ns(timestamp_ns: int) -> dict:
    sec, nanosec = divmod(int(timestamp_ns), NSEC)
    if not -(2**31) <= sec < 2**31:
        raise OverflowError("Timestamp exceeds ROS 2 Time int32 seconds range")
    return {"sec": sec, "nanosec": nanosec}


def stamp_to_ns(stamp: dict) -> int:
    sec, nanosec = stamp["sec"], stamp["nanosec"]
    if type(sec) is not int or type(nanosec) is not int:
        raise TypeError("sec and nanosec must be integers")
    if not -(2**31) <= sec < 2**31 or not 0 <= nanosec < NSEC:
        raise ValueError("Invalid ROS 2 Time sec/nanosec")
    return sec * NSEC + nanosec


def clock_pair() -> tuple[int, int, int]:
    before = time.monotonic_ns()
    wall = time.time_ns()
    after = time.monotonic_ns()
    return (before + after) // 2, wall, after - before


class SharedClock:
    def __init__(self, namespace=None, mode=None, *, directory=None):
        self.namespace = namespace or os.environ.get("REALUS_CLOCK_NAMESPACE", "default")
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", self.namespace):
            raise ValueError("REALUS_CLOCK_NAMESPACE must contain 1..80 letters/digits/_/-")
        requested_mode = mode or os.environ.get("REALUS_CLOCK_MODE")
        if requested_mode is not None and requested_mode not in ("epoch", "elapsed"):
            raise ValueError("REALUS_CLOCK_MODE must be epoch or elapsed")
        directory = Path(directory or os.environ.get("REALUS_CLOCK_DIR", "/dev/shm"))
        directory.mkdir(parents=True, exist_ok=True)
        base = directory / f"realus_clock_{os.getuid()}_{self.namespace}_v1"
        self.path = base.with_suffix(".json")
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        fd = os.open(str(base) + ".lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        self.created = False
        try:
            deadline = time.monotonic() + 3.0
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Shared clock initialization lock timed out")
                    time.sleep(0.005)
            data = None
            if self.path.exists():
                try:
                    data = json.loads(self.path.read_text())
                    required = ("clock_id", "boot_id", "mode", "anchor_monotonic_ns",
                                "anchor_wall_time_ns", "anchor_time_ns", "pair_uncertainty_ns")
                    if data.get("schema") != "realus_shared_clock_v1" or any(k not in data for k in required):
                        raise ValueError("missing clock fields")
                except (ValueError, TypeError, AttributeError) as exc:
                    raise RuntimeError(f"Invalid shared clock {self.path}; use a new namespace") from exc
            if data is None or data["boot_id"] != boot_id:
                mono, wall, uncertainty = min((clock_pair() for _ in range(10)), key=lambda p: p[2])
                selected_mode = requested_mode or "epoch"
                data = dict(schema="realus_shared_clock_v1", namespace=self.namespace,
                            clock_id=uuid.uuid4().hex, boot_id=boot_id, mode=selected_mode,
                            anchor_monotonic_ns=mono, anchor_wall_time_ns=wall,
                            anchor_time_ns=wall if selected_mode == "epoch" else 0,
                            pair_uncertainty_ns=uncertainty, creator_pid=os.getpid(),
                            host=os.uname().nodename)
                temp_fd, temp_name = tempfile.mkstemp(prefix=base.name + ".", dir=directory)
                try:
                    with os.fdopen(temp_fd, "w") as handle:
                        json.dump(data, handle, sort_keys=True)
                        handle.flush()
                    os.replace(temp_name, self.path)
                finally:
                    if os.path.exists(temp_name):
                        os.unlink(temp_name)
                self.created = True
            elif requested_mode is not None and data["mode"] != requested_mode:
                raise RuntimeError(f"Clock {self.namespace!r} already uses {data['mode']}; "
                                   "join that mode or select a new namespace in every participant")
            self.description = data
        finally:
            os.close(fd)
        self.clock_id = data["clock_id"]
        self.mode = data["mode"]
        self.anchor_monotonic_ns = int(data["anchor_monotonic_ns"])
        self.anchor_wall_time_ns = int(data["anchor_wall_time_ns"])
        self.offset_ns = int(data["anchor_time_ns"]) - self.anchor_monotonic_ns

    def from_monotonic_ns(self, monotonic_ns: int) -> int:
        return int(monotonic_ns) + self.offset_ns

    def now_ns(self) -> int:
        return self.from_monotonic_ns(time.monotonic_ns())

    def header(self, frame_id: str, *, monotonic_ns=None, timestamp_ns=None) -> dict:
        if monotonic_ns is not None and timestamp_ns is not None:
            raise ValueError("Pass monotonic_ns or timestamp_ns, not both")
        if timestamp_ns is None:
            timestamp_ns = self.now_ns() if monotonic_ns is None else self.from_monotonic_ns(monotonic_ns)
        return {"stamp": stamp_from_ns(timestamp_ns), "frame_id": str(frame_id)}

    def metadata(self, frame_id: str, *, monotonic_ns=None, timestamp_ns=None) -> dict:
        header = self.header(frame_id, monotonic_ns=monotonic_ns, timestamp_ns=timestamp_ns)
        return dict(header=header, timestamp_ns=stamp_to_ns(header["stamp"]),
                    clock_id=self.clock_id, clock_domain="realus_shared", clock_mode=self.mode)


def get_clock(namespace=None, mode=None) -> SharedClock:
    key = (os.getpid(), namespace or os.environ.get("REALUS_CLOCK_NAMESPACE", "default"),
           mode or os.environ.get("REALUS_CLOCK_MODE"), os.environ.get("REALUS_CLOCK_DIR", "/dev/shm"))
    # Fast path takes no mutex. Call once at node startup and retain the object.
    found = _CLOCKS.get(key)
    if found is not None:
        return found
    with _CACHE_LOCK:
        if key not in _CLOCKS:
            _CLOCKS[key] = SharedClock(namespace, mode)
        return _CLOCKS[key]


# Companion SHM preserves every existing controller payload ABI. The versioned
# header carries the matching payload sequence and source monotonic stamp.
_GENERATION = struct.Struct("<Q")
_HEADER_BODY = struct.Struct("<qiIqQ32s96s")
HEADER_SIZE = _GENERATION.size + _HEADER_BODY.size


def header_path(name: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,180}", name):
        raise ValueError("Invalid SHM name")
    return Path("/dev/shm") / (name + "_header_v1")


class HeaderPublisher:
    """Single-writer stamped companion; no filesystem work on publish()."""
    def __init__(self, name: str, frame_id: str, clock=None):
        self.clock = clock or get_clock()
        self.frame_id = frame_id
        self._frame_bytes = frame_id.encode("utf8")
        if len(self._frame_bytes) >= 96:
            raise ValueError("frame_id is too long for the SHM header")
        self.path = header_path(name)
        fd, temporary = tempfile.mkstemp(prefix=self.path.name + ".", dir=self.path.parent)
        try:
            os.ftruncate(fd, HEADER_SIZE)
            self.mapping = mmap.mmap(fd, HEADER_SIZE)
            self.identity = (os.fstat(fd).st_dev, os.fstat(fd).st_ino)
            os.replace(temporary, self.path)
        finally:
            os.close(fd)
            if os.path.exists(temporary):
                os.unlink(temporary)
        self.generation = 0

    def publish(self, monotonic_ns: int, source_seq: int, *, timestamp_ns=None):
        timestamp_ns = self.clock.from_monotonic_ns(monotonic_ns) if timestamp_ns is None else int(timestamp_ns)
        stamp = stamp_from_ns(timestamp_ns)
        body = _HEADER_BODY.pack(timestamp_ns, stamp["sec"], stamp["nanosec"], int(monotonic_ns),
                                 int(source_seq), self.clock.clock_id.encode("ascii"), self._frame_bytes)
        self.generation += 2
        _GENERATION.pack_into(self.mapping, 0, self.generation - 1)
        self.mapping[8:] = body
        _GENERATION.pack_into(self.mapping, 0, self.generation)

    def close(self):
        self.mapping.close()
        try:
            stat = self.path.stat()
            if (stat.st_dev, stat.st_ino) == self.identity:
                self.path.unlink()
        except FileNotFoundError:
            pass


class HeaderReader:
    def __init__(self, name: str, clock=None):
        self.clock = clock or get_clock()
        self.path = header_path(name)
        self.mapping = None
        self.identity = None
        self.next_probe = 0.0

    def close(self):
        if self.mapping is not None:
            self.mapping.close()
        self.mapping = None
        self.identity = None

    def read(self, *, source_seq=None, monotonic_ns=None):
        now = time.monotonic()
        if self.mapping is None or now >= self.next_probe:
            self.next_probe = now + 1.0
            try:
                stat = self.path.stat()
                identity = (stat.st_dev, stat.st_ino)
                if stat.st_size != HEADER_SIZE:
                    raise RuntimeError(f"Header ABI mismatch: {self.path}")
                if identity != self.identity:
                    self.close()
                    with self.path.open("rb") as handle:
                        self.mapping = mmap.mmap(handle.fileno(), HEADER_SIZE, prot=mmap.PROT_READ)
                    self.identity = identity
            except FileNotFoundError:
                self.close()
                return None
        for _ in range(3):
            first = _GENERATION.unpack_from(self.mapping)[0]
            if not first or first % 2:
                continue
            body = _HEADER_BODY.unpack(self.mapping[8:])
            if first != _GENERATION.unpack_from(self.mapping)[0]:
                continue
            stamp_ns, sec, nanosec, mono, seq, clock_id, frame_id = body
            clock_id = clock_id.rstrip(b"\0").decode("ascii")
            if clock_id != self.clock.clock_id:
                raise RuntimeError(f"Clock domain mismatch on {self.path}: {clock_id} != {self.clock.clock_id}")
            if source_seq is not None and seq != int(source_seq):
                return None
            if monotonic_ns is not None and abs(mono - int(monotonic_ns)) > 1:
                return None
            if sec * NSEC + nanosec != stamp_ns:
                raise RuntimeError(f"Invalid timestamp fields on {self.path}")
            return dict(header={"stamp": {"sec": sec, "nanosec": nanosec},
                                "frame_id": frame_id.rstrip(b"\0").decode("utf8")},
                        timestamp_ns=stamp_ns, clock_id=clock_id, clock_domain="realus_shared",
                        source_seq=seq, source_monotonic_ns=mono)
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", default=None)
    parser.add_argument("--mode", choices=("epoch", "elapsed"), default=None)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    clock = get_clock(args.namespace, args.mode)
    print(json.dumps(dict(clock.description, created=clock.created, path=str(clock.path)), indent=2))
    try:
        while True:
            print(json.dumps(clock.metadata("clock")), flush=True)
            if not args.watch:
                return 0
            time.sleep(0.5)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
