"""Shared-memory robot state relay for split-process digital twin (same host).

Controller process owns UDP push and publishes latest frames via a background
thread. Twin process subscribes read-only — no Realman TCP/UDP.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Any

import numpy as np

from rm75_control.control.admittance_common.async_state import AsyncStateSnapshot, RealtimeStateObserver
from rm75_control.control.admittance_common.shm_util import (
    attach_named_shm,
    close_attached_shm,
    close_named_shm,
    create_named_shm,
)
from rm75_control.control.admittance_common.state_bus import RobotStateBus, expand_q_meas_8dof

DEFAULT_RELAY_NAME = "rm75_state"
DEFAULT_RELAY_HZ = 200.0

_HEADER_DTYPE = np.dtype([("active", "<u8"), ("global_seq", "<u8"), ("session_id", "<u8")])
_SLOT_DTYPE = np.dtype(
    [
        ("seq", "<u8"),
        ("t_s", "<f8"),
        ("q_deg", "<f8", (7,)),
        ("pose", "<f8", (6,)),
        ("force", "<f8", (6,)),
        ("rail_m", "<f8"),
        ("ok", "u1"),
    ],
    align=True,
)
_LAYOUT_DTYPE = np.dtype([("header", _HEADER_DTYPE), ("slots", _SLOT_DTYPE, (2,))])
SHM_SIZE = int(_LAYOUT_DTYPE.itemsize)


@dataclass(frozen=True)
class StateRelayConfig:
    enabled: bool = False
    name: str = DEFAULT_RELAY_NAME
    hz: float = DEFAULT_RELAY_HZ


def parse_state_relay_config(raw: dict[str, Any] | None) -> StateRelayConfig:
    raw = raw or {}
    section = raw.get("state_relay", {})
    return StateRelayConfig(
        enabled=bool(section.get("enabled", False)),
        name=str(section.get("name", DEFAULT_RELAY_NAME)),
        hz=float(section.get("hz", DEFAULT_RELAY_HZ)),
    )


def normalize_relay_name(name: str) -> str:
    name = str(name).strip()
    if name.startswith("shm://"):
        return name[len("shm://") :]
    return name


def relay_shm_has_publisher(name: str = DEFAULT_RELAY_NAME) -> bool:
    """True when a controller is publishing on the state relay segment."""
    name = normalize_relay_name(name)
    try:
        probe = attach_named_shm(name)
        view = _ShmView(probe)
        sid = int(view.header["session_id"])
        gseq = int(view.header["global_seq"])
        view.release()
        close_attached_shm(probe)
        return sid != 0 and gseq != 0
    except (FileNotFoundError, ValueError, OSError):
        return False


class _ShmView:
    def __init__(self, shm: shared_memory.SharedMemory) -> None:
        if shm.size < SHM_SIZE:
            raise ValueError(f"shared memory too small: {shm.size} < {SHM_SIZE}")
        self._shm = shm
        self._arr = np.ndarray((), dtype=_LAYOUT_DTYPE, buffer=shm.buf)

    @property
    def header(self):
        return self._arr["header"]

    @property
    def slots(self):
        return self._arr["slots"]

    def release(self) -> None:
        self._arr = None
        self._shm = None

    def close(self) -> None:
        if self._shm is not None:
            self._shm.close()
        self.release()

    def unlink(self) -> None:
        self._shm.unlink()


def _write_slot(
    slot,
    *,
    seq: int,
    snap: AsyncStateSnapshot,
    rail_m: float,
    pose_override: np.ndarray | None = None,
) -> None:
    slot["seq"] = np.uint64(seq)
    slot["t_s"] = float(snap.t_s)
    if snap.q_deg is not None:
        slot["q_deg"][:] = np.asarray(snap.q_deg, dtype=float)[:7]
    pose = pose_override if pose_override is not None else snap.pose
    if pose is not None:
        slot["pose"][:] = np.asarray(pose, dtype=float)[:6]
    slot["force"][:] = np.asarray(snap.force_raw, dtype=float)[:6]
    slot["rail_m"] = float(rail_m)
    has_pose = pose is not None or snap.pose is not None
    slot["ok"] = np.uint8(1 if snap.ok and has_pose and snap.q_deg is not None else 0)


def _read_slot(slot) -> tuple[int, AsyncStateSnapshot, float]:
    seq = int(slot["seq"])
    ok = bool(slot["ok"])
    q_deg = np.asarray(slot["q_deg"], dtype=float).copy()
    pose = np.asarray(slot["pose"], dtype=float).copy()
    force_raw = np.asarray(slot["force"], dtype=float).copy()
    rail_m = float(slot["rail_m"])
    snap = AsyncStateSnapshot(
        pose=pose,
        q_deg=q_deg,
        force_raw=force_raw,
        t_s=float(slot["t_s"]),
        ok=ok,
        seq=seq,
    )
    return seq, snap, rail_m


class StateRelayPublisher:
    """Background publisher: ``RobotStateBus.read()`` -> shared memory @ hz."""

    def __init__(
        self,
        bus: RobotStateBus,
        *,
        name: str = DEFAULT_RELAY_NAME,
        hz: float = DEFAULT_RELAY_HZ,
        rail_m_fn: Callable[[], float] | None = None,
        kin: Any | None = None,
    ) -> None:
        self._bus = bus
        self._name = normalize_relay_name(name)
        self._hz = max(float(hz), 1.0)
        self._rail_m_fn = rail_m_fn or (lambda: 0.0)
        # Optional Pinocchio kinematics: overwrite RealMan UDP pose (often
        # ArmTip/link_7) with gripper-TCP fk_pose(q, rail).
        self._kin = kin
        self._kin_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._shm: shared_memory.SharedMemory | None = None
        self._view: _ShmView | None = None
        self._seq = 0
        self._session_id = 0
        self._udp_listener = None
        self._last_pub_mono = 0.0
        self._last_good_snap: AsyncStateSnapshot | None = None
        self._rail_thread: threading.Thread | None = None
        self._pub_lock = threading.Lock()
        # Publish-rate probe (measurement only).
        self._pub_n = 0
        self._pub_rail_n = 0
        self._pub_window_t0 = 0.0
        self._rate_log_period_s = 5.0
        self._last_logged_rail = float("nan")

    def set_kin(self, kin: Any | None) -> None:
        """Hot-swap TCP kinematics used for SHM pose (e.g. after tool sync)."""
        with self._kin_lock:
            self._kin = kin

    def _pose_from_kin(self, snap: AsyncStateSnapshot, rail_m: float) -> np.ndarray | None:
        with self._kin_lock:
            kin = self._kin
        if kin is None or snap.q_deg is None:
            return None
        try:
            q8 = expand_q_meas_8dof(snap.q_deg, rail_m)
            return np.asarray(kin.fk_pose(q8), dtype=float).reshape(6)
        except Exception:
            return None

    @property
    def name(self) -> str:
        return self._name

    @property
    def hz(self) -> float:
        return self._hz

    @property
    def session_id(self) -> int:
        return int(self._session_id)

    def start(self) -> None:
        if self._view is not None:
            return
        self._shm = create_named_shm(self._name, SHM_SIZE)
        self._view = _ShmView(self._shm)
        self._view.header["active"] = np.uint64(0)
        self._view.header["global_seq"] = np.uint64(0)
        self._session_id = int(time.time_ns() & ((1 << 64) - 1)) or 1
        self._view.header["session_id"] = np.uint64(self._session_id)
        self._seq = 0
        self._stop.clear()

        def _on_udp(snap: AsyncStateSnapshot) -> None:
            if self._stop.is_set() or self._view is None:
                return
            try:
                self._publish_snap(snap, source="udp")
            except Exception:
                pass

        self._udp_listener = _on_udp
        self._bus.observer.add_listener(_on_udp)

        # Real robot: UDP callback publishes arm frames; a light rail refresh
        # thread keeps encoder rail_m at ~50 Hz so the twin does not look ~10 Hz.
        if self._thread is None or not self._thread.is_alive():
            target = (
                self._run_watchdog
                if isinstance(self._bus.observer, RealtimeStateObserver)
                else self._run
            )
            self._thread = threading.Thread(target=target, name="state-relay-pub", daemon=True)
            self._thread.start()
        if self._rail_thread is None or not self._rail_thread.is_alive():
            self._rail_thread = threading.Thread(
                target=self._run_rail_refresh, name="state-relay-rail", daemon=True
            )
            self._rail_thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._udp_listener is not None:
            self._bus.observer.remove_listener(self._udp_listener)
            self._udp_listener = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._rail_thread is not None:
            self._rail_thread.join(timeout=1.0)
            self._rail_thread = None
        if self._view is not None:
            try:
                self._view.header["global_seq"] = np.uint64(0)
                self._view.header["session_id"] = np.uint64(0)
            except (OSError, ValueError):
                pass
            self._view = None
        close_named_shm(self._shm)
        self._shm = None
        self._session_id = 0

    def _publish_snap(self, snap: AsyncStateSnapshot, *, source: str = "thread") -> None:
        assert self._view is not None
        if snap.ok and snap.pose is not None and snap.q_deg is not None:
            self._last_good_snap = snap
        try:
            rail_m = float(self._rail_m_fn())
        except Exception:
            rail_m = 0.0
        # Never publish garbage encoder (e.g. -1474 mm) into SHM/twin.
        if not np.isfinite(rail_m) or rail_m < -0.05 or rail_m > 0.85:
            rail_m = float(self._last_logged_rail) if np.isfinite(self._last_logged_rail) else 0.0
        pose_override = self._pose_from_kin(snap, rail_m)
        with self._pub_lock:
            self._seq += 1
            active = int(self._view.header["active"])
            inactive = 1 - active
            _write_slot(
                self._view.slots[inactive],
                seq=self._seq,
                snap=snap,
                rail_m=rail_m,
                pose_override=pose_override,
            )
            self._view.header["active"] = np.uint64(inactive)
            self._view.header["global_seq"] = np.uint64(self._seq)
            self._last_pub_mono = time.monotonic()
            # Rate probe
            now = self._last_pub_mono
            if self._pub_window_t0 <= 0.0:
                self._pub_window_t0 = now
            self._pub_n += 1
            if source == "rail":
                self._pub_rail_n += 1
            if (
                not (self._last_logged_rail == self._last_logged_rail)
                or abs(rail_m - self._last_logged_rail) > 1e-7
            ):
                self._last_logged_rail = rail_m
            elapsed = now - self._pub_window_t0
            if elapsed >= self._rate_log_period_s:
                pub_hz = self._pub_n / max(elapsed, 1e-6)
                rail_hz = self._pub_rail_n / max(elapsed, 1e-6)
                print(
                    f"rm75 state-relay: publish {pub_hz:.0f} Hz "
                    f"(rail-refresh={rail_hz:.0f} Hz, last_rail={rail_m * 1000:.1f} mm)",
                    flush=True,
                )
                self._pub_n = 0
                self._pub_rail_n = 0
                self._pub_window_t0 = now

    def _run_rail_refresh(self) -> None:
        """Republish last arm snap with fresh encoder rail @ 50 Hz for twin smoothness."""
        period = 0.02
        while not self._stop.wait(period):
            snap = self._last_good_snap
            if snap is None or self._view is None:
                continue
            if time.monotonic() - self._last_pub_mono < 0.012:
                continue
            try:
                self._publish_snap(snap, source="rail")
            except Exception:
                pass

    def _publish_once(self) -> None:
        obs = self._bus.observer
        snap = obs.read()
        if not snap.ok:
            if self._last_good_snap is not None:
                self._publish_snap(self._last_good_snap, source="watchdog_hold")
            return
        if isinstance(obs, RealtimeStateObserver):
            self._publish_snap(snap, source="watchdog")
            return
        if self._udp_listener is not None:
            return
        self._publish_snap(snap, source="thread")

    def _run_watchdog(self) -> None:
        """Republish only when UDP push stalls (RealtimeStateObserver)."""
        while not self._stop.is_set():
            try:
                if time.monotonic() - self._last_pub_mono > 0.1:
                    self._publish_once()
            except Exception:
                pass
            self._stop.wait(0.05)

    def _run(self) -> None:
        try:
            import os

            os.nice(10)
        except Exception:
            pass
        period = 1.0 / self._hz
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                self._publish_once()
            except Exception:
                pass
            delay = period - (time.monotonic() - t0)
            if delay > 0.0:
                self._stop.wait(delay)


class RelayStateBus:
    """Read-only subscriber with the same surface as ``RobotStateBus``."""

    def __init__(self, name: str = DEFAULT_RELAY_NAME) -> None:
        self._name = normalize_relay_name(name)
        self._shm: shared_memory.SharedMemory | None = None
        self._view: _ShmView | None = None
        self._last_rail_m = 0.0
        self._attached_session_id = 0
        self._last_reattach_t = 0.0
        self._last_live_seq = 0
        self._last_live_t = 0.0

    @property
    def name(self) -> str:
        return self._name

    @property
    def session_id(self) -> int:
        return int(self._attached_session_id)

    @property
    def push_period_ms(self) -> float:
        return 1000.0 / max(float(DEFAULT_RELAY_HZ), 1.0)

    @property
    def observer(self):
        return self

    def _detach(self) -> None:
        if self._view is not None:
            self._view.release()
            self._view = None
        close_attached_shm(self._shm)
        self._shm = None
        self._attached_session_id = 0

    def ensure_attached(self, *, force: bool = False) -> bool:
        """Attach or re-attach when controller (re)starts publishing."""
        now = time.monotonic()
        stale = (now - self._last_live_t) > 0.5
        if not force and not stale and self._view is not None and self._attached_session_id != 0:
            try:
                sid = int(self._view.header["session_id"])
                gseq = int(self._view.header["global_seq"])
                if sid == self._attached_session_id and sid != 0 and gseq != 0:
                    return True
            except Exception:
                pass
            self._detach()
        try:
            probe = attach_named_shm(self._name)
            view = _ShmView(probe)
            sid = int(view.header["session_id"])
            if sid == 0:
                probe.close()
                self._detach()
                self._last_reattach_t = now
                return False
            if self._view is not None and sid == self._attached_session_id:
                probe.close()
                self._last_reattach_t = now
                return True
            self._detach()
            self._shm = probe
            self._view = view
            self._attached_session_id = sid
            self._last_reattach_t = now
            self._last_live_seq = 0
            self._last_live_t = 0.0
            return True
        except FileNotFoundError:
            self._detach()
            self._last_reattach_t = now
            return False

    def is_live(self) -> bool:
        """True when attached and frames are advancing."""
        if not self.ensure_attached():
            return False
        snap = self.read()
        if not snap.ok:
            return False
        now = time.monotonic()
        if snap.seq != self._last_live_seq:
            self._last_live_seq = int(snap.seq)
            self._last_live_t = now
            return True
        return (now - self._last_live_t) < 1.0

    def start(self) -> None:
        pass

    def stop(self) -> None:
        self._detach()

    def read(self) -> AsyncStateSnapshot:
        if not self.ensure_attached():
            return AsyncStateSnapshot()
        snap = self._read_once()
        if snap.ok:
            return snap
        self._detach()
        if self.ensure_attached(force=True):
            return self._read_once()
        return AsyncStateSnapshot()

    def _read_once(self) -> AsyncStateSnapshot:
        if self._view is None:
            return AsyncStateSnapshot()
        for _ in range(8):
            active = int(self._view.header["active"])
            global_seq = int(self._view.header["global_seq"])
            sid = int(self._view.header["session_id"])
            if sid == 0 or global_seq == 0 or sid != self._attached_session_id:
                break
            seq, snap, rail_m = _read_slot(self._view.slots[active])
            if seq == global_seq and int(self._view.header["active"]) == active:
                self._last_rail_m = rail_m
                if snap.ok:
                    self._last_live_seq = int(snap.seq)
                    self._last_live_t = time.monotonic()
                return snap
        return AsyncStateSnapshot()

    def wait_first_pose(self, timeout_s: float | None = 10.0) -> np.ndarray:
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        while deadline is None or time.monotonic() < deadline:
            if not relay_shm_has_publisher(self._name):
                time.sleep(0.15)
                continue
            if self.ensure_attached(force=True):
                snap = self.read()
                if snap.pose is not None and snap.ok:
                    return snap.pose.copy()
            time.sleep(0.05)
        if not relay_shm_has_publisher(self._name):
            raise TimeoutError(
                f"RelayStateBus: shm {self._name!r} has no publisher "
                f"(start window A with --state-relay)"
            )
        raise TimeoutError(
            f"RelayStateBus: no live frame on shm {self._name!r} within {timeout_s:.1f}s "
            f"(restart window A if it was running during an older client exit)"
        )

    @property
    def last_rail_m(self) -> float:
        return self._last_rail_m

    def q_meas_8dof(self, rail_m: float = 0.0) -> np.ndarray | None:
        del rail_m  # rail position comes from the relay frame
        snap = self.read()
        if snap.q_deg is None or not snap.ok:
            return None
        return expand_q_meas_8dof(snap.q_deg, self._last_rail_m)
