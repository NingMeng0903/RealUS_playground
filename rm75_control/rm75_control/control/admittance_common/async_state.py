"""Robot state feedback via Realman UDP realtime push (no TCP polling).

The previous ``AsyncStateObserver`` polled ``rm_get_current_arm_state`` and
``rm_get_force_data`` on a background thread.  That contended with the main
thread's 200 Hz ``rm_movej_canfd`` stream on the same TCP connection and
collapsed effective joint feedback to ~10 Hz.  This module uses the SDK's
UDP push API instead (``rm_set_realtime_push`` + callback), which runs in
parallel with CANFD and matches the control-loop period (default 5 ms).
"""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from Robotic_Arm.rm_ctypes_wrap import (
    rm_realtime_arm_state_callback_ptr,
    rm_realtime_push_config_t,
)


@dataclass
class AsyncStateSnapshot:
    pose: np.ndarray | None = None
    q_deg: np.ndarray | None = None
    force_raw: np.ndarray = field(default_factory=lambda: np.zeros(6))
    t_s: float = 0.0
    ok: bool = False
    seq: int = 0


@dataclass(frozen=True)
class RealtimePushConfig:
    """UDP arm-state push settings (``cycle`` is in multiples of 5 ms)."""

    cycle: int = 1
    port: int = 8098
    ip: str | None = None
    force_coordinate: int = 0


def local_ip_toward(peer_ip: str) -> str:
    """Pick the local IPv4 on the route toward ``peer_ip`` (same subnet)."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect((peer_ip, 1))
        return sock.getsockname()[0]
    finally:
        sock.close()


def pose_from_waypoint(waypoint) -> np.ndarray:
    """6D pose [x,y,z,rx,ry,rz] (m, rad) from an SDK ``rm_pose_t`` waypoint."""
    pos = waypoint.position
    euler = waypoint.euler
    return np.array(
        [pos.x, pos.y, pos.z, euler.rx, euler.ry, euler.rz],
        dtype=float,
    )


def parse_realtime_push_config(raw: dict[str, Any] | None) -> RealtimePushConfig:
    """Build push config from a YAML ``timing`` / ``realtime_push`` section."""
    raw = raw or {}
    timing = raw.get("timing", {})
    rp = raw.get("realtime_push", {})
    dt_ms = float(timing.get("dt_ms", 5.0))
    default_cycle = max(1, int(round(dt_ms / 5.0)))
    return RealtimePushConfig(
        cycle=int(rp.get("cycle", default_cycle)),
        port=int(rp.get("port", 8098)),
        ip=rp.get("ip"),
        force_coordinate=int(rp.get("force_coordinate", 0)),
    )


class RealtimeStateObserver:
    """UDP push observer — same read API as the legacy TCP poller."""

    def __init__(
        self,
        robot,
        *,
        config: RealtimePushConfig | None = None,
        robot_ip: str | None = None,
    ) -> None:
        self.robot = robot
        self.config = config or RealtimePushConfig()
        self._robot_ip = robot_ip
        self._lock = threading.Lock()
        self._slots: list[AsyncStateSnapshot] = [AsyncStateSnapshot(), AsyncStateSnapshot()]
        self._active = 0
        self._seq = 0
        self._running = False
        self._callback_ref = None
        self._target_ip = ""
        self._listeners: list[Callable[[AsyncStateSnapshot], None]] = []

    def add_listener(self, fn: Callable[[AsyncStateSnapshot], None]) -> None:
        if fn not in self._listeners:
            self._listeners.append(fn)

    def remove_listener(self, fn: Callable[[AsyncStateSnapshot], None]) -> None:
        try:
            self._listeners.remove(fn)
        except ValueError:
            pass

    def _store_snap(self, snap: AsyncStateSnapshot) -> None:
        with self._lock:
            inactive = 1 - self._active
            self._slots[inactive] = snap
            self._active = inactive
            self._seq += 1
        for fn in self._listeners:
            try:
                fn(snap)
            except Exception:
                pass

    def _snapshot_copy(self) -> AsyncStateSnapshot:
        """Copy latest frame; array copy happens outside the short lock."""
        for _ in range(8):
            with self._lock:
                active = self._active
                seq = self._seq
            s = self._slots[active]
            if s.pose is None:
                return AsyncStateSnapshot(
                    force_raw=s.force_raw.copy(),
                    t_s=s.t_s,
                    ok=False,
                    seq=seq,
                )
            out = AsyncStateSnapshot(
                pose=s.pose.copy(),
                q_deg=s.q_deg.copy() if s.q_deg is not None else None,
                force_raw=s.force_raw.copy(),
                t_s=s.t_s,
                ok=s.ok,
                seq=seq,
            )
            with self._lock:
                if self._active == active and self._seq == seq:
                    return out
        s = self._slots[self._active]
        if s.pose is None:
            return AsyncStateSnapshot(force_raw=s.force_raw.copy(), t_s=s.t_s, ok=False, seq=self._seq)
        return AsyncStateSnapshot(
            pose=s.pose.copy(),
            q_deg=s.q_deg.copy() if s.q_deg is not None else None,
            force_raw=s.force_raw.copy(),
            t_s=s.t_s,
            ok=s.ok,
            seq=self._seq,
        )

    @property
    def push_period_ms(self) -> float:
        return float(self.config.cycle) * 5.0

    def start(
        self,
        *,
        retries: int = 3,
        retry_delay_s: float = 1.0,
    ) -> None:
        if self._running:
            return
        peer = self._robot_ip or self.config.ip
        if not peer:
            raise ValueError("robot_ip or realtime_push.ip is required for UDP feedback")
        self._target_ip = self.config.ip or local_ip_toward(peer)

        def _on_state(data) -> None:
            if data.errCode != 0:
                return
            t_s = time.monotonic()
            q_deg = np.asarray(
                [data.joint_status.joint_position[i] for i in range(7)],
                dtype=float,
            )
            pose = pose_from_waypoint(data.waypoint)
            force_raw = np.asarray(
                [data.force_sensor.force[i] for i in range(6)],
                dtype=float,
            )
            self._store_snap(
                AsyncStateSnapshot(
                    pose=pose,
                    q_deg=q_deg,
                    force_raw=force_raw,
                    t_s=t_s,
                    ok=True,
                    seq=0,
                )
            )

        self._callback_ref = rm_realtime_arm_state_callback_ptr(_on_state)
        self.robot.rm_realtime_arm_state_call_back(self._callback_ref)

        push_on = rm_realtime_push_config_t(
            self.config.cycle,
            True,
            self.config.port,
            self.config.force_coordinate,
            self._target_ip,
        )
        push_off = rm_realtime_push_config_t(
            self.config.cycle,
            False,
            self.config.port,
            self.config.force_coordinate,
            self._target_ip,
        )

        last_ret: int | None = None
        attempts = max(1, int(retries))
        for attempt in range(attempts):
            if attempt > 0:
                try:
                    self.robot.rm_set_realtime_push(push_off)
                except Exception:
                    pass
                time.sleep(retry_delay_s)
            ret = self.robot.rm_set_realtime_push(push_on)
            if ret == 0:
                self._running = True
                return
            last_ret = ret
            if attempt + 1 < attempts:
                time.sleep(retry_delay_s)

        raise RuntimeError(
            f"rm_set_realtime_push failed: {last_ret} "
            f"(cycle={self.config.cycle}, port={self.config.port}, "
            f"ip={self._target_ip!r}, force_coord={self.config.force_coordinate}, "
            f"attempts={attempts}). "
            "Ensure robot.thread_mode=2 (triple thread), realtime_push.ip is the "
            "robot-reachable PC address, only one controller owns the session, and "
            "firewall allows UDP."
        )

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        try:
            off = rm_realtime_push_config_t(
                self.config.cycle,
                False,
                self.config.port,
                self.config.force_coordinate,
                self._target_ip,
            )
            self.robot.rm_set_realtime_push(off)
        except Exception:
            pass

    def wait_first_pose(self, timeout_s: float = 5.0) -> np.ndarray:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            snap = self.read()
            if snap.pose is not None and snap.ok:
                return snap.pose.copy()
            time.sleep(0.001)
        raise TimeoutError(
            f"RealtimeStateObserver: no UDP pose within {timeout_s:.1f}s "
            f"(target {self._target_ip}:{self.config.port})"
        )

    def read(self) -> AsyncStateSnapshot:
        return self._snapshot_copy()


def create_state_observer(
    robot,
    raw: dict[str, Any] | None = None,
    *,
    robot_ip: str | None = None,
) -> RealtimeStateObserver:
    """Factory: YAML dict -> configured UDP observer."""
    cfg = parse_realtime_push_config(raw)
    ip = robot_ip or (raw or {}).get("robot", {}).get("ip")
    return RealtimeStateObserver(robot, config=cfg, robot_ip=ip)


# Backward-compatible alias — all call sites now get UDP push, not TCP poll.
AsyncStateObserver = RealtimeStateObserver
