"""Joint-limit data, command-step accel clamp, and the stall Watchdog.

Velocity / acceleration / position boxes live in the QP.  The full
post-solve SafetyLimiter is gone (it rewrote the certified command).
A one-rule command-step clamp remains: ``|dq_k - dq_{k-1}| <= a_max * dt_nom^2``.
The Watchdog still trips if the control loop stops feeding heartbeats.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class SafetyLimits:
    q_lower: np.ndarray
    q_upper: np.ndarray
    v_max: np.ndarray                       # rad/s (per joint)
    a_max: np.ndarray | None = None         # rad/s^2 (per joint); None disables accel clamp
    # Back-off from the hard limit; scalar (rad) or per-joint vector.  Units
    # are per joint: rad for revolute joints, METRES for a prismatic rail —
    # a scalar rad margin silently stole 3.5 cm of rail travel (2 deg = 35 mm).
    position_margin: float | np.ndarray = 0.017

    @classmethod
    def from_kinematics(
        cls,
        kin,
        *,
        v_scale: float = 1.0,
        a_max: np.ndarray | float | None = None,
        position_margin: float | np.ndarray = 0.017,
    ) -> "SafetyLimits":
        v_max = np.asarray(kin.v_max, dtype=float) * float(v_scale)
        if a_max is not None and np.isscalar(a_max):
            a_max = np.full_like(v_max, float(a_max))
        return cls(
            q_lower=np.asarray(kin.q_lower, dtype=float),
            q_upper=np.asarray(kin.q_upper, dtype=float),
            v_max=v_max,
            a_max=None if a_max is None else np.asarray(a_max, dtype=float),
            position_margin=position_margin,
        )


_INTEGRATION_OVERRUN_FRAC = 1.25


def integration_period(dt_nom: float, dt_wall_s: float | None) -> float:
    """Wall period used to integrate ``qdot`` into the next absolute target.

    ``rm_movej_canfd`` has no period argument, so a long wall tick would
    otherwise emit a double-size step.  Clip to ``[dt_nom, 1.25 * dt_nom]``.
    """
    nominal = float(dt_nom)
    if dt_wall_s is None:
        return nominal
    wall = float(dt_wall_s)
    if not np.isfinite(wall) or wall <= 0.0:
        raise ValueError("dt_wall_s must be finite and > 0")
    return float(np.clip(wall, nominal, _INTEGRATION_OVERRUN_FRAC * nominal))


def clamp_command_step(
    q_prev: np.ndarray,
    q_desired: np.ndarray,
    dq_prev: np.ndarray | None,
    a_max: np.ndarray | None,
    dt_nom: float,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Bound ``|dq_k - dq_{k-1}|`` by ``a_max * dt_nom^2``.

    Returns ``(q_safe, dq, acc_clamped)``.  First tick (no previous step)
    or a missing ``a_max`` is a no-op.
    """
    q_prev = np.asarray(q_prev, dtype=float).reshape(-1)
    q_desired = np.asarray(q_desired, dtype=float).reshape(-1)
    dq = q_desired - q_prev
    acc_clamped = False
    if a_max is not None and dq_prev is not None:
        prev = np.asarray(dq_prev, dtype=float).reshape(-1)
        if prev.shape == dq.shape:
            dt2 = float(dt_nom) * float(dt_nom)
            ddq_max = np.asarray(a_max, dtype=float).reshape(-1) * dt2
            dq_new = prev + np.clip(dq - prev, -ddq_max, ddq_max)
            if np.any(np.abs(dq_new - dq) > 1.0e-15):
                acc_clamped = True
            dq = dq_new
    return q_prev + dq, dq, acc_clamped


class Watchdog:
    """Independent heartbeat monitor.

    The control loop calls `beat()` every tick.  If no beat arrives within
    `timeout_s`, the watchdog fires `on_stall` exactly once (e.g. slow-stop the
    arm / latch a hold).  Runs as a daemon thread so it survives a stuck loop.
    """

    def __init__(
        self,
        timeout_s: float,
        on_stall: Callable[[], None],
        *,
        poll_s: float = 0.005,
        name: str = "ja-watchdog",
    ) -> None:
        self.timeout_s = float(timeout_s)
        self.on_stall = on_stall
        self.poll_s = float(poll_s)
        self._name = name
        self._last_beat = time.perf_counter()
        self._stop = threading.Event()
        self._fired = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def beat(self) -> bool:
        """Refresh a healthy watchdog; a fired watchdog stays latched."""

        with self._lock:
            if self._fired.is_set():
                return False
            self._last_beat = time.perf_counter()
            return True

    def arm(self) -> None:
        """Explicitly arm a new inactive-to-active control phase."""

        with self._lock:
            self._last_beat = time.perf_counter()
            self._fired.clear()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self.arm()
        self._thread = threading.Thread(target=self._run, name=self._name, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    @property
    def fired(self) -> bool:
        return self._fired.is_set()

    def _run(self) -> None:
        while not self._stop.is_set():
            should_fire = False
            with self._lock:
                dt = time.perf_counter() - self._last_beat
                if dt > self.timeout_s and not self._fired.is_set():
                    self._fired.set()
                    should_fire = True
            if should_fire:
                try:
                    self.on_stall()
                except Exception:
                    pass
            time.sleep(self.poll_s)

    def __enter__(self) -> "Watchdog":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()
