"""Genesis digital twin: mirror real robot joint state via shared state bus."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from rm75_control.control.joint_admittance_8dof.viewer.scene import RailGenesisScene


class StateBusView(Protocol):
    def q_meas_8dof(self, rail_m: float = 0.0): ...


def _is_viewer_closed(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "viewer closed" in msg or "genesisexception" in type(exc).__name__.lower()


class DigitalTwinMirror:
    """Kinematic Genesis viewer driven by a shared UDP or SHM state bus (read-only).

    Rail (q[0]) is short-horizon extrapolated between SHM updates so a ~40–50 Hz
    encoder feed still looks continuous at 60 Hz render rate.
    """

    def __init__(
        self,
        bus: StateBusView,
        scene: RailGenesisScene,
        *,
        hz: float = 30.0,
        rail_m_fn: Callable[[], float] | None = None,
        rail_extrapolate_s: float = 0.04,
    ) -> None:
        self._bus = bus
        self._scene = scene
        self._hz = max(float(hz), 1.0)
        self._rail_m_fn = rail_m_fn or (lambda: 0.0)
        self._rail_extrapolate_s = max(0.0, float(rail_extrapolate_s))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._viewer_closed = False
        self._last_seq = -1
        self._rail_x = 0.0
        self._rail_v = 0.0
        self._rail_t = 0.0
        self._rail_sample = 0.0
        self._rail_have = False
        # Sync-rate probe (measurement only; does not change refresh).
        self._sync_ok_n = 0
        self._sync_fail_n = 0
        self._sync_rail_change_n = 0
        self._sync_last_rail = float("nan")
        self._sync_window_t0 = 0.0
        self._rate_log_period_s = 5.0

    @property
    def viewer_closed(self) -> bool:
        return self._viewer_closed

    def _extrapolate_rail(self, rail_meas: float, now: float) -> float:
        """Constant-velocity hold between SHM encoder updates (≤ rail_extrapolate_s)."""
        # Quantize to 0.1 mm so encoder LSB chatter at mechanical zero does not
        # invent a velocity and make the twin jitter.
        quant = 1.0e-4
        x = round(float(rail_meas) / quant) * quant
        if not self._rail_have:
            self._rail_x = x
            self._rail_sample = x
            self._rail_v = 0.0
            self._rail_t = now
            self._rail_have = True
            return x

        if abs(x - self._rail_sample) > quant * 0.5:
            dt = max(now - self._rail_t, 1e-4)
            v_inst = (x - self._rail_x) / dt
            self._rail_v = 0.5 * self._rail_v + 0.5 * v_inst
            self._rail_x = x
            self._rail_sample = x
            self._rail_t = now
            return x

        # No real motion: kill residual velocity so hold does not drift/jitter.
        self._rail_v = 0.0
        return self._rail_x

    def _note_sync(self, ok: bool, rail_raw: float | None = None) -> None:
        now = time.monotonic()
        if self._sync_window_t0 <= 0.0:
            self._sync_window_t0 = now
        if ok:
            self._sync_ok_n += 1
            if rail_raw is not None:
                quant = 1.0e-4
                rr = round(float(rail_raw) / quant) * quant
                if (
                    not (self._sync_last_rail == self._sync_last_rail)
                    or abs(rr - float(self._sync_last_rail)) > quant * 0.5
                ):
                    if self._sync_last_rail == self._sync_last_rail:  # not NaN
                        self._sync_rail_change_n += 1
                    self._sync_last_rail = rr
        else:
            self._sync_fail_n += 1
        elapsed = now - self._sync_window_t0
        if elapsed >= self._rate_log_period_s:
            sync_hz = self._sync_ok_n / max(elapsed, 1e-6)
            rail_hz = self._sync_rail_change_n / max(elapsed, 1e-6)
            if self._sync_last_rail == self._sync_last_rail:  # finite
                rail_note = (
                    f"rail SHM updates {rail_hz:.1f} Hz "
                    f"(last={self._sync_last_rail * 1000:.1f} mm)"
                )
            else:
                rail_note = f"rail SHM updates {rail_hz:.1f} Hz (no sample yet)"
            print(
                f"rm75 twin: sync {sync_hz:.1f} Hz "
                f"(target={self._hz:.0f}, fail={self._sync_fail_n}) | {rail_note}",
                flush=True,
            )
            self._sync_ok_n = 0
            self._sync_fail_n = 0
            self._sync_rail_change_n = 0
            self._sync_window_t0 = now

    def sync_once(self) -> bool:
        if self._viewer_closed:
            return False
        q8 = self._bus.q_meas_8dof(self._rail_m_fn())
        if q8 is None:
            self._note_sync(False)
            return False
        try:
            q = np.asarray(q8, dtype=float).reshape(-1).copy()
            rail_raw = float(q[0]) if q.size >= 1 else float("nan")
            # Reject garbage encoder before rendering (never fly twin to -1474 mm).
            if q.size >= 1 and (
                not np.isfinite(rail_raw) or rail_raw < -0.05 or rail_raw > 0.85
            ):
                if self._rail_have:
                    rail_raw = float(self._rail_x)
                    q[0] = rail_raw
                else:
                    self._note_sync(False)
                    return False
            if q.size >= 1:
                now = time.monotonic()
                q[0] = self._extrapolate_rail(float(q[0]), now)
            self._scene.set_joint_positions(q)
            self._scene.step()
            self._note_sync(True, rail_raw=rail_raw)
        except AssertionError:
            # Genesis/quadrants fastcache race after A restart while B stays up.
            self._note_sync(False)
            return False
        except Exception as exc:
            if _is_viewer_closed(exc):
                self._viewer_closed = True
                return False
            raise
        return True

    def _run(self) -> None:
        period = 1.0 / self._hz
        while not self._stop.is_set():
            if self._viewer_closed:
                self._stop.wait(0.5)
                continue
            t0 = time.monotonic()
            try:
                self.sync_once()
            except Exception:
                self._note_sync(False)
            delay = period - (time.monotonic() - t0)
            if delay > 0.0:
                self._stop.wait(delay)

    def start_background(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="genesis-digital-twin", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def feed(self, q8) -> None:
        """Offline replay: push an 8-DOF vector without the state bus."""
        self._scene.set_joint_positions(q8)
        self._scene.step()
