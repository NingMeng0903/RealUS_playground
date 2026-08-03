#!/usr/bin/env python3
"""Live tool-Z force from window A ``rm75_f_ext`` SHM.

  source env.sh
  # Start window A, then:
  python apps/joint_admittance_8dof/viz_tool_z_force.py --desired-z 2.0
"""

from __future__ import annotations

import argparse
import time
from collections import deque

import numpy as np

DEFAULT_RELAY = "rm75_state"
WINDOW_TITLE = "toolz fext"


class ToolZForcePlot:
    def __init__(
        self,
        *,
        window_s: float = 20.0,
        refresh_hz: float = 10.0,
        desired_z: float | None = None,
    ) -> None:
        import matplotlib

        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt

        self.window_s = float(window_s)
        self.refresh_interval = 1.0 / max(float(refresh_hz), 1.0)
        self._last_refresh = 0.0
        n = max(int(window_s * refresh_hz * 2) + 50, 400)
        self._t: deque[float] = deque(maxlen=n)
        self._fz: deque[float] = deque(maxlen=n)
        self._status = "waiting…"
        self._t0: float | None = None
        self._closed = False

        plt.ion()
        self._fig, self._ax = plt.subplots(1, 1, figsize=(11, 4.5))
        try:
            self._fig.canvas.manager.set_window_title(WINDOW_TITLE)
        except Exception:
            pass
        self._fig.suptitle(WINDOW_TITLE)
        (self._line_fz,) = self._ax.plot(
            [], [], color="#2563eb", lw=1.5, label="fext z"
        )
        self._ax.axhline(0.0, color="k", lw=0.5, alpha=0.35)
        if desired_z is not None:
            self._ax.axhline(
                float(desired_z),
                color="#16a34a",
                lw=1.2,
                ls="--",
                alpha=0.8,
                label=f"desired={desired_z:g} N",
            )
        self._ax.set_xlabel("t (s)")
        self._ax.set_ylabel("N")
        self._ax.grid(True, alpha=0.3)
        self._ax.legend(loc="upper right", fontsize=9)
        self._text = self._fig.text(
            0.01, 0.01, "", fontsize=9, family="monospace"
        )
        self._fig.tight_layout(rect=(0, 0.04, 1, 0.94))

        def _on_close(_evt) -> None:
            self._closed = True

        self._fig.canvas.mpl_connect("close_event", _on_close)
        self._fig.show()
        try:
            self._fig.canvas.draw_idle()
        except Exception:
            pass

    def set_status(self, msg: str) -> None:
        self._status = str(msg)

    def append(self, t_s: float, fz: float) -> None:
        if not np.isfinite(fz):
            return
        if self._t0 is None:
            self._t0 = float(t_s) if np.isfinite(t_s) else time.time()
        self._t.append(float(t_s) - float(self._t0))
        self._fz.append(float(fz))

    def refresh(self, now: float) -> bool:
        if self._closed:
            return False
        import matplotlib.pyplot as plt

        try:
            if not plt.fignum_exists(self._fig.number):
                self._closed = True
                return False
            if now - self._last_refresh < self.refresh_interval:
                return True
            self._last_refresh = now

            if not self._t:
                self._text.set_text(self._status)
            else:
                ts = np.asarray(self._t, dtype=float)
                fz = np.asarray(self._fz, dtype=float)
                t_end = float(ts[-1])
                t_start = max(float(ts[0]), t_end - self.window_s)
                mask = ts >= t_start
                xs, ys = ts[mask], fz[mask]
                self._line_fz.set_data(xs, ys)
                if ys.size:
                    y0, y1 = float(ys.min()), float(ys.max())
                    pad = max(0.4, 0.12 * (y1 - y0 + 1e-6))
                    self._ax.set_ylim(y0 - pad, y1 + pad)
                self._ax.set_xlim(t_start, max(t_end, t_start + 1.0))
                self._text.set_text(
                    f"{self._status}  |  fz={float(ys[-1]):7.3f} N"
                )

            self._fig.canvas.draw_idle()
            try:
                self._fig.canvas.start_event_loop(0.001)
            except Exception:
                pass
            return True
        except KeyboardInterrupt:
            self._closed = True
            return False
        except Exception:
            return not self._closed

    def close(self) -> None:
        import matplotlib.pyplot as plt

        self._closed = True
        try:
            plt.close(self._fig)
        except Exception:
            pass
        try:
            plt.ioff()
        except Exception:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Live toolz fext from rm75_f_ext")
    parser.add_argument("--relay", type=str, default=DEFAULT_RELAY)
    parser.add_argument("--f-ext-shm", type=str, default=None)
    parser.add_argument("--window-s", type=float, default=20.0)
    parser.add_argument("--refresh-hz", type=float, default=10.0)
    parser.add_argument("--desired-z", type=float, default=None)
    args = parser.parse_args()

    from rm75_control.control.admittance_common.state_relay import (
        ForceExtBus,
        f_ext_name_for_relay,
        f_ext_shm_has_publisher,
    )

    f_ext_name = args.f_ext_shm or f_ext_name_for_relay(args.relay)
    bus = ForceExtBus(name=f_ext_name)
    plot = ToolZForcePlot(
        window_s=args.window_s,
        refresh_hz=args.refresh_hz,
        desired_z=args.desired_z,
    )
    print(f"Listening shm://{f_ext_name}  title={WINDOW_TITLE}", flush=True)

    last_seq = -1
    try:
        while True:
            now = time.time()
            try:
                if not f_ext_shm_has_publisher(f_ext_name):
                    plot.set_status("no publisher (start A)")
                    bus.ensure_attached(force=True)
                else:
                    ok, seq, _t_s, f_ext = bus.read()
                    if ok and seq != last_seq:
                        last_seq = seq
                        fz = float(f_ext[2]) if f_ext.size >= 3 else float("nan")
                        if np.isfinite(fz):
                            plot.append(now, fz)
                            plot.set_status(f"seq={seq}")
                    elif not ok:
                        plot.set_status("waiting")
            except Exception:
                bus.ensure_attached(force=True)

            if not plot.refresh(now):
                break
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        bus.stop()
        plot.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
