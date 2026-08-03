#!/usr/bin/env python3
"""Live tool-Z force from controller ``f_ext`` on ``rm75_f_ext``.

Reads the compensated tool-frame wrench published by window A (same value
the force loop uses), not raw sensor force. Uses a *separate* SHM from
``rm75_state`` so the twin layout (rail) stays stable.

  source env.sh
  # Restart window A once, then:
  python apps/joint_admittance_8dof/viz_tool_z_force.py
  python apps/joint_admittance_8dof/viz_tool_z_force.py --desired-z 2.0
"""

from __future__ import annotations

import argparse
import time
from collections import deque

import numpy as np

DEFAULT_RELAY = "rm75_state"


class ToolZForcePlot:
    def __init__(
        self,
        *,
        window_s: float = 20.0,
        refresh_hz: float = 20.0,
        desired_z: float | None = None,
        title_shm: str = "rm75_f_ext",
    ) -> None:
        import matplotlib.pyplot as plt

        self.window_s = float(window_s)
        self.refresh_interval = 1.0 / max(float(refresh_hz), 1.0)
        self._last_refresh = 0.0
        n = max(int(window_s * refresh_hz * 2) + 50, 400)
        self._t: deque[float] = deque(maxlen=n)
        self._fz: deque[float] = deque(maxlen=n)
        self._status = "waiting for controller f_ext…"
        self._t0: float | None = None

        plt.ion()
        self._fig, self._ax = plt.subplots(1, 1, figsize=(11, 4.5))
        self._fig.suptitle(f"Tool-Z F_ext  (controller → {title_shm})")
        (self._line_fz,) = self._ax.plot(
            [], [], color="#2563eb", lw=1.5, label="F_ext,z"
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
        self._ax.set_ylabel("F_ext,z (N)")
        self._ax.grid(True, alpha=0.3)
        self._ax.legend(loc="upper right", fontsize=9)
        self._text = self._fig.text(
            0.01, 0.01, "", fontsize=9, family="monospace"
        )
        self._fig.tight_layout(rect=(0, 0.04, 1, 0.94))
        try:
            self._fig.canvas.manager.set_window_title("RM75 tool-Z F_ext")
        except Exception:
            pass
        self._fig.show()
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()

    def set_status(self, msg: str) -> None:
        self._status = msg

    def append(self, t_s: float, fz: float) -> None:
        if not np.isfinite(fz):
            return
        if self._t0 is None:
            if np.isfinite(t_s) and t_s > 1.0:
                self._t0 = float(t_s)
            else:
                self._t0 = time.time()
                t_s = self._t0
        if np.isfinite(t_s) and t_s > 1.0:
            t_rel = float(t_s) - float(self._t0)
        else:
            t_rel = time.time() - float(self._t0)
        self._t.append(t_rel)
        self._fz.append(float(fz))

    def refresh(self, now: float) -> bool:
        import matplotlib.pyplot as plt

        if not plt.fignum_exists(self._fig.number):
            return False
        if now - self._last_refresh < self.refresh_interval:
            return True
        self._last_refresh = now

        if not self._t:
            self._text.set_text(self._status)
            self._fig.canvas.draw_idle()
            self._fig.canvas.flush_events()
            return True

        ts = np.asarray(self._t, dtype=float)
        fz = np.asarray(self._fz, dtype=float)
        t_end = float(ts[-1])
        t_start = max(float(ts[0]), t_end - self.window_s)
        mask = ts >= t_start
        xs = ts[mask]
        ys = fz[mask]
        self._line_fz.set_data(xs, ys)
        if ys.size:
            y0, y1 = float(ys.min()), float(ys.max())
            pad = max(0.4, 0.12 * (y1 - y0 + 1e-6))
            self._ax.set_ylim(y0 - pad, y1 + pad)
        self._ax.set_xlim(t_start, max(t_end, t_start + 1.0))

        fz_now = float(ys[-1])
        self._text.set_text(f"{self._status}  |  F_ext,z={fz_now:7.3f} N")
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()
        return True

    def close(self) -> None:
        import matplotlib.pyplot as plt

        plt.close(self._fig)
        plt.ioff()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Live tool-Z F_ext from controller via rm75_f_ext"
    )
    parser.add_argument(
        "--relay",
        type=str,
        default=DEFAULT_RELAY,
        help="State relay name (f_ext companion derived from this)",
    )
    parser.add_argument("--f-ext-shm", type=str, default=None, help="Override f_ext SHM name")
    parser.add_argument("--window-s", type=float, default=20.0)
    parser.add_argument("--refresh-hz", type=float, default=20.0)
    parser.add_argument(
        "--desired-z",
        type=float,
        default=None,
        help="Optional constant guide line (N)",
    )
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
        title_shm=f_ext_name,
    )
    print(
        f"Listening shm://{f_ext_name} f_ext[2] "
        "(restart window A; run a force task so f_ext is published). "
        "Close plot or Ctrl+C to stop.",
        flush=True,
    )

    last_seq = -1
    try:
        while True:
            now = time.time()
            if not f_ext_shm_has_publisher(f_ext_name):
                plot.set_status(f"{f_ext_name}: no publisher (start A)")
            else:
                ok, seq, t_s, f_ext = bus.read()
                if ok:
                    fz = float(f_ext[2]) if f_ext.size >= 3 else float("nan")
                    if seq != last_seq:
                        last_seq = seq
                        if np.isfinite(fz):
                            plot.append(float(t_s), fz)
                            plot.set_status(f"{f_ext_name}  seq={seq}")
                        else:
                            plot.set_status(
                                f"{f_ext_name} seq={seq}  "
                                "f_ext unset (start force task on C)"
                            )
                else:
                    plot.set_status(f"{f_ext_name}: waiting for f_ext (run force task)")

            if not plot.refresh(now):
                break
            time.sleep(0.005)
    except KeyboardInterrupt:
        print("\nstopped.", flush=True)
    finally:
        bus.stop()
        plot.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
