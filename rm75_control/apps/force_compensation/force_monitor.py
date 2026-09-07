#!/usr/bin/env python3
"""
Live 6D force compensation monitor — raw vs F_ext after φ.

Window A running (recommended while scanning):

  source env.sh
  python apps/force_compensation/force_monitor.py

That reads rm75_state (no second robot TCP). Without A, it opens RobotSession
and you drag the arm in free space.

Config: configs/force_compensation/force_id.yaml, data/force_compensation/logs/force_id_phi.json
"""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock

import numpy as np
import yaml

from rm75_control.force.compensation import regressor as fid
from rm75_control.force.compensation.id_config import load_config
from rm75_control.force.compensation.paths import CONFIG_ID, CONFIG_ROBOT

AXIS_LABELS = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]
FORCE_IDX = (0, 1, 2)
MOM_IDX = (3, 4, 5)


def load_phi(path: Path, source: str) -> tuple[np.ndarray, str]:
    data = json.loads(path.read_text())
    if source not in data:
        raise SystemExit(f"Key '{source}' not in {path}. Keys: {list(data.keys())}")
    phi = np.array([data[source][k] for k in fid.PHI_NAMES])
    return phi, source


@dataclass
class SampleBuffer:
    max_len: int
    t: deque = field(default_factory=deque)
    pose: deque = field(default_factory=deque)
    force: deque = field(default_factory=deque)

    def __post_init__(self) -> None:
        self.t = deque(maxlen=self.max_len)
        self.pose = deque(maxlen=self.max_len)
        self.force = deque(maxlen=self.max_len)

    def append(self, t_s: float, pose6: np.ndarray, force6: np.ndarray) -> None:
        self.t.append(t_s)
        self.pose.append(np.asarray(pose6, dtype=float))
        self.force.append(np.asarray(force6, dtype=float))

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return np.asarray(self.t), np.asarray(self.pose), np.asarray(self.force)


def compensate_latest(
    buf: SampleBuffer,
    phi: np.ndarray,
    cfg: fid.FrameConfig,
    fc: float,
    *,
    use_inertia: bool,
    min_samples: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    if len(buf.t) < min_samples:
        return None
    t, pose, force = buf.arrays()
    W, Y = fid.build_dataset(pose, force, t, cfg, fc=fc, use_inertia=use_inertia)
    k = len(t) - 1
    sl = slice(6 * k, 6 * k + 6)
    return Y[sl].copy(), (Y[sl] - W[sl] @ phi).reshape(6)


class CompMonitor:
    def __init__(self, *, window_s: float, refresh_hz: float = 12.0) -> None:
        import matplotlib.pyplot as plt

        self.window_s = window_s
        self.refresh_interval = 1.0 / refresh_hz
        self._lock = Lock()
        max_pts = max(int(window_s * refresh_hz * 1.5) + 20, 200)
        self._t: deque[float] = deque(maxlen=max_pts)
        self._raw: list[deque[float]] = [deque(maxlen=max_pts) for _ in range(6)]
        self._ext: list[deque[float]] = [deque(maxlen=max_pts) for _ in range(6)]
        self._status = "Collecting buffer..."
        self._last_refresh = 0.0

        plt.ion()
        self._fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
        self._fig.suptitle("6D force: raw (signed, filtered) vs compensated F_ext")
        self._axes = axes.ravel()
        self._line_raw: list = []
        self._line_ext: list = []
        for i, ax in enumerate(self._axes):
            unit = "N" if i < 3 else "N·m"
            (lr,) = ax.plot([], [], color="#2563eb", linewidth=1.2, alpha=0.85, label="raw")
            (le,) = ax.plot([], [], color="#ea580c", linewidth=1.4, label="F_ext")
            ax.axhline(0.0, color="k", linewidth=0.5, alpha=0.35)
            ax.set_ylabel(f"{AXIS_LABELS[i]} ({unit})")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right", fontsize=8)
            self._line_raw.append(lr)
            self._line_ext.append(le)
        for ax in self._axes[3:]:
            ax.set_xlabel("Time (s)")
        self._text = self._fig.text(0.01, 0.01, "", fontsize=9, family="monospace")
        self._fig.tight_layout(rect=(0, 0.03, 1, 0.96))
        try:
            self._fig.canvas.manager.set_window_title("RM75 force compensation monitor")
        except Exception:
            pass
        self._fig.show()
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()

    def append(self, t_s: float, raw6: np.ndarray, ext6: np.ndarray | None = None) -> None:
        with self._lock:
            self._t.append(t_s)
            for i in range(6):
                self._raw[i].append(float(raw6[i]))
                self._ext[i].append(float(ext6[i]) if ext6 is not None else float("nan"))

    def set_status(self, msg: str) -> None:
        with self._lock:
            self._status = msg

    def refresh(self, now: float) -> None:
        if now - self._last_refresh < self.refresh_interval:
            return
        self._last_refresh = now
        with self._lock:
            if not self._t:
                return
            ts = np.asarray(self._t)
            t_end = float(ts[-1])
            t_start = max(0.0, t_end - self.window_s)
            mask = ts >= t_start
            xs = ts[mask]
            status = self._status
            raw_pts = [np.asarray(self._raw[i])[mask] for i in range(6)]
            ext_pts = [np.asarray(self._ext[i])[mask] for i in range(6)]

        for i in range(6):
            self._line_raw[i].set_data(xs, raw_pts[i])
            self._line_ext[i].set_data(xs, ext_pts[i])
            vals = np.concatenate([raw_pts[i], ext_pts[i]])
            finite = vals[np.isfinite(vals)]
            if len(finite):
                y0, y1 = float(np.min(finite)), float(np.max(finite))
                pad = max(0.5, 0.15 * (y1 - y0 + 1e-6))
                self._axes[i].set_ylim(y0 - pad, y1 + pad)
            self._axes[i].set_xlim(t_start, max(t_end, t_start + 1.0))

        if len(xs) >= 5:
            rf = float(np.sqrt(np.nanmean(
                np.sum(np.stack([raw_pts[j] for j in FORCE_IDX], axis=1) ** 2, axis=1)
            )))
            ef = np.stack([ext_pts[j] for j in FORCE_IDX], axis=1)
            ef_ok = np.isfinite(ef).all(axis=1)
            if np.any(ef_ok):
                re = float(np.sqrt(np.mean(np.sum(ef[ef_ok] ** 2, axis=1))))
                em = np.stack([ext_pts[j] for j in MOM_IDX], axis=1)
                em_ok = np.isfinite(em).all(axis=1)
                rm = float(np.sqrt(np.mean(np.sum(em[em_ok] ** 2, axis=1)))) if np.any(em_ok) else float("nan")
                status = f"{status}  |  |F| raw={rf:.2f}N ext={re:.2f}N  |M| ext={rm:.3f}N·m"
            else:
                status = f"{status}  |  |F| raw={rf:.2f}N  (F_ext warming up)"
        self._text.set_text(status)
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()

    def close(self) -> None:
        import matplotlib.pyplot as plt
        plt.close(self._fig)
        plt.ioff()


def main() -> int:
    parser = argparse.ArgumentParser(description="Live 6D compensated force plot")
    parser.add_argument("--id-config", type=Path, default=CONFIG_ID)
    parser.add_argument("--phi", type=Path, default=None)
    parser.add_argument("--phi-source", type=str, default=None)
    parser.add_argument("--10p-only", dest="only_10p", action="store_true")
    parser.add_argument(
        "--source",
        choices=("auto", "shm", "robot"),
        default="auto",
        help="auto uses Window A SHM if live, else a second robot TCP",
    )
    args = parser.parse_args()

    id_cfg = load_config(args.id_config)
    mc = id_cfg.monitor
    fc_cfg = id_cfg.fit
    phi_path = args.phi or fc_cfg.phi_output
    phi_src = args.phi_source or mc.phi_source

    phi, src = load_phi(phi_path, phi_src)
    if args.only_10p:
        phi = phi.copy()
        phi[4:10] = 0.0
        src = f"{src} (I=0)"

    frame = fid.FrameConfig.from_yaml(fc_cfg.force_sensor)
    fc = float(yaml.safe_load(fc_cfg.force_sensor.read_text()).get("filtfilt_cutoff_hz", 2.5))
    use_inertia = mc.use_inertia and float(np.max(np.abs(phi[4:10]))) > 1e-9

    max_buf = max(mc.min_samples + 10, int(mc.buffer_s * 1000 / mc.poll_ms) + 5)
    buf = SampleBuffer(max_len=max_buf)
    sign = np.array(frame.force_sign, dtype=float)

    source = _resolve_source(str(args.source))
    print(f"φ ({src}) from {phi_path}  m={phi[0]:.3f} kg")
    print(f"source={source}  poll={mc.poll_ms}ms  window={mc.window_s}s  buffer≈{mc.buffer_s}s")
    if source == "shm":
        print("Reading Window A rm75_state. Close plot or Ctrl+C to stop.")
    else:
        print("Drag arm in FREE SPACE (no Window A). Close plot or Ctrl+C to stop.")

    try:
        monitor = CompMonitor(window_s=mc.window_s, refresh_hz=mc.refresh_hz)
    except ModuleNotFoundError as exc:
        if "matplotlib" in str(exc):
            print(
                "[ERR] matplotlib missing in this python. Install into rm75:\n"
                "      PYTHONNOUSERSITE=1 python -m pip install matplotlib",
                flush=True,
            )
            return 1
        raise
    dt_s = mc.poll_ms / 1000.0

    try:
        if source == "shm":
            _run_from_shm(
                monitor, buf, phi, frame, fc, sign,
                use_inertia=use_inertia, min_samples=mc.min_samples, dt_s=dt_s,
            )
        else:
            _run_from_robot(
                monitor, buf, phi, frame, fc, sign,
                use_inertia=use_inertia, min_samples=mc.min_samples, dt_s=dt_s,
            )
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        monitor.close()
    return 0


def _resolve_source(wanted: str) -> str:
    if wanted == "robot":
        return "robot"
    if wanted == "shm":
        return "shm"
    from rm75_control.control.admittance_common.state_relay import relay_shm_has_publisher

    return "shm" if relay_shm_has_publisher() else "robot"


def _push_sample(
    monitor: CompMonitor,
    buf: SampleBuffer,
    t_s: float,
    pose: np.ndarray,
    force: np.ndarray,
    phi: np.ndarray,
    frame,
    fc: float,
    sign: np.ndarray,
    *,
    use_inertia: bool,
    min_samples: int,
    status: str,
) -> None:
    buf.append(t_s, pose, force)
    comp = compensate_latest(
        buf, phi, frame, fc, use_inertia=use_inertia, min_samples=min_samples,
    )
    if comp is None:
        monitor.set_status(f"{status}  buffer {len(buf.t)}/{min_samples}")
        monitor.append(t_s, force * sign, None)
    else:
        raw_show, ext_show = comp
        monitor.set_status(f"{status}  n={len(buf.t)}")
        monitor.append(t_s, raw_show, ext_show)


def _poll_wait(monitor: CompMonitor, next_poll: float, dt_s: float) -> float | None:
    import matplotlib.pyplot as plt

    if not plt.fignum_exists(monitor._fig.number):
        print("Plot closed — exiting.")
        return None
    now = time.monotonic()
    if now < next_poll:
        time.sleep(min(0.02, next_poll - now))
        return next_poll
    return next_poll + dt_s


def _run_from_shm(
    monitor: CompMonitor,
    buf: SampleBuffer,
    phi: np.ndarray,
    frame,
    fc: float,
    sign: np.ndarray,
    *,
    use_inertia: bool,
    min_samples: int,
    dt_s: float,
) -> None:
    from rm75_control.control.admittance_common.state_relay import RelayStateBus

    bus = RelayStateBus()
    if not bus.ensure_attached():
        raise SystemExit("no rm75_state — start Window A, or use --source robot")
    t0 = time.monotonic()
    next_poll = t0
    last_seq = -1
    try:
        while True:
            nxt = _poll_wait(monitor, next_poll, dt_s)
            if nxt is None:
                return
            if nxt == next_poll:
                continue
            next_poll = nxt
            now = time.monotonic()
            if not bus.ensure_attached():
                monitor.set_status("waiting rm75_state")
                monitor.refresh(now)
                continue
            snap = bus.read()
            if not snap.ok or snap.pose is None:
                monitor.set_status("shm stale")
                monitor.refresh(now)
                continue
            if int(snap.seq) == last_seq:
                monitor.refresh(now)
                continue
            last_seq = int(snap.seq)
            _push_sample(
                monitor, buf, now - t0, snap.pose, snap.force_raw, phi, frame, fc, sign,
                use_inertia=use_inertia, min_samples=min_samples, status="shm",
            )
            monitor.refresh(now)
    finally:
        bus.stop()


def _run_from_robot(
    monitor: CompMonitor,
    buf: SampleBuffer,
    phi: np.ndarray,
    frame,
    fc: float,
    sign: np.ndarray,
    *,
    use_inertia: bool,
    min_samples: int,
    dt_s: float,
) -> None:
    from rm75_control import RobotSession

    with RobotSession(config=CONFIG_ROBOT) as bot:
        t0 = time.monotonic()
        next_poll = t0
        while True:
            nxt = _poll_wait(monitor, next_poll, dt_s)
            if nxt is None:
                return
            if nxt == next_poll:
                continue
            next_poll = nxt
            now = time.monotonic()
            ret_s, st = bot.robot.rm_get_current_arm_state()
            ret_f, fd = bot.robot.rm_get_force_data()
            if ret_s != 0 or ret_f != 0:
                monitor.set_status(f"API err s={ret_s} f={ret_f}")
                monitor.refresh(now)
                continue
            pose = np.asarray(st["pose"][:6], dtype=float)
            force = np.asarray(fd["force_data"][:6], dtype=float)
            _push_sample(
                monitor, buf, now - t0, pose, force, phi, frame, fc, sign,
                use_inertia=use_inertia, min_samples=min_samples, status="robot",
            )
            monitor.refresh(now)


if __name__ == "__main__":
    raise SystemExit(main())
