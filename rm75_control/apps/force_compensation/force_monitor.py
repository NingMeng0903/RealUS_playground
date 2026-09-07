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
import time
from collections import deque
from pathlib import Path
from threading import Lock

if __name__ == "__main__":
    from rm75_control.control.admittance_common.observer_runtime import prepare_observer_process
    prepare_observer_process()

import numpy as np
from scipy.spatial.transform import Rotation as Rsc

from rm75_control.control.admittance_common.observer import CompensatedForceObserver, ForceObserverConfig
from rm75_control.force.compensation.tool_pose import read_active_tool_offset, read_tool_offset_cache
from rm75_control.force.compensation.v2.frames import (
    link7_pose_from_tcp_pose, wrench_link7_to_sensor, wrench_link7_to_tcp, wrench_tcp_to_link7,
)
from rm75_control.force.compensation.id_config import load_config
from rm75_control.force.compensation.paths import CONFIG_ID, CONFIG_ROBOT

AXIS_LABELS = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]
FORCE_IDX = (0, 1, 2)
MOM_IDX = (3, 4, 5)


class MonitorEstimator:
    """One gravity-only observer, with an explicit pose and display frame."""

    def __init__(self, observer, offset: np.ndarray, display_frame: str = "tcp") -> None:
        self.observer = observer
        self.offset = np.asarray(offset, dtype=float).reshape(6)
        self.R_LT = Rsc.from_euler(observer.frame.euler_order, self.offset[3:]).as_matrix()
        self.r_LT = self.offset[:3]
        self.display_frame = display_frame

    def from_link7(self, wrench: np.ndarray) -> np.ndarray:
        if self.display_frame == "link7":
            return np.asarray(wrench, dtype=float).copy()
        if self.display_frame == "sensor":
            return wrench_link7_to_sensor(wrench, self.observer.contract)
        return wrench_link7_to_tcp(wrench, R_LT=self.R_LT, r_LT_L=self.r_LT)

    def from_tcp(self, wrench: np.ndarray) -> np.ndarray:
        if self.display_frame == "tcp":
            return np.asarray(wrench, dtype=float).copy()
        return self.from_link7(wrench_tcp_to_link7(wrench, R_LT=self.R_LT, r_LT_L=self.r_LT))

    def sample(self, t_s: float, pose_tcp: np.ndarray, force_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pose_link7 = link7_pose_from_tcp_pose(
            pose_tcp, R_LT=self.R_LT, r_LT_L=self.r_LT,
            euler_order=self.observer.frame.euler_order,
        )
        signed, ext = self.observer.update(t_s, pose_link7, force_raw)
        return self.from_link7(signed), self.from_link7(ext)


class CompMonitor:
    def __init__(self, *, window_s: float, refresh_hz: float = 12.0, display_frame: str = "tcp") -> None:
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
        self._fig.suptitle(f"6D wrench at {display_frame} origin, in {display_frame} axes: raw vs compensated")
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
                pad = max(0.5 if i < 3 else 0.005, 0.15 * (y1 - y0 + 1e-6))
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
    parser = argparse.ArgumentParser(description="Live 6D wrench; default uses controller TCP compensation")
    parser.add_argument("--id-config", type=Path, default=CONFIG_ID)
    parser.add_argument("--phi", type=Path, default=None, help="alternate model; implies --recompute")
    parser.add_argument("--phi-source", type=str, default=None, help="alternate parameter block; implies --recompute")
    parser.add_argument("--10p-only", dest="only_10p", action="store_true", help="compatibility flag; monitor replay is gravity-only")
    parser.add_argument("--source", choices=("auto", "shm", "robot"), default="auto")
    parser.add_argument("--frame", choices=("tcp", "link7", "sensor"), default="tcp", help="axes AND moment origin for all six plotted channels")
    parser.add_argument("--recompute", action="store_true", help="SHM: show independent gravity-only replay instead of controller F_ext")
    args = parser.parse_args()
    id_cfg = load_config(args.id_config)
    mc, fc_cfg = id_cfg.monitor, id_cfg.fit
    observer = CompensatedForceObserver(ForceObserverConfig(
        phi_path=args.phi or fc_cfg.phi_output,
        phi_source=args.phi_source or mc.phi_source,
        force_sensor=fc_cfg.force_sensor,
        poll_hz=1000.0 / mc.poll_ms,
        use_dynamic_kinematics=False, use_rotational_inertia=False,
        dynamic_kinematics_mode="off", min_samples=1,
    ))
    source = _resolve_source(args.source)
    print(f"phi={observer.model_path} source={observer.cfg.phi_source} rev={observer.model_revision} m={observer.phi[0]:.4f}kg parameters=link_7")
    print(f"source={source} display axes+origin={args.frame}; independent replay=gravity+bias only")
    print("No automatic zeroing. Raw and compensated curves use the same axes and moment origin.")
    monitor = CompMonitor(window_s=mc.window_s, refresh_hz=mc.refresh_hz, display_frame=args.frame)
    try:
        if source == "shm":
            replay_requested = args.recompute or args.phi is not None or args.phi_source is not None or args.only_10p
            _run_from_shm(monitor, observer, display_frame=args.frame, recompute=replay_requested, dt_s=mc.poll_ms / 1000.0)
        else:
            _run_from_robot(monitor, observer, display_frame=args.frame, dt_s=mc.poll_ms / 1000.0)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        monitor.close()
    return 0


def _resolve_source(wanted: str) -> str:
    if wanted != "auto":
        return wanted
    from rm75_control.control.admittance_common.state_relay import relay_shm_has_publisher
    return "shm" if relay_shm_has_publisher() else "robot"


def _poll_wait(monitor: CompMonitor, next_poll: float, dt_s: float) -> float | None:
    import matplotlib.pyplot as plt
    if not plt.fignum_exists(monitor._fig.number):
        return None
    now = time.monotonic()
    if now < next_poll:
        time.sleep(min(0.02, next_poll - now))
        return next_poll
    return max(next_poll + dt_s, now)


def _run_from_shm(monitor, observer, *, display_frame: str, recompute: bool, dt_s: float) -> None:
    from rm75_control.control.admittance_common.state_relay import ForceExtBus, RelayStateBus
    cached = read_tool_offset_cache()
    if cached is None:
        raise SystemExit("No live tool transform cache; start Window A first. Cannot interpret TCP pose as link_7.")
    tool_name, offset = cached
    est = MonitorEstimator(observer, offset, display_frame)
    print(f"tool={tool_name} tcp_offset={offset.tolist()}")
    state, ext_bus = RelayStateBus(), ForceExtBus()
    if not state.ensure_attached():
        raise SystemExit("no rm75_state — start Window A, or use --source robot")
    t0 = next_poll = time.monotonic()
    last_sample_t = None
    ctrl_seq, ctrl_seen_t = -1, float("-inf")
    next_model_check = t0
    try:
        while True:
            nxt = _poll_wait(monitor, next_poll, dt_s)
            if nxt is None:
                return
            if nxt == next_poll:
                continue
            next_poll = nxt
            now = time.monotonic()
            snap = state.read()
            if not snap.ok or snap.pose is None or float(snap.t_s) == last_sample_t:
                monitor.set_status("state stale / waiting fresh sensor sample")
                monitor.refresh(now)
                continue
            last_sample_t = float(snap.t_s)
            # This process is a read-only monitor, so local replay can reload.
            # Window A's observer is deliberately not changed from here.
            if now >= next_model_check:
                observer.reload_if_changed()
                live_cache = read_tool_offset_cache()
                if live_cache is not None and (
                    live_cache[0] != tool_name or not np.allclose(live_cache[1], est.offset, atol=1e-10, rtol=0)
                ):
                    tool_name, offset = live_cache
                    est = MonitorEstimator(observer, offset, display_frame)
                next_model_check = now + 1.0
            raw, replay = est.sample(float(snap.t_s), snap.pose, snap.force_raw)
            ok, seq, _stamp, ctrl_tcp = ext_bus.read()
            if ok and seq != ctrl_seq:
                ctrl_seq, ctrl_seen_t = seq, now
            ctrl_fresh = ok and now - ctrl_seen_t < 0.2 and np.isfinite(ctrl_tcp).all()
            shown = replay if recompute else None
            status = "independent gravity replay" if recompute else "waiting controller F_ext"
            if ctrl_fresh:
                controller = est.from_tcp(ctrl_tcp)
                if not recompute:
                    shown = controller
                    status = "controller F_ext"
                diff = controller - replay
                status += f"; controller-replay |dF|={np.linalg.norm(diff[:3]):.3f}N |dM|={np.linalg.norm(diff[3:]):.4f}Nm"
            status += f"; frame={display_frame}; replay m={observer.phi[0]:.4f}kg rev={observer.model_revision}"
            if observer.reload_error:
                status += "; phi reload failed: " + observer.reload_error
            monitor.set_status(status)
            monitor.append(now - t0, raw, shown)
            monitor.refresh(now)
    finally:
        state.stop()
        ext_bus.stop()


def _run_from_robot(monitor, observer, *, display_frame: str, dt_s: float) -> None:
    from rm75_control import RobotSession
    with RobotSession(config=CONFIG_ROBOT) as bot:
        name, offset = read_active_tool_offset(bot.robot)
        print(f"tool={name} tcp_offset={offset.tolist()}")
        est = MonitorEstimator(observer, offset, display_frame)
        t0 = next_poll = time.monotonic()
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
            if ret_s or ret_f:
                monitor.set_status(f"API err s={ret_s} f={ret_f}")
                monitor.refresh(now)
                continue
            raw, ext = est.sample(now, np.asarray(st["pose"][:6]), np.asarray(fd["force_data"][:6]))
            monitor.set_status(f"independent gravity+bias; frame={display_frame}; m={observer.phi[0]:.4f}kg")
            monitor.append(now - t0, raw, ext)
            monitor.refresh(now)


if __name__ == "__main__":
    raise SystemExit(main())
