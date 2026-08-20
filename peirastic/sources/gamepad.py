"""Pad → filtered 6D twist. Not a controller mode."""

from __future__ import annotations

import threading
import time
from dataclasses import replace

import numpy as np

from rm75_control.control.joint_admittance_8dof.teleop.gamepad_twist import (
    GamepadTwistConfig,
    compose_inner_twist,
    map_pad_to_world_lin_tool_ang,
    pad_hold_active,
    slew_axes_jerk,
)
from rm75_control.control.joint_admittance_8dof.teleop.xbox_pad import (
    PadState,
    XboxPad,
)

LOGICAL_L3 = 6
LOGICAL_R3 = 7


class GamepadTwistSource:
    """Background reader: deadzone → LPF → jerk slew → inner-frame twist."""

    def __init__(
        self,
        pad=None,
        cfg: GamepadTwistConfig | None = None,
        *,
        pose_fn=None,
    ) -> None:
        self.pad = pad if pad is not None else XboxPad()
        self.cfg = cfg or GamepadTwistConfig()
        self.pose_fn = pose_fn
        self._lock = threading.Lock()
        self._twist = np.zeros(6, dtype=float)
        self._axes = np.zeros(6, dtype=float)
        self._buttons = np.zeros(16, dtype=float)
        self._hz = float("nan")
        self._l3 = False
        self._r3 = False
        self._l3_prev = False
        self._r3_prev = False
        self._l3_edge = False
        self._r3_edge = False
        self._connected = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._mapped_out = np.zeros(6, dtype=float)
        self._mapped_acc = np.zeros(6, dtype=float)
        self._lpf = np.zeros(6, dtype=float)
        self._latched = False
        self._stamps: list[float] = []
        self._t0 = time.monotonic()
        self._armed = str(getattr(pad, "transport", "") or "") == "fake"

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="peirastic-pad", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        close = getattr(self.pad, "close", None)
        if close is not None:
            close()

    def snapshot(self) -> dict:
        with self._lock:
            l3_edge = self._l3_edge
            r3_edge = self._r3_edge
            self._l3_edge = False
            self._r3_edge = False
            return {
                "twist": self._twist.copy(),
                "axes": self._axes.copy(),
                "buttons": self._buttons.copy(),
                "hz": float(self._hz),
                "l3": self._l3,
                "r3": self._r3,
                "l3_edge": l3_edge,
                "r3_edge": r3_edge,
                "connected": self._connected,
                "layout": self._layout_name(),
                "armed": self._armed,
            }

    def _loop(self) -> None:
        dt = float(self.cfg.dt)
        while not self._stop.is_set():
            t0 = time.perf_counter()
            self._tick()
            remain = dt - (time.perf_counter() - t0)
            if remain > 0.0:
                time.sleep(remain)

    def _tick(self) -> None:
        state: PadState = self.pad.read()
        now = time.monotonic()
        self._stamps.append(now)
        if len(self._stamps) > 40:
            self._stamps = self._stamps[-40:]
        hz = float("nan")
        if len(self._stamps) >= 4:
            span = self._stamps[-1] - self._stamps[0]
            if span > 1e-6:
                hz = (len(self._stamps) - 1) / span
        axes = np.zeros(6, dtype=float)
        raw_ax = np.asarray(state.axes, dtype=float).reshape(-1)
        axes[: min(6, raw_ax.size)] = raw_ax[:6]
        buttons = np.zeros(16, dtype=float)
        raw_b = np.asarray(state.buttons, dtype=float).reshape(-1)
        buttons[: min(16, raw_b.size)] = raw_b[:16]
        l3 = bool(state.button(LOGICAL_L3))
        r3 = bool(state.button(LOGICAL_R3))
        layout = getattr(self.pad, "layout", None)
        z_sign = int(getattr(layout, "z_sign", getattr(self.cfg, "z_sign", 1)) or 1)
        cfg = replace(self.cfg, z_sign=z_sign)
        v_raw, w_raw = map_pad_to_world_lin_tool_ang(state, cfg)
        requested = pad_hold_active(state, self.cfg, self._latched)
        self._latched = bool(requested)
        if requested:
            blended = self._lpf_pad(np.concatenate([v_raw, w_raw]))
            v_w, w_t = blended[:3], blended[3:6]
        else:
            self._lpf[:] = 0.0
            v_w = np.zeros(3)
            w_t = np.zeros(3)
        if not self._armed and (time.monotonic() - self._t0) >= 0.25:
            self._armed = True
        if not self._armed:
            self._lpf[:] = 0.0
            self._mapped_out[:] = 0.0
            self._mapped_acc[:] = 0.0
            v_w = np.zeros(3)
            w_t = np.zeros(3)
        v_s, w_s = self._slew(v_w, w_t)
        pose = np.zeros(6)
        if self.pose_fn is not None:
            pose = np.asarray(self.pose_fn(), dtype=float).reshape(6)
        twist, _base = compose_inner_twist(
            v_s,
            w_s,
            pose,
            euler_order=self.cfg.euler_order,
            control_frame=self.cfg.control_frame,
        )
        with self._lock:
            self._l3_edge = self._l3_edge or (l3 and not self._l3_prev)
            self._r3_edge = self._r3_edge or (r3 and not self._r3_prev)
            self._l3_prev = l3
            self._r3_prev = r3
            self._l3 = l3
            self._r3 = r3
            self._twist = twist
            self._axes = axes
            self._buttons = buttons
            self._hz = hz
            self._connected = bool(getattr(self.pad, "connected", True))

    def _layout_name(self) -> str:
        layout = getattr(self.pad, "layout", None)
        name = getattr(layout, "name", None) if layout is not None else None
        return str(name or "pending")

    def _lpf_pad(self, raw: np.ndarray) -> np.ndarray:
        dt = float(self.cfg.dt)
        fc = float(self.cfg.pad_lpf_hz)
        x = np.asarray(raw, dtype=float).reshape(-1)
        if fc <= 1e-9 or dt <= 0.0:
            self._lpf[: x.size] = x[: self._lpf.size]
            return x
        tau = 1.0 / (2.0 * np.pi * fc)
        a = dt / (tau + dt)
        n = min(x.size, self._lpf.size)
        self._lpf[:n] = (1.0 - a) * self._lpf[:n] + a * x[:n]
        out = x.copy()
        out[:n] = self._lpf[:n]
        return out

    def _slew(self, v_world: np.ndarray, w_tool: np.ndarray):
        dt = float(self.cfg.dt)
        raw = np.zeros(6)
        raw[:3] = v_world
        raw[3:6] = w_tool
        lin, a_lin = slew_axes_jerk(
            self._mapped_out[:3],
            self._mapped_acc[:3],
            raw[:3],
            self.cfg.trans_a_max_m_s2,
            self.cfg.trans_j_max_m_s3,
            dt,
        )
        ang, a_ang = slew_axes_jerk(
            self._mapped_out[3:6],
            self._mapped_acc[3:6],
            raw[3:6],
            self.cfg.rot_a_max_rad_s2,
            self.cfg.rot_j_max_rad_s3,
            dt,
        )
        self._mapped_out[:3] = lin[:3]
        self._mapped_out[3:6] = ang[:3]
        self._mapped_acc[:3] = a_lin[:3]
        self._mapped_acc[3:6] = a_ang[:3]
        return self._mapped_out[:3].copy(), self._mapped_out[3:6].copy()
