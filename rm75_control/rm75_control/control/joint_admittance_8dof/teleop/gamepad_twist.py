"""Map Xbox sticks onto a QPIK tool-frame twist (inner-loop v_cmd).

Translation is world/base XY + world Z. Rotation is TCP-frame.

  Left stick X (left) → world +Y
  Left stick Y (up)   → world +X
  LB                  → world +Z
  LT                  → world −Z
  Right stick X (right) → TCP +ωx
  Right stick Y (up)    → TCP −ωy
  RB / RT               → TCP ±ωz
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as Rsc

from rm75_control.control.admittance_common.pose_math import pose_error
from rm75_control.control.joint_admittance_8dof.teleop.xbox_pad import (
    XBOX_BUTTON_LB,
    XBOX_BUTTON_RB,
    PadState,
)

MAPPING_HELP = (
    "gamepad vcmd (inner QPIK): "
    "left stick XY world (left=+Y, up=+X); LB=+Z LT=-Z; "
    "right stick TCP rot; RB=+wz RT=-wz. Ctrl+C / window-C stop to exit."
)


@dataclass
class GamepadTwistConfig:
    trans_m_s: float = 0.10
    rot_rad_s: float = 0.60
    deadzone: float = 0.18
    max_lin_vel_m_s: float = 0.24
    max_ang_vel_rad_s: float = 1.20
    dt: float = 0.005
    euler_order: str = "xyz"
    control_frame: str = "tool"
    # Idle hold gain on latched pose_d (1/s).  Zero disables the P term.
    hold_k_task: float = 4.0
    trans_a_max_m_s2: float = 0.8
    rot_a_max_rad_s2: float = 4.0
    trans_j_max_m_s3: float = 8.0
    rot_j_max_rad_s3: float = 16.0
    pad_lpf_hz: float = 16.0
    deadzone_release: float = 0.12
    hold_v_max_m_s: float = 0.03
    hold_w_max_rad_s: float = 0.20
    hold_deadband_m: float = 0.001
    hold_deadband_rad: float = 0.005
    hold_settle_v_m_s: float = 0.005
    hold_relatch_on_settle: bool = True
    trigger_deadzone: float = 0.08


def pad_hold_active(
    state: PadState,
    cfg: GamepadTwistConfig,
    latched: bool,
) -> bool:
    """Stick/button activity with release hysteresis so the deadzone does not chatter."""

    axes = np.asarray(state.axes, dtype=float).reshape(-1)
    if axes.size < 6:
        padded = np.zeros(6, dtype=float)
        padded[: axes.size] = axes
        axes = padded
    enter = max(float(cfg.deadzone), 0.0)
    exit_ = min(enter, max(float(cfg.deadzone_release), 0.0))
    thr = exit_ if latched else enter
    sticks = any(abs(float(axes[i])) > thr for i in (0, 1, 3, 4))
    trig_enter = max(float(cfg.trigger_deadzone), 0.0)
    trig_thr = trig_enter * (exit_ / enter) if enter > 1.0e-12 and latched else trig_enter
    triggers = (
        normalize_trigger(float(axes[2]), trig_thr) > 0.0
        or normalize_trigger(float(axes[5]), trig_thr) > 0.0
    )
    buttons = bool(state.button(XBOX_BUTTON_LB) or state.button(XBOX_BUTTON_RB))
    return bool(sticks or triggers or buttons)


def apply_deadzone(value: float, deadzone: float) -> float:
    mag = abs(float(value))
    dz = float(deadzone)
    if mag <= dz or dz >= 1.0:
        return 0.0
    scaled = (mag - dz) / (1.0 - dz)
    return float(np.copysign(scaled, value))


def normalize_trigger(raw: float, deadzone: float = 0.08) -> float:
    """Linux Xbox trigger rests at −1; map to [0, 1] with a rest deadzone."""
    u = float(np.clip((float(raw) + 1.0) * 0.5, 0.0, 1.0))
    dz = float(deadzone)
    if dz <= 0.0:
        return u
    if dz >= 1.0 or u <= dz:
        return 0.0
    return (u - dz) / (1.0 - dz)


def _cap_vec(vec: np.ndarray, limit: float) -> np.ndarray:
    out = np.asarray(vec, dtype=float).reshape(-1).copy()
    nrm = float(np.linalg.norm(out))
    lim = float(limit)
    if lim > 0.0 and nrm > lim:
        out *= lim / nrm
    return out


def map_pad_to_world_lin_tool_ang(
    state: PadState,
    cfg: GamepadTwistConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(v_world[3], w_tool[3])`` in m/s and rad/s."""
    axes = np.asarray(state.axes, dtype=float).reshape(-1)
    if axes.size < 6:
        padded = np.zeros(6, dtype=float)
        padded[: axes.size] = axes
        axes = padded
    lx = apply_deadzone(float(axes[0]), cfg.deadzone)
    ly = apply_deadzone(float(axes[1]), cfg.deadzone)
    rx = apply_deadzone(float(axes[3]), cfg.deadzone)
    ry = apply_deadzone(float(axes[4]), cfg.deadzone)
    lt = normalize_trigger(float(axes[2]), cfg.trigger_deadzone)
    rt = normalize_trigger(float(axes[5]), cfg.trigger_deadzone)
    lb = 1.0 if state.button(XBOX_BUTTON_LB) else 0.0
    rb = 1.0 if state.button(XBOX_BUTTON_RB) else 0.0

    v_world = np.zeros(3, dtype=float)
    v_world[0] = (-ly) * float(cfg.trans_m_s)
    v_world[1] = (-lx) * float(cfg.trans_m_s)
    v_world[2] = (lb - lt) * float(cfg.trans_m_s)
    v_world = _cap_vec(v_world, cfg.max_lin_vel_m_s)

    w_tool = np.zeros(3, dtype=float)
    w_tool[0] = rx * float(cfg.rot_rad_s)
    w_tool[1] = ry * float(cfg.rot_rad_s)
    w_tool[2] = (rb - rt) * float(cfg.rot_rad_s)
    w_tool = _cap_vec(w_tool, cfg.max_ang_vel_rad_s)
    return v_world, w_tool


def compose_inner_twist(
    v_world: np.ndarray,
    w_tool: np.ndarray,
    current_pose: np.ndarray,
    *,
    euler_order: str,
    control_frame: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Pack the mixed-frame command for ``JointIkController.update``.

    Returns ``(twist_for_inner, twist_base)``.
    """
    pose = np.asarray(current_pose, dtype=float).reshape(6)
    rotation = Rsc.from_euler(euler_order, pose[3:6], degrees=False).as_matrix()
    v_w = np.asarray(v_world, dtype=float).reshape(3)
    w_t = np.asarray(w_tool, dtype=float).reshape(3)
    twist_base = np.zeros(6, dtype=float)
    twist_base[:3] = v_w
    twist_base[3:6] = rotation @ w_t
    if str(control_frame) == "tool":
        twist = np.zeros(6, dtype=float)
        twist[:3] = rotation.T @ v_w
        twist[3:6] = w_t
        return twist, twist_base
    return twist_base.copy(), twist_base


def slew_vec(prev: np.ndarray, target: np.ndarray, a_max: float, dt: float) -> np.ndarray:
    """Rate-limit a 3-vector without changing its direction."""

    out = np.asarray(target, dtype=float).reshape(-1).copy()
    prev_v = np.asarray(prev, dtype=float).reshape(-1)
    n = min(out.size, prev_v.size)
    if n == 0:
        return out
    step = out[:n] - prev_v[:n]
    lim = max(float(a_max), 0.0) * max(float(dt), 0.0)
    mag = float(np.linalg.norm(step))
    if lim > 0.0 and mag > lim:
        out[:n] = prev_v[:n] + step * (lim / mag)
    return out


def slew_vec_jerk(
    prev_v: np.ndarray,
    prev_a: np.ndarray,
    target: np.ndarray,
    a_max: float,
    j_max: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Accel + jerk limit with a braking approach that does not ring.

    ``j_max <= 0`` falls back to accel-only slew.
    """

    tgt = np.asarray(target, dtype=float).reshape(-1).copy()
    vel = np.asarray(prev_v, dtype=float).reshape(-1).copy()
    acc = np.asarray(prev_a, dtype=float).reshape(-1).copy()
    n = min(tgt.size, vel.size, acc.size)
    if n == 0:
        return tgt, acc
    if float(j_max) <= 0.0:
        vel[:n] = slew_vec(vel[:n], tgt[:n], a_max, dt)
        acc[:n] = 0.0
        return vel, acc
    h = max(float(dt), 1.0e-9)
    a_lim = max(float(a_max), 0.0)
    j_mag = max(float(j_max), 0.0)
    j_lim = j_mag * h
    err = tgt[:n] - vel[:n]
    e_n = float(np.linalg.norm(err))
    tau = max(h, 0.050)
    if e_n <= 1.0e-12:
        a_des = np.zeros(n)
    else:
        a_track = e_n / tau
        a_stop = float(np.sqrt(max(2.0 * j_mag * e_n, 0.0)))
        a_mag = a_track
        if a_lim > 0.0:
            a_mag = min(a_mag, a_lim)
        a_mag = min(a_mag, a_stop)
        a_des = err * (a_mag / e_n)
    da = a_des - acc[:n]
    da_n = float(np.linalg.norm(da))
    if j_lim > 0.0 and da_n > j_lim:
        da *= j_lim / da_n
    acc[:n] = acc[:n] + da
    acc_n = float(np.linalg.norm(acc[:n]))
    if a_lim > 0.0 and acc_n > a_lim:
        acc[:n] *= a_lim / acc_n
    nxt = vel[:n] + acc[:n] * h
    vel[:n] = nxt
    return vel, acc


def slew_axes_jerk(
    prev_v: np.ndarray,
    prev_a: np.ndarray,
    target: np.ndarray,
    a_max: float,
    j_max: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-axis accel+jerk limit so LT/ROLL do not share a 3-vector budget."""

    tgt = np.asarray(target, dtype=float).reshape(-1).copy()
    vel = np.asarray(prev_v, dtype=float).reshape(-1).copy()
    acc = np.asarray(prev_a, dtype=float).reshape(-1).copy()
    n = min(tgt.size, vel.size, acc.size)
    if n == 0:
        return tgt, acc
    for i in range(n):
        vi, ai = slew_vec_jerk(vel[i : i + 1], acc[i : i + 1], tgt[i : i + 1], a_max, j_max, dt)
        vel[i] = float(vi[0])
        acc[i] = float(ai[0])
    return vel, acc


def apply_hold_deadband(err: np.ndarray, dead_m: float, dead_rad: float) -> np.ndarray:
    out = np.asarray(err, dtype=float).reshape(6).copy()
    if float(np.linalg.norm(out[:3])) <= max(float(dead_m), 0.0):
        out[:3] = 0.0
    if float(np.linalg.norm(out[3:6])) <= max(float(dead_rad), 0.0):
        out[3:6] = 0.0
    return out


class GamepadTwistOuterLoop:
    """Outer loop that sends stick velocity into the QPIK inner loop.

    Stick motion is open-loop v_cmd.  World linear and tool angular axes are
    jerk-limited separately, then composed.  When the pad is idle, ``pose_d``
    is latched after TCP settle (or immediately if relatch is off) so residual
    drift has a capped closed-loop hold.  ``last_twist_base`` is always world.
    """

    def __init__(self, pad, cfg: GamepadTwistConfig | None = None) -> None:
        self.pad = pad
        self.cfg = cfg or GamepadTwistConfig()
        self.last_err_mm: float = 0.0
        self.last_vel_ff: np.ndarray | None = None
        self.last_pose_d: np.ndarray | None = None
        self.last_path_twist = np.zeros(6, dtype=float)
        self.last_feedback_twist = np.zeros(6, dtype=float)
        self.last_v_world = np.zeros(3, dtype=float)
        self.last_w_tool = np.zeros(3, dtype=float)
        self.last_pad_axes = np.zeros(6, dtype=float)
        self.last_pad_buttons = np.zeros(8, dtype=float)
        self.last_twist_base = np.zeros(6, dtype=float)
        self.last_pad_connected = False
        self.last_twist_slewed = False
        self._idle_hold_active = False
        self._hold_p_active = False
        self._coast_until_settle = False
        self._coast_ticks = 0
        self._prev_pose: np.ndarray | None = None
        self._mapped_out = np.zeros(6, dtype=float)
        self._mapped_acc = np.zeros(6, dtype=float)
        self._pad_lpf = np.zeros(6, dtype=float)
        self._pad_latched = False

    def set_origin(self, pose0: np.ndarray, *, t_s: float | None = None) -> None:
        del t_s
        pose = np.asarray(pose0, dtype=float).reshape(6).copy()
        self.last_pose_d = pose
        self.last_vel_ff = np.zeros(6, dtype=float)
        self._idle_hold_active = True
        self._hold_p_active = True
        self._coast_until_settle = False
        self._coast_ticks = 0
        self._prev_pose = pose.copy()
        self._mapped_out[:] = 0.0
        self._mapped_acc[:] = 0.0
        self._pad_lpf[:] = 0.0
        self._pad_latched = False
        self.last_twist_slewed = False

    def _lpf_pad(self, raw: np.ndarray) -> np.ndarray:
        dt = float(self.cfg.dt)
        fc = float(self.cfg.pad_lpf_hz)
        x = np.asarray(raw, dtype=float).reshape(-1)
        if fc <= 1.0e-9 or dt <= 0.0:
            self._pad_lpf[: x.size] = x[: self._pad_lpf.size]
            return x
        tau = 1.0 / (2.0 * np.pi * fc)
        a = dt / (tau + dt)
        n = min(x.size, self._pad_lpf.size)
        self._pad_lpf[:n] = (1.0 - a) * self._pad_lpf[:n] + a * x[:n]
        out = x.copy()
        out[:n] = self._pad_lpf[:n]
        return out

    def _slew_mapped(
        self, v_world: np.ndarray, w_tool: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        dt = float(self.cfg.dt)
        raw = np.zeros(6, dtype=float)
        raw[:3] = np.asarray(v_world, dtype=float).reshape(3)
        raw[3:6] = np.asarray(w_tool, dtype=float).reshape(3)
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
        out = np.zeros(6, dtype=float)
        out[:3] = lin[:3]
        out[3:6] = ang[:3]
        self._mapped_acc[:3] = a_lin[:3]
        self._mapped_acc[3:6] = a_ang[:3]
        self.last_twist_slewed = float(np.linalg.norm(out - raw)) > 1.0e-9
        self._mapped_out = out
        return out[:3].copy(), out[3:6].copy()

    def _publish_mapped(
        self,
        v_world: np.ndarray,
        w_tool: np.ndarray,
        pose: np.ndarray,
        *,
        feedback: np.ndarray | None = None,
        err_mm: float = 0.0,
        vel_ff: np.ndarray | None = None,
    ) -> np.ndarray:
        twist, twist_base = compose_inner_twist(
            v_world,
            w_tool,
            pose,
            euler_order=self.cfg.euler_order,
            control_frame=self.cfg.control_frame,
        )
        self.last_path_twist = twist.copy()
        self.last_twist_base = twist_base.copy()
        self.last_v_world = np.asarray(v_world, dtype=float).reshape(3).copy()
        self.last_w_tool = np.asarray(w_tool, dtype=float).reshape(3).copy()
        self.last_feedback_twist = (
            np.asarray(feedback, dtype=float).reshape(6).copy()
            if feedback is not None
            else np.zeros(6, dtype=float)
        )
        self.last_err_mm = float(err_mm)
        self.last_vel_ff = (
            np.asarray(vel_ff, dtype=float).reshape(6).copy()
            if vel_ff is not None
            else twist_base.copy()
        )
        return twist

    def sample(self, t_s: float, current_pose: np.ndarray, f_ext: np.ndarray) -> np.ndarray:
        del t_s, f_ext
        state = self.pad.read()
        self.last_pad_axes = np.asarray(state.axes, dtype=float).reshape(-1)[:6].copy()
        if self.last_pad_axes.size < 6:
            padded = np.zeros(6, dtype=float)
            padded[: self.last_pad_axes.size] = self.last_pad_axes
            self.last_pad_axes = padded
        self.last_pad_buttons = np.asarray(state.buttons, dtype=float).reshape(-1)[:8].copy()
        if self.last_pad_buttons.size < 8:
            padded_b = np.zeros(8, dtype=float)
            padded_b[: self.last_pad_buttons.size] = self.last_pad_buttons
            self.last_pad_buttons = padded_b
        self.last_pad_connected = bool(getattr(self.pad, "connected", True))
        v_raw, w_raw = map_pad_to_world_lin_tool_ang(state, self.cfg)
        requested = pad_hold_active(state, self.cfg, self._pad_latched)
        self._pad_latched = bool(requested)
        if requested:
            blended = self._lpf_pad(np.concatenate([v_raw, w_raw]))
            v_world = blended[:3].copy()
            w_tool = blended[3:6].copy()
        else:
            self._pad_lpf[:] = 0.0
            v_world = np.zeros(3, dtype=float)
            w_tool = np.zeros(3, dtype=float)
        pose = np.asarray(current_pose, dtype=float).reshape(6).copy()
        dt = float(self.cfg.dt)
        tcp_speed = 0.0
        if self._prev_pose is not None and dt > 1.0e-9:
            tcp_speed = float(np.linalg.norm(pose[:3] - self._prev_pose[:3]) / dt)
        self._prev_pose = pose.copy()
        if not requested:
            just_released = not self._idle_hold_active
            self._idle_hold_active = True
            if just_released and bool(self.cfg.hold_relatch_on_settle):
                self._coast_until_settle = True
                self._hold_p_active = False
                self._coast_ticks = 0
                self.last_pose_d = None
            elif just_released:
                self._coast_until_settle = False
                self._hold_p_active = True
                self._coast_ticks = 0
                self.last_pose_d = pose.copy()
            if self._coast_until_settle:
                v_s, w_s = self._slew_mapped(np.zeros(3), np.zeros(3))
                self._coast_ticks += 1
                settled = (
                    self._coast_ticks >= 2
                    and tcp_speed <= float(self.cfg.hold_settle_v_m_s)
                )
                if settled:
                    self.last_pose_d = pose.copy()
                    self._coast_until_settle = False
                    self._hold_p_active = True
                return self._publish_mapped(
                    v_s, w_s, pose, vel_ff=np.zeros(6, dtype=float)
                )
            if self.last_pose_d is None or not np.all(np.isfinite(self.last_pose_d)):
                self.last_pose_d = pose.copy()
            pose_d = np.asarray(self.last_pose_d, dtype=float).reshape(6).copy()
            err = apply_hold_deadband(
                pose_error(pose_d, pose, self.cfg.euler_order),
                self.cfg.hold_deadband_m,
                self.cfg.hold_deadband_rad,
            )
            k = max(float(self.cfg.hold_k_task), 0.0)
            fb_base = k * err
            fb_base[:3] = _cap_vec(fb_base[:3], self.cfg.hold_v_max_m_s)
            fb_base[3:6] = _cap_vec(fb_base[3:6], self.cfg.hold_w_max_rad_s)
            rotation = Rsc.from_euler(
                self.cfg.euler_order, pose[3:6], degrees=False
            ).as_matrix()
            v_hold = fb_base[:3].copy()
            w_hold = rotation.T @ fb_base[3:6]
            if not self._hold_p_active:
                v_hold[:] = 0.0
                w_hold[:] = 0.0
            v_s, w_s = self._slew_mapped(v_hold, w_hold)
            self.last_pose_d = pose_d
            twist = self._publish_mapped(
                v_s,
                w_s,
                pose,
                feedback=np.concatenate([v_s, w_s]),
                err_mm=float(np.linalg.norm(err[:3]) * 1000.0),
                vel_ff=np.zeros(6, dtype=float),
            )
            return twist

        if self._idle_hold_active:
            self._idle_hold_active = False
            self._coast_until_settle = False
            self._hold_p_active = False
            self._coast_ticks = 0
        v_s, w_s = self._slew_mapped(v_world, w_tool)
        twist, twist_base = compose_inner_twist(
            v_s,
            w_s,
            pose,
            euler_order=self.cfg.euler_order,
            control_frame=self.cfg.control_frame,
        )
        pose_d = pose.copy()
        pose_d[:3] = pose[:3] + twist_base[:3] * dt
        if float(np.linalg.norm(twist_base[3:6])) > 1.0e-12:
            delta = Rsc.from_rotvec(twist_base[3:6] * dt)
            cur = Rsc.from_euler(self.cfg.euler_order, pose[3:6], degrees=False)
            pose_d[3:6] = (delta * cur).as_euler(self.cfg.euler_order, degrees=False)
        self.last_pose_d = pose_d
        return self._publish_mapped(v_s, w_s, pose, vel_ff=twist_base)
