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
    hold_v_max_m_s: float = 0.03
    hold_w_max_rad_s: float = 0.20
    hold_deadband_m: float = 0.001
    hold_deadband_rad: float = 0.005
    hold_settle_v_m_s: float = 0.005
    hold_relatch_on_settle: bool = True
    trigger_deadzone: float = 0.08


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


def apply_hold_deadband(err: np.ndarray, dead_m: float, dead_rad: float) -> np.ndarray:
    out = np.asarray(err, dtype=float).reshape(6).copy()
    if float(np.linalg.norm(out[:3])) <= max(float(dead_m), 0.0):
        out[:3] = 0.0
    if float(np.linalg.norm(out[3:6])) <= max(float(dead_rad), 0.0):
        out[3:6] = 0.0
    return out


class GamepadTwistOuterLoop:
    """Outer loop that sends stick velocity into the QPIK inner loop.

    Stick motion is open-loop v_cmd.  When the pad is idle, ``pose_d`` is
    latched after TCP settle (or immediately if relatch is off) so residual
    drift has a capped closed-loop hold.  A single slew limiter is the only
    rate-limit on both branches.
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
        self._prev_pose: np.ndarray | None = None
        self._twist_out = np.zeros(6, dtype=float)

    def set_origin(self, pose0: np.ndarray, *, t_s: float | None = None) -> None:
        del t_s
        pose = np.asarray(pose0, dtype=float).reshape(6).copy()
        self.last_pose_d = pose
        self.last_vel_ff = np.zeros(6, dtype=float)
        self._idle_hold_active = True
        self._hold_p_active = True
        self._coast_until_settle = False
        self._prev_pose = pose.copy()
        self._twist_out = np.zeros(6, dtype=float)
        self.last_twist_slewed = False

    def _slew_twist(self, twist: np.ndarray) -> np.ndarray:
        dt = float(self.cfg.dt)
        raw = np.asarray(twist, dtype=float).reshape(6)
        lin = slew_vec(self._twist_out[:3], raw[:3], self.cfg.trans_a_max_m_s2, dt)
        ang = slew_vec(self._twist_out[3:6], raw[3:6], self.cfg.rot_a_max_rad_s2, dt)
        out = np.zeros(6, dtype=float)
        out[:3] = lin[:3]
        out[3:6] = ang[:3]
        slewed = float(np.linalg.norm(out - raw)) > 1.0e-9
        self.last_twist_slewed = slewed
        self._twist_out = out
        return out

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
        v_world, w_tool = map_pad_to_world_lin_tool_ang(state, self.cfg)
        pose = np.asarray(current_pose, dtype=float).reshape(6).copy()
        dt = float(self.cfg.dt)
        tcp_speed = 0.0
        if self._prev_pose is not None and dt > 1.0e-9:
            tcp_speed = float(np.linalg.norm(pose[:3] - self._prev_pose[:3]) / dt)
        self._prev_pose = pose.copy()
        requested = (
            float(np.linalg.norm(v_world)) > 1.0e-12
            or float(np.linalg.norm(w_tool)) > 1.0e-12
        )
        if not requested:
            just_released = not self._idle_hold_active
            self._idle_hold_active = True
            if just_released and bool(self.cfg.hold_relatch_on_settle):
                self._coast_until_settle = True
                self._hold_p_active = False
            elif just_released:
                self._coast_until_settle = False
                self._hold_p_active = True
                if self.last_pose_d is None or not np.all(np.isfinite(self.last_pose_d)):
                    self.last_pose_d = pose.copy()
            if self._coast_until_settle:
                if tcp_speed <= float(self.cfg.hold_settle_v_m_s):
                    self.last_pose_d = pose.copy()
                    self._coast_until_settle = False
                    self._hold_p_active = True
                else:
                    self.last_vel_ff = np.zeros(6, dtype=float)
                    self.last_twist_base = np.zeros(6, dtype=float)
                    self.last_path_twist = np.zeros(6, dtype=float)
                    out = self._slew_twist(np.zeros(6, dtype=float))
                    self.last_feedback_twist = out.copy()
                    self.last_v_world = np.zeros(3, dtype=float)
                    self.last_w_tool = np.zeros(3, dtype=float)
                    self.last_err_mm = 0.0
                    return out
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
            if str(self.cfg.control_frame) == "tool":
                feedback = np.zeros(6, dtype=float)
                feedback[:3] = rotation.T @ fb_base[:3]
                feedback[3:6] = rotation.T @ fb_base[3:6]
            else:
                feedback = fb_base
            if not self._hold_p_active:
                feedback = np.zeros(6, dtype=float)
            self.last_pose_d = pose_d
            self.last_vel_ff = np.zeros(6, dtype=float)
            self.last_twist_base = np.zeros(6, dtype=float)
            self.last_path_twist = np.zeros(6, dtype=float)
            out = self._slew_twist(feedback)
            self.last_feedback_twist = out.copy()
            self.last_v_world = np.zeros(3, dtype=float)
            self.last_w_tool = np.zeros(3, dtype=float)
            self.last_err_mm = float(np.linalg.norm(err[:3]) * 1000.0)
            return out

        if self._idle_hold_active:
            self._idle_hold_active = False
            self._coast_until_settle = False
            self._hold_p_active = False
        twist, twist_base = compose_inner_twist(
            v_world,
            w_tool,
            current_pose,
            euler_order=self.cfg.euler_order,
            control_frame=self.cfg.control_frame,
        )
        out = self._slew_twist(twist)
        slewed_base = out.copy()
        if str(self.cfg.control_frame) == "tool":
            rotation = Rsc.from_euler(
                self.cfg.euler_order, pose[3:6], degrees=False
            ).as_matrix()
            slewed_base[:3] = rotation @ out[:3]
            slewed_base[3:6] = rotation @ out[3:6]
        pose_d = pose.copy()
        pose_d[:3] = pose[:3] + slewed_base[:3] * dt
        if float(np.linalg.norm(slewed_base[3:6])) > 1.0e-12:
            delta = Rsc.from_rotvec(slewed_base[3:6] * dt)
            cur = Rsc.from_euler(self.cfg.euler_order, pose[3:6], degrees=False)
            pose_d[3:6] = (delta * cur).as_euler(self.cfg.euler_order, degrees=False)
        self.last_pose_d = pose_d
        self.last_vel_ff = slewed_base
        self.last_twist_base = slewed_base
        self.last_path_twist = out.copy()
        self.last_feedback_twist = np.zeros(6, dtype=float)
        self.last_v_world = v_world
        self.last_w_tool = w_tool
        self.last_err_mm = 0.0
        return out
