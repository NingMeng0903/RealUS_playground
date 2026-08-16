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
    trans_m_s: float = 0.08
    rot_rad_s: float = 0.60
    deadzone: float = 0.18
    max_lin_vel_m_s: float = 0.16
    max_ang_vel_rad_s: float = 1.20
    dt: float = 0.005
    euler_order: str = "xyz"
    control_frame: str = "tool"


def apply_deadzone(value: float, deadzone: float) -> float:
    mag = abs(float(value))
    dz = float(deadzone)
    if mag <= dz or dz >= 1.0:
        return 0.0
    scaled = (mag - dz) / (1.0 - dz)
    return float(np.copysign(scaled, value))


def normalize_trigger(raw: float) -> float:
    """Linux Xbox trigger rests at −1; map to [0, 1]."""
    return float(np.clip((float(raw) + 1.0) * 0.5, 0.0, 1.0))


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
    lt = normalize_trigger(float(axes[2]))
    rt = normalize_trigger(float(axes[5]))
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


class GamepadTwistOuterLoop:
    """Outer loop that sends stick velocity straight into the QPIK inner loop.

    No pose-PD: ``sample()`` is the v_cmd. Existing inner constraints
    (limits, CBF, rail pin/escape, nullspace) stay on ``JointIkController``.
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

    def set_origin(self, pose0: np.ndarray, *, t_s: float | None = None) -> None:
        del t_s
        pose = np.asarray(pose0, dtype=float).reshape(6).copy()
        self.last_pose_d = pose
        self.last_vel_ff = np.zeros(6, dtype=float)

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
        self.last_v_world = v_world
        self.last_w_tool = w_tool
        twist, twist_base = compose_inner_twist(
            v_world,
            w_tool,
            current_pose,
            euler_order=self.cfg.euler_order,
            control_frame=self.cfg.control_frame,
        )
        pose = np.asarray(current_pose, dtype=float).reshape(6).copy()
        dt = float(self.cfg.dt)
        pose_d = pose.copy()
        pose_d[:3] = pose[:3] + twist_base[:3] * dt
        if float(np.linalg.norm(twist_base[3:6])) > 1.0e-12:
            delta = Rsc.from_rotvec(twist_base[3:6] * dt)
            cur = Rsc.from_euler(self.cfg.euler_order, pose[3:6], degrees=False)
            pose_d[3:6] = (delta * cur).as_euler(self.cfg.euler_order, degrees=False)
        self.last_pose_d = pose_d
        self.last_vel_ff = twist_base
        self.last_twist_base = twist_base
        self.last_path_twist = np.asarray(twist, dtype=float).copy()
        self.last_feedback_twist = np.zeros(6, dtype=float)
        self.last_err_mm = 0.0
        return twist
