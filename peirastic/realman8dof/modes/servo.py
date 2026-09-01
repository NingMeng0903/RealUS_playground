"""Pure velocity and velocity-position hybrid. No pad in this layer."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.spatial.transform import Rotation as Rsc

from rm75_control.control.admittance_common.pose_math import pose_error
from rm75_control.control.joint_admittance_8dof.teleop.gamepad_twist import (
    apply_hold_deadband,
    compose_inner_twist,
    slew_vec_jerk,
)

TwistFn = Callable[[], np.ndarray]


class TwistSlew:
    """Receive-side a/j limit on v*. First sample seeds; later ticks slew."""

    def __init__(
        self,
        *,
        a_lin_m_s2: float = 0.8,
        j_lin_m_s3: float = 16.0,
        a_ang_rad_s2: float = 4.0,
        j_ang_rad_s3: float = 32.0,
    ) -> None:
        self.a_lin_m_s2 = float(a_lin_m_s2)
        self.j_lin_m_s3 = float(j_lin_m_s3)
        self.a_ang_rad_s2 = float(a_ang_rad_s2)
        self.j_ang_rad_s3 = float(j_ang_rad_s3)
        self._v: np.ndarray | None = None
        self._a = np.zeros(6, dtype=float)
        self._t_prev: float | None = None

    def reset(self) -> None:
        self._v = None
        self._a[:] = 0.0
        self._t_prev = None

    def step(self, target: np.ndarray, t_s: float) -> np.ndarray:
        raw = np.asarray(target, dtype=float).reshape(6).copy()
        if self._v is None:
            self._v = raw
            self._a[:] = 0.0
            self._t_prev = float(t_s)
            return raw
        dt = 0.005
        if self._t_prev is not None:
            raw_dt = float(t_s) - float(self._t_prev)
            if np.isfinite(raw_dt) and raw_dt > 1.0e-4:
                dt = float(np.clip(raw_dt, 0.001, 0.05))
        self._t_prev = float(t_s)
        lin, a_lin = slew_vec_jerk(
            self._v[:3], self._a[:3], raw[:3], self.a_lin_m_s2, self.j_lin_m_s3, dt
        )
        ang, a_ang = slew_vec_jerk(
            self._v[3:], self._a[3:], raw[3:], self.a_ang_rad_s2, self.j_ang_rad_s3, dt
        )
        self._v[:3] = lin
        self._v[3:] = ang
        self._a[:3] = a_lin
        self._a[3:] = a_ang
        return self._v.copy()


def _as_twist(value) -> np.ndarray:
    if callable(value):
        value = value()
    out = np.asarray(value, dtype=float).reshape(-1)
    if out.size < 6:
        pad = np.zeros(6, dtype=float)
        pad[: out.size] = out
        return pad
    return out[:6].copy()


class ServoTwistOuter:
    """v* = v_cmd. No pose lock, no pad."""

    def __init__(
        self,
        source: TwistFn | np.ndarray,
        *,
        control_frame: str = "tool",
        euler_order: str = "xyz",
    ) -> None:
        self.source = source
        self.control_frame = str(control_frame)
        self.euler_order = str(euler_order)
        self.last_err_mm = 0.0
        self.last_vel_ff = np.zeros(6, dtype=float)
        self.last_pose_d: np.ndarray | None = None
        self.last_path_twist = np.zeros(6, dtype=float)
        self.last_feedback_twist = np.zeros(6, dtype=float)
        self._slew = TwistSlew()

    def set_origin(self, pose0: np.ndarray, *, t_s: float | None = None) -> None:
        del t_s
        self.last_pose_d = np.asarray(pose0, dtype=float).reshape(6).copy()
        self._slew.reset()

    def sample(self, t_s: float, current_pose: np.ndarray, f_ext: np.ndarray) -> np.ndarray:
        del f_ext
        twist = self._slew.step(_as_twist(self.source), float(t_s))
        self.last_path_twist = twist.copy()
        self.last_feedback_twist[:] = 0.0
        self.last_vel_ff = twist.copy()
        self.last_err_mm = 0.0
        # Keep pose_d on the live TCP. set_origin alone used to freeze Y at the
        # start, which made mid-ranging and tool_y_err chase the origin.
        self.last_pose_d = np.asarray(current_pose, dtype=float).reshape(6).copy()
        return twist


class ServoTwistHoldOuter:
    """Pass v_cmd while it is live; latch pose_d and P-hold when quiet."""

    def __init__(
        self,
        source: TwistFn | np.ndarray,
        *,
        control_frame: str = "tool",
        euler_order: str = "xyz",
        hold_k_task: float = 4.0,
        hold_v_max_m_s: float = 0.03,
        hold_w_max_rad_s: float = 0.20,
        hold_deadband_m: float = 0.001,
        hold_deadband_rad: float = 0.005,
        hold_settle_v_m_s: float = 0.005,
        quiet_lin_m_s: float = 0.002,
        quiet_rot_rad_s: float = 0.02,
        dt: float = 0.005,
    ) -> None:
        self.source = source
        self.control_frame = str(control_frame)
        self.euler_order = str(euler_order)
        self.hold_k_task = float(hold_k_task)
        self.hold_v_max_m_s = float(hold_v_max_m_s)
        self.hold_w_max_rad_s = float(hold_w_max_rad_s)
        self.hold_deadband_m = float(hold_deadband_m)
        self.hold_deadband_rad = float(hold_deadband_rad)
        self.hold_settle_v_m_s = float(hold_settle_v_m_s)
        self.quiet_lin_m_s = float(quiet_lin_m_s)
        self.quiet_rot_rad_s = float(quiet_rot_rad_s)
        self.dt = float(dt)
        self.last_err_mm = 0.0
        self.last_vel_ff = np.zeros(6, dtype=float)
        self.last_pose_d: np.ndarray | None = None
        self.last_path_twist = np.zeros(6, dtype=float)
        self.last_feedback_twist = np.zeros(6, dtype=float)
        self._holding = False
        self._coast = 0
        self._prev_pose: np.ndarray | None = None
        self._slew = TwistSlew()

    def set_origin(self, pose0: np.ndarray, *, t_s: float | None = None) -> None:
        del t_s
        pose = np.asarray(pose0, dtype=float).reshape(6).copy()
        self.last_pose_d = pose
        self._holding = True
        self._coast = 0
        self._prev_pose = pose.copy()
        self._slew.reset()

    def sample(self, t_s: float, current_pose: np.ndarray, f_ext: np.ndarray) -> np.ndarray:
        del f_ext
        pose = np.asarray(current_pose, dtype=float).reshape(6).copy()
        raw = _as_twist(self.source)
        lin = float(np.linalg.norm(raw[:3]))
        rot = float(np.linalg.norm(raw[3:6]))
        live = lin > self.quiet_lin_m_s or rot > self.quiet_rot_rad_s
        twist = self._slew.step(raw, float(t_s)) if live else raw
        tcp = 0.0
        if self._prev_pose is not None and self.dt > 1e-9:
            tcp = float(np.linalg.norm(pose[:3] - self._prev_pose[:3]) / self.dt)
        self._prev_pose = pose.copy()

        if live:
            self._holding = False
            self._coast = 0
            self.last_path_twist = twist.copy()
            self.last_feedback_twist[:] = 0.0
            self.last_vel_ff = twist.copy()
            self.last_err_mm = 0.0
            pose_d = pose.copy()
            pose_d[:3] = pose[:3] + twist[:3] * self.dt
            self.last_pose_d = pose_d
            return twist

        self._slew.reset()
        if not self._holding:
            self._coast += 1
            if self._coast >= 2 and tcp <= self.hold_settle_v_m_s:
                self.last_pose_d = pose.copy()
                self._holding = True
            self.last_path_twist[:] = 0.0
            self.last_feedback_twist[:] = 0.0
            self.last_vel_ff[:] = 0.0
            return np.zeros(6, dtype=float)

        if self.last_pose_d is None:
            self.last_pose_d = pose.copy()
        pose_d = np.asarray(self.last_pose_d, dtype=float).reshape(6)
        err = apply_hold_deadband(
            pose_error(pose_d, pose, self.euler_order),
            self.hold_deadband_m,
            self.hold_deadband_rad,
        )
        fb = self.hold_k_task * err
        lin_n = float(np.linalg.norm(fb[:3]))
        if lin_n > self.hold_v_max_m_s > 0.0:
            fb[:3] *= self.hold_v_max_m_s / lin_n
        rot_n = float(np.linalg.norm(fb[3:6]))
        if rot_n > self.hold_w_max_rad_s > 0.0:
            fb[3:6] *= self.hold_w_max_rad_s / rot_n
        v_world = fb[:3]
        w_tool = Rsc.from_euler(self.euler_order, pose[3:6], degrees=False).as_matrix().T @ fb[3:6]
        twist_out, _base = compose_inner_twist(
            v_world,
            w_tool,
            pose,
            euler_order=self.euler_order,
            control_frame=self.control_frame,
        )
        self.last_path_twist[:] = 0.0
        self.last_feedback_twist = twist_out.copy()
        self.last_vel_ff[:] = 0.0
        self.last_err_mm = float(np.linalg.norm(err[:3]) * 1000.0)
        return twist_out
