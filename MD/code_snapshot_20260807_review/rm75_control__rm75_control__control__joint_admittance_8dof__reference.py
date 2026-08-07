"""Motion references for the joint-admittance loop (pure kinematics / scipy).

HoldReference, JointSmoothMoveReference, SrsSmoothMoveReference (branch-locked
quintic in pose/ψ), RailSmoothMoveReference, SinToolYReference.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial.transform import Rotation as Rsc

from rm75_control.control.admittance_common.reference import MotionReference


class HoldReference:
    """Hold the start pose: pose_d = pose0, vel_ff = 0."""

    def __init__(self) -> None:
        self._pose0: np.ndarray | None = None

    def set_origin(self, pose0: np.ndarray) -> None:
        self._pose0 = np.asarray(pose0, dtype=float).copy()

    def sample(self, t_s: float) -> MotionReference:
        if self._pose0 is None:
            raise RuntimeError("HoldReference.set_origin must be called first")
        return MotionReference.from_pose_hold(self._pose0)


def smoothstep_scalar(t_s: float, duration_s: float) -> tuple[float, float]:
    """Quintic smoothstep s(u), ds/dt with s''(0)=s''(1)=0 (C² endpoints)."""
    if duration_s <= 0.0:
        return 1.0, 0.0
    u = float(np.clip(t_s / duration_s, 0.0, 1.0))
    u2 = u * u
    u3 = u2 * u
    u4 = u3 * u
    u5 = u4 * u
    s = 10.0 * u3 - 15.0 * u4 + 6.0 * u5
    ds_du = 30.0 * u2 - 60.0 * u3 + 30.0 * u4
    ds_dt = ds_du / duration_s
    return s, ds_dt


class JointSmoothMoveReference:
    """Joint-space smoothstep (q_start→q_target) exposed via FK/J as Cartesian ref.

    Does no IK — interpolate a pre-resolved ``q_target`` only.
    """

    def __init__(
        self,
        kin,
        q_start_rad: np.ndarray,
        q_target_rad: np.ndarray,
        duration_s: float,
    ) -> None:
        self.kin = kin
        self.q_start = np.asarray(q_start_rad, dtype=float).copy()
        self.q_target = np.asarray(q_target_rad, dtype=float).copy()
        self.duration_s = float(duration_s)

    def set_origin(self, pose0: np.ndarray) -> None:
        # q_start already anchors this reference; pose0 is implied by FK(q_start).
        del pose0

    def sample_q(self, t_s: float) -> tuple[np.ndarray, np.ndarray]:
        """Joint-space (q_ref(t), qdot_ff(t)) for Phase.qdot_ff_provider."""
        from rm75_control.control.joint_admittance_8dof.model import joint_ptp_delta

        s, ds_dt = smoothstep_scalar(t_s, self.duration_s)
        # Limit-aware delta (not shortest-angle wrap) — see joint_ptp_delta.
        q_lo = getattr(self.kin, "q_lower", None)
        q_hi = getattr(self.kin, "q_upper", None)
        dq = joint_ptp_delta(self.q_start, self.q_target, q_lo, q_hi)
        q = self.q_start + s * dq
        qdot = ds_dt * dq
        return q, qdot

    def sample(self, t_s: float) -> MotionReference:
        """Cartesian (pose, vel_ff) via FK/Jacobian."""
        q, qdot = self.sample_q(t_s)
        pose = self.kin.fk_pose(q)
        vel = self.kin.jacobian(q) @ qdot
        return MotionReference(pose, vel, t_ref=t_s)

    def done(self, t_s: float) -> bool:
        return t_s >= self.duration_s


def srs_move_duration_s(
    q_start_rad: np.ndarray,
    q_target_rad: np.ndarray,
    *,
    max_qdot_rad_s: float | np.ndarray = 1.0,
    peak_v_frac: float = 0.60,
    duration_min_s: float = 0.5,
    q_lower: np.ndarray | None = None,
    q_upper: np.ndarray | None = None,
) -> float:
    """Auto duration so quintic peak ``1.875·|dq|/T`` stays under ``peak_v_frac·v_max``."""
    from rm75_control.control.joint_admittance_8dof.model import joint_ptp_delta

    dq = np.abs(joint_ptp_delta(q_start_rad, q_target_rad, q_lower, q_upper))
    if np.isscalar(max_qdot_rad_s):
        vmax_vec = np.full_like(dq, float(max_qdot_rad_s))
    else:
        vmax_vec = np.asarray(max_qdot_rad_s, dtype=float)
    vmax_vec = np.maximum(vmax_vec * float(peak_v_frac), 1e-6)
    t_per_joint = 1.875 * dq / vmax_vec
    return max(float(duration_min_s), float(np.max(t_per_joint)))


class SrsSmoothMoveReference:
    """Branch-locked quintic in (pose, ψ, y_rail); each tick ``srs_ik`` on q_start branch.

    Cartesian path is line+slerp; ψ is C²-smooth; no mid-move J1/J4 flip.
    """

    def __init__(
        self,
        kin,
        q_start_rad: np.ndarray,
        pose_target: np.ndarray,
        *,
        y_rail_target_m: float,
        psi_target_rad: float,
        duration_s: float,
        branch_id: int | None = None,
        euler_order: str = "xyz",
        d_wt: float | None = None,
        max_ik_fail_streak: int = 5,
    ) -> None:
        from rm75_control.kinematics.srs_ik import branch_from_q, d_wt_from_kin, psi_from_q

        self.kin = kin
        self.q_start = np.asarray(q_start_rad, dtype=float).copy()
        self.pose_start = np.asarray(self.kin.fk_pose(self.q_start), dtype=float)
        self.pose_target = np.asarray(pose_target, dtype=float).copy()
        self.y_start = float(self.q_start[0])
        self.y_target = float(y_rail_target_m)
        self.duration_s = float(duration_s)
        q_arm_start = self.q_start[1:]
        self.branch_id = int(branch_id) if branch_id is not None else int(branch_from_q(q_arm_start))
        self.psi_start = float(psi_from_q(q_arm_start))
        self.psi_target = float(psi_target_rad)
        # Shortest-arc unwrap so ψ does not travel the long way around ±π.
        self.psi_delta = float(
            (self.psi_target - self.psi_start + np.pi) % (2.0 * np.pi) - np.pi
        )
        self.euler_order = str(euler_order)
        self.d_wt = float(d_wt_from_kin(kin) if d_wt is None else d_wt)
        R_start = Rsc.from_euler(self.euler_order, self.pose_start[3:])
        R_target = Rsc.from_euler(self.euler_order, self.pose_target[3:])
        self._R_start = R_start
        self._delta_rotvec = (R_target * R_start.inv()).as_rotvec()
        self._last_q = self.q_start.copy()
        self._ik_fail_streak = 0
        self._max_ik_fail_streak = int(max(1, max_ik_fail_streak))

    def reseed_start(self, q_start_rad: np.ndarray) -> None:
        """Re-anchor start from live encoders; keep pose/y/ψ targets."""
        from rm75_control.kinematics.srs_ik import branch_from_q, psi_from_q

        self.q_start = np.asarray(q_start_rad, dtype=float).copy()
        self.pose_start = np.asarray(self.kin.fk_pose(self.q_start), dtype=float)
        self.y_start = float(self.q_start[0])
        q_arm_start = self.q_start[1:]
        self.branch_id = int(branch_from_q(q_arm_start))
        self.psi_start = float(psi_from_q(q_arm_start))
        self.psi_delta = float(
            (self.psi_target - self.psi_start + np.pi) % (2.0 * np.pi) - np.pi
        )
        R_start = Rsc.from_euler(self.euler_order, self.pose_start[3:])
        R_target = Rsc.from_euler(self.euler_order, self.pose_target[3:])
        self._R_start = R_start
        self._delta_rotvec = (R_target * R_start.inv()).as_rotvec()
        self._last_q = self.q_start.copy()
        self._ik_fail_streak = 0

    def _pose_at(self, s: float) -> np.ndarray:
        pos = self.pose_start[:3] + s * (self.pose_target[:3] - self.pose_start[:3])
        R_at = Rsc.from_rotvec(s * self._delta_rotvec) * self._R_start
        pose = np.zeros(6)
        pose[:3] = pos
        pose[3:] = R_at.as_euler(self.euler_order)
        return pose

    def _q_at(self, s: float) -> np.ndarray:
        from rm75_control.kinematics.srs_ik import srs_ik

        pose_s = self._pose_at(s)
        psi_s = self.psi_start + s * self.psi_delta
        y_s = self.y_start + s * (self.y_target - self.y_start)
        q_arm = srs_ik(
            pose_s,
            psi_s,
            self.branch_id,
            y_rail=y_s,
            euler_order=self.euler_order,
            check_limits=False,
            d_wt=self.d_wt,
        )
        q = np.zeros_like(self.q_start)
        q[0] = y_s
        if q_arm is None:
            self._ik_fail_streak += 1
            if self._ik_fail_streak >= self._max_ik_fail_streak:
                raise RuntimeError(
                    f"SrsSmoothMoveReference: srs_ik returned None for "
                    f"{self._ik_fail_streak} consecutive samples "
                    f"(s={s:.3f}, branch={self.branch_id}, "
                    f"psi={np.degrees(psi_s):.1f}deg). "
                    f"Refusing silent joint hold (would freeze TCP governor). "
                    f"Use joint PTP recovery for cross-branch moves."
                )
            q = self._last_q.copy()
            q[0] = y_s
        else:
            self._ik_fail_streak = 0
            q[1:] = q_arm
            self._last_q = q.copy()
        return q

    def sample_q(self, t_s: float) -> tuple[np.ndarray, np.ndarray]:
        s, _ds_dt = smoothstep_scalar(t_s, self.duration_s)
        q = self._q_at(s)
        # qdot_ff via central-diff on the smoothstep clock so the loop's
        # Phase.qdot_ff_provider gets a consistent (q, qdot) pair even at t=0/T.
        h = 1.0e-3
        s_plus, _ = smoothstep_scalar(min(t_s + h, self.duration_s), self.duration_s)
        s_minus, _ = smoothstep_scalar(max(t_s - h, 0.0), self.duration_s)
        q_plus = self._q_at(s_plus)
        q_minus = self._q_at(s_minus)
        denom = max(1e-9, (min(t_s + h, self.duration_s) - max(t_s - h, 0.0)))
        qdot = (q_plus - q_minus) / denom
        return q, qdot

    def sample(self, t_s: float) -> MotionReference:
        s, _ = smoothstep_scalar(t_s, self.duration_s)
        pose = self._pose_at(s)
        h = 1.0e-3
        s_plus, _ = smoothstep_scalar(min(t_s + h, self.duration_s), self.duration_s)
        s_minus, _ = smoothstep_scalar(max(t_s - h, 0.0), self.duration_s)
        pose_plus = self._pose_at(s_plus)
        pose_minus = self._pose_at(s_minus)
        denom = max(1e-9, (min(t_s + h, self.duration_s) - max(t_s - h, 0.0)))
        vel = np.zeros(6)
        vel[:3] = (pose_plus[:3] - pose_minus[:3]) / denom
        R_plus = Rsc.from_euler(self.euler_order, pose_plus[3:])
        R_minus = Rsc.from_euler(self.euler_order, pose_minus[3:])
        vel[3:] = (R_plus * R_minus.inv()).as_rotvec() / denom
        return MotionReference(pose_d=pose, vel_ff=vel, t_ref=t_s)

    def sample_psi(self, t_s: float) -> float:
        s, _ = smoothstep_scalar(t_s, self.duration_s)
        return float(self.psi_start + s * self.psi_delta)

    def set_origin(self, pose0: np.ndarray) -> None:
        del pose0  # q_start anchors this reference

    def done(self, t_s: float) -> bool:
        return t_s >= self.duration_s


def auto_rail_move_duration_s(
    q_start_m: float,
    q_target_m: float,
    *,
    v_max_m_s: float,
    peak_v_frac: float = 0.50,
    duration_min_s: float = 0.5,
) -> float:
    """Duration for quintic rail smoothstep (peak speed 1.875·|dq|/T)."""
    dq = abs(float(q_target_m) - float(q_start_m))
    v_lim = max(float(v_max_m_s) * float(peak_v_frac), 1e-6)
    from_rail = 1.875 * dq / v_lim
    return max(float(duration_min_s), from_rail)


class RailSmoothMoveReference:
    """Quintic smoothstep on rail_y only; arm joints held at q_start[1:]."""

    def __init__(
        self,
        q_start: np.ndarray,
        q_target_m: float,
        duration_s: float,
    ) -> None:
        self.q_start = np.asarray(q_start, dtype=float).copy()
        self.q_target_m = float(q_target_m)
        self.duration_s = float(duration_s)
        self._q_arm = self.q_start[1:].copy()

    @property
    def q_target(self) -> np.ndarray:
        q = self.q_start.copy()
        q[0] = self.q_target_m
        return q

    def sample_q(self, t_s: float) -> tuple[np.ndarray, np.ndarray]:
        s, ds_dt = smoothstep_scalar(t_s, self.duration_s)
        dq_rail = self.q_target_m - float(self.q_start[0])
        q = np.zeros_like(self.q_start)
        q[0] = float(self.q_start[0]) + s * dq_rail
        q[1:] = self._q_arm
        qdot = np.zeros_like(self.q_start)
        qdot[0] = ds_dt * dq_rail
        return q, qdot

    def done(self, t_s: float) -> bool:
        return t_s >= self.duration_s


def sin_period_for_peak_vel(amplitude_m: float, max_vel_m_s: float) -> float:
    if amplitude_m <= 0.0 or max_vel_m_s <= 0.0:
        return 1.0
    return 2.0 * math.pi * amplitude_m / max_vel_m_s


def quintic_move_s_for_peak_vel(amplitude_m: float, max_vel_m_s: float) -> float:
    """Half-stroke duration so quintic peak speed 1.875·(2A)/T equals ``max_vel``."""
    if amplitude_m <= 0.0 or max_vel_m_s <= 0.0:
        return 1.0
    return float(1.875 * (2.0 * amplitude_m) / max_vel_m_s)


def sin_y_motion(
    t_s: float,
    amplitude_m: float,
    omega: float,
    *,
    soft_start: bool,
    ramp_s: float = 2.0,
) -> tuple[float, float]:
    """(dy, vy) with C1 soft start via time-warp tau(t) (pose/vel stay consistent)."""
    if soft_start and ramp_s > 0.0:
        if t_s < ramp_s:
            # tau(t) = int_0^t sin(pi*u/(2*ramp)) du
            tau = (2.0 * ramp_s / math.pi) * (1.0 - math.cos(0.5 * math.pi * t_s / ramp_s))
            tau_dot = math.sin(0.5 * math.pi * t_s / ramp_s)
        else:
            tau = t_s - ramp_s + (2.0 * ramp_s / math.pi)
            tau_dot = 1.0
    else:
        tau = t_s
        tau_dot = 1.0
    dy = amplitude_m * math.sin(omega * tau)
    vy = amplitude_m * omega * math.cos(omega * tau) * tau_dot
    return dy, vy


def _unit_quintic_dwell_profile(
    t_s: float,
    move_s: float,
    dwell_s: float,
) -> tuple[float, float]:
    """Periodic unit scan in [-1, 1]: quintic cruise, zero vel/accel at ends + dwell."""
    move_s = max(float(move_s), 1e-3)
    dwell_s = max(float(dwell_s), 0.0)
    half = move_s + dwell_s
    cycle = 2.0 * half
    tau = float(t_s) % cycle
    if tau < move_s:
        s, ds = smoothstep_scalar(tau, move_s)
        return -1.0 + 2.0 * s, 2.0 * ds
    if tau < move_s + dwell_s:
        return 1.0, 0.0
    tau_back = tau - move_s - dwell_s
    if tau_back < move_s:
        s, ds = smoothstep_scalar(tau_back, move_s)
        return 1.0 - 2.0 * s, -2.0 * ds
    return -1.0, 0.0


def quintic_dwell_y_motion(
    t_s: float,
    amplitude_m: float,
    move_s: float,
    dwell_s: float,
    *,
    soft_start: bool,
    ramp_s: float = 2.0,
) -> tuple[float, float]:
    """(dy, vy) round-trip with C² endpoints and optional end dwell."""
    p, pdot = _unit_quintic_dwell_profile(t_s, move_s, dwell_s)
    if soft_start and ramp_s > 0.0 and t_s < ramp_s:
        amp_scale, amp_dot = smoothstep_scalar(t_s, ramp_s)
    else:
        amp_scale, amp_dot = 1.0, 0.0
    a = float(amplitude_m)
    dy = a * amp_scale * p
    vy = a * (amp_dot * p + amp_scale * pdot)
    return dy, vy


class SinToolYReference:
    """Tool-frame Y scan about a fixed origin (orientation held).

    ``profile``:
    * ``sine`` — classic sinusoid (max |a| at turnaround where v=0);
    * ``quintic_dwell`` — C² quintic half-strokes with end dwell (v=a=0 at ends).
    """

    def __init__(
        self,
        amplitude_m: float,
        *,
        period_s: float | None = None,
        max_vel_m_s: float | None = None,
        soft_start: bool = True,
        ramp_s: float = 2.0,
        euler_order: str = "xyz",
        profile: str = "quintic_dwell",
        dwell_s: float = 0.20,
    ) -> None:
        self.amplitude_m = float(amplitude_m)
        self.profile = str(profile).strip().lower()
        self.dwell_s = max(float(dwell_s), 0.0)
        self.soft_start = soft_start
        self.ramp_s = ramp_s
        self.euler_order = euler_order
        self._origin: np.ndarray | None = None
        # Phase anchor for teach re-origin: sample uses (t_s - _t_anchor) so a
        # mid-scan set_origin() does not double-apply the accumulated sin offset.
        self._t_anchor: float = 0.0

        if self.profile not in ("sine", "quintic_dwell"):
            raise ValueError(f"unknown scan profile {profile!r}")

        if self.profile == "sine":
            if period_s is None:
                if max_vel_m_s is None:
                    raise ValueError("provide either period_s or max_vel_m_s")
                period_s = sin_period_for_peak_vel(amplitude_m, max_vel_m_s)
            self.period_s = float(period_s)
            self.omega = 2.0 * math.pi / self.period_s if self.period_s > 0 else 0.0
            self.move_s = 0.0
        else:
            if period_s is not None:
                half = 0.5 * float(period_s)
                move_s = max(half - self.dwell_s, 1e-3)
                if half <= self.dwell_s:
                    move_s = max(half * 0.8, 1e-3)
                    self.dwell_s = max(half - move_s, 0.0)
            else:
                if max_vel_m_s is None:
                    raise ValueError("provide either period_s or max_vel_m_s")
                move_s = quintic_move_s_for_peak_vel(amplitude_m, max_vel_m_s)
            self.move_s = float(move_s)
            self.period_s = 2.0 * (self.move_s + self.dwell_s)
            self.omega = 0.0

    def set_origin(self, pose0: np.ndarray, *, t_s: float | None = None) -> None:
        self._origin = np.asarray(pose0, dtype=float).copy()
        if t_s is not None:
            self._t_anchor = float(t_s)

    def sample(self, t_s: float) -> MotionReference:
        if self._origin is None:
            raise RuntimeError("SinToolYReference.set_origin must be called first")
        t_eff = float(t_s) - float(self._t_anchor)
        if self.profile == "sine":
            dy, vy = sin_y_motion(
                t_eff,
                self.amplitude_m,
                self.omega,
                soft_start=self.soft_start,
                ramp_s=self.ramp_s,
            )
        else:
            dy, vy = quintic_dwell_y_motion(
                t_eff,
                self.amplitude_m,
                self.move_s,
                self.dwell_s,
                soft_start=self.soft_start,
                ramp_s=self.ramp_s,
            )
        r_mat = Rsc.from_euler(self.euler_order, self._origin[3:6], degrees=False).as_matrix()
        pose = self._origin.copy()
        pose[:3] = self._origin[:3] + r_mat @ np.array([0.0, dy, 0.0])
        vel = np.zeros(6, dtype=float)
        vel[:3] = r_mat @ np.array([0.0, vy, 0.0])
        return MotionReference(pose_d=pose, vel_ff=vel, t_ref=t_s)

