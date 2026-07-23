"""Motion references for the joint-admittance loop.

Re-uses admittance_common.MotionReference so any existing MotionReferenceSource
(demo trajectories, planners) is equally usable with the joint-space loop.

Provided here, self-contained (no robot handle needed - pure kinematics/scipy):

* HoldReference          - hold the start pose (bring-up default).
* JointSmoothMoveReference - smoothstep interpolation IN JOINT SPACE from q_start
  to q_target (from our pose_ik.solve_pose_ik, NOT vendor IK).  Exposed to the
  loop as FK/J(q_ref) Cartesian references via sample(), plus sample_q() whose
  qdot goes to Phase.qdot_ff_provider (nullspace feedforward).
* SrsSmoothMoveReference - Bug-5 replacement for JointSmoothMoveReference in
  ``phase_cartesian_goto``: quintic smoothstep in (pose, ψ) space with the
  SRS branch locked to q_start; each tick calls ``srs_ik`` to get q(t) with
  guaranteed branch consistency (no J1 flip mid-move).  Also exposes
  ``sample_psi(t)`` so the loop can drive ``inner.arm_task.set_reference``
  every tick and the arm-angle task tracks ψ_ref(t) continuously.
* SinToolYReference      - tool-frame Y sinusoid about a fixed origin (analogue
  of the tmp/Velocity_Admittance BuiltinTrajectorySource "sin_tool_y" mode, but
  computed directly instead of via robot.rm_algo_pose_move).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial.transform import Rotation as Rsc

from rm75_control.control.admittance_common.reference import MotionReference


class HoldReference:
    """Hold the start pose: pose_d = pose0, vel_ff = 0 (bring-up default).

    With force enabled and force_axes = tool-Z, this yields a pure constant-force
    hold - the safest first on-robot test of the cascade.
    """

    def __init__(self) -> None:
        self._pose0: np.ndarray | None = None

    def set_origin(self, pose0: np.ndarray) -> None:
        self._pose0 = np.asarray(pose0, dtype=float).copy()

    def sample(self, t_s: float) -> MotionReference:
        if self._pose0 is None:
            raise RuntimeError("HoldReference.set_origin must be called first")
        return MotionReference.from_pose_hold(self._pose0)


def smoothstep_scalar(t_s: float, duration_s: float) -> tuple[float, float]:
    """Quintic smoothstep s(u) in [0, 1] and ds/dt, u = clip(t/T, 0, 1).

    Uses the C² Perlin/quintic form s = 10u³ − 15u⁴ + 6u⁵ instead of the
    classic cubic 3u² − 2u³.  Both are monotone with s(0)=0, s(1)=1,
    s'(0)=s'(1)=0, but only the quintic also has s''(0)=s''(1)=0 — no
    acceleration step at either endpoint.

    The cubic form's s''(1) = −6/T² injected a 6°/s² peak deceleration
    burst into qdot_plan at plan end which the QP saw as a jerk step; on
    long joint moves through a σ dip, the arm couldn't fully decelerate
    within one accel-box tick and the TCP crossed the target by ~5–10 mm
    before PD pulled it back.  Quintic removes that pattern for free (no
    peak-velocity or peak-accel penalty — quintic peak qdot = 15/8·|dq|/T
    vs cubic 3/2·|dq|/T, only 25% higher, still miles under v_max on this
    arm).
    """
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
    """Smoothstep move in JOINT SPACE (q_start -> q_target), exposed as a Cartesian
    MotionReferenceSource via FK/Jacobian - i.e. the "free-planned, natural motion"
    analogue of MoveJ (smooth joint interpolation, whatever curved Cartesian path
    that implies), rather than a forced Cartesian straight line.

    Feeding (pose(t), vel_ff(t) = J(q(t)) @ qdot(t)) into the QP inner loop
    makes it track a target that is EXACTLY consistent with smooth joint motion,
    so the resulting q_cmd closely follows q(t) itself - the tracking correction
    only has to cancel small linearization residuals, not fight a Cartesian
    constraint.  Requires q_target to already be resolved via
    ``pose_ik.solve_pose_ik`` (self-developed WBC iterative IK - NEVER the
    vendor ``rm_algo_inverse_kinematics``) - this class itself does no IK, it
    only interpolates.
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
        """Joint-space (q_ref(t), qdot_ff(t)); qdot_ff feeds the QP nullspace via
        Phase.qdot_ff_provider so the redundant DOF follows this smoothstep."""
        from rm75_control.control.joint_admittance_8dof.model import wrap_joint_delta

        s, ds_dt = smoothstep_scalar(t_s, self.duration_s)
        dq = wrap_joint_delta(self.q_start, self.q_target)
        q = self.q_start + s * dq
        qdot = ds_dt * dq
        return q, qdot

    def sample(self, t_s: float) -> MotionReference:
        """Cartesian (pose, vel_ff) view via FK/Jacobian - feed through
        CartesianTrackOuterLoop; pair with qdot_ff_provider for nullspace tracking."""
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
) -> float:
    """Auto quintic duration bounded by per-joint rate limits.

    Quintic smoothstep peak speed on joint i is ``1.875·|dq_i|/T``.  We size T
    so no joint exceeds ``peak_v_frac · max_qdot_i`` at the mid-move peak, then
    take the worst-case joint as the binding constraint (a proper safety
    envelope, matching the plan's Bug-5 note ``T_i ≥ 1.875·|dq_i|/v_max_i``).
    """
    from rm75_control.control.joint_admittance_8dof.model import wrap_joint_delta

    dq = np.abs(wrap_joint_delta(q_start_rad, q_target_rad))
    if np.isscalar(max_qdot_rad_s):
        vmax_vec = np.full_like(dq, float(max_qdot_rad_s))
    else:
        vmax_vec = np.asarray(max_qdot_rad_s, dtype=float)
    vmax_vec = np.maximum(vmax_vec * float(peak_v_frac), 1e-6)
    t_per_joint = 1.875 * dq / vmax_vec
    return max(float(duration_min_s), float(np.max(t_per_joint)))


class SrsSmoothMoveReference:
    """Quintic smoothstep move in (pose, ψ, y_rail) space with SRS branch lock.

    Unlike :class:`JointSmoothMoveReference` (pure linear joint interp), this
    reference makes the Cartesian PATH straight-line in tool position + slerp
    in tool orientation, while the redundant DOF (ψ) is quintic-interpolated
    between start and target ψ.  Every tick, closed-form ``srs_ik`` yields
    q_ref(t) on the branch of ``q_start``, so:

    * primary-task tracking is a pure line-slerp (no jitter from IK residuals),
    * ψ transitions are C^2-smooth and constrained by the planner's max-swing,
    * no J1/J4 flip mid-move (branch is locked; :func:`branch_from_q` on
      ``q_start`` fixes the elbow/wrist configuration for the whole segment).

    The loop drives ``inner.arm_task.set_reference(sample_psi(t))`` every tick
    so the arm-angle secondary task tracks the ψ trajectory continuously.
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
        self.euler_order = str(euler_order)
        self.d_wt = float(d_wt_from_kin(kin) if d_wt is None else d_wt)
        R_start = Rsc.from_euler(self.euler_order, self.pose_start[3:])
        R_target = Rsc.from_euler(self.euler_order, self.pose_target[3:])
        self._R_start = R_start
        self._delta_rotvec = (R_target * R_start.inv()).as_rotvec()
        self._last_q = self.q_start.copy()

    def reseed_start(self, q_start_rad: np.ndarray) -> None:
        """Re-anchor the quintic at live encoders (soft-start, no Cartesian lurch).

        Keeps ``pose_target`` / ``y_target`` / ``psi_target``; recomputes start
        pose, rail, ψ, and branch lock from ``q_start_rad``.
        """
        from rm75_control.kinematics.srs_ik import branch_from_q, psi_from_q

        self.q_start = np.asarray(q_start_rad, dtype=float).copy()
        self.pose_start = np.asarray(self.kin.fk_pose(self.q_start), dtype=float)
        self.y_start = float(self.q_start[0])
        q_arm_start = self.q_start[1:]
        self.branch_id = int(branch_from_q(q_arm_start))
        self.psi_start = float(psi_from_q(q_arm_start))
        R_start = Rsc.from_euler(self.euler_order, self.pose_start[3:])
        R_target = Rsc.from_euler(self.euler_order, self.pose_target[3:])
        self._R_start = R_start
        self._delta_rotvec = (R_target * R_start.inv()).as_rotvec()
        self._last_q = self.q_start.copy()

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
        psi_s = self.psi_start + s * (self.psi_target - self.psi_start)
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
            q = self._last_q.copy()
            q[0] = y_s
        else:
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
        return float(self.psi_start + s * (self.psi_target - self.psi_start))

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


def sin_y_motion(
    t_s: float,
    amplitude_m: float,
    omega: float,
    *,
    soft_start: bool,
    ramp_s: float = 2.0,
) -> tuple[float, float]:
    """(dy, vy) of the sinusoid, with a C1-consistent soft start.

    The soft start is a TIME WARP tau(t): tau_dot ramps 0 -> 1 as
    sin(pi*t/(2*ramp_s)), so dy = A*sin(omega*tau) and vy = dy/dt stay exactly
    consistent (vy = A*omega*cos(omega*tau) * tau_dot).  Scaling only the
    velocity while leaving the position on the unwarped clock (the old
    behaviour) made pose_d and vel_ff contradict each other for the first
    ramp_s seconds - the tracking loop had to serve the whole initial
    transient from feedback (~15 mm error spikes on hardware).
    """
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


class SinToolYReference:
    """Tool-frame Y sinusoid about a fixed origin (orientation held constant).

    origin is set once via ``set_origin`` (e.g. pose D once the arm has arrived);
    pose = origin + R(origin) @ [0, amplitude*sin(wt), 0], matching a pure
    tool-frame translation delta (equivalent to rm_algo_pose_move with a
    translation-only delta in tool frame, computed directly - no robot RPC).
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
    ) -> None:
        if period_s is None:
            if max_vel_m_s is None:
                raise ValueError("provide either period_s or max_vel_m_s")
            period_s = sin_period_for_peak_vel(amplitude_m, max_vel_m_s)
        self.amplitude_m = float(amplitude_m)
        self.period_s = float(period_s)
        self.omega = 2.0 * math.pi / self.period_s if self.period_s > 0 else 0.0
        self.soft_start = soft_start
        self.ramp_s = ramp_s
        self.euler_order = euler_order
        self._origin: np.ndarray | None = None
        # Phase anchor for teach re-origin: sample uses (t_s - _t_anchor) so a
        # mid-scan set_origin() does not double-apply the accumulated sin offset.
        self._t_anchor: float = 0.0

    def set_origin(self, pose0: np.ndarray, *, t_s: float | None = None) -> None:
        self._origin = np.asarray(pose0, dtype=float).copy()
        if t_s is not None:
            self._t_anchor = float(t_s)

    def sample(self, t_s: float) -> MotionReference:
        if self._origin is None:
            raise RuntimeError("SinToolYReference.set_origin must be called first")
        t_eff = float(t_s) - float(self._t_anchor)
        dy, vy = sin_y_motion(
            t_eff, self.amplitude_m, self.omega, soft_start=self.soft_start, ramp_s=self.ramp_s
        )
        r_mat = Rsc.from_euler(self.euler_order, self._origin[3:6], degrees=False).as_matrix()
        pose = self._origin.copy()
        pose[:3] = self._origin[:3] + r_mat @ np.array([0.0, dy, 0.0])
        vel = np.zeros(6, dtype=float)
        vel[:3] = r_mat @ np.array([0.0, vy, 0.0])
        return MotionReference(pose_d=pose, vel_ff=vel, t_ref=t_s)
