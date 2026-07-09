"""Motion references for the joint-admittance loop.

Re-uses admittance_common.MotionReference so any existing MotionReferenceSource
(demo trajectories, planners) is equally usable with the joint-space loop.

Provided here, self-contained (no robot handle needed - pure kinematics/scipy):

* HoldReference          - hold the start pose (bring-up default).
* JointSmoothMoveReference - smoothstep interpolation IN JOINT SPACE from q_start
  to q_target (from our pose_ik.solve_pose_ik, NOT vendor IK).  Exposed to the
  loop as FK/J(q_ref) Cartesian references via sample(), plus sample_q() whose
  qdot goes to Phase.qdot_ff_provider (nullspace feedforward).  This is the only
  point-to-point reference: moves are always planned in joint space, never as a
  forced Cartesian straight line (a line constraint fights redundancy
  resolution on a 7-DOF arm).
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

    C² Perlin form s = 10u³ − 15u⁴ + 6u⁵: s'(0)=s'(1)=0 and s''(0)=s''(1)=0,
    so no acceleration step at plan endpoints (cubic had s''(1)=−6/T² jerk).
    Peak joint speed is 15/8·|dq|/T vs cubic 1.5·|dq|/T (+25%).
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
        from rm75_control.control.joint_admittance.model import wrap_joint_delta

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

    def set_origin(self, pose0: np.ndarray) -> None:
        self._origin = np.asarray(pose0, dtype=float).copy()

    def sample(self, t_s: float) -> MotionReference:
        if self._origin is None:
            raise RuntimeError("SinToolYReference.set_origin must be called first")
        dy, vy = sin_y_motion(
            t_s, self.amplitude_m, self.omega, soft_start=self.soft_start, ramp_s=self.ramp_s
        )
        r_mat = Rsc.from_euler(self.euler_order, self._origin[3:6], degrees=False).as_matrix()
        pose = self._origin.copy()
        pose[:3] = self._origin[:3] + r_mat @ np.array([0.0, dy, 0.0])
        vel = np.zeros(6, dtype=float)
        vel[:3] = r_mat @ np.array([0.0, vy, 0.0])
        return MotionReference(pose, vel, t_ref=t_s)
