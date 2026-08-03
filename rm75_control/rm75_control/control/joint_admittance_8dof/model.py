"""Pinocchio kinematics engine for RM75-F on Y-axis rail (8 DOF: rail_y + arm).

* Joint order  : rail_y (prismatic, m) then joint_1..joint_7 (rad).
* Realman API  : still 7 arm joints at the CANFD boundary; rail is sim/extra axis.
* Cartesian    : TCP twist / Jacobian in rail_base frame (LOCAL_WORLD_ALIGNED).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation as Rsc

from rm75_control.control.admittance_common.pose_math import (
    pose_error,
    pose_track_error_mm_deg,
)

DEFAULT_URDF = (
    Path(__file__).resolve().parents[2]
    / "assets"
    / "robots"
    / "rm75_6f_8dof"
    / "RM75-6F-8dof.urdf"
)

RAIL_JOINT_NAME = "rail_y"
ARM_JOINT_NAMES = [f"joint_{i}" for i in range(1, 8)]
JOINT_NAMES = [RAIL_JOINT_NAME, *ARM_JOINT_NAMES]
TCP_JOINT_NAME = "link_7_to_tcp"
RAIL_INDEX = 0
ARM_Q_INDICES = slice(1, 8)
N_ARM = 7
EXPECTED_NQ = 8


def deg2rad(q_deg: np.ndarray) -> np.ndarray:
    return np.asarray(q_deg, dtype=float) * (np.pi / 180.0)


def rad2deg(q_rad: np.ndarray) -> np.ndarray:
    return np.asarray(q_rad, dtype=float) * (180.0 / np.pi)


def wrap_joint_delta(q_from: np.ndarray, q_to: np.ndarray) -> np.ndarray:
    """Shortest signed joint delta; prismatic rail uses linear diff, arm uses (-pi, pi]."""
    a = np.asarray(q_from, dtype=float)
    b = np.asarray(q_to, dtype=float)
    d = b - a
    if d.size >= 1:
        d[0] = b[0] - a[0]
    if d.size > 1:
        arm = (b[1:] - a[1:] + np.pi) % (2.0 * np.pi) - np.pi
        d[1:] = arm
    return d


def arm_q_from_full(q_full: np.ndarray) -> np.ndarray:
    """Extract 7 arm joints (rad) for Realman CANFD."""
    return np.asarray(q_full, dtype=float)[ARM_Q_INDICES]


def full_q_from_arm(q_arm_rad: np.ndarray, rail_m: float = 0.0) -> np.ndarray:
    """Build 8-DOF state from rail position + 7 arm joints."""
    q = np.zeros(EXPECTED_NQ, dtype=float)
    q[0] = float(rail_m)
    q[1:] = np.asarray(q_arm_rad, dtype=float)[:N_ARM]
    return q


def max_joint_err_deg(q_a: np.ndarray, q_b: np.ndarray) -> float:
    """Max wrapped |dq| in degrees between two joint vectors."""
    return float(np.rad2deg(np.max(np.abs(wrap_joint_delta(q_a, q_b)))))


def pose_distance(
    pose_a: np.ndarray, pose_b: np.ndarray, euler_order: str = "xyz"
) -> tuple[float, float]:
    """Position distance (mm) and orientation distance (deg) between two pose6."""
    a = np.asarray(pose_a, dtype=float)
    b = np.asarray(pose_b, dtype=float)
    d_mm = float(np.linalg.norm(a[:3] - b[:3]) * 1000.0)
    ra = Rsc.from_euler(euler_order, a[3:6], degrees=False).as_matrix()
    rb = Rsc.from_euler(euler_order, b[3:6], degrees=False).as_matrix()
    d_deg = float(np.degrees(np.linalg.norm(Rsc.from_matrix(ra @ rb.T).as_rotvec())))
    return d_mm, d_deg


def auto_move_duration_s(
    kin: "RobotKinematics",
    q0_rad: np.ndarray,
    q_target_rad: np.ndarray,
    pose_target: np.ndarray,
    *,
    v_scale: float,
    v_max_rad_s: np.ndarray,
    peak_joint_v_frac: float = 0.80,
    max_lin_vel_m_s: float = 0.4,
    peak_lin_v_frac: float = 0.55,
    duration_min_s: float = 2.5,
    duration_max_s: float = 20.0,
    approach_dz_m: float | None = None,
    sigma_ref: float = 0.08,
    euler_order: str = "xyz",
) -> tuple[float, dict]:
    """Accuracy-first duration for a joint smoothstep move.

    Quintic smoothstep (``reference.smoothstep_scalar``) has peak joint speed
    ``15/8 · |dq|/T = 1.875 · |dq|/T`` per joint (vs the previous cubic 1.5·|dq|/T).
    ``T`` is chosen from joint kinematics only; TCP chord length from
    FK(q0)→pose_D is capped at the taught standoff (``approach_dz``) because
    the arm follows a joint path, not a straight-line TCP jump.  A hard
    ``duration_max_s`` prevents runaway plans when σ₀ is numerically tiny.
    """
    dq = wrap_joint_delta(q0_rad, q_target_rad)
    v_lim = np.asarray(v_max_rad_s, dtype=float) * float(v_scale) * float(peak_joint_v_frac)
    with np.errstate(divide="ignore", invalid="ignore"):
        per_joint = np.where(v_lim > 1e-6, 1.875 * np.abs(dq) / v_lim, 0.0)
    from_joints_s = float(np.max(per_joint))

    pose0 = kin.fk_pose(q0_rad)
    tcp_mm, _ = pose_distance(pose0, pose_target, euler_order)
    if approach_dz_m is not None and approach_dz_m > 0.0:
        tcp_mm = min(tcp_mm, float(approach_dz_m) * 1000.0 * 1.15)
    lin_cap = max(float(max_lin_vel_m_s) * float(peak_lin_v_frac), 1e-6)
    from_tcp_s = (tcp_mm / 1000.0) / lin_cap

    max_dq_deg = float(np.rad2deg(np.max(np.abs(dq))))
    joint_headroom = 1.0 + min(0.35, max(0.0, (max_dq_deg - 50.0) / 100.0))

    J0 = kin.jacobian(q0_rad)
    sigma0 = float(kin.singular_values(J0).min())

    raw = max(from_joints_s, from_tcp_s, float(duration_min_s))
    duration_s = min(float(duration_max_s), raw * joint_headroom)
    meta = {
        "from_joints_s": from_joints_s,
        "from_tcp_s": from_tcp_s,
        "joint_headroom": joint_headroom,
        "singularity_factor": 1.0,
        "max_dq_deg": max_dq_deg,
        "tcp_mm": tcp_mm,
        "sigma0": sigma0,
    }
    return duration_s, meta


class RobotKinematics:
    """Thin Pinocchio wrapper exposing FK, Jacobian and manipulability at the TCP."""

    def __init__(
        self,
        urdf_path: str | Path | None = None,
        tcp_frame: str = "tcp",
        euler_order: str = "xyz",
    ) -> None:
        self.urdf_path = Path(urdf_path) if urdf_path is not None else DEFAULT_URDF
        if not self.urdf_path.exists():
            raise FileNotFoundError(f"URDF not found: {self.urdf_path}")
        self.model = pin.buildModelFromUrdf(str(self.urdf_path))
        self.data = self.model.createData()
        self.euler_order = euler_order

        if not self.model.existFrame(tcp_frame):
            raise ValueError(f"frame {tcp_frame!r} not in URDF {self.urdf_path}")
        self.tcp_frame = tcp_frame
        self.tcp_id = self.model.getFrameId(tcp_frame)

        self._link7_id = (
            self.model.getFrameId("link_7") if self.model.existFrame("link_7") else None
        )
        self._tcp_offset_pose = self._read_tcp_offset_pose()
        self._R_link7_tcp, self._r_link7_tcp = self._compute_link7_to_tcp_kinematics()

        self.nq = self.model.nq
        self.nv = self.model.nv
        if self.nq != EXPECTED_NQ or self.nv != EXPECTED_NQ:
            raise ValueError(f"expected {EXPECTED_NQ}-DOF model, got nq={self.nq} nv={self.nv}")
        self.rail_index = RAIL_INDEX
        self.arm_q_indices = ARM_Q_INDICES

        # Position / velocity limits (radians, rad/s) straight from the URDF.
        self.q_lower = np.asarray(self.model.lowerPositionLimit, dtype=float).copy()
        self.q_upper = np.asarray(self.model.upperPositionLimit, dtype=float).copy()
        self.v_max = np.asarray(self.model.velocityLimit, dtype=float).copy()

    # ---- forward kinematics ------------------------------------------------
    def fk_placement(self, q_rad: np.ndarray) -> pin.SE3:
        q = np.asarray(q_rad, dtype=float)
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacement(self.model, self.data, self.tcp_id)
        return self.data.oMf[self.tcp_id]

    def fk_pose(self, q_rad: np.ndarray) -> np.ndarray:
        """TCP pose as [x, y, z, rx, ry, rz] (m, rad; intrinsic xyz Euler)."""
        M = self.fk_placement(q_rad)
        pose = np.zeros(6, dtype=float)
        pose[:3] = M.translation
        pose[3:6] = Rsc.from_matrix(M.rotation).as_euler(self.euler_order, degrees=False)
        return pose

    def fk_position_quat(self, q_rad: np.ndarray) -> np.ndarray:
        """TCP pose as [x, y, z, qx, qy, qz, qw] (handy for logging / comparisons)."""
        M = self.fk_placement(q_rad)
        quat = Rsc.from_matrix(M.rotation).as_quat()  # [x, y, z, w]
        return np.concatenate([M.translation, quat])

    def _read_tcp_offset_pose(self) -> np.ndarray:
        M = self.model.frames[self.tcp_id].placement
        pose = np.zeros(6, dtype=float)
        pose[:3] = np.asarray(M.translation, dtype=float)
        pose[3:6] = Rsc.from_matrix(M.rotation).as_euler(self.euler_order, degrees=False)
        return pose

    def apply_link7_to_tcp_offset(
        self,
        pose6: np.ndarray,
        *,
        euler_order: str | None = None,
    ) -> np.ndarray:
        """Set link_7->tcp frame from RealMan tool offset [x,y,z,rx,ry,rz] (m, rad)."""
        pose6 = np.asarray(pose6, dtype=float).reshape(6)
        order = str(euler_order or self.euler_order)
        R = Rsc.from_euler(order, pose6[3:6], degrees=False).as_matrix()
        self.model.frames[self.tcp_id].placement = pin.SE3(R, pose6[:3])
        self._tcp_offset_pose = pose6.copy()
        self._R_link7_tcp, self._r_link7_tcp = self._compute_link7_to_tcp_kinematics()
        return self._tcp_offset_pose.copy()

    @property
    def tcp_offset_pose(self) -> np.ndarray:
        return np.asarray(self._tcp_offset_pose, dtype=float).copy()

    def _compute_link7_to_tcp_kinematics(self) -> tuple[np.ndarray, np.ndarray]:
        """Rotation and translation (link_7 frame) from URDF tcp joint placement."""
        M = self.model.frames[self.tcp_id].placement
        R = np.asarray(M.rotation, dtype=float)
        r = np.asarray(M.translation, dtype=float)
        if self._link7_id is not None:
            q = pin.neutral(self.model)
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacement(self.model, self.data, self._link7_id)
            pin.updateFramePlacement(self.model, self.data, self.tcp_id)
            R7 = self.data.oMf[self._link7_id].rotation
            Rt = self.data.oMf[self.tcp_id].rotation
            pt = self.data.oMf[self.tcp_id].translation - self.data.oMf[self._link7_id].translation
            R = np.asarray(R7.T @ Rt, dtype=float)
            r = np.asarray(R7.T @ pt, dtype=float)
        return R, r

    def wrench_link7_to_tcp(self, wrench: np.ndarray) -> np.ndarray:
        """Express a link_7/sensor wrench at the tcp origin, in tcp tool coordinates."""
        w = np.asarray(wrench, dtype=float).reshape(6).copy()
        R = self._R_link7_tcp
        r = self._r_link7_tcp
        f_s = w[:3]
        m_s = w[3:6]
        # Transport moment to tcp origin (same frame), then rotate into tcp axes.
        m_at_tcp = m_s + np.cross(r, f_s)
        f_tcp = R.T @ f_s
        m_tcp = R.T @ m_at_tcp
        return np.concatenate([f_tcp, m_tcp])

    def frame_placement(self, q_rad: np.ndarray, frame_name: str) -> pin.SE3:
        """SE3 of an arbitrary frame (e.g. 'link_7' flange) in the base frame."""
        if not self.model.existFrame(frame_name):
            raise ValueError(f"frame {frame_name!r} not in URDF {self.urdf_path}")
        fid = self.model.getFrameId(frame_name)
        q = np.asarray(q_rad, dtype=float)
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacement(self.model, self.data, fid)
        return self.data.oMf[fid]

    def frame_pose(self, q_rad: np.ndarray, frame_name: str) -> np.ndarray:
        """Pose [x, y, z, rx, ry, rz] of an arbitrary frame in the base frame."""
        M = self.frame_placement(q_rad, frame_name)
        pose = np.zeros(6, dtype=float)
        pose[:3] = M.translation
        pose[3:6] = Rsc.from_matrix(M.rotation).as_euler(self.euler_order, degrees=False)
        return pose

    # ---- differential kinematics ------------------------------------------
    def jacobian(self, q_rad: np.ndarray) -> np.ndarray:
        """6×nv TCP Jacobian, LOCAL_WORLD_ALIGNED (linear on top, angular below).

        Maps joint velocity (rad/s) -> [v_lin(base), omega(base)].
        """
        q = np.asarray(q_rad, dtype=float)
        pin.computeJointJacobians(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        J = pin.getFrameJacobian(
            self.model, self.data, self.tcp_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        return np.asarray(J, dtype=float)

    @staticmethod
    def manipulability(J: np.ndarray) -> float:
        """Yoshikawa measure sqrt(det(J J^T)); 0 at a singularity."""
        JJt = J @ J.T
        det = float(np.linalg.det(JJt))
        return float(np.sqrt(max(det, 0.0)))

    @staticmethod
    def singular_values(J: np.ndarray) -> np.ndarray:
        return np.linalg.svd(J, compute_uv=False)

    def mass_matrix(self, q_rad: np.ndarray) -> np.ndarray:
        """Joint-space inertia matrix M(q) via Pinocchio CRBA (nv x nv, symmetric)."""
        q = np.asarray(q_rad, dtype=float)
        pin.crba(self.model, self.data, q)
        M = np.array(self.data.M, dtype=float)
        # CRBA returns upper triangle only
        return M + M.T - np.diag(np.diag(M))

    def clamp_to_limits(self, q_rad: np.ndarray, margin: float = 0.0) -> np.ndarray:
        return np.clip(q_rad, self.q_lower + margin, self.q_upper - margin)
