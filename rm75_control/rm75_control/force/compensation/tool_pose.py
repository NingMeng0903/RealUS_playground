"""Map poses.yaml slot TCP into the active tool frame (FK from q_deg)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation as Rsc

if TYPE_CHECKING:
    from rm75_control.control.joint_admittance.model import RobotKinematics

# Scan standoff: poses.yaml slot d is force-ID Arm_Tip at contact; runtime with
# gripper active applies +approach_dz along the *active* tool Z at teach q_deg.
DEFAULT_SCAN_APPROACH_DZ_M = 0.220
DEFAULT_EULER_ORDER = "xyz"


def get_active_tool_name(robot) -> str:
    ret, cur = robot.rm_get_current_tool_frame()
    if ret != 0:
        return ""
    return str(cur.get("name", ""))


def poses_calib_tool_frame(poses_data: dict, *, default: str = "Arm_Tip") -> str:
    return str(poses_data.get("pose_tool_frame", default))


def slot_tcp_pose(
    robot,
    q_deg: np.ndarray,
    pose_stored: np.ndarray,
    *,
    calib_tool: str,
) -> np.ndarray:
    """
    TCP pose in base frame for the **active** tool at slot ``q_deg``.

    ``poses.yaml`` ``pose_base`` is recorded with ``calib_tool`` active (e.g. Arm_Tip).
    When the active tool differs (e.g. gripper), use ``rm_algo_forward_kinematics``.
    """
    q_deg = np.asarray(q_deg, dtype=float)
    pose_stored = np.asarray(pose_stored, dtype=float)
    active = get_active_tool_name(robot)
    if active and calib_tool and active != calib_tool:
        fk = robot.rm_algo_forward_kinematics(q_deg.tolist(), flag=1)
        return np.asarray(fk[:6], dtype=float)
    return pose_stored.copy()


def tool_frame_delta_pose(
    robot,
    pose_ref: np.ndarray,
    dx: float,
    dy: float,
    dz: float,
) -> np.ndarray:
    """Apply a translation delta in the tool frame of ``pose_ref`` (Realman frameMode=1)."""
    delta = [float(dx), float(dy), float(dz), 0.0, 0.0, 0.0]
    out = robot.rm_algo_pose_move(list(np.asarray(pose_ref, dtype=float)), delta, frameMode=1)
    return np.asarray(out[:6], dtype=float)


def slot_scan_approach_pose_kin(
    kin: "RobotKinematics",
    pose_arm_tip_contact: np.ndarray,
    q_deg: np.ndarray,
    *,
    approach_dz_m: float = DEFAULT_SCAN_APPROACH_DZ_M,
    euler_order: str = DEFAULT_EULER_ORDER,
    rail_m: float = 0.0,
) -> np.ndarray:
    """Scan standoff D in Pinocchio ``tcp`` frame (matches inner WBC IK).

    ``pose_arm_tip_contact`` is Arm_Tip at tissue contact (a base-frame point,
    tool-agnostic); +dz is applied along the **Pinocchio URDF ``tcp`` frame's**
    +Z at ``q_deg``, deliberately NOT the physically-active Realman tool frame
    - this keeps the standoff direction defined in EXACTLY the frame the WBC
    inner loop (``CartesianTrackOuterLoop``/``solve_pose_ik``) tracks, so the
    executed motion matches the planned one.

    IMPORTANT ASSUMPTION this relies on: the URDF's ``tcp`` frame geometrically
    represents whatever tool is *physically mounted and active* right now (this
    module was built around ``active == "gripper"``). If a different tool is
    active, the Z direction (and thus the whole WBC kinematic chain, not just
    this one offset) silently stops matching the physical robot - use
    ``pose_kin_vs_active_drift_mm`` to catch that BEFORE trusting the result,
    do not rely on eyeballing the printed pose.
    """
    from rm75_control.control.joint_admittance.model import deg2rad

    del euler_order  # kin carries euler_order; FK orientation is from model
    contact = np.asarray(pose_arm_tip_contact, dtype=float)
    q_deg_arr = np.asarray(q_deg, dtype=float)
    nq = int(getattr(getattr(kin, "model", None), "nq", q_deg_arr.size))
    if nq == 8 and q_deg_arr.size == 7:
        from rm75_control.control.joint_admittance_8dof.model import (
            deg2rad as deg2rad8,
            full_q_from_arm,
        )

        q_rad = full_q_from_arm(deg2rad8(q_deg_arr), float(rail_m))
    else:
        q_rad = deg2rad(q_deg_arr[:nq] if q_deg_arr.size > nq else q_deg_arr)
    fk = kin.fk_pose(q_rad)
    R = kin.fk_placement(q_rad).rotation
    out = fk.copy()
    out[:3] = contact[:3] + R @ np.array([0.0, 0.0, float(approach_dz_m)])
    return out


def pose_kin_vs_active_drift_mm(
    robot,
    pose_kin: np.ndarray,
    pose_arm_tip_contact: np.ndarray,
    q_deg: np.ndarray,
    *,
    approach_dz_m: float = DEFAULT_SCAN_APPROACH_DZ_M,
    calib_tool: str = "Arm_Tip",
    euler_order: str = DEFAULT_EULER_ORDER,
) -> float:
    """Position drift (mm) between ``slot_scan_approach_pose_kin`` and the
    firmware-computed standoff for the tool that is ACTUALLY active right now.

    Only meaningful (and only called) when ``active != calib_tool`` - when they
    match, both paths degenerate to the same contact-frame math and drift is
    ~0 by construction.  A large drift means the URDF ``tcp`` frame's Z axis at
    ``q_deg`` does not represent the currently-active physical tool - the
    "位置用 Arm_Tip、姿态用 Pin tcp" cross-frame assumption
    ``slot_scan_approach_pose_kin`` depends on is violated, and ``pose_kin``
    should NOT be trusted as an IK target (see ``resolve_scan_pose_d`` in
    ``apps/joint_admittance/d_sin_tool_y.py`` for the hard-abort threshold this
    feeds).
    """
    pose_rm = slot_scan_approach_pose(
        robot,
        pose_arm_tip_contact,
        approach_dz_m=approach_dz_m,
        q_deg=q_deg,
        calib_tool=calib_tool,
        euler_order=euler_order,
    )
    return float(np.linalg.norm(np.asarray(pose_kin, dtype=float)[:3] - pose_rm[:3]) * 1000.0)


def slot_scan_approach_pose(
    robot,
    pose_arm_tip_contact: np.ndarray,
    *,
    approach_dz_m: float = DEFAULT_SCAN_APPROACH_DZ_M,
    q_deg: np.ndarray | None = None,
    calib_tool: str = "Arm_Tip",
    euler_order: str = DEFAULT_EULER_ORDER,
) -> np.ndarray:
    """
    Scan standoff pose D from a force-ID slot teach pose.

    ``pose_arm_tip_contact`` is ``poses.yaml`` ``pose_base`` with ``calib_tool``
    (Arm_Tip) at the tissue contact point.

    When the active tool is **gripper** (or any tool != calib_tool):
      contact stays at the Arm_Tip-teach base position, but +dz is applied along
      the **active** tool Z at ``q_deg`` (gripper FK orientation), not Arm_Tip axes.

    When active == calib_tool: legacy path via ``rm_algo_pose_move`` on teach pose.
    """
    contact = np.asarray(pose_arm_tip_contact, dtype=float)
    active = get_active_tool_name(robot)
    if q_deg is None or not active or active == calib_tool:
        return tool_frame_delta_pose(robot, contact, 0.0, 0.0, approach_dz_m)

    q_deg = np.asarray(q_deg, dtype=float)
    fk_active = np.asarray(robot.rm_algo_forward_kinematics(q_deg.tolist(), flag=1)[:6])
    R = Rsc.from_euler(euler_order, fk_active[3:6], degrees=False).as_matrix()
    out = fk_active.copy()
    out[:3] = contact[:3] + R @ np.array([0.0, 0.0, float(approach_dz_m)])
    return out
