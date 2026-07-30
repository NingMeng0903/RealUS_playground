"""Shared sin-tool-Y program builder and executor (window A and C)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml
from scipy.spatial.transform import Rotation as Rsc

from rm75_control.control.admittance_common.controller import AdmittanceController
from rm75_control.control.admittance_common.phase_ipc import SinToolYTaskParams
from rm75_control.control.joint_admittance_8dof.api import (
    ArmAngleSpec,
    CompileContext,
    GovernorSpec,
    SecondaryPolicy,
    compile_phases,
    phase_hold_at_pose,
    phase_hybrid_track,
    phase_rail_reposition,
    scale_admittance_for_desired_z,
)
from rm75_control.control.joint_admittance_8dof.config import build_joint_ik_config
from rm75_control.control.joint_admittance_8dof.loop import (
    JointIkController,
    LoopResult,
    run_joint_admittance_phases,
)
from rm75_control.control.joint_admittance_8dof.model import (
    RobotKinematics,
    deg2rad,
    full_q_from_arm,
    wrap_joint_delta,
)
from rm75_control.control.joint_admittance_8dof.reference import (
    HoldReference,
    SinToolYReference,
)
from rm75_control.control.joint_admittance_8dof.wbc_arm import WbcArm
from rm75_control.control.joint_admittance_8dof.pose_ik import solve_pose_ik
from rm75_control.control.joint_admittance_8dof.tasks.arm_angle import _wrap_pi
from rm75_control.force.compensation import excitation as ex
from rm75_control.force.compensation.id_config import load_config as load_force_id_config
from rm75_control.force.compensation.paths import CONFIG_ID
from rm75_control.force.compensation.tool_pose import (
    DEFAULT_SCAN_APPROACH_DZ_M,
    get_active_tool_name,
    maybe_sync_kin_tcp_from_config,
    poses_calib_tool_frame,
    slot_scan_approach_pose_kin,
)

MAX_POSE_KIN_DRIFT_MM = 25.0


@dataclass
class ScanTargetD:
    """Planned move->D target independent of RealMan published TCP (optional modes)."""

    q_slot_deg: np.ndarray
    pose_d: np.ndarray
    pose_id: np.ndarray
    q_target_rad: np.ndarray
    d_target: str


def load_slot_joints_only(slot: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load taught ``q_deg`` / ``pose_base`` from poses.yaml without RealMan FK."""
    fid = load_force_id_config(CONFIG_ID)
    data = ex.load_poses_yaml(fid.poses_yaml)
    rec = ex.get_slot_record(data, slot)
    if rec is None:
        raise RuntimeError(f"Pose slot {slot!r} missing in {fid.poses_yaml}")
    q_deg = np.asarray(rec["q_deg"], dtype=float)
    pose_id = np.asarray(rec["pose_base"], dtype=float)
    return q_deg, pose_id, rec


def resolve_scan_target_at_d(
    slot: str,
    kin: RobotKinematics,
    *,
    d_target: str = "legacy",
    approach_dz_m: float = DEFAULT_SCAN_APPROACH_DZ_M,
    use_force_id_pose: bool = False,
    euler_order: str = "xyz",
    rail_m: float = 0.0,
    robot=None,
    qp_cfg=None,
    nullspace_cfg=None,
) -> ScanTargetD:
    """Resolve scan pose D and joint target for the move->D phase.

    ``d_target`` modes (all produce a Cartesian ``pose_d``; move execution is
    still ``--move-mode``, default cartesian/SRS):

    * ``legacy`` — RealMan active-tool FK + Pin standoff + pose IK (original).
    * ``joints`` — taught ``q_deg`` with j7+90° (ArmTip +X → TCP +Z), then fold
      approach into a world-vertical plane, IK, Cartesian SRS.
    * ``kin-fk`` — Pinocchio standoff ``pose_d`` from taught contact frame;
      optional pose IK; never uses ``rm_algo_forward_kinematics``.
    """
    mode = str(d_target).strip().lower()
    if mode == "legacy":
        return _resolve_scan_target_legacy(
            slot,
            kin,
            approach_dz_m=approach_dz_m,
            use_force_id_pose=use_force_id_pose,
            euler_order=euler_order,
            rail_m=rail_m,
            robot=robot,
            qp_cfg=qp_cfg,
            nullspace_cfg=nullspace_cfg,
        )
    if mode == "joints":
        travel = 0.80
        try:
            travel = float(kin.q_upper[0])
        except Exception:
            pass
        return _resolve_scan_target_joints(
            slot,
            kin,
            rail_m=rail_m,
            travel_m=travel,
            qp_cfg=qp_cfg,
            nullspace_cfg=nullspace_cfg,
            euler_order=euler_order,
        )
    if mode in {"kin-fk", "kin_fk", "kinfk"}:
        return _resolve_scan_target_kin_fk(
            slot,
            kin,
            approach_dz_m=approach_dz_m,
            use_force_id_pose=use_force_id_pose,
            euler_order=euler_order,
            rail_m=rail_m,
            qp_cfg=qp_cfg,
            nullspace_cfg=nullspace_cfg,
        )
    raise ValueError(f"unknown d_target mode {d_target!r}; use legacy, joints, or kin-fk")


def _pick_wellconditioned_rail_m(
    kin: RobotKinematics,
    q_arm_rad: np.ndarray,
    *,
    travel_m: float,
    prefer_m: float | None = None,
    n_samples: int = 21,
) -> tuple[float, float]:
    """Pick rail_y that maximizes σ_min for a fixed taught arm posture.

    ``prefer_m`` (e.g. mid-stroke from ``--rail-scan-center``) breaks ties and
    softly biases toward the caller's prior when σ is nearly flat.
    Returns ``(y_rail_m, sigma_min)``.
    """
    travel = max(float(travel_m), 1e-3)
    prefer = float(prefer_m) if prefer_m is not None else 0.5 * travel
    prefer = float(np.clip(prefer, 0.0, travel))
    best_y = prefer
    best_sig = -1.0
    best_score = -1e9
    q_arm = np.asarray(q_arm_rad, dtype=float).reshape(-1)
    for i in range(max(3, int(n_samples))):
        y = travel * i / (n_samples - 1)
        q = full_q_from_arm(q_arm, float(y))
        try:
            sig = float(kin.singular_values(kin.jacobian(q)).min())
        except Exception:
            continue
        # Soft prefer prior: 2 cm of rail ≈ 0.01 of σ_min (tie-break only).
        score = sig - 0.5 * abs(y - prefer)
        if score > best_score:
            best_score = score
            best_sig = sig
            best_y = y
    return float(best_y), float(best_sig)


def _remap_taught_q_armtip_x_to_tcp_z(q_arm_rad: np.ndarray) -> np.ndarray:
    """Map ArmTip-+X approach teach onto probe TCP-+Z (= ArmTip -Y).

    Slot ``d`` was taught with ArmTip +X oblique-down in the symmetry plane.
    Probe URDF TCP has +Z = ArmTip -Y, so the same joint vector leaves the tip
    sideways.  Adding +π/2 on wrist joint 7 is ``R ← R·Rz(+π/2)`` and makes
    ArmTip -Y (and TCP +Z) inherit the old +X world direction.
    """
    q = np.asarray(q_arm_rad, dtype=float).reshape(-1).copy()
    if q.size < 7:
        raise ValueError(f"expected 7 arm joints, got {q.size}")
    q[6] = float(q[6] + 0.5 * np.pi)
    # Keep a principal value so SRS / limit checks stay sane.
    q[6] = float(np.arctan2(np.sin(q[6]), np.cos(q[6])))
    return q


def _fold_flange_into_world_vertical_plane(R_l7: np.ndarray) -> tuple[np.ndarray, float]:
    """Fold link_7 so TCP+Z (= -Y) and flange +Z lie in a world-vertical plane.

    Taught D's ArmTip +X already had ~16° of world-Y lean; j7+90° kept that lean
    on TCP+Z.  Project approach into a constant-Y vertical plane (normal = ê_y),
    rebuild a right-handed flange frame with +Z also in that plane.
    Returns ``(R_l7_new, approach_fold_deg)``.
    """
    R = np.asarray(R_l7, dtype=float).reshape(3, 3)
    ey = np.array([0.0, 1.0, 0.0])
    # TCP +Z = ArmTip -Y
    approach = -R[:, 1]
    n = float(np.linalg.norm(approach))
    if n < 1e-9:
        return R.copy(), 0.0
    approach = approach / n
    a_proj = approach - (approach @ ey) * ey
    na = float(np.linalg.norm(a_proj))
    if na < 1e-9:
        return R.copy(), 0.0
    a_proj = a_proj / na
    fold_deg = float(np.degrees(np.arccos(np.clip(approach @ a_proj, -1.0, 1.0))))

    y_axis = -a_proj  # ArmTip +Y after fold
    # Flange +Z in the same vertical plane, ⊥ Y; pick the branch near the old Z.
    z_axis = np.cross(ey, y_axis)
    nz = float(np.linalg.norm(z_axis))
    if nz < 1e-9:
        return R.copy(), fold_deg
    z_axis = z_axis / nz
    if float(z_axis @ R[:, 2]) < 0.0:
        z_axis = -z_axis
    x_axis = np.cross(y_axis, z_axis)
    x_axis = x_axis / max(float(np.linalg.norm(x_axis)), 1e-12)
    # Re-orthogonalize Z in case of drift.
    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / max(float(np.linalg.norm(z_axis)), 1e-12)
    R_new = np.column_stack((x_axis, y_axis, z_axis))
    return R_new, fold_deg


def _tcp_pose_from_link7(
    kin: RobotKinematics,
    p_l7: np.ndarray,
    R_l7: np.ndarray,
    *,
    euler_order: str = "xyz",
) -> np.ndarray:
    """Compose world TCP pose from link_7 pose and URDF link_7→tcp offset."""
    R_off = np.asarray(kin._R_link7_tcp, dtype=float).reshape(3, 3)
    t_off = np.asarray(kin._r_link7_tcp, dtype=float).reshape(3)
    R_tcp = R_l7 @ R_off
    p_tcp = np.asarray(p_l7, dtype=float).reshape(3) + R_l7 @ t_off
    pose = np.zeros(6, dtype=float)
    pose[:3] = p_tcp
    pose[3:6] = Rsc.from_matrix(R_tcp).as_euler(euler_order, degrees=False)
    return pose


def _resolve_scan_target_joints(
    slot: str,
    kin: RobotKinematics,
    *,
    rail_m: float = 0.0,
    travel_m: float = 0.80,
    refine_rail: bool = True,
    qp_cfg=None,
    nullspace_cfg=None,
    euler_order: str = "xyz",
) -> ScanTargetD:
    q_deg_taught, pose_id, _rec = load_slot_joints_only(slot)
    q_arm = _remap_taught_q_armtip_x_to_tcp_z(deg2rad(q_deg_taught))
    y_rail = float(rail_m)
    sig_note = ""
    if refine_rail:
        y_rail, sig = _pick_wellconditioned_rail_m(
            kin,
            q_arm,
            travel_m=float(travel_m),
            prefer_m=float(rail_m),
        )
        sig_note = f" rail→{y_rail * 1000:.0f}mm (σ_min={sig:.3f}, prefer={float(rail_m) * 1000:.0f}mm)"
    q_seed = full_q_from_arm(q_arm, y_rail)

    Ml7 = kin.frame_placement(q_seed, "link_7")
    R_fold, fold_deg = _fold_flange_into_world_vertical_plane(Ml7.rotation)
    pose_d = _tcp_pose_from_link7(
        kin, Ml7.translation, R_fold, euler_order=euler_order
    )

    q_target_rad = q_seed
    ik_note = ""
    if qp_cfg is not None:
        q_target_rad, _ok, rep = solve_pose_ik(
            kin,
            q_seed,
            pose_d,
            qp_cfg=qp_cfg,
            nullspace_cfg=nullspace_cfg,
            attractor_q=q_seed,
        )
        # Keep Cartesian target as FK of the solved q (consistent with build()).
        pose_d = np.asarray(kin.fk_pose(q_target_rad), dtype=float)
        ik_note = (
            f" IK pos={rep.pos_err_mm:.2f}mm rot={rep.rot_err_deg:.2f}deg"
        )
    else:
        # No QP: still publish the folded Cartesian; SRS will pull toward it.
        pass

    q_deg = np.rad2deg(q_target_rad[1:])
    off = np.asarray(kin.tcp_offset_pose, dtype=float).reshape(6)
    print(
        f"D target=joints→Cartesian pose_d slot={slot} "
        f"taught_q_deg={np.round(q_deg_taught, 2).tolist()} "
        f"→ j7+90° + foldΔ={fold_deg:.1f}° into world-vertical plane "
        f"q_deg={np.round(q_deg, 2).tolist()} "
        f"xyz(mm)={np.round(pose_d[:3] * 1000.0, 1).tolist()} "
        f"rpy(deg)={np.round(np.degrees(pose_d[3:6]), 1).tolist()} "
        f"| tool_offset xyz(mm)={np.round(off[:3] * 1000.0, 1).tolist()} "
        f"rpy(deg)={np.round(np.degrees(off[3:6]), 1).tolist()} "
        f"(Cartesian/SRS){sig_note}{ik_note}",
        flush=True,
    )
    return ScanTargetD(
        q_slot_deg=q_deg,
        pose_d=pose_d,
        pose_id=pose_id,
        q_target_rad=q_target_rad,
        d_target="joints",
    )


def _resolve_scan_target_kin_fk(
    slot: str,
    kin: RobotKinematics,
    *,
    approach_dz_m: float,
    use_force_id_pose: bool,
    euler_order: str,
    rail_m: float,
    qp_cfg,
    nullspace_cfg,
) -> ScanTargetD:
    q_deg, pose_id, _rec = load_slot_joints_only(slot)
    q_slot_rad = full_q_from_arm(deg2rad(q_deg), float(rail_m))
    if use_force_id_pose:
        pose_d = np.asarray(kin.fk_pose(q_slot_rad), dtype=float)
    else:
        pose_d = slot_scan_approach_pose_kin(
            kin,
            pose_id,
            q_deg,
            approach_dz_m=approach_dz_m,
            euler_order=euler_order,
            rail_m=rail_m,
        )
    q_target_rad, _ok, rep = solve_pose_ik(
        kin,
        q_slot_rad,
        pose_d,
        qp_cfg=qp_cfg,
        nullspace_cfg=nullspace_cfg,
        attractor_q=q_slot_rad,
    )
    if rep.pos_err_mm > 5.0 or rep.rot_err_deg > 2.0 or not rep.within_limits:
        raise RuntimeError(
            f"kin-fk pose IK did not converge: pos={rep.pos_err_mm:.2f}mm, "
            f"rot={rep.rot_err_deg:.2f}deg, within_limits={rep.within_limits}"
        )
    print(
        f"D target=kin-fk dz={approach_dz_m * 1000:.0f}mm pin_tcp z={pose_d[2]:.3f}m "
        "(RealMan TCP ignored)",
        flush=True,
    )
    return ScanTargetD(
        q_slot_deg=q_deg,
        pose_d=pose_d,
        pose_id=pose_id,
        q_target_rad=q_target_rad,
        d_target="kin-fk",
    )


def _resolve_scan_target_legacy(
    slot: str,
    kin: RobotKinematics,
    *,
    approach_dz_m: float,
    use_force_id_pose: bool,
    euler_order: str,
    rail_m: float,
    robot,
    qp_cfg,
    nullspace_cfg,
) -> ScanTargetD:
    from rm75_control.force.compensation.collection import load_slot
    from rm75_control.force.compensation.tool_pose import pose_kin_vs_active_drift_mm

    fid = load_force_id_config(CONFIG_ID)
    poses_data = ex.load_poses_yaml(fid.poses_yaml)
    calib_tool = poses_calib_tool_frame(poses_data)
    active = get_active_tool_name(robot) if robot is not None else ""

    q_deg, fk_pose, rec = load_slot(fid, slot, robot, calib_tool=calib_tool)
    pose_id = np.asarray(rec["pose_base"], dtype=float)

    if use_force_id_pose:
        pose_d = fk_pose.copy()
    else:
        pose_d = slot_scan_approach_pose_kin(
            kin,
            pose_id,
            q_deg,
            approach_dz_m=approach_dz_m,
            euler_order=euler_order,
            rail_m=rail_m,
        )
        if robot is not None and active and calib_tool and active != calib_tool:
            d_mm = pose_kin_vs_active_drift_mm(
                robot,
                pose_d,
                pose_id,
                q_deg,
                approach_dz_m=approach_dz_m,
                calib_tool=calib_tool,
                euler_order=euler_order,
            )
            if d_mm > MAX_POSE_KIN_DRIFT_MM:
                raise RuntimeError(
                    f"pose D Pinocchio-tcp vs Realman {active!r} drift {d_mm:.1f}mm > "
                    f"{MAX_POSE_KIN_DRIFT_MM:.0f}mm safety bound"
                )
            if d_mm > 5.0:
                print(
                    f"warn: D pose Pinocchio vs Realman {active!r} {d_mm:.1f}mm "
                    "(loop tracks Pinocchio tcp)",
                    flush=True,
                )
    tool_note = f"tool={active!r}" if active else "tool=Pin-tcp"
    if active and calib_tool and active != calib_tool:
        tool_note += " (contact Arm_Tip teach, +dz Pin tcp @ q)"
    print(f"D target=legacy dz={approach_dz_m * 1000:.0f}mm {tool_note} z={pose_d[2]:.3f}", flush=True)

    q_slot_rad = full_q_from_arm(deg2rad(q_deg), float(rail_m))
    q_target_rad, _ok, rep = solve_pose_ik(
        kin,
        q_slot_rad,
        pose_d,
        qp_cfg=qp_cfg,
        nullspace_cfg=nullspace_cfg,
        attractor_q=q_slot_rad,
    )
    if rep.pos_err_mm > 5.0 or rep.rot_err_deg > 2.0 or not rep.within_limits:
        raise RuntimeError(
            f"pose IK did not converge: pos={rep.pos_err_mm:.2f}mm, "
            f"rot={rep.rot_err_deg:.2f}deg, within_limits={rep.within_limits}"
        )
    return ScanTargetD(
        q_slot_deg=q_deg,
        pose_d=pose_d,
        pose_id=pose_id,
        q_target_rad=q_target_rad,
        d_target="legacy",
    )


def load_yaml(path: Path | str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_psi_sides(
    psi_center: float,
    *,
    side_offset_rad: float = np.deg2rad(90.5),
    psi_left_rad: float | None = None,
    psi_right_rad: float | None = None,
) -> tuple[float, float, float]:
    """Center swivel at pose D; left/right = center ± offset (same branch)."""
    center = float(_wrap_pi(psi_center))
    if psi_left_rad is not None and psi_right_rad is not None:
        return (
            center,
            float(_wrap_pi(psi_left_rad)),
            float(_wrap_pi(psi_right_rad)),
        )
    off = abs(float(side_offset_rad))
    return center, float(_wrap_pi(center + off)), float(_wrap_pi(center - off))


def resolve_psi_sides_live(
    psi_center: float,
    psi_live: float,
    *,
    fallback_offset_rad: float = np.deg2rad(90.5),
    min_offset_rad: float = np.deg2rad(10.0),
) -> tuple[float, float, float]:
    """Center @ D; left = live Realman psi; right mirrored: center - (left - center)."""
    center = float(_wrap_pi(psi_center))
    left = float(_wrap_pi(psi_live))
    delta = _wrap_pi(left - center)
    if abs(delta) < min_offset_rad:
        return resolve_psi_sides(center, side_offset_rad=fallback_offset_rad)
    right = float(_wrap_pi(center - delta))
    return center, left, right


def plan_q_toggle_at_pose(
    kin: RobotKinematics,
    pose_d: np.ndarray,
    q_center_rad: np.ndarray,
    q_live_rad: np.ndarray,
    *,
    qp_cfg,
    nullspace_cfg,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """IK @ pose D: center, left (seed=live teach), right (mirror joint delta)."""
    q_center = np.asarray(q_center_rad, dtype=float).reshape(-1)
    pose_d = np.asarray(pose_d, dtype=float).reshape(6)
    q_left, ok_l, rep_l = solve_pose_ik(
        kin, q_live_rad, pose_d, qp_cfg=qp_cfg, nullspace_cfg=nullspace_cfg
    )
    if not ok_l or rep_l.pos_err_mm > 5.0:
        return q_center, q_center, q_center
    delta = q_left - q_center
    q_right, ok_r, rep_r = solve_pose_ik(
        kin, q_center - delta, pose_d, qp_cfg=qp_cfg, nullspace_cfg=nullspace_cfg
    )
    if not ok_r or rep_r.pos_err_mm > 5.0:
        q_right = q_center - delta
    return q_center, np.asarray(q_left, dtype=float).reshape(-1), np.asarray(q_right, dtype=float).reshape(-1)


def plan_psi_sides_ik_at_pose(
    kin: RobotKinematics,
    inner: JointIkController,
    pose_d: np.ndarray,
    q_center_rad: np.ndarray,
    q_live_rad: np.ndarray,
    *,
    qp_cfg,
    nullspace_cfg,
    side_offset_rad: float = np.deg2rad(90.5),
) -> tuple[float, float, float]:
    """ψ labels for logging (from IK q targets at fixed TCP)."""
    q_c, q_l, q_r = plan_q_toggle_at_pose(
        kin, pose_d, q_center_rad, q_live_rad, qp_cfg=qp_cfg, nullspace_cfg=nullspace_cfg
    )
    if inner.arm_task is None:
        return resolve_psi_sides(0.0, side_offset_rad=side_offset_rad)
    psi_c = float(inner.arm_task.arm_angle(q_c))
    if np.max(np.abs(q_l - q_c)) < 1e-6:
        return resolve_psi_sides(psi_c, side_offset_rad=side_offset_rad)
    psi_l = float(inner.arm_task.arm_angle(q_l))
    psi_r = float(inner.arm_task.arm_angle(q_r))
    return psi_c, psi_l, psi_r


def plan_psi_toggle_sides(
    inner: JointIkController,
    q_live_rad: np.ndarray,
    psi_center: float,
    *,
    side_offset_rad: float = np.deg2rad(90.5),
    psi_left_rad: float | None = None,
    psi_right_rad: float | None = None,
    psi_live_left: bool = True,
    kin: RobotKinematics | None = None,
    pose_d: np.ndarray | None = None,
    q_center_rad: np.ndarray | None = None,
    qp_cfg=None,
    nullspace_cfg=None,
) -> tuple[float, float, float]:
    """Plan center/left/right ψ for hybrid @ D (IK-feasible at fixed TCP when possible)."""
    if psi_left_rad is not None and psi_right_rad is not None:
        return resolve_psi_sides(
            psi_center,
            psi_left_rad=psi_left_rad,
            psi_right_rad=psi_right_rad,
        )
    if (
        psi_live_left
        and kin is not None
        and pose_d is not None
        and q_center_rad is not None
        and qp_cfg is not None
        and nullspace_cfg is not None
        and inner.arm_task is not None
    ):
        return plan_psi_sides_ik_at_pose(
            kin,
            inner,
            pose_d,
            q_center_rad,
            q_live_rad,
            qp_cfg=qp_cfg,
            nullspace_cfg=nullspace_cfg,
            side_offset_rad=side_offset_rad,
        )
    if psi_live_left and inner.arm_task is not None:
        psi_live = float(inner.arm_task.arm_angle(q_live_rad))
        return resolve_psi_sides_live(
            psi_center,
            psi_live,
            fallback_offset_rad=side_offset_rad,
        )
    return resolve_psi_sides(psi_center, side_offset_rad=side_offset_rad)


def attach_hybrid_posture_toggle(
    phases: list,
    inner: JointIkController,
    *,
    q_center: np.ndarray,
    q_left: np.ndarray,
    q_right: np.ndarray,
    period_s: float,
    verbose: bool = True,
    filter_alpha: float = 0.02,
    ramp_duration_s: float = 4.0,
    k_center_scale: float = 2.5,
    max_qdot_frac: float = 0.35,
) -> None:
    """Ramp joint centering targets (same TCP) — visible multi-DOF posture change."""
    if period_s <= 0.0:
        return
    q_center = np.asarray(q_center, dtype=float).reshape(-1)
    q_left = np.asarray(q_left, dtype=float).reshape(-1)
    q_right = np.asarray(q_right, dtype=float).reshape(-1)

    inner.set_arm_task_suppressed(True)
    k_saved = float(inner.centering_task.cfg.k_center)
    inner.centering_task.cfg.k_center = k_saved * float(k_center_scale)
    frac_saved = float(inner.secondary.max_qdot_frac)
    inner.secondary.max_qdot_frac = float(max_qdot_frac)
    inner.centering_task.q_target = q_center.copy()

    ramp_s = max(0.5, min(float(ramp_duration_s), float(period_s) * 0.85))
    toggle_state = {
        "last_bucket": -1,
        "current_q": q_center.copy(),
        "ramp_from": q_center.copy(),
        "ramp_to": q_center.copy(),
        "ramp_t0": 0.0,
    }

    def _q_for_bucket(bucket: int) -> tuple[np.ndarray, str]:
        if bucket == 0:
            return q_center, "center"
        if bucket % 2 == 1:
            return q_left, "left"
        return q_right, "right"

    def on_tick(t_ref: float, _step, _q_meas: np.ndarray) -> None:
        bucket = int(t_ref / period_s)
        if bucket != toggle_state["last_bucket"]:
            toggle_state["last_bucket"] = bucket
            target, tag = _q_for_bucket(bucket)
            toggle_state["ramp_from"] = toggle_state["current_q"].copy()
            toggle_state["ramp_to"] = target.copy()
            toggle_state["ramp_t0"] = t_ref
            if verbose:
                print(f"  posture ramp start @ {t_ref:.1f}s -> {tag} over {ramp_s:.1f}s", flush=True)

        dt_ramp = t_ref - toggle_state["ramp_t0"]
        u = float(np.clip(dt_ramp / ramp_s, 0.0, 1.0))
        u2, u3, u4, u5 = u * u, u * u * u, u * u * u * u, u * u * u * u * u
        s = 10.0 * u3 - 15.0 * u4 + 6.0 * u5
        delta = wrap_joint_delta(toggle_state["ramp_from"], toggle_state["ramp_to"])
        target_q = toggle_state["ramp_from"] + s * delta

        diff = wrap_joint_delta(toggle_state["current_q"], target_q)
        toggle_state["current_q"] = toggle_state["current_q"] + filter_alpha * diff
        toggle_state["current_q"][0] = q_center[0]
        inner.centering_task.q_target = toggle_state["current_q"].copy()

    hybrid_labels = ("scan", "hybrid@D")
    for phase in phases:
        if phase.label in hybrid_labels:
            phase.on_tick = on_tick
            return
    raise RuntimeError(f"attach_hybrid_posture_toggle: no phase in {hybrid_labels}")


def attach_scan_psi_toggle(
    phases: list,
    inner: JointIkController,
    *,
    psi_center: float,
    psi_left: float,
    psi_right: float,
    period_s: float,
    verbose: bool = True,
    filter_alpha: float = 0.01,
    ramp_duration_s: float = 4.0,
    k_psi_scale: float = 0.35,
) -> None:
    """Hybrid phase: hold center, then quintic-ramp left / right arm-angle targets."""
    if period_s <= 0.0:
        return
    if inner.arm_task is None:
        raise RuntimeError("psi toggle requires arm_angle secondary task")

    k_psi_saved = float(inner.arm_task.cfg.k_psi)
    inner.arm_task.cfg.k_psi = k_psi_saved * float(k_psi_scale)

    ramp_s = max(0.5, min(float(ramp_duration_s), float(period_s) * 0.85))
    toggle_state = {
        "last_bucket": -1,
        "current_psi": psi_center,
        "ramp_from": psi_center,
        "ramp_to": psi_center,
        "ramp_t0": 0.0,
    }

    def _target_for_bucket(bucket: int) -> tuple[float, str]:
        if bucket == 0:
            return psi_center, "center"
        if bucket % 2 == 1:
            return psi_left, "left"
        return psi_right, "right"

    def on_tick(t_ref: float, _step, _q_meas: np.ndarray) -> None:
        bucket = int(t_ref / period_s)
        if bucket != toggle_state["last_bucket"]:
            toggle_state["last_bucket"] = bucket
            target, tag = _target_for_bucket(bucket)
            toggle_state["ramp_from"] = toggle_state["current_psi"]
            toggle_state["ramp_to"] = target
            toggle_state["ramp_t0"] = t_ref
            if verbose:
                print(
                    f"  psi ramp start @ {t_ref:.1f}s -> {np.degrees(target):+.1f}deg ({tag}) "
                    f"over {ramp_s:.1f}s",
                    flush=True,
                )

        dt_ramp = t_ref - toggle_state["ramp_t0"]
        u = float(np.clip(dt_ramp / ramp_s, 0.0, 1.0))
        u2, u3, u4, u5 = u * u, u * u * u, u * u * u * u, u * u * u * u * u
        s = 10.0 * u3 - 15.0 * u4 + 6.0 * u5
        delta = _wrap_pi(toggle_state["ramp_to"] - toggle_state["ramp_from"])
        target = _wrap_pi(toggle_state["ramp_from"] + s * delta)

        current = toggle_state["current_psi"]
        diff = _wrap_pi(target - current)
        toggle_state["current_psi"] = _wrap_pi(current + filter_alpha * diff)
        inner.arm_task.set_reference(toggle_state["current_psi"])

    hybrid_labels = ("scan", "hybrid@D")
    for phase in phases:
        if phase.label in hybrid_labels:
            phase.on_tick = on_tick
            return
    raise RuntimeError(
        f"attach_scan_psi_toggle: no phase in {hybrid_labels}"
    )


@dataclass
class BuiltSinToolYProgram:
    phases: list
    compiled: list
    inner: JointIkController
    kin: RobotKinematics
    force_observer: Any


def build_sin_tool_y_program(
    params: SinToolYTaskParams,
    *,
    raw: dict | None = None,
) -> BuiltSinToolYProgram:
    """Build phase list from precomputed task params (same on C and A)."""
    raw = raw if raw is not None else load_yaml(params.config_path)
    kin = RobotKinematics()
    maybe_sync_kin_tcp_from_config(
        kin,
        raw,
        tcp_offset_pose=params.tcp_offset_pose if params.tcp_offset_pose else None,
    )
    inner_cfg = build_joint_ik_config(raw)
    inner = JointIkController(kin, inner_cfg)
    max_lin = (
        float(params.cartesian_max_lin_vel)
        if params.cartesian_max_lin_vel is not None
        else 0.4
    )
    rail_m = float(inner_cfg.rail.q_ref_m)

    q_target_rad = np.asarray(params.q_target_rad, dtype=float).reshape(-1)
    q0_rad = np.asarray(params.q0_rad, dtype=float).reshape(-1)
    # Wait/SRS target must be FK(q_target) after TCP sync — raw params.pose_d can
    # still carry an ArmTip/IK residual orientation that blocks arrival forever
    # while track_err_mm (position-only) looks fine.
    pose_d = np.asarray(kin.fk_pose(q_target_rad), dtype=float).reshape(6)
    pose_in = np.asarray(params.pose_d, dtype=float).reshape(6)
    dpos = float(np.linalg.norm(pose_d[:3] - pose_in[:3]))
    if dpos > 0.005:
        print(
            f"  build: pose_d←FK(q_target) (|Δpos| vs task pose={dpos * 1000:.1f} mm; "
            f"TCP z={float(kin.tcp_offset_pose[2]) * 1000:.1f} mm)",
            flush=True,
        )
    move_mode = str(params.plan_move_mode)
    if move_mode == "joint":
        move_phase = WbcArm.make_movej_phase(
            kin,
            q0_rad,
            q_target_rad,
            duration_s=float(params.plan_duration_s),
            label=f"movej->{params.slot}",
            move_kp=float(params.move_kp),
            gov_joint_max_deg=float(params.plan_gov_joint_max_deg),
            force_observer=None,
        )
    else:
        move_phase = WbcArm.make_movel_phase(
            kin,
            q0_rad,
            pose_d,
            q_target_rad,
            duration_s=float(params.plan_duration_s),
            label=f"movel->{params.slot}",
            move_kp=float(params.move_kp),
            max_lin_vel_m_s=max_lin,
            gov_joint_max_deg=float(params.plan_gov_joint_max_deg),
            force_observer=None,
            euler_order=inner_cfg.euler_order,
        )

    force_observer = None
    if params.enable_force and params.scan_duration > 0.0:
        from rm75_control.control.admittance_common.observer import CompensatedForceObserver

        force_observer = CompensatedForceObserver.from_yaml(raw)
        move_phase.force_observer = force_observer

    ctx = CompileContext(
        kin=kin,
        inner=inner,
        euler_order=inner_cfg.euler_order,
        control_frame=inner_cfg.control_frame,
        v_scale=inner_cfg.v_scale,
    )

    specs = [move_phase]

    if params.hold_at_d_s > 0.0:
        specs.append(
            phase_hold_at_pose(
                params.hold_at_d_s,
                label="hold@D",
                force_observer=force_observer,
            )
        )

    if params.rail_move_cm > 0.0:
        sign = 1.0 if params.rail_move_dir == "+y" else -1.0
        rail0 = float(inner_cfg.rail.q_ref_m if inner_cfg.rail.q_ref_m is not None else 0.0)
        delta_m = sign * float(params.rail_move_cm) * 0.01
        rail_target = rail0 + delta_m
        lo, hi = 0.0, float(inner_cfg.rail.travel_m)
        if not (lo <= rail_target <= hi):
            raise RuntimeError(
                f"rail target {rail_target * 100:.1f}cm outside travel "
                f"[{lo * 100:.0f}, {hi * 100:.0f}]cm"
            )
        q_rail_start = full_q_from_arm(q_target_rad, rail_m=rail0)
        rail_style = str(params.rail_move_mode)
        specs.append(
            phase_rail_reposition(
                rail_target,
                q_rail_start,
                kin,
                label=f"rail{params.rail_move_dir}{params.rail_move_cm:.0f}cm_{rail_style}",
                style=rail_style,
                force_observer=force_observer,
                v_max_m_s=inner_cfg.rail.v_max_m_s,
            )
        )

    if params.scan_duration > 0.0:
        dt = float(raw.get("timing", {}).get("dt_ms", 5.0)) / 1000.0
        outer_ctrl = AdmittanceController(
            dt, scale_admittance_for_desired_z(raw, float(params.desired_z))
        )
        desired_force = np.zeros(6)
        desired_force[2] = float(params.desired_z)
        psi = None if params.psi_tgt is None or not np.isfinite(params.psi_tgt) else float(params.psi_tgt)
        if params.scan_hybrid_hold:
            hybrid_ref: HoldReference | SinToolYReference = HoldReference()
            hybrid_label = "hybrid@D"
            hybrid_sec = SecondaryPolicy(
                preset="hold",
                arm_angle=ArmAngleSpec(psi_rad=psi) if psi is not None else None,
                qdot_ff="off",
            )
            hybrid_gov = GovernorSpec(err_ok_mm=15.0, err_max_mm=80.0)
        else:
            amplitude_m = float(params.y_pp_cm) * 0.01 / 2.0
            max_vel_m_s = float(params.max_vel_cm_s) * 0.01
            hybrid_ref = SinToolYReference(
                amplitude_m,
                period_s=params.period_s,
                max_vel_m_s=None if params.period_s is not None else max_vel_m_s,
                soft_start=True,
                ramp_s=2.0,
                euler_order=inner_cfg.euler_order,
            )
            hybrid_label = "scan"
            # COUPLED: let the QP-IK freely distribute the tool-Y sweep between the
            # rail and the arm (rail slides, arm reaches out) — exactly the old
            # controller-driven-rail behaviour. The velocity-mode motor just follows
            # the resulting smooth q_cmd[0]; no rail pinning, no arm-only contortion.
            hybrid_sec = SecondaryPolicy(preset="track", qdot_ff="off")
            hybrid_gov = GovernorSpec(err_ok_mm=10.0, err_max_mm=40.0)
        specs.append(
            phase_hybrid_track(
                hybrid_ref,
                outer_ctrl,
                desired_force=desired_force,
                label=hybrid_label,
                duration_s=float(params.scan_duration),
                force_observer=force_observer,
                psi_rad_on_enter=psi,
                secondary=hybrid_sec,
                governor=hybrid_gov,
            )
        )

    compiled = compile_phases(specs, ctx)
    phases = [c.phase for c in compiled]
    if params.psi_toggle_period_s > 0.0 and params.scan_duration > 0.0:
        q_c = np.asarray(params.q_target_rad, dtype=float).reshape(-1)
        has_q = (
            len(params.q_toggle_left_rad) >= q_c.size
            and len(params.q_toggle_right_rad) >= q_c.size
        )
        if has_q:
            attach_hybrid_posture_toggle(
                phases,
                inner,
                q_center=q_c,
                q_left=np.asarray(params.q_toggle_left_rad, dtype=float).reshape(-1),
                q_right=np.asarray(params.q_toggle_right_rad, dtype=float).reshape(-1),
                period_s=float(params.psi_toggle_period_s),
                filter_alpha=float(params.psi_filter_alpha),
                ramp_duration_s=float(params.psi_ramp_s),
            )
        elif params.psi_tgt is not None and np.isfinite(params.psi_tgt):
            psi_center, psi_left, psi_right = resolve_psi_sides(
                float(params.psi_tgt),
                side_offset_rad=float(params.psi_side_offset_rad),
                psi_left_rad=params.psi_left_rad,
                psi_right_rad=params.psi_right_rad,
            )
            attach_scan_psi_toggle(
                phases,
                inner,
                psi_center=psi_center,
                psi_left=psi_left,
                psi_right=psi_right,
                period_s=float(params.psi_toggle_period_s),
                filter_alpha=float(params.psi_filter_alpha),
                ramp_duration_s=float(params.psi_ramp_s),
            )
        else:
            raise RuntimeError("psi toggle requires q_toggle_left/right or psi_tgt")
    return BuiltSinToolYProgram(
        phases=phases,
        compiled=compiled,
        inner=inner,
        kin=kin,
        force_observer=force_observer,
    )


def execute_sin_tool_y_program(
    session,
    state_bus,
    params: SinToolYTaskParams,
    *,
    raw: dict | None = None,
    built: BuiltSinToolYProgram | None = None,
    on_step: Callable | None = None,
    stop_check: Callable[[], bool] | None = None,
    verbose: bool = True,
    rail_bridge=None,
) -> LoopResult:
    """Run WBC on window A (direct UDP feedback + direct CANFD)."""
    raw = raw if raw is not None else load_yaml(params.config_path)
    startup = raw.get("startup", {})
    dt = float(raw.get("timing", {}).get("dt_ms", 5.0)) / 1000.0
    if built is None:
        built = build_sin_tool_y_program(params, raw=raw)

    return run_joint_admittance_phases(
        session,
        built.phases,
        built.inner,
        q_start_deg=None,
        dt=dt,
        follow=bool(startup.get("follow", True)),
        move_speed=int(startup.get("move_speed", 20)),
        realtime=bool(startup.get("realtime", False)),
        watchdog_timeout_s=float(startup.get("watchdog_timeout_s", 0.1)),
        on_step=on_step,
        log_csv=params.log_csv,
        state_bus=state_bus,
        canfd_proxy=None,
        stop_check=stop_check,
        verbose=verbose,
        rail_bridge=rail_bridge,
    )


def make_task_params_from_args(
    args,
    *,
    config_path: str,
    q0_rad: np.ndarray,
    q_target_rad: np.ndarray,
    pose_d: np.ndarray,
    plan,
    psi_tgt: float | None,
    desired_z: float,
    enable_force: bool,
    psi_left_rad: float | None = None,
    psi_right_rad: float | None = None,
    q_toggle_left_rad: np.ndarray | None = None,
    q_toggle_right_rad: np.ndarray | None = None,
    tcp_offset_pose: np.ndarray | None = None,
) -> SinToolYTaskParams:
    return SinToolYTaskParams(
        config_path=config_path,
        slot=str(args.slot),
        move_kp=float(args.move_kp),
        y_pp_cm=float(args.y_pp_cm),
        max_vel_cm_s=float(args.max_vel_cm_s),
        period_s=args.period_s,
        desired_z=float(desired_z),
        scan_duration=float(args.scan_duration),
        hold_at_d_s=float(args.hold_at_d_s),
        rail_move_cm=float(args.rail_move_cm),
        rail_move_mode=str(args.rail_move_mode),
        rail_move_dir=str(args.rail_move_dir),
        enable_force=bool(enable_force),
        log_csv=args.log_csv,
        rail_log_csv=getattr(args, "rail_log_csv", None),
        cartesian_max_lin_vel=args.cartesian_max_lin_vel,
        q0_rad=np.asarray(q0_rad, dtype=float).reshape(-1).tolist(),
        q_target_rad=np.asarray(q_target_rad, dtype=float).reshape(-1).tolist(),
        pose_d=np.asarray(pose_d, dtype=float).reshape(6).tolist(),
        plan_duration_s=float(plan.duration_s),
        plan_move_mode=str(plan.move_mode),
        plan_gov_joint_max_deg=float(plan.gov_joint_max_deg),
        psi_tgt=psi_tgt,
        psi_toggle_period_s=float(getattr(args, "psi_toggle_period", 0.0) or 0.0),
        psi_side_offset_rad=np.deg2rad(
            float(getattr(args, "psi_side_offset_deg", 90.5))
        ),
        psi_left_rad=(
            float(psi_left_rad)
            if psi_left_rad is not None
            else (
                np.deg2rad(float(args.psi_left_deg))
                if getattr(args, "psi_left_deg", None) is not None
                else None
            )
        ),
        psi_right_rad=(
            float(psi_right_rad)
            if psi_right_rad is not None
            else (
                np.deg2rad(float(args.psi_right_deg))
                if getattr(args, "psi_right_deg", None) is not None
                else None
            )
        ),
        psi_filter_alpha=float(getattr(args, "psi_toggle_alpha", 0.02)),
        psi_ramp_s=float(getattr(args, "psi_ramp_s", 4.0)),
        scan_hybrid_hold=bool(getattr(args, "hybrid_hold_at_d", False)),
        q_toggle_left_rad=(
            np.asarray(q_toggle_left_rad, dtype=float).reshape(-1).tolist()
            if q_toggle_left_rad is not None
            else []
        ),
        q_toggle_right_rad=(
            np.asarray(q_toggle_right_rad, dtype=float).reshape(-1).tolist()
            if q_toggle_right_rad is not None
            else []
        ),
        tcp_offset_pose=(
            np.asarray(tcp_offset_pose, dtype=float).reshape(6).tolist()
            if tcp_offset_pose is not None
            else []
        ),
    )
