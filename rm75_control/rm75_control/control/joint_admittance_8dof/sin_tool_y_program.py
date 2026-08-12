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
)
from rm75_control.control.joint_admittance_8dof.reference import (
    HoldReference,
    SinToolYReference,
)
from rm75_control.control.joint_admittance_8dof.wbc_arm import WbcArm
from rm75_control.control.joint_admittance_8dof.pose_ik import resolve_pose_ik_srs
from rm75_control.kinematics.srs_ik import psi_from_q
from rm75_control.force.compensation import excitation as ex
from rm75_control.force.compensation.id_config import load_config as load_force_id_config
from rm75_control.force.compensation.paths import CONFIG_ID
from rm75_control.force.compensation.tool_pose import maybe_sync_kin_tcp_from_config


@dataclass
class ScanTargetD:
    """Planned move->D target from taught joints (Pinocchio FK / pose IK)."""

    q_slot_deg: np.ndarray
    pose_d: np.ndarray
    pose_id: np.ndarray
    q_target_rad: np.ndarray


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
    euler_order: str = "xyz",
    rail_m: float = 0.0,
    q_seed_rad: np.ndarray | None = None,
) -> ScanTargetD:
    """Resolve scan pose D and joint target for the move->D phase.

    Joints-only: taught ``q_deg`` with j7+90° (ArmTip +X → TCP +Z), fold
    approach into a world-vertical plane, optional pose IK. Move execution is
    still ``--move-mode`` (joint MoveJ or cartesian/SRS).
    """
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
        q_seed_rad=q_seed_rad,
        euler_order=euler_order,
    )


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
    q_seed_rad: np.ndarray | None = None,
    euler_order: str = "xyz",
) -> ScanTargetD:
    q_deg_taught, pose_id, _rec = load_slot_joints_only(slot)
    q_arm = _remap_taught_q_armtip_x_to_tcp_z(deg2rad(q_deg_taught))
    y_rail = float(rail_m)
    if not 0.0 <= y_rail <= float(travel_m):
        raise ValueError(
            f"requested rail {y_rail:.6f}m is outside [0, {float(travel_m):.6f}]"
        )
    q_seed = full_q_from_arm(q_arm, y_rail)

    Ml7 = kin.frame_placement(q_seed, "link_7")
    R_fold, _fold_deg = _fold_flange_into_world_vertical_plane(Ml7.rotation)
    pose_d = _tcp_pose_from_link7(
        kin, Ml7.translation, R_fold, euler_order=euler_order
    )

    path_seed = q_seed if q_seed_rad is None else np.asarray(q_seed_rad, dtype=float)
    if path_seed.shape != (kin.nv,) or not np.isfinite(path_seed).all():
        raise ValueError(f"q_seed_rad must be a finite {(kin.nv,)} vector")
    q_target_rad, ok, report = resolve_pose_ik_srs(
        kin,
        path_seed,
        pose_d,
        y_rail_target=y_rail,
        psi_home_rad=psi_from_q(q_seed),
        euler_order=euler_order,
        require_path=True,
    )
    if not ok:
        raise RuntimeError(
            "SRS target or connecting path is invalid: "
            f"position={report.pos_err_mm:.3f}mm, rotation={report.rot_err_deg:.3f}deg"
        )
    pose_d = np.asarray(kin.fk_pose(q_target_rad), dtype=float)

    q_deg = np.rad2deg(q_target_rad[1:])
    return ScanTargetD(
        q_slot_deg=q_deg,
        pose_d=pose_d,
        pose_id=pose_id,
        q_target_rad=q_target_rad,
    )



def load_yaml(path: Path | str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


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
    q_target_rad = np.asarray(params.q_target_rad, dtype=float).reshape(-1)
    q0_rad = np.asarray(params.q0_rad, dtype=float).reshape(-1)
    rail_m = (
        float(inner_cfg.rail.q_ref_m)
        if inner_cfg.rail.q_ref_m is not None
        else float(q0_rad[0])
    )
    # Wait/SRS target must be FK(q_target) after TCP sync — raw params.pose_d can
    # still carry an ArmTip/IK residual orientation that blocks arrival forever
    # while track_err_mm (position-only) looks fine.
    pose_d = np.asarray(kin.fk_pose(q_target_rad), dtype=float).reshape(6)
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
        rail0 = (
            float(inner_cfg.rail.q_ref_m)
            if inner_cfg.rail.q_ref_m is not None
            else float(q0_rad[0])
        )
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
            # TCP force hold (no Y motion); rail stays a continuous variable
            # in the whole-body QP.
            hybrid_ref: HoldReference | SinToolYReference = HoldReference()
            hybrid_label = "hybrid@D"
            hybrid_sec = SecondaryPolicy(
                preset="track",
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
                profile=str(getattr(params, "scan_profile", "quintic_dwell")),
                dwell_s=float(getattr(params, "scan_dwell_s", 0.20)),
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
        raise RuntimeError(
            "contact-time posture toggle was removed: submit a branch-locked "
            "continuous PostureGuide through the generic posture planner"
        )
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
    verbose: bool = False,
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
        scan_profile=str(getattr(args, "scan_profile", "quintic_dwell")),
        scan_dwell_s=float(getattr(args, "scan_dwell_s", 0.20)),
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
