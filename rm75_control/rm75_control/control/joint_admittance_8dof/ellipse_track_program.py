"""Build a no-force Cartesian TRACKING ellipse program (window A / standalone)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from rm75_control.control.admittance_common.phase_ipc import SinToolYTaskParams
from rm75_control.control.joint_admittance_8dof.api import (
    CompileContext,
    GovernorSpec,
    SecondaryPolicy,
    compile_phases,
    phase_cartesian_track,
)
from rm75_control.control.joint_admittance_8dof.config import build_joint_ik_config
from rm75_control.control.joint_admittance_8dof.loop import JointIkController
from rm75_control.control.joint_admittance_8dof.model import (
    RobotKinematics,
    shared_robot_kinematics,
)
from rm75_control.control.joint_admittance_8dof.reference import EllipseToolXYReference
from rm75_control.control.joint_admittance_8dof.wbc_arm import WbcArm
from rm75_control.force.compensation.tool_pose import maybe_sync_kin_tcp_from_config


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


@dataclass
class BuiltEllipseTrackProgram:
    phases: list
    compiled: list
    inner: JointIkController
    kin: RobotKinematics
    force_observer: Any
    reference: EllipseToolXYReference


def build_ellipse_track_program(
    params: SinToolYTaskParams,
    *,
    raw: dict | None = None,
) -> BuiltEllipseTrackProgram:
    raw = raw if raw is not None else load_yaml(params.config_path)
    kin = shared_robot_kinematics()
    maybe_sync_kin_tcp_from_config(
        kin,
        raw,
        attach_mode=True,
        tcp_offset_pose=params.tcp_offset_pose if params.tcp_offset_pose else None,
    )
    inner_cfg = build_joint_ik_config(raw)
    inner = JointIkController(kin, inner_cfg)
    ctx = CompileContext(
        kin=kin,
        inner=inner,
        euler_order=inner_cfg.euler_order,
        control_frame=inner_cfg.control_frame,
        v_scale=inner_cfg.v_scale,
    )

    specs = []
    q_target = np.asarray(params.q_target_rad, dtype=float).reshape(-1)
    q0 = np.asarray(params.q0_rad, dtype=float).reshape(-1)
    if float(params.plan_duration_s) > 1.0e-9 and q_target.size == q0.size and q0.size > 0:
        move_mode = str(params.plan_move_mode)
        if move_mode == "joint":
            specs.append(
                WbcArm.make_movej_phase(
                    kin,
                    q0,
                    q_target,
                    duration_s=float(params.plan_duration_s),
                    label=f"movej->{params.slot}",
                    move_kp=SinToolYTaskParams.optional_move_kp(params.move_kp),
                    gov_joint_max_deg=float(params.plan_gov_joint_max_deg),
                )
            )
        else:
            pose_d = np.asarray(kin.fk_pose(q_target), dtype=float).reshape(6)
            specs.append(
                WbcArm.make_movel_phase(
                    kin,
                    q0,
                    pose_d,
                    q_target,
                    duration_s=float(params.plan_duration_s),
                    label=f"movel->{params.slot}",
                    move_kp=SinToolYTaskParams.optional_move_kp(params.move_kp),
                    max_lin_vel_m_s=(
                        float(params.cartesian_max_lin_vel)
                        if params.cartesian_max_lin_vel is not None
                        else 0.4
                    ),
                    gov_joint_max_deg=float(params.plan_gov_joint_max_deg),
                    euler_order=inner_cfg.euler_order,
                )
            )

    ax_m = float(getattr(params, "x_pp_cm", 0.0) or 0.0) * 0.01 / 2.0
    ay_m = float(params.y_pp_cm) * 0.01 / 2.0
    max_vel_m_s = float(params.max_vel_cm_s) * 0.01
    track_ref = EllipseToolXYReference(
        ax_m,
        ay_m,
        period_s=params.period_s,
        max_vel_m_s=None if params.period_s is not None else max_vel_m_s,
        soft_start=True,
        ramp_s=2.0,
        euler_order=inner_cfg.euler_order,
    )
    track_lin = (
        float(params.cartesian_max_lin_vel)
        if params.cartesian_max_lin_vel is not None
        else max(0.15, 3.0 * max_vel_m_s)
    )
    duration = float(params.scan_duration)
    specs.append(
        phase_cartesian_track(
            track_ref,
            label="ellipse_track",
            duration_s=None if duration <= 0.0 else duration,
            move_kp=SinToolYTaskParams.optional_move_kp(params.move_kp),
            max_lin_vel_m_s=track_lin,
            secondary=SecondaryPolicy(preset="track", qdot_ff="off"),
            governor=GovernorSpec(err_ok_mm=10.0, err_max_mm=40.0),
        )
    )

    compiled = compile_phases(specs, ctx)
    return BuiltEllipseTrackProgram(
        phases=[item.phase for item in compiled],
        compiled=compiled,
        inner=inner,
        kin=kin,
        force_observer=None,
        reference=track_ref,
    )
