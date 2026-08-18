"""Build a QPIK inner-loop gamepad v_cmd program (window A / standalone)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from rm75_control.control.admittance_common.phase_ipc import SinToolYTaskParams
from rm75_control.control.joint_admittance_8dof.api import (
    CompileContext,
    SecondaryPolicy,
    compile_phases,
)
from rm75_control.control.joint_admittance_8dof.config import build_joint_ik_config
from rm75_control.control.joint_admittance_8dof.loop import JointIkController, Phase
from rm75_control.control.joint_admittance_8dof.model import RobotKinematics
from rm75_control.control.joint_admittance_8dof.teleop.gamepad_twist import (
    GamepadTwistConfig,
    GamepadTwistOuterLoop,
)
from rm75_control.control.joint_admittance_8dof.teleop.xbox_pad import XboxPad
from rm75_control.control.joint_admittance_8dof.wbc_arm import WbcArm
from rm75_control.force.compensation.tool_pose import maybe_sync_kin_tcp_from_config


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


@dataclass
class BuiltGamepadVcmdProgram:
    phases: list
    compiled: list
    inner: JointIkController
    kin: RobotKinematics
    force_observer: Any
    pad: Any = None


def close_built_pad(built: Any) -> None:
    pad = getattr(built, "pad", None)
    if pad is None:
        return
    try:
        pad.close()
    except Exception:
        pass


def build_gamepad_vcmd_program(
    params: SinToolYTaskParams,
    *,
    raw: dict | None = None,
    pad: Any | None = None,
) -> BuiltGamepadVcmdProgram:
    raw = raw if raw is not None else load_yaml(params.config_path)
    kin = RobotKinematics()
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

    compiled = compile_phases(specs, ctx) if specs else []
    phases = [item.phase for item in compiled]

    if pad is None:
        idx = int(getattr(params, "gamepad_device_index", -1))
        pad = XboxPad(
            device_index=max(idx, 0),
            auto_select=idx < 0,
            allow_missing=True,
        )
    twist_cfg = GamepadTwistConfig(
        trans_m_s=float(getattr(params, "gamepad_trans_m_s", 0.10)),
        rot_rad_s=float(getattr(params, "gamepad_rot_rad_s", 0.60)),
        deadzone=float(getattr(params, "gamepad_deadzone", 0.18)),
        max_lin_vel_m_s=2.0 * float(getattr(params, "gamepad_trans_m_s", 0.10)),
        max_ang_vel_rad_s=2.0 * float(getattr(params, "gamepad_rot_rad_s", 0.60)),
        dt=float(inner_cfg.dt),
        euler_order=str(inner_cfg.euler_order),
        control_frame=str(inner_cfg.control_frame),
        trans_a_max_m_s2=float(getattr(params, "gamepad_trans_a_max_m_s2", 0.8)),
        rot_a_max_rad_s2=float(getattr(params, "gamepad_rot_a_max_rad_s2", 4.0)),
        hold_v_max_m_s=float(getattr(params, "gamepad_hold_v_max_m_s", 0.03)),
        hold_w_max_rad_s=float(getattr(params, "gamepad_hold_w_max_rad_s", 0.20)),
        hold_deadband_m=float(getattr(params, "gamepad_hold_deadband_m", 0.001)),
        hold_deadband_rad=float(getattr(params, "gamepad_hold_deadband_rad", 0.005)),
        hold_settle_v_m_s=float(getattr(params, "gamepad_hold_settle_v_m_s", 0.005)),
        hold_relatch_on_settle=bool(
            getattr(params, "gamepad_hold_relatch_on_settle", True)
        ),
    )
    outer = GamepadTwistOuterLoop(pad, twist_cfg)

    def _enter() -> None:
        SecondaryPolicy(preset="track", qdot_ff="off").apply(inner)
        if not getattr(pad, "connected", True):
            print("gamepad: no device — v_cmd stays zero until a pad appears", flush=True)

    duration = float(params.scan_duration)
    phase = Phase(
        outer=outer,
        label="gamepad_vcmd",
        duration_s=None if duration <= 0.0 else duration,
        governor_err_ok_mm=15.0,
        governor_err_max_mm=0.0,
        governor_joint_err_max_deg=0.0,
        on_enter=_enter,
        scale_qdot_ff_with_governor=False,
    )
    phases.append(phase)
    return BuiltGamepadVcmdProgram(
        phases=phases,
        compiled=compiled,
        inner=inner,
        kin=kin,
        force_observer=None,
        pad=pad,
    )
