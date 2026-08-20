"""Build JointIkController + kinematics from yaml. Starts wbc_rt when native."""

from __future__ import annotations

from pathlib import Path

import yaml

from rm75_control.control.joint_admittance_8dof.api import CompileContext
from rm75_control.control.joint_admittance_8dof.config import build_joint_ik_config
from rm75_control.control.joint_admittance_8dof.loop import JointIkController
from rm75_control.control.joint_admittance_8dof.model import (
    RobotKinematics,
    shared_robot_kinematics,
)
from rm75_control.force.compensation.tool_pose import maybe_sync_kin_tcp_from_config


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def bind_controller(
    raw: dict,
    *,
    backend: str | None = None,
    shm_prefix: str | None = None,
) -> tuple[RobotKinematics, JointIkController, CompileContext]:
    kin = shared_robot_kinematics()
    maybe_sync_kin_tcp_from_config(kin, raw, attach_mode=True)
    cfg = build_joint_ik_config(raw)
    if backend is not None:
        cfg.backend = str(backend)
    if shm_prefix is not None:
        cfg.native_shm_prefix = str(shm_prefix)
    inner = JointIkController(kin, cfg)
    ctx = CompileContext(
        kin=kin,
        inner=inner,
        euler_order=cfg.euler_order,
        control_frame=cfg.control_frame,
        v_scale=cfg.v_scale,
    )
    return kin, inner, ctx
