"""Escande slack-QP WBC for the RM75 arm and prismatic rail."""

from __future__ import annotations

__all__ = [
    "RobotKinematics",
    "JointIkController",
    "JointIkConfig",
    "QpConfig",
    "QpIkController",
    "TaskMode",
    "SecondaryPolicy",
    "JointPhaseSpec",
    "CompileContext",
    "CompiledPhase",
    "compile_phase",
    "compile_phases",
    "phase_cartesian_goto",
    "phase_hybrid_track",
    "compute_move_plan",
    "scale_admittance_for_desired_z",
    "WbcArm",
]


def __getattr__(name: str):
    if name in ("RobotKinematics",):
        from rm75_control.control.joint_admittance_8dof.model import RobotKinematics

        return RobotKinematics
    if name in ("JointIkController", "JointIkConfig"):
        from rm75_control.control.joint_admittance_8dof import loop

        return getattr(loop, name)
    if name in ("QpConfig", "QpIkController"):
        from rm75_control.control.joint_admittance_8dof.solver.qp_builder import (
            QpConfig,
            QpIkController,
        )

        return QpConfig if name == "QpConfig" else QpIkController
    if name in (
        "TaskMode",
        "SecondaryPolicy",
        "JointPhaseSpec",
        "CompileContext",
        "CompiledPhase",
        "compile_phase",
        "compile_phases",
        "phase_cartesian_goto",
        "phase_hybrid_track",
        "compute_move_plan",
        "scale_admittance_for_desired_z",
    ):
        from rm75_control.control.joint_admittance_8dof import api

        return getattr(api, name)
    if name == "WbcArm":
        from rm75_control.control.joint_admittance_8dof.wbc_arm import WbcArm

        return WbcArm
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
