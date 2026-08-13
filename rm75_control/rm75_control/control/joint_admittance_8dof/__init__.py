"""Fixed single-shot QPIK for the RM75 arm and prismatic rail."""

from __future__ import annotations

__all__ = [
    "RobotKinematics",
    "JointIkController",
    "JointIkConfig",
    "RobotState",
    "HardConstraintRow",
    "LinearConstraintSet",
    "TaskSpaceConstraintRow",
    "CartesianQpCommand",
    "SingleQpikConfig",
    "SingleQpikController",
    "SingleQpikResult",
    "HealthMonitor",
    "HealthState",
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
    if name in (
        "RobotState",
        "HardConstraintRow",
        "LinearConstraintSet",
    ):
        from rm75_control.control.joint_admittance_8dof import generic_tasks

        return getattr(generic_tasks, name)
    if name == "TaskSpaceConstraintRow":
        from rm75_control.control.joint_admittance_8dof import task_adapter

        return getattr(task_adapter, name)
    if name in (
        "CartesianQpCommand",
        "SingleQpikConfig",
        "SingleQpikController",
        "SingleQpikResult",
    ):
        from rm75_control.control.joint_admittance_8dof.solver import single_qpik

        return getattr(single_qpik, name)
    if name in ("HealthMonitor", "HealthState"):
        from rm75_control.control.joint_admittance_8dof import health_monitor

        return getattr(health_monitor, name)
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
