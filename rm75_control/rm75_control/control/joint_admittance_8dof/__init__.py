"""Generic task-priority QPIK for an arm with an optional prismatic rail.

The servo-facing API is trajectory agnostic: applications declare protected
and scalable task rows, while robot-specific posture planning remains an
optional lowest-priority plugin.
"""

from __future__ import annotations

__all__ = [
    "RobotKinematics",
    "JointIkController",
    "JointIkConfig",
    "RobotState",
    "ProtectedTask",
    "ScalableTask",
    "PostureGuide",
    "HardConstraintRow",
    "LinearConstraintSet",
    "CartesianTaskProfile",
    "ScalableRowGroup",
    "TaskSpaceConstraintRow",
    "TwoLevelQpikConfig",
    "TwoLevelQpikController",
    "TwoLevelQpikResult",
    "ReferenceGovernor",
    "ReferenceHorizon",
    "HealthMonitor",
    "HealthState",
    "PosturePlanner",
    "PosturePlanningRequest",
    "Rm75SrsPosturePlanner",
    "Rm75SrsPlannerConfig",
    "TaskMode",
    "SecondaryPolicy",
    "ArmAngleSpec",
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
        "ProtectedTask",
        "ScalableTask",
        "PostureGuide",
        "HardConstraintRow",
        "LinearConstraintSet",
        "ReferenceHorizon",
    ):
        from rm75_control.control.joint_admittance_8dof import generic_tasks

        return getattr(generic_tasks, name)
    if name in (
        "CartesianTaskProfile",
        "ScalableRowGroup",
        "TaskSpaceConstraintRow",
    ):
        from rm75_control.control.joint_admittance_8dof import task_adapter

        return getattr(task_adapter, name)
    if name in (
        "TwoLevelQpikConfig",
        "TwoLevelQpikController",
        "TwoLevelQpikResult",
    ):
        from rm75_control.control.joint_admittance_8dof.solver import two_level_qpik

        return getattr(two_level_qpik, name)
    if name == "ReferenceGovernor":
        from rm75_control.control.joint_admittance_8dof import reference_governor

        return getattr(reference_governor, name)
    if name in ("HealthMonitor", "HealthState"):
        from rm75_control.control.joint_admittance_8dof import health_monitor

        return getattr(health_monitor, name)
    if name in ("PosturePlanner", "PosturePlanningRequest"):
        from rm75_control.control.joint_admittance_8dof import posture_planner

        return getattr(posture_planner, name)
    if name in ("Rm75SrsPosturePlanner", "Rm75SrsPlannerConfig"):
        from rm75_control.control.joint_admittance_8dof import rm75_srs_planner

        return getattr(rm75_srs_planner, name)
    if name in (
        "TaskMode",
        "SecondaryPolicy",
        "ArmAngleSpec",
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
