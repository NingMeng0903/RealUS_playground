"""Joint-space WBC inner loop (Pinocchio slack-QP IK) for RM75-F on Y-axis rail.

8 DOF: rail_y (prismatic) + joint_1..joint_7.  See MD/JOINT_ADMITTANCE_8DOF.md.
"""

from __future__ import annotations

__all__ = [
    "RobotKinematics",
    "QpIkController",
    "QpConfig",
    "JointIkController",
    "JointIkConfig",
    "IkStepResult",
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
    if name in ("QpIkController", "QpConfig"):
        from rm75_control.control.joint_admittance_8dof.solver import qp_builder

        return getattr(qp_builder, name)
    if name in ("JointIkController", "JointIkConfig"):
        from rm75_control.control.joint_admittance_8dof import loop

        return getattr(loop, name)
    if name in ("IkStepResult",):
        from rm75_control.control.joint_admittance_8dof.ik_types import IkStepResult

        return IkStepResult
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
