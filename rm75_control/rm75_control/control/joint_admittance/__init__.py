"""Joint-space WBC inner loop (Pinocchio slack-QP IK) for RM75-F.

Cascaded controller: the task-space admittance outer loop produces a 6D Cartesian
twist; this package converts it to absolute joint angles streamed via rm_movej_canfd.

Imports are lazy — Pinocchio loads only when submodules are imported.
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
    "phase_joint_reset",
    "phase_cartesian_goto",
    "phase_cartesian_track",
    "phase_hybrid_track",
    "compute_move_plan",
    "scale_admittance_for_desired_z",
]


def __getattr__(name: str):
    if name in ("RobotKinematics",):
        from rm75_control.control.joint_admittance.model import RobotKinematics

        return RobotKinematics
    if name in ("QpIkController", "QpConfig"):
        from rm75_control.control.joint_admittance.solver import qp_builder

        return getattr(qp_builder, name)
    if name in ("JointIkController", "JointIkConfig"):
        from rm75_control.control.joint_admittance import loop

        return getattr(loop, name)
    if name in ("IkStepResult",):
        from rm75_control.control.joint_admittance.ik_types import IkStepResult

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
        "phase_joint_reset",
        "phase_cartesian_goto",
        "phase_cartesian_track",
        "phase_hybrid_track",
        "compute_move_plan",
        "scale_admittance_for_desired_z",
    ):
        from rm75_control.control.joint_admittance import api

        return getattr(api, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
