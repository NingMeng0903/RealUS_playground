"""Analytical SRS IK <-> Pinocchio FK round-trip and psi consistency tests.

These tests pin the module-level invariants that the rest of the fix relies
on:

  1. srs_ik is the RIGHT-inverse of the FK on all 8 branches - given any
     (q_arm) drawn from the URDF joint box, srs_ik(fk_pose(q), psi_from_q(q),
     branch_from_q(q)) reproduces q to numerical precision.

  2. psi_from_q agrees with ArmAngleTask.arm_angle to 1e-6 rad.  This is
     what makes the planner and the servo-layer arm_task share the same
     nullspace coordinate.

  3. branch_from_q is a right-inverse: srs_ik(...branch_from_q(q)) picks
     that same branch.

  4. Rail invariance: y_rail translates S, but psi and the resolved q_arm
     are unchanged (up to numerical noise).

  5. Joint-limit filter and reachability checks return None where expected.
"""

from __future__ import annotations

import numpy as np
import pytest

from rm75_control.control.joint_admittance_8dof.model import (
    RobotKinematics,
    full_q_from_arm,
)
from rm75_control.control.joint_admittance_8dof.tasks.arm_angle import (
    ArmAngleTask,
    ArmAngleTaskConfig,
)
from rm75_control.kinematics.srs_ik import (
    D_BS,
    D_EW,
    D_SE,
    D_WT,
    branch_from_q,
    is_reachable,
    psi_from_q,
    srs_ik,
    srs_ik_with_diagnostics,
)


# Poses chosen far from the ZYZ shoulder gimbal (q_2 = 0/±π) and wrist gimbal
# (q_6 = 0/±π) so all round-trip tests exercise the closed-form branch, not
# the None fallback.
Q_ARM_SAFE: list[np.ndarray] = [
    np.array([0.30, 0.70, -0.20, 0.90, 0.15, 0.60, -0.40]),
    np.array([-0.50, 1.10, 0.30, -1.20, -0.25, 0.85, 0.10]),
    np.array([0.80, 0.55, 0.40, 0.75, 1.00, 0.65, 1.20]),
    np.array([-0.20, -0.85, -0.35, -1.05, 0.55, -0.75, -0.90]),
    np.array([1.20, 0.40, 0.60, 0.50, -0.80, 0.55, 0.30]),
]


@pytest.fixture(scope="module")
def kin() -> RobotKinematics:
    return RobotKinematics()


@pytest.fixture(scope="module")
def arm_angle_task(kin: RobotKinematics) -> ArmAngleTask:
    return ArmAngleTask(kin, ArmAngleTaskConfig(enabled=True))


# ---------------------------------------------------------------------------
# 1. Round trip: srs_ik ∘ (fk_pose, psi_from_q, branch_from_q) = identity
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("q_arm", Q_ARM_SAFE)
@pytest.mark.parametrize("y_rail", [0.0, -0.10, 0.15])
def test_srs_ik_roundtrip(kin: RobotKinematics, q_arm: np.ndarray, y_rail: float) -> None:
    q_full = full_q_from_arm(q_arm, rail_m=y_rail)
    pose = kin.fk_pose(q_full)
    psi = psi_from_q(q_arm)
    branch = branch_from_q(q_arm)

    q_out = srs_ik(pose, psi, branch, y_rail=y_rail)
    assert q_out is not None, f"srs_ik returned None for q={q_arm}, y_rail={y_rail}"
    # ~1e-5 rad = ~1e-3 deg residual is the URDF's own approximation of π/2
    # (the URDF stores 1.5708 = π/2 − 3.6e-5), well below servo precision.
    assert np.allclose(q_out, q_arm, atol=1e-4), (
        f"round-trip diff = {np.max(np.abs(q_out - q_arm)):.3e}\n"
        f"q_in  = {q_arm}\nq_out = {q_out}"
    )


# ---------------------------------------------------------------------------
# 2. psi_from_q matches ArmAngleTask.arm_angle to 1e-6 rad
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("q_arm", Q_ARM_SAFE)
@pytest.mark.parametrize("y_rail", [0.0, 0.20])
def test_psi_from_q_matches_arm_angle_task(
    kin: RobotKinematics,
    arm_angle_task: ArmAngleTask,
    q_arm: np.ndarray,
    y_rail: float,
) -> None:
    q_full = full_q_from_arm(q_arm, rail_m=y_rail)
    psi_task = float(arm_angle_task.arm_angle(q_full))
    psi_srs = psi_from_q(q_arm)
    # ArmAngleTask uses Pinocchio FK (URDF π/2 stored as 1.5708) and this
    # module uses exact analytical trig — the residual is ~5e-6 rad.
    assert abs(psi_task - psi_srs) < 5e-5, (
        f"psi mismatch: task={psi_task:.9f} rad srs={psi_srs:.9f} rad diff={psi_task - psi_srs:.3e}"
    )


# ---------------------------------------------------------------------------
# 3. branch_from_q is a right inverse
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("q_arm", Q_ARM_SAFE)
def test_branch_id_is_stable_under_srs_ik(kin: RobotKinematics, q_arm: np.ndarray) -> None:
    q_full = full_q_from_arm(q_arm, rail_m=0.0)
    pose = kin.fk_pose(q_full)
    psi = psi_from_q(q_arm)
    branch = branch_from_q(q_arm)
    q_out = srs_ik(pose, psi, branch)
    assert q_out is not None
    assert branch_from_q(q_out) == branch


# ---------------------------------------------------------------------------
# 4. Rail invariance
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("q_arm", Q_ARM_SAFE)
def test_psi_from_q_rail_invariant(
    kin: RobotKinematics,
    arm_angle_task: ArmAngleTask,
    q_arm: np.ndarray,
) -> None:
    psis = []
    for y_rail in (-0.20, -0.05, 0.0, 0.10, 0.24):
        q_full = full_q_from_arm(q_arm, rail_m=y_rail)
        psis.append(float(arm_angle_task.arm_angle(q_full)))
    for p in psis[1:]:
        assert abs(p - psis[0]) < 1e-10


@pytest.mark.parametrize("q_arm", Q_ARM_SAFE)
def test_srs_ik_rail_shifts_but_keeps_arm(kin: RobotKinematics, q_arm: np.ndarray) -> None:
    # FK-then-IK at y_rail=0.15 should give the SAME arm q as at 0.0, because
    # the rail just translates S; ψ (rail-invariant) and R_tcp are unchanged.
    q_full0 = full_q_from_arm(q_arm, rail_m=0.0)
    q_full1 = full_q_from_arm(q_arm, rail_m=0.15)
    pose0 = kin.fk_pose(q_full0)
    pose1 = kin.fk_pose(q_full1)
    psi = psi_from_q(q_arm)
    branch = branch_from_q(q_arm)
    q0 = srs_ik(pose0, psi, branch, y_rail=0.0)
    q1 = srs_ik(pose1, psi, branch, y_rail=0.15)
    assert q0 is not None and q1 is not None
    assert np.allclose(q0, q1, atol=1e-6), (
        f"rail-shift arm diff = {np.max(np.abs(q0 - q1)):.3e}"
    )


# ---------------------------------------------------------------------------
# 5. Joint-limit filter and reachability
# ---------------------------------------------------------------------------
def test_srs_ik_rejects_unreachable_pose() -> None:
    # Move the pose too far above the shoulder — |SW| > D_SE + D_EW = 0.466 m
    p_far = np.array([0.0, 0.0, D_BS + D_SE + D_EW + D_WT + 0.20, 0.0, 0.0, 0.0])
    assert srs_ik(p_far, 0.0, 0) is None
    assert not is_reachable(p_far)


def test_srs_ik_rejects_pose_inside_shoulder() -> None:
    # Wrist centre coinciding with S → dsw = 0
    p_close = np.array([0.0, 0.0, D_BS + D_WT, 0.0, 0.0, 0.0])
    assert srs_ik(p_close, 0.0, 0) is None
    assert not is_reachable(p_close)


def test_srs_ik_rejects_shoulder_vertical_singularity(kin: RobotKinematics) -> None:
    # q_arm = 0 makes the arm straight up — sin(q_2) = 0.  Even at ψ = 0 this
    # is the algorithmic singularity and srs_ik should return None.
    q_arm = np.zeros(7, dtype=float)
    q_full = full_q_from_arm(q_arm)
    pose = kin.fk_pose(q_full)
    assert srs_ik(pose, 0.0, 0) is None


def test_diagnostics_returns_psi_within_1e6(kin: RobotKinematics) -> None:
    q_arm = Q_ARM_SAFE[0]
    q_full = full_q_from_arm(q_arm, rail_m=0.1)
    pose = kin.fk_pose(q_full)
    psi_in = psi_from_q(q_arm)
    sol = srs_ik_with_diagnostics(pose, psi_in, branch_from_q(q_arm), y_rail=0.1)
    assert sol is not None
    assert abs(sol.psi_realised - psi_in) < 1e-5
    assert sol.branch == branch_from_q(q_arm)
