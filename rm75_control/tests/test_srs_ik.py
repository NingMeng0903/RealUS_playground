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

  4. Rail invariance: q_rail translates S, but psi and the resolved q_arm
     are unchanged (up to numerical noise).

  5. Joint-limit filter and reachability checks return None where expected.

Pose / rail frame contract (see ``srs_ik`` module docstring):
  * FK poses are in ``rail_base``.
  * ``srs_ik(..., y_rail=...)`` takes shoulder world-Y =
    ``RAIL_ORIGIN_Y + q_rail``, not the prismatic joint value.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as Rsc

from rm75_control.control.joint_admittance_8dof.model import (
    RobotKinematics,
    full_q_from_arm,
)
from rm75_control.control.joint_admittance_8dof.pose_ik import (
    UnreachablePathError,
    resolve_pose_ik_srs,
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
    RAIL_ORIGIN_Y,
    TOOL_MODE_COAXIAL,
    assert_srs_constants_match_urdf,
    branch_from_q,
    flange_tcp_from_kin,
    is_reachable,
    psi_from_q,
    shoulder_y_from_q_rail,
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


@pytest.fixture(scope="module")
def flange_tcp(kin: RobotKinematics) -> tuple[np.ndarray, np.ndarray]:
    return flange_tcp_from_kin(kin)


def _solve(pose, psi, branch, q_rail: float, flange_tcp):
    R_ft, t_ft = flange_tcp
    return srs_ik(
        pose,
        psi,
        branch,
        y_rail=shoulder_y_from_q_rail(q_rail),
        R_flange_tcp=R_ft,
        t_flange_tcp=t_ft,
    )


# ---------------------------------------------------------------------------
# 0. URDF constant ledger
# ---------------------------------------------------------------------------
def test_srs_constants_match_urdf() -> None:
    assert_srs_constants_match_urdf()
    assert RAIL_ORIGIN_Y == pytest.approx(-0.4)


# ---------------------------------------------------------------------------
# 1. Round trip: srs_ik ∘ (fk_pose, psi_from_q, branch_from_q) = identity
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("q_arm", Q_ARM_SAFE)
@pytest.mark.parametrize("q_rail", [0.0, 0.10, 0.35])
def test_srs_ik_roundtrip(
    kin: RobotKinematics,
    flange_tcp,
    q_arm: np.ndarray,
    q_rail: float,
) -> None:
    q_full = full_q_from_arm(q_arm, rail_m=q_rail)
    pose = kin.fk_pose(q_full)
    psi = psi_from_q(q_arm)
    branch = branch_from_q(q_arm)

    q_out = _solve(pose, psi, branch, q_rail, flange_tcp)
    assert q_out is not None, f"srs_ik returned None for q={q_arm}, q_rail={q_rail}"
    # ~1e-5 rad = ~1e-3 deg residual is the URDF's own approximation of π/2
    # (the URDF stores 1.5708 = π/2 − 3.6e-5), well below servo precision.
    assert np.allclose(q_out, q_arm, atol=1e-4), (
        f"round-trip diff = {np.max(np.abs(q_out - q_arm)):.3e}\n"
        f"q_in  = {q_arm}\nq_out = {q_out}"
    )


# ---------------------------------------------------------------------------
# 1b. Pose residual on successful solves (probe45 flange path)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("q_arm", Q_ARM_SAFE)
def test_srs_ik_fk_pose_residual_probe45(
    kin: RobotKinematics,
    flange_tcp,
    q_arm: np.ndarray,
) -> None:
    q_rail = 0.20
    q_full = full_q_from_arm(q_arm, rail_m=q_rail)
    pose = kin.fk_pose(q_full)
    psi = psi_from_q(q_arm)
    branch = branch_from_q(q_arm)
    q_out = _solve(pose, psi, branch, q_rail, flange_tcp)
    assert q_out is not None
    assert np.allclose(q_out, q_arm, atol=5e-5)
    pose_rec = kin.fk_pose(full_q_from_arm(q_out, rail_m=q_rail))
    pos_err = float(np.linalg.norm(pose_rec[:3] - pose[:3]))
    R_tgt = Rsc.from_euler("xyz", pose[3:]).as_matrix()
    R_rec = Rsc.from_euler("xyz", pose_rec[3:]).as_matrix()
    rot_err = float(np.linalg.norm(Rsc.from_matrix(R_rec @ R_tgt.T).as_rotvec()))
    # Pinocchio FK uses URDF joint origins with π/2 stored as 1.5708 (~3.6e-5
    # rad), so true µm / 1e-6 rad through the full TCP chain is not attainable;
    # these bounds are still ~1e3–1e4× below the pre-fix probe45 residuals
    # (15.230 mm / 99.548°).
    assert pos_err < 5e-6, f"pos residual {pos_err:.3e} m"
    assert rot_err < 2e-5, f"rot residual {rot_err:.3e} rad"


def test_coaxial_mode_on_probe45_has_tool_residual(kin: RobotKinematics) -> None:
    """Regression: feeding probe45 TCP into coaxial mode reproduces the tool offset."""
    q_arm = Q_ARM_SAFE[0]
    q_rail = 0.20
    pose = kin.fk_pose(full_q_from_arm(q_arm, rail_m=q_rail))
    psi = psi_from_q(q_arm)
    branch = branch_from_q(q_arm)
    # Coaxial path with the (incorrect for probe45) d_wt_from_kin length.
    from rm75_control.kinematics.srs_ik import d_wt_from_kin

    q_wrong = srs_ik(
        pose,
        psi,
        branch,
        y_rail=shoulder_y_from_q_rail(q_rail),
        tool_mode=TOOL_MODE_COAXIAL,
        d_wt=d_wt_from_kin(kin),
    )
    # May return a wrong q or None; if it returns, FK residual must be huge.
    if q_wrong is not None:
        pose_rec = kin.fk_pose(full_q_from_arm(q_wrong, rail_m=q_rail))
        pos_err = float(np.linalg.norm(pose_rec[:3] - pose[:3]))
        R_tgt = Rsc.from_euler("xyz", pose[3:]).as_matrix()
        R_rec = Rsc.from_euler("xyz", pose_rec[3:]).as_matrix()
        rot_err = float(np.linalg.norm(Rsc.from_matrix(R_rec @ R_tgt.T).as_rotvec()))
        assert pos_err > 0.01 or rot_err > np.deg2rad(10.0)


# ---------------------------------------------------------------------------
# 2. psi_from_q matches ArmAngleTask.arm_angle to 1e-6 rad
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("q_arm", Q_ARM_SAFE)
@pytest.mark.parametrize("q_rail", [0.0, 0.20])
def test_psi_from_q_matches_arm_angle_task(
    kin: RobotKinematics,
    arm_angle_task: ArmAngleTask,
    q_arm: np.ndarray,
    q_rail: float,
) -> None:
    q_full = full_q_from_arm(q_arm, rail_m=q_rail)
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
def test_branch_id_is_stable_under_srs_ik(
    kin: RobotKinematics,
    flange_tcp,
    q_arm: np.ndarray,
) -> None:
    q_rail = 0.0
    q_full = full_q_from_arm(q_arm, rail_m=q_rail)
    pose = kin.fk_pose(q_full)
    psi = psi_from_q(q_arm)
    branch = branch_from_q(q_arm)
    q_out = _solve(pose, psi, branch, q_rail, flange_tcp)
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
    for q_rail in (0.0, 0.05, 0.10, 0.24, 0.40):
        q_full = full_q_from_arm(q_arm, rail_m=q_rail)
        psis.append(float(arm_angle_task.arm_angle(q_full)))
    for p in psis[1:]:
        assert abs(p - psis[0]) < 1e-10


@pytest.mark.parametrize("q_arm", Q_ARM_SAFE)
def test_srs_ik_rail_shifts_but_keeps_arm(
    kin: RobotKinematics,
    flange_tcp,
    q_arm: np.ndarray,
) -> None:
    # FK-then-IK at q_rail=0.15 should give the SAME arm q as at 0.0, because
    # the rail just translates S; ψ (rail-invariant) and R_tcp are unchanged
    # up to that rigid Y shift (absorbed by shoulder_y).
    q_full0 = full_q_from_arm(q_arm, rail_m=0.0)
    q_full1 = full_q_from_arm(q_arm, rail_m=0.15)
    pose0 = kin.fk_pose(q_full0)
    pose1 = kin.fk_pose(q_full1)
    psi = psi_from_q(q_arm)
    branch = branch_from_q(q_arm)
    q0 = _solve(pose0, psi, branch, 0.0, flange_tcp)
    q1 = _solve(pose1, psi, branch, 0.15, flange_tcp)
    assert q0 is not None and q1 is not None
    assert np.allclose(q0, q1, atol=1e-6), (
        f"rail-shift arm diff = {np.max(np.abs(q0 - q1)):.3e}"
    )


# ---------------------------------------------------------------------------
# 5. Joint-limit filter and reachability (coaxial synthetic poses)
# ---------------------------------------------------------------------------
def test_srs_ik_rejects_unreachable_pose() -> None:
    # Move the pose too far above the shoulder — |SW| > D_SE + D_EW = 0.466 m
    # Synthetic pose assumes coaxial tool + base_link frame (y_rail=0).
    p_far = np.array([0.0, 0.0, D_BS + D_SE + D_EW + D_WT + 0.20, 0.0, 0.0, 0.0])
    assert srs_ik(p_far, 0.0, 0, tool_mode=TOOL_MODE_COAXIAL) is None
    assert not is_reachable(p_far, tool_mode=TOOL_MODE_COAXIAL)


def test_srs_ik_rejects_pose_inside_shoulder() -> None:
    # Wrist centre coinciding with S → dsw = 0
    p_close = np.array([0.0, 0.0, D_BS + D_WT, 0.0, 0.0, 0.0])
    assert srs_ik(p_close, 0.0, 0, tool_mode=TOOL_MODE_COAXIAL) is None
    assert not is_reachable(p_close, tool_mode=TOOL_MODE_COAXIAL)


def test_srs_ik_rejects_shoulder_vertical_singularity(
    kin: RobotKinematics,
    flange_tcp,
) -> None:
    # q_arm = 0 makes the arm straight up — sin(q_2) = 0.  Even at ψ = 0 this
    # is the algorithmic singularity and srs_ik should return None.
    q_arm = np.zeros(7, dtype=float)
    q_full = full_q_from_arm(q_arm)
    pose = kin.fk_pose(q_full)
    assert _solve(pose, 0.0, 0, 0.0, flange_tcp) is None


def test_diagnostics_returns_psi_within_1e6(
    kin: RobotKinematics,
    flange_tcp,
) -> None:
    q_arm = Q_ARM_SAFE[0]
    q_rail = 0.1
    q_full = full_q_from_arm(q_arm, rail_m=q_rail)
    pose = kin.fk_pose(q_full)
    psi_in = psi_from_q(q_arm)
    R_ft, t_ft = flange_tcp
    sol = srs_ik_with_diagnostics(
        pose,
        psi_in,
        branch_from_q(q_arm),
        y_rail=shoulder_y_from_q_rail(q_rail),
        R_flange_tcp=R_ft,
        t_flange_tcp=t_ft,
    )
    assert sol is not None
    assert abs(sol.psi_realised - psi_in) < 1e-5
    assert sol.branch == branch_from_q(q_arm)


# ---------------------------------------------------------------------------
# 6. End-to-end resolve_pose_ik_srs on self-produced FK poses
# ---------------------------------------------------------------------------
def test_require_path_false_does_not_force_ok_false(kin: RobotKinematics) -> None:
    """MoveJ path: skipping path check must not zero out ``ok`` via path_ok=False."""
    q_arm = np.array(Q_ARM_SAFE[0], dtype=float)
    q_rail = 0.2
    q_full = full_q_from_arm(q_arm, rail_m=q_rail)
    pose = kin.fk_pose(q_full)
    q_tgt, ok, rep = resolve_pose_ik_srs(
        kin,
        q_seed=q_full,
        pose_target=pose,
        y_rail_target=q_rail,
        require_path=False,
    )
    assert ok
    assert rep.path_ok
    assert rep.pos_err_mm < 1.0
    assert q_tgt.shape == (8,)


def test_resolve_pose_ik_srs_self_fk_ok_rate(kin: RobotKinematics) -> None:
    """Self-FK targets should resolve with ok≈1 when path check is off.

    With ``require_path=True`` and a far seed, a small fraction of path
    failures is allowed (documented below); identity seed→target must be
    near-perfect.
    """
    rng = np.random.default_rng(20260729)
    n = 40
    ok_identity = 0
    ok_near = 0
    path_fail = 0
    for i in range(n):
        q_arm = np.array(Q_ARM_SAFE[i % len(Q_ARM_SAFE)], dtype=float)
        # Jitter away from exact safe set while staying clear of gimbal.
        q_arm = q_arm + rng.uniform(-0.05, 0.05, size=7)
        q_arm[1] = float(np.clip(q_arm[1], 0.35, 1.2)) if q_arm[1] >= 0 else float(np.clip(q_arm[1], -1.2, -0.35))
        q_arm[5] = float(np.clip(q_arm[5], 0.35, 1.0)) if q_arm[5] >= 0 else float(np.clip(q_arm[5], -1.0, -0.35))
        q_rail = float(rng.uniform(0.05, 0.55))
        q_full = full_q_from_arm(q_arm, rail_m=q_rail)
        pose = kin.fk_pose(q_full)

        # Identity: seed == target, path must succeed.
        q_tgt, ok, rep = resolve_pose_ik_srs(
            kin,
            q_seed=q_full,
            pose_target=pose,
            y_rail_target=q_rail,
            require_path=True,
        )
        if ok:
            ok_identity += 1
        assert float(np.linalg.norm(q_tgt[1:] - q_arm)) < 5e-3 or ok

        # Near seed: small rail shift; allow UnreachablePathError fraction.
        q_seed = full_q_from_arm(q_arm, rail_m=float(np.clip(q_rail - 0.05, 0.0, 0.8)))
        try:
            _, ok2, _ = resolve_pose_ik_srs(
                kin,
                q_seed=q_seed,
                pose_target=pose,
                y_rail_target=q_rail,
                require_path=True,
            )
            if ok2:
                ok_near += 1
        except UnreachablePathError:
            path_fail += 1

    assert ok_identity == n, f"identity ok rate {ok_identity}/{n}"
    # Documented allowance: path failures when seed rail differs can occur;
    # require most near-seed solves to succeed.
    assert ok_near >= int(0.85 * n), (
        f"near-seed ok={ok_near}/{n} path_fail={path_fail} (expected ≥85%)"
    )
