"""Focused safety tests for the fixed-rail numerical pose IK helper."""

from __future__ import annotations

import numpy as np
import pytest

from rm75_control.control.joint_admittance_8dof.model import (
    RobotKinematics,
    deg2rad,
    full_q_from_arm,
)
from rm75_control.control.joint_admittance_8dof.numerical_pose_ik import (
    NumericalPoseIkConfig,
    NumericalPoseIkError,
    solve_fixed_rail_pose_ik,
    solve_numerical_pose_ik,
)
from rm75_control.control.joint_admittance_8dof.sin_tool_y_program import (
    _fold_flange_into_world_vertical_plane,
    _remap_taught_q_armtip_x_to_tcp_z,
    _tcp_pose_from_link7,
    load_slot_joints_only,
)
from rm75_control.kinematics.srs_ik import branch_from_q, psi_from_q, srs_ik


def _folded_d(kin: RobotKinematics, rail_m: float) -> tuple[np.ndarray, np.ndarray]:
    q_deg, _pose_id, _record = load_slot_joints_only("d")
    q_arm = _remap_taught_q_armtip_x_to_tcp_z(deg2rad(q_deg))
    q_seed = full_q_from_arm(q_arm, rail_m=rail_m)
    link7 = kin.frame_placement(q_seed, "link_7")
    folded_r, _fold_deg = _fold_flange_into_world_vertical_plane(link7.rotation)
    pose_d = _tcp_pose_from_link7(kin, link7.translation, folded_r)
    return q_seed, pose_d


def test_fixed_rail_numerical_ik_solves_folded_slot_d() -> None:
    """The folded D orientation is intentionally outside the SRS candidate set."""

    kin = RobotKinematics()
    rail = 0.40
    q_seed, pose_d = _folded_d(kin, rail)
    assert srs_ik(
        pose_d,
        psi_from_q(q_seed[1:]),
        branch_from_q(q_seed[1:]),
        y_rail=rail,
    ) is None

    result = solve_numerical_pose_ik(
        kin,
        q_seed,
        pose_d,
        rail_m=rail,
        config=NumericalPoseIkConfig(path_check_samples=12, max_step_rad=0.10),
    )
    q, ok, report = result
    assert ok, report
    assert report.path_ok and report.within_limits and report.rail_exact
    assert q[0] == rail  # exact equality is intentional: rail is not an IK variable
    np.testing.assert_allclose(kin.fk_pose(q)[:3], pose_d[:3], atol=1e-8)
    assert report.pos_err_m < 1e-7
    assert report.rot_err_rad < 1e-7
    assert np.all(q >= kin.q_lower - 1e-10)
    assert np.all(q <= kin.q_upper + 1e-10)


def test_requested_rail_is_preserved_for_every_fk_and_collision_query() -> None:
    kin = RobotKinematics()
    q_seed, _pose = _folded_d(kin, 0.21)
    requested_rail = 0.57
    q_at_target_rail = q_seed.copy()
    q_at_target_rail[0] = requested_rail
    target = kin.fk_pose(q_at_target_rail)
    seen: list[np.ndarray] = []

    def collision_check(q: np.ndarray) -> bool:
        seen.append(q.copy())
        return True

    result = solve_fixed_rail_pose_ik(
        kin,
        q_seed,
        target,
        rail_target_m=requested_rail,
        collision_check=collision_check,
        config=NumericalPoseIkConfig(path_check_samples=4),
    )
    assert result.ok, result.report
    assert result.q[0] == requested_rail
    assert seen
    assert all(float(q[0]) == requested_rail for q in seen)
    assert all(np.isfinite(q).all() for q in seen)
    assert all(np.all(q >= kin.q_lower - 1e-9) for q in seen)
    assert all(np.all(q <= kin.q_upper + 1e-9) for q in seen)


def test_collision_rejection_is_fail_loud_and_never_returns_unvalidated_q() -> None:
    kin = RobotKinematics()
    q_seed, pose_d = _folded_d(kin, 0.40)
    calls: list[np.ndarray] = []

    # The seed is valid, but every non-zero J1 move is rejected.  This forces
    # the optimizer to encounter a rejected generated state and verifies that
    # the helper does not silently return that state as a target.
    def collision_check(q: np.ndarray) -> bool:
        calls.append(q.copy())
        return abs(float(q[1] - q_seed[1])) < 1e-14

    result = solve_numerical_pose_ik(
        kin,
        q_seed,
        pose_d,
        rail_m=0.40,
        collision_checker=collision_check,
        config=NumericalPoseIkConfig(max_iters=8),
    )
    assert not result.ok
    assert result.report.invalid_evaluations > 0
    assert result.report.reason
    assert calls
    assert result.q[0] == 0.40
    assert np.all(result.q >= kin.q_lower - 1e-9)
    assert np.all(result.q <= kin.q_upper + 1e-9)


def test_path_sample_bound_fails_loudly_after_convergence() -> None:
    kin = RobotKinematics()
    q_seed, pose_d = _folded_d(kin, 0.40)
    cfg = NumericalPoseIkConfig(
        path_check_samples=1,
        max_step_rad=1.0e-3,
        max_path_samples=2,
    )
    result = solve_numerical_pose_ik(kin, q_seed, pose_d, rail_m=0.40, config=cfg)
    assert not result.ok
    assert result.report.target_reached
    assert not result.report.path_ok
    assert "max_path_samples" in result.report.reason


def test_invalid_inputs_raise_before_solver() -> None:
    kin = RobotKinematics()
    q = np.zeros(8)
    pose = kin.fk_pose(q)
    with pytest.raises(NumericalPoseIkError, match="pose_target"):
        solve_numerical_pose_ik(kin, q, np.full(6, np.nan))
    with pytest.raises(NumericalPoseIkError, match="outside allowed"):
        solve_numerical_pose_ik(kin, q, pose, rail_m=float(kin.q_upper[0]) + 1.0)
