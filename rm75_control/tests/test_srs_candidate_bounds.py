"""Controller position-box filtering for the closed-form SRS planner."""

from __future__ import annotations

import numpy as np
import pytest

from rm75_control.control.joint_admittance_8dof.model import (
    RobotKinematics,
    full_q_from_arm,
)
from rm75_control.control.joint_admittance_8dof.pose_ik import (
    UnreachablePathError,
    resolve_pose_ik_srs,
)
from rm75_control.kinematics.srs_ik import psi_from_q


def _target(kin: RobotKinematics) -> tuple[np.ndarray, np.ndarray, float]:
    """Return a well-conditioned reachable target and its ψ seed."""
    q_arm = np.array([0.30, 0.70, -0.20, 0.90, 0.15, 0.60, -0.40])
    q_full = full_q_from_arm(q_arm, rail_m=0.20)
    return q_full, kin.fk_pose(q_full), float(psi_from_q(q_arm))


def _resolve(
    kin: RobotKinematics,
    q_seed: np.ndarray,
    pose_target: np.ndarray,
    psi_home: float,
    **kwargs,
):
    return resolve_pose_ik_srs(
        kin,
        q_seed=q_seed,
        pose_target=pose_target,
        y_rail_target=0.20,
        psi_home_rad=psi_home,
        max_psi_swing_rad=np.pi,
        require_path=False,
        **kwargs,
    )


def test_rejects_high_score_candidate_and_selects_next_bounded_solution() -> None:
    """A reachable high-score ψ outside the controller box must be skipped."""
    kin = RobotKinematics()
    q_seed, pose_target, psi_home = _target(kin)

    q_default, ok_default, _ = _resolve(kin, q_seed, pose_target, psi_home)
    assert ok_default
    # The unbounded planner selects ψ=175° for this target, whose J1 is below
    # the custom lower bound used below.
    assert q_default[1] < 0.30

    q_lower = kin.q_lower.copy()
    q_upper = kin.q_upper.copy()
    q_lower[1] = 0.30
    q_bounded, ok_bounded, report = _resolve(
        kin,
        q_seed,
        pose_target,
        psi_home,
        q_lower=q_lower,
        q_upper=q_upper,
    )
    assert ok_bounded
    assert report.within_limits
    assert np.all(q_bounded >= q_lower)
    assert np.all(q_bounded <= q_upper)
    # The next high-scoring reachable ψ=170° candidate is selected instead.
    assert q_bounded[1] >= 0.30
    assert not np.allclose(q_bounded, q_default)


def test_all_reachable_candidates_rejected_reports_custom_bounds() -> None:
    kin = RobotKinematics()
    q_seed, pose_target, psi_home = _target(kin)
    q_lower = kin.q_lower.copy()
    q_upper = kin.q_upper.copy()
    # All reachable candidates for this pose have J1 < 0.67 rad.  This is a
    # valid position box, deliberately disjoint from those candidates.
    q_lower[1] = 1.0
    q_upper[1] = 1.1

    with pytest.raises(UnreachablePathError, match="all 70.*custom q_lower/q_upper"):
        _resolve(
            kin,
            q_seed,
            pose_target,
            psi_home,
            q_lower=q_lower,
            q_upper=q_upper,
        )


@pytest.mark.parametrize(
    ("q_lower", "q_upper", "message"),
    [
        (np.zeros(8), None, "provided together"),
        (np.zeros(7), np.ones(7), "shape"),
        (np.full(8, np.nan), np.ones(8), "finite"),
        (np.ones(8), np.ones(8), "strictly less"),
    ],
)
def test_candidate_bounds_are_validated(
    q_lower: np.ndarray,
    q_upper: np.ndarray | None,
    message: str,
) -> None:
    kin = RobotKinematics()
    q_seed, pose_target, psi_home = _target(kin)
    with pytest.raises(ValueError, match=message):
        _resolve(
            kin,
            q_seed,
            pose_target,
            psi_home,
            q_lower=q_lower,
            q_upper=q_upper,
        )

