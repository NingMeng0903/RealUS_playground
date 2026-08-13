from __future__ import annotations

import numpy as np
import pytest

from projects.genesis_ue_sync.anatomy_retarget.deep_flex_poses_v12 import (
    HINGES_V12,
    build_deep_flex_poses_v12,
    hinge_angle_deg,
    measure_hinges_deg,
    mirror_axis_angle,
    solve_hinge_magnitude,
    verify_deep_flex_poses_v12,
)


_SEGMENT = np.asarray([0.0, 0.0, -0.4], dtype=np.float64)


def _rotation_x(angle_rad: float) -> np.ndarray:
    cos, sin = np.cos(angle_rad), np.sin(angle_rad)
    return np.asarray(
        [[1.0, 0.0, 0.0], [0.0, cos, -sin], [0.0, sin, cos]], dtype=np.float64
    )


def _fake_joints_of(pose: np.ndarray) -> np.ndarray:
    """Two-link chain per hinge whose angle equals the rotation magnitude.

    Lets the solver be tested without loading SMPL-X: the anatomical angle is
    exactly ``|axis_angle|``, so a 120 deg target must come back as 2.094 rad.
    """

    values = np.asarray(pose, dtype=np.float64).reshape(55, 3)
    joints = np.zeros((55, 3), dtype=np.float64)
    for index, spec in enumerate(HINGES_V12.values()):
        origin = np.asarray([0.2 * index, 0.0, 0.0], dtype=np.float64)
        joints[spec.root] = origin
        joints[spec.joint] = origin + _SEGMENT
        angle = float(np.linalg.norm(values[spec.joint]))
        joints[spec.tip] = joints[spec.joint] + _rotation_x(angle) @ _SEGMENT
    return joints


def _captures() -> dict[str, np.ndarray]:
    """Donor axes long enough to pass the conditioning check."""

    pose_213328 = np.zeros((55, 3), dtype=np.float64)
    pose_213328[HINGES_V12["knee_L"].joint] = [0.6, 0.55, -1.43]
    pose_213712 = np.zeros((55, 3), dtype=np.float64)
    pose_213712[HINGES_V12["elbow_L"].joint] = [0.54, -0.5, -0.11]
    pose_213712[HINGES_V12["elbow_R"].joint] = [0.52, 0.48, 0.13]
    return {"213328": pose_213328, "213712": pose_213712}


def test_hinge_angle_is_zero_when_extended_and_ninety_when_folded() -> None:
    spec = HINGES_V12["knee_L"]
    straight = np.zeros((55, 3), dtype=np.float64)
    straight[spec.root] = [0.0, 0.0, 0.4]
    straight[spec.joint] = [0.0, 0.0, 0.0]
    straight[spec.tip] = [0.0, 0.0, -0.4]
    assert hinge_angle_deg(straight, spec) == pytest.approx(0.0, abs=1.0e-9)

    folded = straight.copy()
    folded[spec.tip] = [0.0, 0.4, 0.0]
    assert hinge_angle_deg(folded, spec) == pytest.approx(90.0, abs=1.0e-9)


def test_mirror_flips_the_off_sagittal_components() -> None:
    assert mirror_axis_angle([0.6, 0.55, -1.43]) == pytest.approx(
        [0.6, -0.55, 1.43]
    )


def test_solver_hits_the_anatomical_target_not_the_axis_angle_norm() -> None:
    axis = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    magnitude, achieved = solve_hinge_magnitude(
        "knee_L", target_deg=120.0, axis=axis, joints_of=_fake_joints_of
    )

    assert achieved == pytest.approx(120.0, abs=0.5)
    # This chain rotates exactly by the axis-angle magnitude, so the solved
    # magnitude must land within the solver's own 0.5 deg angular tolerance.
    assert magnitude == pytest.approx(np.deg2rad(120.0), abs=np.deg2rad(0.5))


def test_solver_refuses_an_unreachable_target() -> None:
    with pytest.raises(ValueError, match="cannot reach"):
        solve_hinge_magnitude(
            "knee_L",
            target_deg=200.0,
            axis=np.asarray([1.0, 0.0, 0.0]),
            joints_of=_fake_joints_of,
        )


def test_built_poses_cover_the_gaps_the_captures_leave() -> None:
    poses = build_deep_flex_poses_v12(
        captures=_captures(), joints_of=_fake_joints_of, target_deg=120.0
    )

    assert set(poses) == {
        "flex_knee_R_120",
        "flex_knee_both_120",
        "flex_knee_elbow_120",
    }

    # The deep RIGHT knee is the case no frozen capture reaches.
    right_only = measure_hinges_deg(
        poses["flex_knee_R_120"], joints_of=_fake_joints_of
    )
    assert right_only["knee_R"] == pytest.approx(120.0, abs=1.0)
    assert right_only["knee_L"] == pytest.approx(0.0, abs=1.0)

    # Knee and elbow deep simultaneously is the other uncovered case.
    combined = measure_hinges_deg(
        poses["flex_knee_elbow_120"], joints_of=_fake_joints_of
    )
    for hinge in ("knee_L", "knee_R", "elbow_L", "elbow_R"):
        assert combined[hinge] == pytest.approx(120.0, abs=1.0)


def test_verification_passes_on_built_poses() -> None:
    poses = build_deep_flex_poses_v12(
        captures=_captures(), joints_of=_fake_joints_of, target_deg=120.0
    )
    report = verify_deep_flex_poses_v12(poses, joints_of=_fake_joints_of)

    assert report["passed"] is True
    assert report["failures"] == []


def test_verification_fails_closed_when_a_hinge_falls_short() -> None:
    poses = build_deep_flex_poses_v12(
        captures=_captures(), joints_of=_fake_joints_of, target_deg=120.0
    )
    tampered = dict(poses)
    half = np.asarray(poses["flex_knee_both_120"], dtype=np.float64).copy()
    half[HINGES_V12["knee_R"].joint] *= 0.5
    tampered["flex_knee_both_120"] = half

    report = verify_deep_flex_poses_v12(tampered, joints_of=_fake_joints_of)

    assert report["passed"] is False
    assert [f["hinge"] for f in report["failures"]] == ["knee_R"]


def test_short_donor_axis_is_rejected_rather_than_guessed() -> None:
    captures = _captures()
    captures["213328"][HINGES_V12["knee_L"].joint] = [0.01, 0.0, 0.0]

    with pytest.raises(ValueError, match="too short to trust"):
        build_deep_flex_poses_v12(captures=captures, joints_of=_fake_joints_of)
