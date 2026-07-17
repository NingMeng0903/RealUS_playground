from __future__ import annotations

import numpy as np

from projects.genesis_ue_sync.anatomy_retarget.validation_matrix import (
    beta_cases,
    pose_cases,
    release_validation_matrix,
)


def _joint_names() -> list[str]:
    names = [
        "pelvis",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
    ]
    names.extend(
        f"{side}_{finger}{level}"
        for side in ("left", "right")
        for finger in ("thumb", "index", "middle", "ring", "pinky")
        for level in (1, 2, 3)
    )
    return names


def test_beta_matrix_contains_zero_real_and_bilateral_extremes() -> None:
    cases = beta_cases(np.asarray((0.2, -0.4, 0.1)), principal_dimensions=2)
    assert set(cases) == {
        "beta_zero",
        "beta_real",
        "beta_00_plus_2sigma",
        "beta_00_minus_2sigma",
        "beta_01_plus_2sigma",
        "beta_01_minus_2sigma",
    }
    assert cases["beta_00_plus_2sigma"][0] == 2.0
    assert cases["beta_00_minus_2sigma"][0] == -2.0


def test_pose_matrix_exercises_twist_and_all_fingers() -> None:
    names = _joint_names()
    cases = pose_cases(names)
    assert np.linalg.norm(cases["pose_axial_twist"][names.index("left_hip")]) > 0.0
    finger = cases["pose_finger_flex"]
    assert np.count_nonzero(np.linalg.norm(finger, axis=1)) == 30
    matrix = release_validation_matrix(
        np.zeros(3, dtype=np.float32),
        names,
        principal_dimensions=1,
    )
    assert len(matrix) == 4 * len(cases)
