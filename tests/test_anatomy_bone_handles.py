from __future__ import annotations

import numpy as np
import pytest

from projects.genesis_ue_sync.anatomy_retarget.bone_handles import (
    build_internal_joint_handles,
)


def test_internal_handles_encode_subject_joint_displacement() -> None:
    names = ["pelvis", "left_hip", "right_hip"]
    neutral = np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)))
    subject = neutral + np.asarray(((0.0, 0.1, 0.0), (0.2, 0.0, 0.0), (-0.2, 0.0, 0.0)))
    handles = build_internal_joint_handles(
        names,
        neutral,
        subject,
        selected_names=("pelvis", "left_hip", "right_hip"),
    )
    np.testing.assert_allclose(handles.points, neutral)
    np.testing.assert_allclose(handles.displacements, subject - neutral)
    assert handles.cache_payload()["names"] == names


def test_internal_handles_fail_on_missing_joint() -> None:
    with pytest.raises(ValueError, match="missing internal handle"):
        build_internal_joint_handles(
            ["pelvis"],
            np.zeros((1, 3)),
            np.zeros((1, 3)),
            selected_names=("pelvis", "head"),
        )
