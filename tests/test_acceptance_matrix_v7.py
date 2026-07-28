from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from projects.genesis_ue_sync.anatomy_retarget.acceptance_matrix_v7 import (
    _json_ready,
    load_matrix_pose_v7,
    load_matrix_subject_v7,
    parse_label_value_pair,
    run_acceptance_matrix_v7,
    synthetic_knee_sweep_poses_v7,
)
from projects.genesis_ue_sync.anatomy_retarget.joint_contact_v7 import (
    FrozenJointMaterialDomainsV7,
)
from projects.genesis_ue_sync.anatomy_retarget.patella_oracle_v7 import (
    load_patella_oracle_v7,
)


_SUBJECT = Path(
    "outputs/anatomy_retarget/v7_candidates/rebuild_003/subject_213328.npz"
)
_DOMAINS = Path(
    "outputs/anatomy_retarget/v7_candidates/joint_rebuild_001/fixed_joint_domains_v7.json"
)
_ORACLE = Path(
    "outputs/anatomy_retarget/v7_candidates/rebuild_003/patella_oracle_v7.npz"
)


def test_load_matrix_pose_zero_and_malformed(tmp_path: Path) -> None:
    pose = load_matrix_pose_v7("tpose", "zero")
    assert pose.label == "tpose"
    assert pose.pose_axis_angle.shape == (55, 3)
    assert pose.transl.shape == (3,)
    assert np.allclose(pose.pose_axis_angle, 0.0)
    assert np.allclose(pose.transl, 0.0)
    assert pose.source == "synthetic"

    bad = tmp_path / "bad_pose.npz"
    np.savez(bad, pose_axis_angle=np.zeros((10, 3), dtype=np.float32))
    with pytest.raises(ValueError):
        load_matrix_pose_v7("bad", bad)


def test_synthetic_knee_sweep_poses() -> None:
    poses = synthetic_knee_sweep_poses_v7(count=5)
    assert len(poses) == 5
    assert np.allclose(poses[0].pose_axis_angle, 0.0)
    magnitudes = [
        float(np.linalg.norm(pose.pose_axis_angle[4])) for pose in poses
    ]
    assert magnitudes == sorted(magnitudes)
    for pose in poses:
        touched = np.flatnonzero(np.any(np.abs(pose.pose_axis_angle) > 0.0, axis=1))
        assert set(touched.tolist()).issubset({4, 5})
        assert np.allclose(pose.pose_axis_angle[4], pose.pose_axis_angle[5])


def test_parse_label_value_pair() -> None:
    assert parse_label_value_pair("a=b/c.npz") == ("a", "b/c.npz")
    with pytest.raises(ValueError):
        parse_label_value_pair("abc")


def test_json_ready_numpy_types() -> None:
    payload = {
        "i": np.int64(3),
        "f": np.float32(1.5),
        "b": np.bool_(True),
        "a": np.asarray([1.0, 2.0], dtype=np.float64),
    }
    ready = _json_ready(payload)
    json.dumps(ready)
    assert ready == {"i": 3, "f": 1.5, "b": True, "a": [1.0, 2.0]}


@pytest.mark.skipif(
    not (_SUBJECT.is_file() and _DOMAINS.is_file() and _ORACLE.is_file()),
    reason="rebuild_003 acceptance inputs are not present",
)
def test_run_acceptance_matrix_is_fail_closed() -> None:
    subjects = [load_matrix_subject_v7("213328", _SUBJECT)]
    poses = [load_matrix_pose_v7("tpose", "zero")]
    domains = FrozenJointMaterialDomainsV7.load_json(_DOMAINS)
    law = load_patella_oracle_v7(_ORACLE)
    report = run_acceptance_matrix_v7(
        subjects=subjects,
        poses=poses,
        domains=domains,
        law=law,
        sweep_count=3,
    )
    json.dumps(_json_ready(report))
    assert len(report["cells"]) == 1
    cell = report["cells"]["213328/tpose"]
    for gate in (
        "controller",
        "local_fk",
        "local_fk_arms",
        "geometry",
        "vessel",
        "compound",
    ):
        assert gate in cell
    # The arm links were once measured, reported, and then left out of the
    # verdict, so a cell could claim passed=true with every elbow link
    # unavailable. Every reported gate must reach both the failure list and the
    # conjunction.
    for gate in ("local_fk", "local_fk_arms"):
        if not cell[gate]["pass"]:
            assert any(
                name.startswith(f"{gate}.") for name in cell["failures"]
            ), gate
            assert cell["passed"] is False
    # The oracle digest cannot be verified without the action export, so the
    # matrix must refuse to pass even if every measured gate were clean.
    assert report["patella_oracle"]["action_digest_verified"] is False
    assert report["passed"] is False
    assert report["publishable"] is False
