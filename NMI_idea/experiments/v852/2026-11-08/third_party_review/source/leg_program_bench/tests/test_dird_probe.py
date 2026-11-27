"""Integration checks using the installed frozen field and actual robot URDF."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dird_probe import CHECKPOINT, DirdProbe, convert_image_poses, dependency_audit


@pytest.fixture(scope="module")
def probe():
    if not CHECKPOINT.exists():
        pytest.skip("installed checkpoint is not available")
    return DirdProbe(allow_stale_audit=True)


@pytest.fixture(scope="module")
def fixture_path(probe):
    # A fixed explicit virtual entry from the controller's nominal arm angles.
    # The small path is produced by current URDF FK, not a hidden vessel curve.
    q = np.r_[.4, np.deg2rad([-89.5, -94.5, 65.2, 96, 89.3, 61, 94.6])]
    kin = probe.kinematics()
    path = []
    for delta in (0., .002, .004, .006):
        qi = q.copy()
        qi[1] += delta
        pose = kin.fk_placement(qi)
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = pose.rotation, pose.translation
        path.append(T)
    return np.asarray(path), q


def test_current_legacy_checkpoint_is_not_silently_accepted():
    with pytest.raises((ValueError, RuntimeError), match="incompatible|mismatch|missing"):
        DirdProbe()


def test_contract_audit_is_explicit_and_json_serializable(probe):
    meta = probe.metadata()
    assert meta["allow_stale_audit"] is True
    assert meta["ready_for_execution"] is False
    assert meta["physical_registration_verified"] is False
    assert meta["status"] == "score_only"
    assert "kinematics_urdf_sha256" in meta["robot_contract_differences"]
    json.dumps(meta, allow_nan=False)


def test_world_frame_change_preserves_dird_rank_and_scores(probe, fixture_path):
    path, _ = fixture_path
    base = np.eye(4)
    change = np.eye(4)
    change[:3, :3] = Rotation.from_euler("xyz", [.2, -.3, .5]).as_matrix()
    change[:3, 3] = [.7, -.2, .4]
    r1 = probe.rank_modes(path, base, [.2, .4, .6])
    r2 = probe.rank_modes(change @ path, change @ base, [.2, .4, .6])
    by1 = {x["mode_id"]: x for x in r1["modes"]}
    by2 = {x["mode_id"]: x for x in r2["modes"]}
    assert [x["mode_id"] for x in r1["modes"]] == [x["mode_id"] for x in r2["modes"]]
    for key in by1:
        np.testing.assert_allclose(by1[key]["waypoint_scores"], by2[key]["waypoint_scores"], atol=2e-4)


def test_out_of_range_rail_is_rejected_without_clipping(probe, fixture_path):
    path, _ = fixture_path
    result = probe.rank_modes(path, np.eye(4), [-.001, .4, .801])
    assert [m["input_index"] for m in result["modes"]] == [1]
    assert [m["input_index"] for m in result["rejected"]] == [0, 2]
    assert "min_score" not in result["rejected"][0]
    assert result["modes"][0]["validated_reachable"] is False


def test_explicit_image_tcp_bridge_rejects_reflection():
    image = np.eye(4)[None]
    reflected = np.diag([1., 1., -1., 1.])
    with pytest.raises(ValueError, match="SO\\(3\\)"):
        convert_image_poses(image, reflected)
    correction = np.eye(4)
    correction[:3, :3] = Rotation.from_euler("x", np.pi).as_matrix()
    correction[:3, 3] = [.001, .002, .003]
    np.testing.assert_allclose(convert_image_poses(image, correction)[0], correction)


def test_srs_lift_recovers_full_pose_without_execution_claim(probe, fixture_path):
    path, q = fixture_path
    result = probe.audit_ik(path, np.eye(4), .4, q)
    assert result["lift_valid"], result
    assert result["max_fk_position_error_m"] < .0005, result
    assert result["max_fk_rotation_error_rad"] < np.deg2rad(1), result
    assert result["joint_limits_passed"], result
    assert result["collision_samples"] > len(path)
    assert np.isfinite(result["minimum_self_collision_distance_m"])
    assert result["ready_for_execution"] is False
    assert result["world_collision_checked"] is False
    assert result["transition_from_entry_checked"] is False
    assert result["retiming_is_measured_duration"] is False
    if result["kinematic_audit_passed"]:
        assert result["nominal_duration_s"] > 0
        assert np.all(np.diff(result["nominal_timestamps_s"]) > 0)
    json.dumps(result, allow_nan=False)


def test_actual_model_construction_works_with_coal_even_without_hppfcl():
    deps = dependency_audit()
    assert deps["pinocchio"]["available"]
    assert deps["robot_model_construction"]["available"], deps
    assert deps["robot_model_construction"]["collision_pairs"] > 0
    assert deps["sqp_available"] is deps["proxsuite"]["available"]
