import json
from pathlib import Path

import numpy as np
import pytest

from projects.genesis_ue_sync.anatomy_retarget.smplx_body_surface_v7 import (
    FROZEN_SMPLX_MALE_SHA256,
    _smplx_joint_kinematics_v7,
    load_smplx_model_v7,
    require_frozen_smplx_male_v7,
)
from projects.genesis_ue_sync.anatomy_retarget.whole_chain_rest_fit_v1 import (
    INVALIDATED_SMPLX_MODEL_SHA256,
    load_whole_chain_rest_fit_v1,
)
from projects.genesis_ue_sync.anatomy_retarget.pose_adapter import (
    easymocap_fit_to_smplx55,
    smplx_pose_hash,
    smplx_shape_hash,
)
from projects.genesis_ue_sync.multiview_realtime.publish.static_smplx_track import (
    build_static_track_payload,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "ref_code_library/EasyMocap/data/smplx/smplx"


def test_frozen_male_model_is_authenticated() -> None:
    path, digest = require_frozen_smplx_male_v7(MODEL_ROOT / "SMPLX_MALE.pkl")
    assert path.name == "SMPLX_MALE.pkl"
    assert digest == FROZEN_SMPLX_MALE_SHA256


def test_neutral_model_is_rejected_for_retarget_review() -> None:
    with pytest.raises(ValueError, match="requires the frozen SMPLX_MALE.pkl"):
        require_frozen_smplx_male_v7(MODEL_ROOT / "SMPLX_NEUTRAL.pkl")


def test_review_station_offset_follows_smplx_joint_pose() -> None:
    model = load_smplx_model_v7(MODEL_ROOT / "SMPLX_MALE.pkl")
    pose = np.zeros((55, 3), dtype=np.float64)
    pose[16, 0] = np.radians(70.0)
    rest, posed, rest_to_pose = _smplx_joint_kinematics_v7(
        model, betas=np.zeros(10), pose_axis_angle=pose
    )
    station_rest = rest[18] + np.asarray((0.0, 0.012, 0.0))
    station_posed = rest_to_pose[18] @ np.append(station_rest, 1.0)
    assert not np.allclose(station_posed[:3], station_rest, atol=1.0e-4)
    assert np.linalg.norm(station_posed[:3] - posed[18, :3, 3]) == pytest.approx(
        0.012, abs=1.0e-9
    )


def test_terminal8_mesh_publish_accepts_neutral_capture_payload(tmp_path: Path) -> None:
    moment = tmp_path / "moment_0000"
    moment.mkdir(parents=True)
    source_vertices = np.asarray(
        [[0.0, 0.1, 0.2], [1.0, 1.1, 1.2], [2.0, 2.1, 2.2], [3.0, 3.1, 3.2]],
        dtype=np.float32,
    )
    source_faces = np.asarray([[0, 1, 2], [1, 2, 3]], dtype=np.int32)
    source_rh = np.asarray([0.01, -0.02, 0.03], dtype=np.float32)
    source_poses = np.linspace(-0.4, 0.4, 165, dtype=np.float32)
    source_poses[:3] = source_rh
    source_shapes = np.linspace(-0.2, 0.2, 10, dtype=np.float32)
    source_th = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
    source_root_offset = np.asarray([0.1, -0.2, 0.3], dtype=np.float32)
    source_pelvis = np.asarray([1.5, 2.5, 3.5], dtype=np.float32)
    result_path = moment / "smplx_result.npz"
    np.savez(
        result_path,
        Rh=source_rh,
        Th=source_th,
        poses=source_poses,
        shapes=source_shapes,
        root_align_offset=source_root_offset,
        pelvis_world_m=source_pelvis,
        vertices=source_vertices,
        faces=source_faces,
    )

    with np.load(result_path) as result:
        payload = build_static_track_payload(moment_dir=moment, gender="neutral")
        expected_pose = easymocap_fit_to_smplx55(result["Rh"], result["poses"], gender="neutral")
        expected_betas = np.asarray(result["shapes"], dtype=np.float32).reshape(-1)[:10]

        assert payload["gender"] == "neutral"
        assert len(payload["vertices"]) == len(source_vertices)
        assert len(payload["faces"]) == len(source_faces)
        assert len(payload["poses"]) == 165
        assert len(payload["smplx_pose_aa_165"]) == 165
        assert payload["shape_hash"] == smplx_shape_hash(expected_betas, gender="neutral")
        assert payload["pose_hash"] == smplx_pose_hash(expected_pose, payload["Th"])
        np.testing.assert_allclose(payload["vertices"], source_vertices, rtol=0.0, atol=0.0)
        np.testing.assert_array_equal(payload["faces"], source_faces)
        np.testing.assert_allclose(payload["poses"], source_poses, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(payload["Th"], source_th + source_root_offset, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(payload["pelvis_world_m"], source_pelvis, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(payload["smplx_pose_aa_165"], expected_pose.reshape(-1), rtol=0.0, atol=0.0)

    # Match the JSON metadata decode performed by TrackPoseSubscriber._recv_loop.
    decoded = json.loads(json.dumps(payload))
    assert str(decoded["gender"]) == "neutral"
    native_pose = np.asarray(decoded["smplx_pose_aa_165"], dtype=np.float32).reshape(-1)
    assert native_pose.size == 165
    assert decoded["shape_hash"] == payload["shape_hash"]
    assert decoded["pose_hash"] == payload["pose_hash"]


def test_whole_chain_loader_rejects_invalidated_neutral_before_npz(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "neutral_artifact"
    artifact.mkdir()
    neutral_sha = next(iter(INVALIDATED_SMPLX_MODEL_SHA256))
    (artifact / "manifest.json").write_text(
        '{"smplx_model_sha256":"' + neutral_sha + '"}', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="explicitly invalidated neutral model"):
        load_whole_chain_rest_fit_v1(
            artifact,
            operator=None,  # Rejection occurs before any trust-root dereference.
            calibration=None,
            smplx_model={},
            smplx_model_sha256=FROZEN_SMPLX_MALE_SHA256,
        )
