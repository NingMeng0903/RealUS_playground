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


def test_terminal8_mesh_publish_rejects_neutral_before_loading_capture(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="frozen to gender=male"):
        build_static_track_payload(moment_dir=tmp_path, gender="neutral")


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
