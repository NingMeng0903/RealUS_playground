from pathlib import Path

import pytest

from projects.genesis_ue_sync.anatomy_retarget.smplx_body_surface_v7 import (
    FROZEN_SMPLX_MALE_SHA256,
    require_frozen_smplx_male_v7,
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


def test_terminal8_mesh_publish_rejects_neutral_before_loading_capture(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="frozen to gender=male"):
        build_static_track_payload(moment_dir=tmp_path, gender="neutral")
