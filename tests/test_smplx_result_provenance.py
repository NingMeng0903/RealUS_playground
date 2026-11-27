from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from projects.genesis_ue_sync.multiview_realtime.easymocap.moment_pipeline import (
    _smplx_result_provenance,
)
from projects.genesis_ue_sync.multiview_realtime.publish.static_smplx_track import (
    build_static_track_payload,
)


def _write_result(moment: Path, *, gender: str | None) -> None:
    moment.mkdir(parents=True, exist_ok=True)
    payload = {
        "Rh": np.zeros(3, dtype=np.float32),
        "Th": np.zeros(3, dtype=np.float32),
        "poses": np.zeros(165, dtype=np.float32),
        "shapes": np.zeros(10, dtype=np.float32),
        "vertices": np.zeros((3, 3), dtype=np.float32),
        "faces": np.asarray([[0, 1, 2]], dtype=np.int32),
    }
    if gender is not None:
        payload.update(
            _smplx_result_provenance(
                gender=gender,
                fit_model="smplx",
                model_type="smplx",
            )
        )
    np.savez(moment / "smplx_result.npz", **payload)


def test_result_provenance_stores_normalized_strings() -> None:
    metadata = _smplx_result_provenance(
        gender=" FEMALE ", fit_model=" EasyMocap ", model_type=" SMPL-X "
    )
    assert {key: str(value.item()) for key, value in metadata.items()} == {
        "gender": "female",
        "fit_model": "easymocap",
        "model_type": "smpl-x",
    }


def test_mesh_publisher_rejects_cli_gender_mismatch(tmp_path: Path) -> None:
    moment = tmp_path / "moment_0000"
    _write_result(moment, gender="female")
    with pytest.raises(ValueError, match="result='female'.*requested='male'"):
        build_static_track_payload(moment_dir=moment, gender="male")


def test_mesh_publisher_accepts_matching_and_legacy_gender(tmp_path: Path) -> None:
    stamped = tmp_path / "stamped"
    _write_result(stamped, gender="female")
    payload = build_static_track_payload(moment_dir=stamped, gender="female")
    assert payload["gender"] == "female"

    legacy = tmp_path / "legacy"
    _write_result(legacy, gender=None)
    legacy_payload = build_static_track_payload(moment_dir=legacy, gender="neutral")
    assert legacy_payload["gender"] == "neutral"

