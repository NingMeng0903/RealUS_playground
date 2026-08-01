from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from src.projects.genesis_ue_sync.anatomy_retarget.anatomical_calibration_v1 import (
    load_anatomical_calibration_v1,
)
from src.projects.genesis_ue_sync.anatomy_retarget.chain_containment_v1 import (
    evaluate_rest_containment_v1,
)
from src.projects.genesis_ue_sync.anatomy_retarget.smplx_body_surface_v7 import (
    load_smplx_model_v7,
    smplx_body_surface_v7,
)
from src.projects.genesis_ue_sync.anatomy_retarget.v8_artifacts import (
    load_source_operator,
    materialize_subject,
)
from src.projects.genesis_ue_sync.anatomy_retarget.whole_chain_rest_fit_v1 import (
    build_whole_chain_rest_fit_v1,
)


ROOT = Path(__file__).resolve().parents[1]
OPERATOR = ROOT / "outputs/anatomy_retarget/v8_candidates/rebuild_012/source_operator_v8"
CALIBRATION = (
    ROOT / "outputs/anatomy_retarget/v8_candidates/chain_retarget_v1_node1_004"
    / "anatomical_calibration_v1"
)
MODEL = ROOT / "ref_code_library/EasyMocap/data/smplx/smplx/SMPLX_MALE.pkl"
CAPTURE = ROOT / "smplx_outputs/20260713_213328/moment_0000/smplx_result.npz"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_terminal_containment_is_exactly_restored_to_142() -> None:
    if not all(path.exists() for path in (OPERATOR, CALIBRATION, MODEL, CAPTURE)):
        pytest.skip("frozen containment inputs are unavailable")
    operator = load_source_operator(OPERATOR, mmap=True)
    calibration = load_anatomical_calibration_v1(
        CALIBRATION, operator=operator, required_scope="lower_chain"
    )
    model = load_smplx_model_v7(MODEL)
    with np.load(CAPTURE, allow_pickle=False) as data:
        betas = np.asarray(data["shapes"]).reshape(-1)[:10]
    value = build_whole_chain_rest_fit_v1(
        operator,
        calibration,
        betas=betas,
        subject_label="213328",
        capture_sha256=_sha(CAPTURE),
        smplx_model=model,
        smplx_model_sha256=_sha(MODEL),
    )
    asset = materialize_subject(operator, betas=betas, gender="male").rigged_asset
    skin, faces = smplx_body_surface_v7(
        model, betas=betas, pose_axis_angle=np.zeros((55, 3), dtype=np.float64)
    )
    report = evaluate_rest_containment_v1(
        value, asset=asset, skin_vertices=skin, skin_faces=faces
    )
    assert report["skin_frame_translation_applied"] is False
    assert report["terminal_rest_byte_exact"] == {
        "terminal_hand": True,
        "terminal_foot": True,
    }
    for name in ("terminal_hand", "terminal_foot"):
        metric = report["regions"][name]
        assert metric["inside_fraction_delta"] == 0.0
        assert abs(metric["max_outside_regression_m"]) <= 1.0e-12
