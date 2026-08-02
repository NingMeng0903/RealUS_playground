from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from src.projects.genesis_ue_sync.anatomy_retarget.anatomical_calibration_v1 import (
    load_anatomical_calibration_v1,
)
from src.projects.genesis_ue_sync.anatomy_retarget.dynamic_main_chain_validation_v2 import (
    dynamic_main_chain_regions_v2,
    evaluate_posed_skin_alignment_v2,
)
from src.projects.genesis_ue_sync.anatomy_retarget.pose_adapter import (
    easymocap_fit_to_smplx55,
)
from src.projects.genesis_ue_sync.anatomy_retarget.pose_map_v1 import build_pose_map_v1
from src.projects.genesis_ue_sync.anatomy_retarget.smplx_body_surface_v7 import (
    load_smplx_model_v7,
)
from src.projects.genesis_ue_sync.anatomy_retarget.v8_artifacts import (
    load_source_operator,
    materialize_subject,
)
from src.projects.genesis_ue_sync.anatomy_retarget.whole_chain_rest_fit_v1 import (
    load_whole_chain_rest_fit_v1,
)


ROOT = Path(__file__).resolve().parents[1]
OPERATOR = ROOT / "outputs/anatomy_retarget/v8_candidates/rebuild_012/source_operator_v8"
CALIBRATION = (
    ROOT / "outputs/anatomy_retarget/v8_candidates/chain_retarget_v1_node1_005"
    / "anatomical_calibration_v1"
)
ORACLE = (
    ROOT / "outputs/anatomy_retarget/v7_candidates/blender_link_oracle_v7_full_001"
    / "blender_link_oracle_v7.npz"
)
MODEL = ROOT / "ref_code_library/EasyMocap/data/smplx/smplx/SMPLX_MALE.pkl"
CAPTURE_213712 = (
    ROOT / "smplx_outputs/20260713_213712/moment_0000/smplx_result.npz"
)
FAILED_SUBJECT_213712 = (
    ROOT / "outputs/anatomy_retarget/v8_candidates/chain_retarget_v1_node2_004"
    / "subject_213712"
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def failed_node2_004():
    required = (OPERATOR, CALIBRATION, ORACLE, MODEL, CAPTURE_213712, FAILED_SUBJECT_213712)
    if not all(path.exists() for path in required):
        pytest.skip("frozen V2 failure evidence is unavailable")
    operator = load_source_operator(OPERATOR, mmap=True)
    calibration = load_anatomical_calibration_v1(
        CALIBRATION, operator=operator, required_scope="full_main_chain"
    )
    model = load_smplx_model_v7(MODEL)
    value = load_whole_chain_rest_fit_v1(
        FAILED_SUBJECT_213712,
        operator=operator,
        calibration=calibration,
        smplx_model=model,
        smplx_model_sha256=_sha(MODEL),
        recheck=False,
    )
    asset = materialize_subject(
        operator, betas=value.betas, gender="male"
    ).rigged_asset
    pose_map = build_pose_map_v1(
        value,
        asset=asset,
        calibration=calibration,
        oracle_path=ORACLE,
        source_operator_digest=operator.runtime_digest(validate=False),
    )
    with np.load(CAPTURE_213712, allow_pickle=False) as capture:
        pose = easymocap_fit_to_smplx55(
            capture["Rh"], capture["poses"], model_path=MODEL
        )
    return value, pose_map, asset, model, pose


def test_regions_cover_each_terminal_compound_without_aliasing(failed_node2_004) -> None:
    _value, _pose_map, asset, _model, _pose = failed_node2_004
    regions = dynamic_main_chain_regions_v2(asset)
    assert {
        "left_hand", "right_hand", "left_foot_major", "right_foot_major",
        "left_toe_phalanges", "right_toe_phalanges",
        "left_upper_core", "right_upper_core", "left_lower_core", "right_lower_core",
        "upper_core", "lower_core", "hand", "foot_major", "toe_phalanges",
    } == set(regions)
    assert not np.intersect1d(regions["left_hand"], regions["right_hand"]).size
    assert not np.intersect1d(
        regions["left_foot_major"], regions["right_foot_major"]
    ).size


def test_exact_node2_004_is_rejected_against_posed_male_skin(failed_node2_004) -> None:
    value, pose_map, asset, model, pose = failed_node2_004
    report = evaluate_posed_skin_alignment_v2(
        value,
        pose_map,
        asset=asset,
        smplx_model=model,
        pose_axis_angle=pose,
        label="pose_213712",
    )
    assert report["passed"] is False
    assert report["male_posed_skin_is_absolute_authority"] is True
    assert report["baseline_142_is_report_only"] is False
    assert report["baseline_roles"]["left_lower_core"] == "gate_reference"
    assert report["regions"]["left_hand"]["candidate"]["inside_fraction"] < 0.10
    assert report["regions"]["right_foot_major"]["candidate"]["inside_fraction"] < 0.90
