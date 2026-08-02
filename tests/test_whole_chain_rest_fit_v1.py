from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from src.projects.genesis_ue_sync.anatomy_retarget.anatomical_calibration_v1 import (
    load_anatomical_calibration_v1,
)
from src.projects.genesis_ue_sync.anatomy_retarget.smplx_body_surface_v7 import (
    load_smplx_model_v7,
)
from src.projects.genesis_ue_sync.anatomy_retarget.v8_artifacts import (
    load_source_operator,
)
from src.projects.genesis_ue_sync.anatomy_retarget.whole_chain_rest_fit_v1 import (
    build_whole_chain_rest_fit_v1,
    check_whole_chain_rest_fit_v1,
)


ROOT = Path(__file__).resolve().parents[1]
OPERATOR = ROOT / "outputs/anatomy_retarget/v8_candidates/rebuild_012/source_operator_v8"
CALIBRATION = (
    ROOT / "outputs/anatomy_retarget/v8_candidates/chain_retarget_v1_node1_005"
    / "anatomical_calibration_v1"
)
MODEL = ROOT / "ref_code_library/EasyMocap/data/smplx/smplx/SMPLX_MALE.pkl"
CAPTURES = {
    label: ROOT / f"smplx_outputs/20260713_{label}/moment_0000/smplx_result.npz"
    for label in ("213328", "213712")
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def matrix():
    if not all(path.exists() for path in (OPERATOR, CALIBRATION, MODEL, *CAPTURES.values())):
        pytest.skip("frozen whole-chain inputs are unavailable")
    operator = load_source_operator(OPERATOR, mmap=True)
    calibration = load_anatomical_calibration_v1(
        CALIBRATION, operator=operator, required_scope="full_main_chain"
    )
    model = load_smplx_model_v7(MODEL)
    model_sha = _sha(MODEL)
    values = {}
    reports = {}
    for label, path in CAPTURES.items():
        with np.load(path, allow_pickle=False) as data:
            betas = np.asarray(data["shapes"]).reshape(-1)[:10]
        value = build_whole_chain_rest_fit_v1(
            operator,
            calibration,
            betas=betas,
            subject_label=label,
            capture_sha256=_sha(path),
            smplx_model=model,
            smplx_model_sha256=model_sha,
        )
        values[label] = value
        reports[label] = check_whole_chain_rest_fit_v1(
            value,
            operator=operator,
            calibration=calibration,
            smplx_model=model,
            smplx_model_sha256=model_sha,
        )
    return values, reports


def test_two_beta_whole_chain_uses_one_sparse_lbs_authority(matrix) -> None:
    _values, reports = matrix
    for report in reports.values():
        assert report["passed"] is True, json.dumps(report, indent=2, sort_keys=True)
        assert report["build_seconds"] < 30.0
        assert report["zero_pose_reproduction"]["rms_m"] <= 1.0e-6
        assert report["zero_pose_reproduction"]["max_m"] <= 1.0e-5
        assert all(report["exact_checks"].values())
        assert report["invariants"]["protected_girdles_byte_exact"] is True
        assert report["invariants"]["pelvis_cage_bounded"] is True
        assert report["invariants"]["tube_rest_transport_exact"] is True
        assert report["invariants"]["tube_transport_application_count"] == 1
        for cap in report["upper_rigid_caps"].values():
            assert cap["pass"] is True
            assert cap["kabsch_rms_m"] <= 0.0005
            assert cap["kabsch_max_m"] <= 0.001
            assert max(abs(scale - 1.0) for scale in cap["radial_scales"]) <= 1.0e-4


def test_upper_anatomical_targets_are_independent_hard_gates(matrix) -> None:
    _values, reports = matrix
    for report in reports.values():
        for metric in report["upper_joints"].values():
            assert metric["pass"] is True
            assert metric["mapped_station_to_axis_m"] <= metric["limit_m"]
            assert "mapped_raw_station_to_axis_m" in metric
            assert "mapped_frozen_offset_target_to_axis_m" in metric


def test_tube_transport_is_applied_once_to_candidate_rest(matrix) -> None:
    _values, reports = matrix
    for report in reports.values():
        transport = report["tube_rest_transport"]
        assert transport["application_count"] == 1
        assert transport["persisted_to_candidate"] is True
        assert transport["max_displacement_m"] > 0.0


def test_terminal_hands_keep_142_rest_geometry_and_global_bind(matrix) -> None:
    values, _reports = matrix
    operator = load_source_operator(OPERATOR, mmap=True)
    asset = operator.template_asset
    names = list(asset.source_bone_names or ())
    parents = np.asarray(asset.source_bone_parents, dtype=np.int64)
    ranges = np.asarray(asset.source_vertex_ranges, dtype=np.int64)

    def descendants(root: int) -> set[int]:
        result = {root}
        while True:
            expanded = result | {
                index for index, parent in enumerate(parents.tolist())
                if int(parent) in result
            }
            if expanded == result:
                return result
            result = expanded

    hand_controllers = set()
    for wrist_name in ("Wrist_Rotate_L", "Wrist_Rotate_R1"):
        hand_controllers.update(descendants(names.index(wrist_name)))
    hand_ids = np.concatenate(
        [
            np.arange(int(start), int(stop), dtype=np.int64)
            for controller, tissue, (start, stop) in zip(
                asset.source_mesh_controller_bones,
                asset.source_tissues,
                ranges.tolist(),
            )
            if str(tissue).strip().lower() == "bone"
            and int(controller) in hand_controllers
        ]
    )
    controller_ids = np.asarray(sorted(hand_controllers), dtype=np.int64)
    for value in values.values():
        assert np.array_equal(value.vertices_final[hand_ids], value.vertices_prefit[hand_ids])
        assert np.array_equal(value.B_final[controller_ids], value.B_prefit[controller_ids])
        assert np.array_equal(
            value.C_bone[controller_ids],
            np.tile(np.eye(4), (len(controller_ids), 1, 1)),
        )
        hand_policies = {
            str(policy)
            for policy in value.mesh_policy.tolist()
            if str(policy) == "copy_142_terminal_hand"
        }
        assert hand_policies == {"copy_142_terminal_hand"}
