from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.projects.genesis_ue_sync.anatomy_retarget.anatomical_calibration_v1 import (
    load_anatomical_calibration_v1,
)
from src.projects.genesis_ue_sync.anatomy_retarget.chain_rest_fit_v1 import (
    build_lower_chain_rest_fit_v1,
    check_chain_rest_fit_v1,
    save_chain_rest_fit_v1,
)
from src.projects.genesis_ue_sync.anatomy_retarget.smplx_body_surface_v7 import (
    load_smplx_model_v7,
)
from src.projects.genesis_ue_sync.anatomy_retarget.v8_artifacts import (
    load_source_operator,
)


ROOT = Path(__file__).resolve().parents[1]
OPERATOR_PATH = ROOT / "outputs/anatomy_retarget/v8_candidates/rebuild_012/source_operator_v8"
CALIBRATION_PATH = (
    ROOT
    / "outputs/anatomy_retarget/v8_candidates/chain_retarget_v1_node1_006"
    / "anatomical_calibration_v1"
)
MODEL_PATH = ROOT / "ref_code_library/EasyMocap/data/smplx/smplx/SMPLX_MALE.pkl"
CAPTURES = {
    label: ROOT / f"smplx_outputs/20260713_{label}/moment_0000/smplx_result.npz"
    for label in ("213328", "213712")
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def matrix():
    required = (OPERATOR_PATH, CALIBRATION_PATH, MODEL_PATH, *CAPTURES.values())
    if not all(path.exists() for path in required):
        pytest.skip("frozen Node 2 inputs are unavailable")
    operator = load_source_operator(OPERATOR_PATH, mmap=True)
    calibration = load_anatomical_calibration_v1(
        CALIBRATION_PATH, operator=operator, required_scope="lower_chain"
    )
    model = load_smplx_model_v7(MODEL_PATH)
    model_sha = _sha(MODEL_PATH)
    values = {}
    reports = {}
    for label, capture in CAPTURES.items():
        with np.load(capture, allow_pickle=False) as data:
            betas = np.asarray(data["shapes"]).reshape(-1)[:10]
        value = build_lower_chain_rest_fit_v1(
            operator,
            calibration,
            betas=betas,
            subject_label=label,
            capture_sha256=_sha(capture),
            smplx_model=model,
            smplx_model_sha256=model_sha,
        )
        values[label] = value
        reports[label] = check_chain_rest_fit_v1(
            value,
            operator=operator,
            calibration=calibration,
            smplx_model=model,
            smplx_model_sha256=model_sha,
        )
    return operator, calibration, model, model_sha, values, reports


def test_two_beta_lower_chain_passes_with_bounded_pelvis_cage_and_untouched_tubes(matrix) -> None:
    _operator, _calibration, _model, _model_sha, values, reports = matrix
    for label in CAPTURES:
        value = values[label]
        report = reports[label]
        assert report["passed"] is True, json.dumps(
            {
                "joints": report["joints"],
                "spans": report["spans"],
                "pelvis_cage_shape": report["pelvis_cage_shape"],
                "invariants": report["invariants"],
            },
            indent=2,
            sort_keys=True,
        )
        assert report["build_seconds"] < 30.0
        assert all(report["source_checks"].values())
        assert all(report["exact_checks"].values())
        assert all(item["pass"] for item in report["joints"].values())
        assert all(item["pass"] for item in report["spans"].values())
        assert all(item["pass"] for item in report["rigid_group_metrics"].values())
        assert report["invariants"]["pelvis_cage_bounded"] is True
        assert report["invariants"]["tube_vertices_byte_exact_in_node2"] is True
        assert report["invariants"]["node3_transport_application_count"] == 0
        assert value.build_report["raw_smplx_joint_translation_target"] is False
        assert 0.85 <= value.build_report["pelvis_cage"]["scale_x"] <= 0.98
        assert value.build_report["pelvis_cage"]["radial_bone_scale"] == 1.0
        for side in ("left", "right"):
            scale = value.build_report["centerlines"][side]["shank_axial_scale"]
            assert 0.97 <= scale <= 1.03


def test_body_frame_mapping_is_per_beta_and_centers_each_limb_laterally(matrix) -> None:
    _operator, _calibration, _model, _model_sha, values, reports = matrix
    assert not np.array_equal(
        values["213328"].station_frame_translation,
        values["213712"].station_frame_translation,
    )
    for report in reports.values():
        for span in report["spans"].values():
            assert span["skin_centerline_lateral_max_m"] <= 0.012


def test_complete_save_rechecks_with_frozen_trust_roots(matrix, tmp_path: Path) -> None:
    operator, calibration, model, model_sha, values, _reports = matrix
    output = save_chain_rest_fit_v1(
        tmp_path / "subject",
        values["213328"],
        operator=operator,
        calibration=calibration,
        smplx_model=model,
        smplx_model_sha256=model_sha,
    )
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["complete"] is True
    assert manifest["accepted_scope"] == "lower_chain_shadow"
    assert manifest["checker_report"]["passed"] is True
    assert manifest["publishable"] is False
    assert manifest["trusted_latest_updated"] is False
    assert manifest["vessel_repair_started"] is False


def test_C_bone_tampering_is_rejected(matrix) -> None:
    _operator, _calibration, _model, _model_sha, values, _reports = matrix
    changed = values["213328"].C_bone.copy()
    changed[2, 0, 3] += 1.0
    with pytest.raises(ValueError, match="C_bone"):
        replace(values["213328"], C_bone=changed).validate()


def test_terminal_feet_keep_142_rest_geometry_and_global_bind(matrix) -> None:
    operator, _calibration, _model, _model_sha, values, _reports = matrix
    asset = operator.template_asset
    names = list(asset.source_bone_names or ())
    ranges = np.asarray(asset.source_vertex_ranges, dtype=np.int64)
    foot_controllers = set()
    for suffix in ("L", "R"):
        foot_controllers.update(
            range(names.index(f"Ankle_Rot_{suffix}"), names.index(f"Patella_Rotate_{suffix}"))
        )
    foot_ids = np.concatenate(
        [
            np.arange(int(start), int(stop), dtype=np.int64)
            for controller, tissue, (start, stop) in zip(
                asset.source_mesh_controller_bones,
                asset.source_tissues,
                ranges.tolist(),
            )
            if str(tissue).strip().lower() == "bone"
            and int(controller) in foot_controllers
        ]
    )
    controller_ids = np.asarray(sorted(foot_controllers), dtype=np.int64)
    for value in values.values():
        assert np.array_equal(value.vertices_final[foot_ids], value.vertices_prefit[foot_ids])
        assert np.array_equal(value.B_final[controller_ids], value.B_prefit[controller_ids])
        assert np.array_equal(
            value.C_bone[controller_ids],
            np.tile(np.eye(4), (len(controller_ids), 1, 1)),
        )
