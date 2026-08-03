from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from src.projects.genesis_ue_sync.anatomy_retarget.anatomical_calibration_v1 import (
    load_anatomical_calibration_v1,
)
from src.projects.genesis_ue_sync.anatomy_retarget.dynamic_chain_validation_v1 import (
    run_dynamic_chain_validation_v1,
    synthetic_chain_sweeps_v1,
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
    build_whole_chain_rest_fit_v1,
)


ROOT = Path(__file__).resolve().parents[1]
OPERATOR = ROOT / "outputs/anatomy_retarget/v8_candidates/rebuild_012/source_operator_v8"
CALIBRATION = (
    ROOT / "outputs/anatomy_retarget/v8_candidates/chain_retarget_v1_node1_006"
    / "anatomical_calibration_v1"
)
ORACLE = (
    ROOT / "outputs/anatomy_retarget/v7_candidates/blender_link_oracle_v7_full_001"
    / "blender_link_oracle_v7.npz"
)
MODEL = ROOT / "ref_code_library/EasyMocap/data/smplx/smplx/SMPLX_MALE.pkl"
CAPTURES = {
    label: ROOT / f"smplx_outputs/20260713_{label}/moment_0000/smplx_result.npz"
    for label in ("213328", "213712")
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def reports():
    if not all(path.exists() for path in (OPERATOR, CALIBRATION, ORACLE, MODEL, *CAPTURES.values())):
        pytest.skip("frozen dynamic matrix inputs are unavailable")
    operator = load_source_operator(OPERATOR, mmap=True)
    calibration = load_anatomical_calibration_v1(
        CALIBRATION, operator=operator, required_scope="full_main_chain"
    )
    model = load_smplx_model_v7(MODEL)
    model_sha = _sha(MODEL)
    recorded = {}
    betas = {}
    for label, capture in CAPTURES.items():
        with np.load(capture, allow_pickle=False) as data:
            betas[label] = np.asarray(data["shapes"]).reshape(-1)[:10]
            recorded[f"pose_{label}"] = easymocap_fit_to_smplx55(
                data["Rh"], data["poses"], model_path=MODEL
            )
    result = {}
    for label in CAPTURES:
        value = build_whole_chain_rest_fit_v1(
            operator,
            calibration,
            betas=betas[label],
            subject_label=label,
            capture_sha256=_sha(CAPTURES[label]),
            smplx_model=model,
            smplx_model_sha256=model_sha,
        )
        asset = materialize_subject(
            operator, betas=betas[label], gender="male"
        ).rigged_asset
        pose_map = build_pose_map_v1(
            value,
            asset=asset,
            calibration=calibration,
            oracle_path=ORACLE,
            source_operator_digest=operator.runtime_digest(validate=False),
        )
        result[label] = run_dynamic_chain_validation_v1(
            value,
            pose_map,
            asset=asset,
            calibration=calibration,
            recorded_poses=recorded,
        )
    return result


def test_sweep_recipe_is_the_frozen_requested_matrix() -> None:
    sweeps = synthetic_chain_sweeps_v1()
    assert len(sweeps) == 17
    assert {name for name in sweeps if name.startswith("knee_")} == {
        "knee_+0deg", "knee_+30deg", "knee_+60deg", "knee_+90deg", "knee_+120deg"
    }
    assert "ankle_-20deg" in sweeps
    assert "elbow_+140deg" in sweeps
    assert "wrist_-45deg" in sweeps
    assert "wrist_+45deg" in sweeps
    assert "shoulder_+120deg" in sweeps


def test_two_beta_cross_pose_and_sweeps_preserve_142_dynamic_authority(reports) -> None:
    for report in reports.values():
        failures = {
            cell_name: {
                "identity_bind_global_max_abs": cell.get("identity_bind_global_max_abs"),
                "nonbone_142_parity_max_m": cell.get("nonbone_142_parity_max_m"),
                "joints_all_pass": cell.get("joints_all_pass"),
            }
            for cell_name, cell in report["cells"].items()
            if not cell["passed"]
        }
        assert report["passed"] is True, json.dumps(failures, indent=2, sort_keys=True)
        assert {"tpose", "pose_213328", "pose_213712"} <= set(report["cells"])
        for cell in report["cells"].values():
            assert cell["passed"] is True
            assert cell["identity_bind_global_max_abs"] <= 3.0e-6
            assert cell["identity_bind_parent_local_basis_max_abs"] <= 3.0e-6
            assert cell["nonbone_142_parity_max_m"] <= 2.0e-7
            assert cell["pose_composition"] == "right_multiply_bind"
            # Anatomical joint motion vs 142 is report-only under right-multiply;
            # terminal hand/foot non-regression is gated separately.
