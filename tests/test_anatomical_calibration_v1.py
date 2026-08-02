from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.projects.genesis_ue_sync.anatomy_retarget.anatomical_calibration_v1 import (
    JOINT_SPECS,
    _calibration_content_digest,
    build_anatomical_calibration_v1,
    check_anatomical_calibration_v1,
    load_anatomical_calibration_v1,
    save_anatomical_calibration_v1,
)
from src.projects.genesis_ue_sync.anatomy_retarget.blender_link_oracle_v7 import (
    EXPECTED_BLEND_SHA256,
    EXPECTED_ORACLE_SHA256,
)
from src.projects.genesis_ue_sync.anatomy_retarget.v8_artifacts import (
    load_source_operator,
)
from src.projects.genesis_ue_sync.anatomy_retarget.whole_chain_rest_fit_v1 import (
    build_whole_chain_rest_fit_v1,
)


_ROOT = Path(__file__).resolve().parents[1]
_OPERATOR = _ROOT / "outputs/anatomy_retarget/v8_candidates/rebuild_012/source_operator_v8"
_ORACLE = (
    _ROOT
    / "outputs/anatomy_retarget/v7_candidates/blender_link_oracle_v7_full_001"
    / "blender_link_oracle_v7.npz"
)


def _real_calibration():
    if not _OPERATOR.is_dir() or not _ORACLE.is_file():
        pytest.skip("frozen 142 operator or Blender oracle is unavailable")
    operator = load_source_operator(_OPERATOR, mmap=True)
    calibration = build_anatomical_calibration_v1(
        operator,
        source_blend_sha256=EXPECTED_BLEND_SHA256,
        blender_oracle_sha256=EXPECTED_ORACLE_SHA256,
    )
    return operator, calibration


def test_source_calibration_is_independent_and_preserves_virtual_pivots() -> None:
    operator, calibration = _real_calibration()
    report = check_anatomical_calibration_v1(calibration, operator=operator)
    assert report["passed"] is True
    assert report["passed_lower_chain"] is True
    assert report["passed_upper_chain"] is True
    assert report["accepted_scope"] == "full_main_chain"
    assert report["joint_count"] == len(JOINT_SPECS) == 12
    assert calibration.build_report["vertices_changed"] is False
    assert calibration.build_report["bind_changed"] is False
    assert calibration.build_report["runtime_changed"] is False
    assert calibration.build_report["raw_smplx_hip_translation_target"] is False
    assert set(calibration.controller_motion_modes.tolist()) <= {
        "bind_follow",
        "station_rigid",
        "hinge",
        "twist",
        "coupled_response",
        "patella_response",
    }

    hip_rows = np.flatnonzero(calibration.joint_kinds == "hip")
    raw_offsets = [
        report["joints"][str(calibration.joint_names[index])][
            "raw_station_to_anatomical_distance_m"
        ]
        for index in hip_rows.tolist()
    ]
    assert min(raw_offsets) > 0.05

    controller = calibration.controller_rest_global
    local = calibration.physical_pivot_controller_local
    reconstructed = (
        np.einsum("bij,bj->bi", controller[:, :3, :3], local)
        + controller[:, :3, 3]
    )
    np.testing.assert_allclose(
        reconstructed,
        calibration.anatomical_rest_global[:, :3, 3],
        atol=1.0e-9,
        rtol=0.0,
    )


def test_calibration_round_trip_and_fail_closed_domain_digest(tmp_path: Path) -> None:
    operator, calibration = _real_calibration()
    report = check_anatomical_calibration_v1(calibration, operator=operator)
    output = save_anatomical_calibration_v1(
        tmp_path / "calibration",
        calibration,
        operator=operator,
        checker_report=report,
        accepted_scope="lower_chain",
    )
    loaded = load_anatomical_calibration_v1(
        output, operator=operator, required_scope="lower_chain"
    )
    np.testing.assert_array_equal(loaded.joint_names, calibration.joint_names)
    np.testing.assert_array_equal(
        loaded.anatomical_rest_global, calibration.anatomical_rest_global
    )
    assert loaded.fixed_domain_digest == calibration.fixed_domain_digest
    assert set(loaded.domains) == set(calibration.domains)
    for name in loaded.domains:
        np.testing.assert_array_equal(loaded.domains[name], calibration.domains[name])

    corrupted = replace(calibration, fixed_domain_digest="0" * 64)
    corrupted_report = check_anatomical_calibration_v1(
        corrupted, operator=operator
    )
    assert corrupted_report["passed"] is False
    assert corrupted_report["source_checks"]["domain_digest"] is False

    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["matrix_convention"] = "row_vector"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest contract"):
        load_anatomical_calibration_v1(
            output, operator=operator, required_scope="lower_chain"
        )


def test_fit_and_validation_domains_are_disjoint() -> None:
    _operator, calibration = _real_calibration()
    for bases in calibration.joint_domain_bases:
        for base in bases.tolist():
            if not base:
                continue
            assert not np.intersect1d(
                calibration.domains[f"{base}.fit"],
                calibration.domains[f"{base}.validation"],
            ).size


@pytest.mark.parametrize(
    "field",
    ("station_from_anatomical", "anatomical_from_controller", "station_rest_global"),
)
def test_calibration_rejects_change_of_basis_tampering(field: str) -> None:
    _operator, calibration = _real_calibration()
    value = np.asarray(getattr(calibration, field)).copy()
    value[0, 0, 3] += 100.0
    tampered = replace(calibration, **{field: value})
    with pytest.raises(ValueError):
        tampered.validate()


def test_checker_rebuilds_domains_and_controller_recipe() -> None:
    operator, calibration = _real_calibration()
    domains = {name: np.asarray(ids).copy() for name, ids in calibration.domains.items()}
    name = "left/femoral_head.fit"
    domains[name][0] = domains[name][-1] + 1
    tampered_domains = replace(calibration, domains=domains)
    report = check_anatomical_calibration_v1(tampered_domains, operator=operator)
    assert report["passed_lower_chain"] is False
    assert report["source_checks"]["domains_exact_from_frozen_operator"] is False

    names = calibration.controller_names.copy()
    names[0] = "not_the_frozen_controller"
    with pytest.raises(ValueError, match="controller_names"):
        replace(calibration, controller_names=names).validate()


def test_independent_rigid_groups_keep_motion_authority() -> None:
    operator, calibration = _real_calibration()
    bone_names = list(operator.template_asset.source_bone_names or ())
    for name in ("Head_Bone", "Scapula_Bone_L", "Scapula_Bone_R"):
        assert calibration.controller_motion_modes[bone_names.index(name)] == "station_rigid"


def test_full_scope_load_rejects_lower_chain_only_artifact(tmp_path: Path) -> None:
    operator, calibration = _real_calibration()
    report = check_anatomical_calibration_v1(calibration, operator=operator)
    output = save_anatomical_calibration_v1(
        tmp_path / "lower_only",
        calibration,
        operator=operator,
        checker_report=report,
        accepted_scope="lower_chain",
    )
    with pytest.raises(ValueError, match="incomplete for the required scope"):
        load_anatomical_calibration_v1(
            output, operator=operator, required_scope="full_main_chain"
        )


def test_forged_checker_dictionary_cannot_authorize_artifact(tmp_path: Path) -> None:
    operator, calibration = _real_calibration()
    modes = calibration.controller_motion_modes.copy()
    bone_names = list(operator.template_asset.source_bone_names or ())
    modes[bone_names.index("Head_Bone")] = "bind_follow"
    forged = replace(
        calibration,
        source_blend_sha256="e" * 64,
        blender_oracle_sha256="f" * 64,
        controller_motion_modes=modes,
    )
    fake_report = check_anatomical_calibration_v1(calibration, operator=operator)
    fake_report["calibration_digest"] = _calibration_content_digest(forged)
    fake_report["passed_lower_chain"] = True
    fake_report["accepted_scope"] = "lower_chain"
    fake_report["source_checks"] = {
        name: True for name in fake_report["source_checks"]
    }
    fake_report["array_checks"] = {
        name: True for name in fake_report["array_checks"]
    }
    with pytest.raises(ValueError):
        save_anatomical_calibration_v1(
            tmp_path / "forged",
            forged,
            operator=operator,
            checker_report=fake_report,
            accepted_scope="lower_chain",
        )


def test_whole_chain_builder_rejects_non_male_before_model_use() -> None:
    operator, calibration = _real_calibration()
    with pytest.raises(ValueError, match="smplx_gender=male"):
        build_whole_chain_rest_fit_v1(
            operator,
            calibration,
            betas=np.zeros(10),
            subject_label="invalid",
            capture_sha256="0" * 64,
            smplx_model={},
            smplx_model_sha256=(
                "af7ebc82e44cf098598685474c0592049ddfaca8e850feb0c2b88343f9aacee3"
            ),
            gender="neutral",
        )
