from __future__ import annotations

from pathlib import Path

import pytest

from projects.genesis_ue_sync.anatomy_retarget.anatomy_semantics import (
    REQUIRED_SEMANTIC_FIELDS,
    SemanticManifestError,
    _parse_simple_yaml,
    load_anatomy_semantics,
    parse_anatomy_semantics,
)


ROOT = Path(__file__).resolve().parents[1]


def _manifest_payload() -> dict:
    return {
        "version": 1,
        "fit_policies": ["rigid", "soft"],
        "driver_policies": ["armature"],
        "quality_profiles": {"default": {}},
        "landmark_recipes": {"generic": {}},
        "global_defaults": {
            "compound_id": "none",
            "side": "none",
            "source_landmarks": ["auto"],
            "target_landmark_recipe": "generic",
            "quality_profile": "default",
        },
        "collection_defaults": {
            "Bones": {"tissue_type": "bone"},
            "Soft": {"tissue_type": "soft"},
        },
        "tissue_defaults": {
            "bone": {"fit_policy": "rigid", "driver_policy": "armature"},
            "soft": {"fit_policy": "soft", "driver_policy": "armature"},
        },
        "meshes": {},
    }


def test_production_manifest_resolves_complete_deterministic_records() -> None:
    manifest = load_anatomy_semantics(
        ROOT / "configs/anatomy/anatomy_semantics.yaml"
    )
    resolved = manifest.resolve("Humerus_L", ["Skeletal_Sys"])
    record = resolved.to_dict()
    assert all(field in record for field in REQUIRED_SEMANTIC_FIELDS)
    assert record["tissue_type"] == "bone"
    assert record["side"] == "left"
    assert record["target_landmark_recipe"] == "long_bone_ends"

    artery = manifest.resolve("Artery", ["Cardiovascular_Sys"])
    assert artery.tissue_type == "vessel"
    assert artery.fit_policy == "soft_volume"


def test_manifest_fails_for_unresolved_and_ambiguous_meshes() -> None:
    manifest = parse_anatomy_semantics(_manifest_payload())
    with pytest.raises(SemanticManifestError, match="unresolved"):
        manifest.resolve("Unknown", ["Other"])
    with pytest.raises(SemanticManifestError, match="ambiguous"):
        manifest.resolve("Shared", ["Soft", "Bones"])


def test_exact_mesh_record_can_resolve_collection_ambiguity() -> None:
    payload = _manifest_payload()
    payload["meshes"]["Shared"] = {"tissue_type": "bone"}
    manifest = parse_anatomy_semantics(payload)
    resolved = manifest.resolve("Shared", ["Soft", "Bones"])
    assert resolved.tissue_type == "bone"
    assert resolved.fit_policy == "rigid"


def test_manifest_rejects_implicit_name_matching_rules() -> None:
    payload = _manifest_payload()
    payload["meshes"]["Femur"] = {"tokens": ["femur"]}
    with pytest.raises(SemanticManifestError, match="forbidden implicit matching"):
        parse_anatomy_semantics(payload)


def test_blender_fallback_yaml_parser_loads_production_manifest() -> None:
    path = ROOT / "configs/anatomy/anatomy_semantics.yaml"
    payload = _parse_simple_yaml(path.read_text(encoding="utf-8"))
    manifest = parse_anatomy_semantics(payload)
    resolved = manifest.resolve("Heart", ["Cardiovascular_Sys"])
    assert resolved.tissue_type == "heart"
    assert resolved.quality_profile == "soft_default"


def test_head_frame_uses_smplhf_eye_joints() -> None:
    exporter = (
        ROOT
        / "src/projects/genesis_ue_sync/anatomy_retarget/blender_scripts"
        / "blender_retarget_script.py"
    ).read_text(encoding="utf-8")
    assert 'joint_index["left_eye_smplhf"]' in exporter
    assert 'joint_index["right_eye_smplhf"]' in exporter
