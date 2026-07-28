from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import matplotlib

matplotlib.use("Agg")

import pytest

from projects.genesis_ue_sync.anatomy_retarget.acceptance_matrix_v7 import (
    load_matrix_pose_v7,
    load_matrix_subject_v7,
)
from projects.genesis_ue_sync.anatomy_retarget.evidence_pack_v7 import (
    REQUIRED_VIEWS_V7,
    evidence_file_stem_v7,
    generate_evidence_pack_v7,
    write_evidence_sidecar_v7,
)
from projects.genesis_ue_sync.anatomy_retarget.joint_contact_v7 import (
    FrozenJointMaterialDomainsV7,
)
from projects.genesis_ue_sync.anatomy_retarget.patella_oracle_v7 import (
    load_patella_oracle_v7,
)
from projects.genesis_ue_sync.anatomy_retarget.v7_artifacts import load_source_operator


_ROOT = Path("outputs/anatomy_retarget/v7_candidates/rebuild_003")
_OPERATOR = _ROOT / "source_operator_v7.npz"
_SUBJECT = _ROOT / "subject_213328.npz"
_DOMAINS = Path(
    "outputs/anatomy_retarget/v7_candidates/joint_rebuild_001/fixed_joint_domains_v7.json"
)
_ORACLE = _ROOT / "patella_oracle_v7.npz"
_POSED = _ROOT / "posed"


def test_evidence_file_stem_v7_prefix_is_eight_chars() -> None:
    stem = evidence_file_stem_v7(
        operator_digest="abcdef0123456789deadbeef",
        beta="213328",
        pose="tpose",
        view="hip_section",
    )
    assert stem == "abcdef01_213328_tpose_hip_section"
    assert stem.split("_", 1)[0] == "abcdef01"
    assert len(stem.split("_", 1)[0]) == 8
    with pytest.raises(ValueError):
        evidence_file_stem_v7(
            operator_digest="abc",
            beta="213328",
            pose="tpose",
            view="hip_section",
        )


def test_write_evidence_sidecar_round_trip(tmp_path: Path) -> None:
    png = tmp_path / "deadbeef_213328_tpose_surface_front.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 64)
    payload = {
        "operator_digest": "deadbeefcafebabe",
        "subject_digest": "subjectdigest",
        "beta": "213328",
        "pose_digest": "posedigest",
        "asset_file_digest": "assetdigest",
        "view": "surface_front",
        "command": "python -m projects... evidence-pack",
        "geometry_source": "apply_subject_pose",
    }
    sidecar = write_evidence_sidecar_v7(png, payload)
    assert sidecar == png.with_suffix(".json")
    loaded = json.loads(sidecar.read_text(encoding="utf-8"))
    for key in (
        "operator_digest",
        "subject_digest",
        "beta",
        "pose_digest",
        "asset_file_digest",
        "view",
        "command",
        "geometry_source",
    ):
        assert key in loaded
        assert loaded[key] == payload[key]


def test_manifest_records_failing_view(tmp_path: Path) -> None:
    subjects = [load_matrix_subject_v7("213328", _SUBJECT)] if _SUBJECT.is_file() else None
    if subjects is None:
        pytest.skip("rebuild_003 subject is not present")
    if not (_DOMAINS.is_file() and _ORACLE.is_file() and _OPERATOR.is_file()):
        pytest.skip("rebuild_003 acceptance inputs are not present")

    operator = load_source_operator(_OPERATOR)
    domains = FrozenJointMaterialDomainsV7.load_json(_DOMAINS)
    law = load_patella_oracle_v7(_ORACLE)
    poses = [load_matrix_pose_v7("tpose", "zero")]

    def _boom(*_args, **_kwargs):
        raise RuntimeError("deliberate hip_section failure")

    with mock.patch(
        "projects.genesis_ue_sync.anatomy_retarget.evidence_pack_v7._render_hip_section",
        side_effect=_boom,
    ):
        manifest = generate_evidence_pack_v7(
            subjects=subjects,
            poses=poses,
            domains=domains,
            law=law,
            operator_digest=str(operator.content_digest()),
            output_dir=tmp_path / "evidence",
            posed_dir=_POSED if _POSED.is_dir() else None,
            sweep_count=3,
            command="test deliberate failure",
        )

    assert manifest["complete"] is False
    missing = [
        item
        for item in manifest["missing_views"]
        if item["view"] == "hip_section"
    ]
    assert missing
    assert "deliberate hip_section failure" in missing[0]["reason"]


@pytest.mark.skipif(
    not (
        _OPERATOR.is_file()
        and _SUBJECT.is_file()
        and _DOMAINS.is_file()
        and _ORACLE.is_file()
    ),
    reason="rebuild_003 acceptance inputs are not present",
)
def test_real_one_cell_evidence_pack(tmp_path: Path) -> None:
    operator = load_source_operator(_OPERATOR)
    subjects = [load_matrix_subject_v7("213328", _SUBJECT)]
    poses = [load_matrix_pose_v7("tpose", "zero")]
    domains = FrozenJointMaterialDomainsV7.load_json(_DOMAINS)
    law = load_patella_oracle_v7(_ORACLE)
    out = tmp_path / "evidence"
    manifest = generate_evidence_pack_v7(
        subjects=subjects,
        poses=poses,
        domains=domains,
        law=law,
        operator_digest=str(operator.content_digest()),
        output_dir=out,
        posed_dir=_POSED if _POSED.is_dir() else None,
        sweep_count=5,
        command="pytest real one-cell evidence pack",
    )
    assert not manifest["missing_views"], manifest["missing_views"]
    assert manifest["complete"] is True
    assert len(manifest["files"]) == len(REQUIRED_VIEWS_V7)
    for item in manifest["files"]:
        png = Path(item["png"])
        sidecar = Path(item["sidecar"])
        assert png.is_file()
        assert png.stat().st_size > 1024
        assert sidecar.is_file()
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        for key in (
            "operator_digest",
            "subject_digest",
            "beta",
            "pose_digest",
            "asset_file_digest",
            "view",
            "command",
            "geometry_source",
        ):
            assert key in payload
        assert payload["view"] == item["view"]
        assert payload["beta"] == "213328"
