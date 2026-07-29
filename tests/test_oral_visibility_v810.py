from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from projects.genesis_ue_sync.anatomy_retarget.anatomy_drawer import (
    render_faces_for_asset_v2,
)
from projects.genesis_ue_sync.anatomy_retarget.reference_fit_v8 import (
    _oral_visibility_policy_v2,
)
from projects.genesis_ue_sync.anatomy_retarget.v8_artifacts import (
    load_source_operator,
)


_REBUILD_012_OPERATOR = Path(
    "outputs/anatomy_retarget/v8_candidates/rebuild_012/source_operator_v8"
)


def _face_ids_digest(face_ids: list[int]) -> str:
    values = np.ascontiguousarray(face_ids, dtype="<i4")
    return hashlib.sha256(values.tobytes()).hexdigest()


def _synthetic_asset() -> SimpleNamespace:
    faces = np.asarray(
        (
            (0, 1, 2),
            (3, 4, 5),
            (6, 7, 8),
            (9, 10, 11),
            (12, 13, 14),
        ),
        dtype=np.int32,
    )
    policy = {
        "hidden_face_count": 1,
        "hidden_face_ids_sha256": _face_ids_digest([0]),
        "hidden_face_source_mesh_names": ["Pharynx"],
        "hidden_face_counts_by_mesh": {"Pharynx": 1},
        "hidden_whole_mesh_face_counts": {"Sublingual_Gland_L": 1},
    }
    return SimpleNamespace(
        vertices_rest=np.zeros((15, 3), dtype=np.float32),
        faces=faces,
        source_mesh_names=[
            "Pharynx",
            "Sublingual_Gland_L",
            "Mandible",
            "Central_Incisor_L",
            "Other_Organ",
        ],
        source_vertex_ranges=np.asarray(
            ((0, 3), (3, 6), (6, 9), (9, 12), (12, 15)),
            dtype=np.int32,
        ),
        source_tissues=["organ", "organ", "bone", "bone", "organ"],
        metadata={
            "show_connective_tissue": True,
            "show_vessels": True,
            "oral_visibility_policy_v2": policy,
            "hidden_mesh_names_v2": ["Sublingual_Gland_L"],
            "hidden_face_ids_v2": [0],
        },
    )


def _real_asset():
    if not (_REBUILD_012_OPERATOR / "manifest.json").exists():
        pytest.skip("rebuild_012 source operator is not available")
    return load_source_operator(_REBUILD_012_OPERATOR).template_asset


def _mesh_face_ids(asset, mesh_name: str) -> np.ndarray:
    index = list(asset.source_mesh_names).index(mesh_name)
    start, stop = (
        int(value)
        for value in np.asarray(asset.source_vertex_ranges)[index]
    )
    faces = np.asarray(asset.faces, dtype=np.int64)
    return np.flatnonzero(np.all((faces >= start) & (faces < stop), axis=1))


def _component_count(faces: np.ndarray) -> int:
    parent = np.arange(len(faces), dtype=np.int64)

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    vertex_owner: dict[int, int] = {}
    for face_index, face in enumerate(np.asarray(faces, dtype=np.int64)):
        for vertex in face.tolist():
            previous = vertex_owner.setdefault(int(vertex), face_index)
            union(previous, face_index)
    return len({find(index) for index in range(len(faces))})


def test_draw_list_combines_reviewed_faces_and_whole_mesh_exclusions() -> None:
    asset = _synthetic_asset()
    rendered = render_faces_for_asset_v2(asset)
    np.testing.assert_array_equal(rendered, asset.faces[[2, 3, 4]])


@pytest.mark.parametrize(
    ("face_ids", "match"),
    (
        ([0, 0], "duplicate"),
        ([99], "invalid face index"),
        ([0.0], "integer face indices"),
    ),
)
def test_draw_list_rejects_invalid_reviewed_face_ids(
    face_ids: list[int],
    match: str,
) -> None:
    asset = _synthetic_asset()
    asset.metadata["hidden_face_ids_v2"] = face_ids
    with pytest.raises(ValueError, match=match):
        render_faces_for_asset_v2(asset)


def test_draw_list_rejects_faces_outside_reviewed_organ_domains() -> None:
    asset = _synthetic_asset()
    asset.metadata["hidden_face_ids_v2"] = [4]
    policy = asset.metadata["oral_visibility_policy_v2"]
    policy["hidden_face_ids_sha256"] = _face_ids_digest([4])
    with pytest.raises(ValueError, match="outside reviewed organ domains"):
        render_faces_for_asset_v2(asset)


def test_rebuild_012_policy_freezes_exact_connected_oral_domains() -> None:
    asset = _real_asset()
    policy = _oral_visibility_policy_v2(asset)

    assert policy["hidden_mesh_names_v2"] == [
        "Sublingual_Ducts_L",
        "Sublingual_Ducts_R",
        "Sublingual_Gland_L",
        "Sublingual_Gland_R",
    ]
    assert policy["hidden_face_source_mesh_names"] == [
        "Pharynx",
        "UNCUT_Digestive_Tract",
    ]
    assert policy["hidden_face_counts_by_mesh"] == {
        "Pharynx": 2618,
        "UNCUT_Digestive_Tract": 2618,
    }
    assert policy["hidden_face_count"] == 5236
    assert policy["hidden_total_face_count"] == 7852
    assert policy["hidden_face_ids_sha256"] == (
        "31929b3ae878e42539dbeb124ab49356e"
        "ffae4ff0ec63f6f0e57f2a0e5400331"
    )
    assert policy["tooth_mesh_count"] == 32
    assert policy["tooth_face_count"] == 11384
    assert policy["preserve_face_counts"] == {
        "Mandible": 4254,
        "Hyoid_Bone": 448,
        "Larynx": 720,
        "Parotid_Gland_L": 548,
        "Parotid_Gland_R": 548,
    }

    hidden_ids = np.asarray(policy["hidden_face_ids_v2"], dtype=np.int64)
    for mesh_name in policy["hidden_face_source_mesh_names"]:
        mesh_face_ids = _mesh_face_ids(asset, mesh_name)
        selected = np.intersect1d(hidden_ids, mesh_face_ids)
        assert len(selected) == policy["hidden_face_counts_by_mesh"][mesh_name]
        assert _component_count(np.asarray(asset.faces)[selected]) == 1


def test_rebuild_012_draw_list_preserves_reviewed_oral_structures() -> None:
    asset = _real_asset()
    policy = _oral_visibility_policy_v2(asset)
    metadata = dict(asset.metadata or {})
    metadata.update(
        {
            "show_connective_tissue": True,
            "show_vessels": True,
            "oral_visibility_policy_v2": policy,
            "hidden_mesh_names_v2": policy["hidden_mesh_names_v2"],
            "hidden_face_ids_v2": policy["hidden_face_ids_v2"],
        }
    )
    reviewed = replace(asset, metadata=metadata)
    rendered = render_faces_for_asset_v2(reviewed)
    faces = np.asarray(asset.faces, dtype=np.int32)

    excluded = np.zeros(len(faces), dtype=bool)
    excluded[np.asarray(policy["hidden_face_ids_v2"], dtype=np.int64)] = True
    for mesh_name in policy["hidden_mesh_names_v2"]:
        excluded[_mesh_face_ids(asset, mesh_name)] = True
    np.testing.assert_array_equal(rendered, faces[~excluded])
    assert len(rendered) == 775004

    preserved_counts = {
        "Mandible": 4254,
        "Hyoid_Bone": 448,
        "Larynx": 720,
        "Parotid_Gland_L": 548,
        "Parotid_Gland_R": 548,
        "Submandibular_Duct_L": 256,
        "Submandibular_Duct_R": 256,
        "Submandibular_Gland_L": 896,
        "Submandibular_Gland_R": 896,
    }
    for mesh_name, expected_count in preserved_counts.items():
        mesh_face_ids = _mesh_face_ids(asset, mesh_name)
        assert len(mesh_face_ids) == expected_count
        assert not np.any(excluded[mesh_face_ids])

    tooth_names = [
        str(name)
        for name in asset.source_mesh_names
        if any(
            token in str(name).lower()
            for token in ("canine", "incisor", "molar", "premolar")
        )
    ]
    tooth_face_ids = np.concatenate(
        [_mesh_face_ids(asset, mesh_name) for mesh_name in tooth_names]
    )
    assert len(tooth_face_ids) == 11384
    assert not np.any(excluded[tooth_face_ids])

    assert np.count_nonzero(
        ~excluded[_mesh_face_ids(asset, "Pharynx")]
    ) == 1878
    assert np.count_nonzero(
        ~excluded[_mesh_face_ids(asset, "UNCUT_Digestive_Tract")]
    ) == 17494
