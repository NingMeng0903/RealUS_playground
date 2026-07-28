from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from projects.genesis_ue_sync.anatomy_retarget.compound_gates_v7 import (
    evaluate_compound_gates_v7,
)
from projects.genesis_ue_sync.anatomy_retarget.rigged_asset import AnatomyRiggedAsset


_REAL_ASSET = Path(
    "outputs/anatomy_retarget/v7_candidates/rebuild_003/subject_213328.npz"
)


def _cube(center: tuple[float, float, float], half: float) -> tuple[np.ndarray, np.ndarray]:
    c = np.asarray(center, dtype=np.float64)
    corners = np.asarray(
        [
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1],
        ],
        dtype=np.float64,
    )
    vertices = c + half * corners
    # Outward winding so igl signed distance is positive outside the box.
    faces = np.asarray(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ],
        dtype=np.int64,
    )
    return vertices, faces


def _compound_stub() -> AnatomyRiggedAsset:
    chunks: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    names: list[str] = []
    tissues: list[str] = []
    ranges: list[tuple[int, int]] = []

    def add(
        name: str,
        tissue: str,
        center: tuple[float, float, float],
        half: float,
    ) -> np.ndarray:
        start = sum(len(chunk) for chunk in chunks)
        vertices, local_faces = _cube(center, half)
        chunks.append(vertices)
        faces.append(local_faces + start)
        stop = start + len(vertices)
        names.append(name)
        tissues.append(tissue)
        ranges.append((start, stop))
        return np.arange(start, stop, dtype=np.int64)

    # Elbow left/right: ~1 mm articular gap at rest for both ulna and radius.
    for side, x in (("L", -0.20), ("R", 0.20)):
        add(f"Humerus_{side}", "bone", (x, 0.00, 0.0), 0.010)
        add(f"Ulna_{side}", "bone", (x, -0.021, 0.0), 0.010)
        add(f"Radius_{side}", "bone", (x, -0.021, 0.002), 0.010)

    # Two ribs seated on both anchors at rest (corner-to-corner contact) and
    # aligned on +x / -x, so a pure translation increases the end-to-end
    # nearest distance one-for-one.  Both ends must be attached at rest for
    # the connection gates to apply.
    add("T1", "bone", (0.0, 0.40, 0.0), 0.008)
    add("T2", "bone", (0.0, 0.42, 0.0), 0.008)
    add("Costal_Cartilage_L", "connective_tissue", (0.032, 0.40, 0.0), 0.008)
    add("Costal_Cartilage_R", "connective_tissue", (-0.032, 0.40, 0.0), 0.008)
    add("Sternum", "bone", (0.0, 0.38, 0.0), 0.006)
    add("Rib_1L", "bone", (0.016, 0.40, 0.0), 0.008)
    add("Rib_1R", "bone", (-0.016, 0.40, 0.0), 0.008)

    # Closed skull box with an intracranial organ strictly inside.
    add("Upper_Skull", "bone", (0.0, 0.80, 0.0), 0.050)
    add("Midbrain", "organ", (0.0, 0.80, 0.0), 0.010)

    vertices = np.concatenate(chunks, axis=0).astype(np.float32)
    triangles = np.concatenate(faces, axis=0).astype(np.int32)
    return AnatomyRiggedAsset(
        vertices_rest=vertices,
        faces=triangles,
        lbs_weights=None,
        joint_names=["root"],
        parents=np.asarray((-1,), dtype=np.int32),
        rest_joints=np.zeros((1, 3), dtype=np.float32),
        inverse_bind=np.eye(4, dtype=np.float32)[None],
        source_mesh_names=names,
        source_vertex_ranges=np.asarray(ranges, dtype=np.int64),
        source_tissues=tissues,
        source_sides=["left" if name.endswith("_L") else "right" if name.endswith("_R") else "mid" for name in names],
        source_bone_names=["Elbow_Rot_L", "Elbow_Rot_R"],
        metadata={},
    )


def _mesh_slice(asset: AnatomyRiggedAsset, name: str) -> slice:
    index = asset.source_mesh_names.index(name)
    start, stop = np.asarray(asset.source_vertex_ranges[index], dtype=np.int64)
    return slice(int(start), int(stop))


def test_identity_passes_elbow_ribs_skull_and_failures_detect_regressions() -> None:
    asset = _compound_stub()
    identity = np.asarray(asset.vertices_rest, dtype=np.float64)
    report = evaluate_compound_gates_v7(asset=asset, posed_vertices=identity)

    assert report["elbow"]["left"]["pass"] is True
    assert report["elbow"]["right"]["pass"] is True
    assert report["ribs"]["pass"] is True
    assert report["skull_brain"]["pass"] is True
    assert report["oral_cavity"]["tongue_present"] is False
    assert report["oral_cavity"]["publish_blocker"] is True
    assert report["pass"] is False

    separated = identity.copy()
    separated[_mesh_slice(asset, "Ulna_L"), 1] -= 0.006
    elbow_fail = evaluate_compound_gates_v7(asset=asset, posed_vertices=separated)
    assert elbow_fail["elbow"]["left"]["humerus_ulna"]["pass"] is False
    assert elbow_fail["elbow"]["left"]["pass"] is False

    pulled = identity.copy()
    rib_slice = _mesh_slice(asset, "Rib_1L")
    # Translate the rib farther from T1 along +x so the vertebral-end gap grows by 5 mm.
    pulled[rib_slice, 0] += 0.005
    rib_fail = evaluate_compound_gates_v7(asset=asset, posed_vertices=pulled)
    assert rib_fail["ribs"]["items"]["Rib_1L"]["vertebral_end"]["pass"] is False
    assert rib_fail["ribs"]["pass"] is False

    outside = identity.copy()
    outside[_mesh_slice(asset, "Midbrain"), 1] += 0.080
    skull_fail = evaluate_compound_gates_v7(asset=asset, posed_vertices=outside)
    assert skull_fail["skull_brain"]["pass"] is False
    assert skull_fail["skull_brain"]["posed_max_outside_m"] > 0.0


def test_unattached_rib_end_is_reported_not_gated() -> None:
    asset = _compound_stub()
    identity = np.asarray(asset.vertices_rest, dtype=np.float64)
    # Move both costal cartilages far anteriorly so no rib reaches them at
    # rest, the floating-rib case: the sternal end must be recorded as ungated
    # instead of failing when the pose separates it further.
    detached = identity.copy()
    for name in ("Costal_Cartilage_L", "Costal_Cartilage_R"):
        detached[_mesh_slice(asset, name), 0] *= 4.0
    unattached = AnatomyRiggedAsset(
        **{**asset.__dict__, "vertices_rest": detached.astype(np.float32)}
    )
    pulled = detached.copy()
    pulled[_mesh_slice(asset, "Rib_1L"), 2] += 0.020

    report = evaluate_compound_gates_v7(asset=unattached, posed_vertices=pulled)
    sternal = report["ribs"]["items"]["Rib_1L"]["sternal_end"]
    assert sternal["attached_at_rest"] is False
    assert sternal["gated"] is False
    assert sternal["increase_m"] > 0.002
    assert sternal["pass"] is True
    assert "Rib_1L" in report["ribs"]["ungated_sternal_ends"]
    # The seated vertebral end still fails on the same displacement.
    assert report["ribs"]["items"]["Rib_1L"]["vertebral_end"]["attached_at_rest"] is True
    assert report["ribs"]["items"]["Rib_1L"]["vertebral_end"]["pass"] is False


def test_missing_tongue_is_publish_blocker() -> None:
    asset = _compound_stub()
    report = evaluate_compound_gates_v7(
        asset=asset,
        posed_vertices=np.asarray(asset.vertices_rest, dtype=np.float64),
    )
    oral = report["oral_cavity"]
    assert oral["available"] is True
    assert oral["tongue_present"] is False
    assert oral["publish_blocker"] is True
    assert oral["pass"] is False
    assert "no tongue mesh" in oral["reason"]
    assert report["pass"] is False


@pytest.mark.skipif(not _REAL_ASSET.is_file(), reason="real V7 subject asset missing")
def test_real_asset_sections_available_and_finite() -> None:
    from projects.genesis_ue_sync.anatomy_retarget.v7_artifacts import (
        apply_subject_pose,
        load_subject_asset,
    )

    subject = load_subject_asset(_REAL_ASSET)
    asset = subject.rigged_asset
    posed = apply_subject_pose(
        subject,
        pose_axis_angle=np.zeros((55, 3), dtype=np.float32),
    )
    report = evaluate_compound_gates_v7(asset=asset, posed_vertices=posed)

    print("thresholds", report["thresholds"])
    for side in ("left", "right"):
        item = report["elbow"][side]
        print(
            f"elbow/{side}",
            "available",
            item["available"],
            "pass",
            item.get("pass"),
            "humerus_ulna",
            item.get("humerus_ulna"),
            "humerus_radius",
            item.get("humerus_radius"),
        )
    rib_items = report["ribs"]["items"]
    ranked = sorted(
        (
            (
                name,
                float(item["vertebral_end"]["increase_m"]),
                float(item["sternal_end"]["increase_m"]),
            )
            for name, item in rib_items.items()
            if item.get("available")
        ),
        key=lambda row: max(row[1], row[2]),
        reverse=True,
    )
    print("ribs worst3", ranked[:3])
    print(
        "skull_brain",
        {
            key: report["skull_brain"].get(key)
            for key in (
                "available",
                "pass",
                "reference_inside_ratio",
                "reference_max_outside_m",
                "posed_inside_ratio",
                "posed_max_outside_m",
                "added_outside_m",
                "worst_structure",
                "reason",
            )
        },
    )
    print("oral_cavity", report["oral_cavity"])

    assert report["elbow"]["left"]["available"] is True
    assert report["elbow"]["right"]["available"] is True
    assert report["ribs"]["available"] is True
    assert report["skull_brain"]["available"] is True
    assert report["oral_cavity"]["available"] is True

    def _assert_finite(value: object) -> None:
        if isinstance(value, dict):
            for child in value.values():
                _assert_finite(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                _assert_finite(child)
        elif isinstance(value, (float, int, np.floating, np.integer)):
            assert np.isfinite(float(value))

    _assert_finite(report["elbow"])
    _assert_finite(report["ribs"])
    _assert_finite(
        {
            key: report["skull_brain"][key]
            for key in (
                "reference_inside_ratio",
                "reference_max_outside_m",
                "posed_inside_ratio",
                "posed_max_outside_m",
                "added_outside_m",
            )
        }
    )
