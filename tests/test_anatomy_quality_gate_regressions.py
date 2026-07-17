from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from projects.genesis_ue_sync.anatomy_retarget.quality_gate import (
    _brain_skull_metrics,
    _cranial_compound_metrics,
    _required_number,
    _soft_mesh_pose_stretch,
)


def test_cranial_gate_includes_upper_teeth_and_deep_brain_meshes() -> None:
    # Three simple tetrahedra: skull defines the similarity and the two head
    # contents must follow it exactly.  This mirrors the historic omission of
    # Upper_Teeth/Fornix from the old name whitelist.
    tetra = np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)), dtype=np.float64)
    source = np.concatenate((tetra, tetra + (2, 0, 0), tetra + (4, 0, 0)))
    final = 0.9 * source + (0.1, 0.2, 0.3)
    final[8:12] += (0.004, 0, 0)  # Upper_Teeth incorrectly skipped by the compound transform.
    asset = SimpleNamespace(
        registration_reference=source,
        vertices_rest=final,
        source_bone_names=None,
        source_bone_parents=None,
        driver_indices=None,
        driver_weights=None,
        source_mesh_names=["Upper_Skull", "Fornix", "Upper_Teeth"],
        source_vertex_ranges=np.asarray(((0, 4), (4, 8), (8, 12))),
    )

    metrics = _cranial_compound_metrics(asset)

    assert metrics["member_count"] == 3
    assert metrics["upper_teeth_meshes"] == ["Upper_Teeth"]
    assert float(metrics["upper_teeth_skull_distance_drift_m"]) > 0.003


def test_soft_edge_gate_reports_each_vessel_mesh_not_global_average() -> None:
    rest = np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0), (3, 0, 0), (4, 0, 0), (3, 1, 0)), dtype=np.float64)
    posed = rest.copy()
    posed[1] = (1.25, 0, 0)
    asset = SimpleNamespace(
        vertices_rest=rest,
        pose_cache_vertices=posed,
        faces=np.asarray(((0, 1, 2), (3, 4, 5)), dtype=np.int64),
        source_mesh_names=["Artery_Bad", "Vein_Good"],
        source_vertex_ranges=np.asarray(((0, 3), (3, 6))),
        source_tissues=["vessel", "vessel"],
    )

    metrics = _soft_mesh_pose_stretch(asset)

    assert float(metrics["Artery_Bad"]["ratio_p999"]) > 1.1
    assert float(metrics["Vein_Good"]["ratio_p999"]) == 1.0


def test_missing_required_quality_field_is_explicit_failure_not_sentinel() -> None:
    failures: list[str] = []

    value = _required_number(
        {},
        "over_limit_count",
        failures=failures,
        label="posed containment",
    )

    assert value is None
    assert failures == ["posed containment field 'over_limit_count' is missing"]


def test_missing_brain_or_skull_reports_unavailable_without_infinity() -> None:
    asset = SimpleNamespace(
        source_mesh_names=["Upper_Skull"],
        source_vertex_ranges=np.asarray(((0, 4),)),
        vertices_rest=np.zeros((4, 3), dtype=np.float64),
    )

    metrics = _brain_skull_metrics(asset)

    assert metrics["available"] is False
    assert metrics["max_outside_m"] is None
