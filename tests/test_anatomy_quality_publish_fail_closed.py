from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import projects.genesis_ue_sync.anatomy_retarget.cli.run_anatomy_retarget as cli


def test_containment_report_includes_heart_and_per_region_critical_tissue(
    monkeypatch,
) -> None:
    asset = SimpleNamespace(
        vertices_rest=np.zeros((4, 3), dtype=np.float32),
        source_mesh_names=["Heart", "Aorta", "Sciatic_Nerve", "Liver"],
        source_vertex_ranges=np.asarray(((0, 1), (1, 2), (2, 3), (3, 4))),
        source_tissues=["heart", "vessel", "nerve", "organ"],
    )
    distances = np.asarray((-0.01, 0.0015, 0.0025, 0.003), dtype=np.float64)

    def fake_signed_distance(*_args, **_kwargs):
        return distances, np.zeros((4, 3)), np.zeros((4, 3))

    monkeypatch.setattr(cli, "signed_distance", fake_signed_distance)
    report = cli._signed_distance_containment_report(
        asset,
        anatomy_vertices=asset.vertices_rest,
        surface_vertices=np.zeros((3, 3)),
        surface_faces=np.asarray(((0, 1, 2),)),
        stage="final_pose",
    )

    assert set(report["over_limit_count"]) == {"heart", "vessel", "nerve", "organ"}
    assert report["over_limit_count"]["heart"] == 0
    assert report["over_limit_count"]["vessel"] == 1
    assert report["over_limit_count"]["nerve"] == 1
    assert report["over_limit_count"]["organ"] == 1
    assert report["per_mesh"]["Aorta"]["over_limit_count"] == 1
    assert report["per_mesh"]["Sciatic_Nerve"]["over_limit_count"] == 1


def test_stale_schema_cache_is_rebuilt_instead_of_loaded(tmp_path) -> None:
    stale = tmp_path / "stale.npz"
    np.savez(stale, schema_version=np.asarray(5, dtype=np.int32))

    assert (
        cli._load_valid_cache(
            stale,
            metadata_key="source_cache_key",
            expected_key="current",
        )
        is None
    )


def test_cache_key_changes_with_solver_or_semantics_content(tmp_path) -> None:
    solver = tmp_path / "solver.py"
    semantics = tmp_path / "semantics.yaml"
    solver.write_text("VERSION = 1\n", encoding="utf-8")
    semantics.write_text("version: 1\n", encoding="utf-8")
    first = cli._cache_key(solver, semantics, extra="schema-6")

    semantics.write_text("version: 2\n", encoding="utf-8")
    second = cli._cache_key(solver, semantics, extra="schema-6")

    assert first != second
