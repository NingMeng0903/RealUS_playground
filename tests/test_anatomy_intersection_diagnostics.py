from types import SimpleNamespace

import numpy as np

from projects.genesis_ue_sync.anatomy_retarget import intersection_diagnostics


def test_intersection_report_separates_face_churn_from_net_new_count(monkeypatch) -> None:
    asset = SimpleNamespace(
        harmonic_reference_vertices=np.zeros((6, 3), dtype=np.float32),
        vertices_rest=np.ones((6, 3), dtype=np.float32),
        faces=np.asarray(((0, 1, 2), (3, 4, 5)), dtype=np.int32),
        source_mesh_names=["Artery", "Femur"],
        source_tissues=["vessel", "bone"],
        source_vertex_ranges=np.asarray(((0, 3), (3, 6)), dtype=np.int32),
    )
    # One harmonic contact moves to a neighbouring face during the bone fit;
    # station follow then adds one genuinely new contact.
    results = iter(
        (
            {(0, 10)},
            {(0, 11)},
            {(0, 11), (0, 12)},
        )
    )
    monkeypatch.setattr(
        intersection_diagnostics,
        "_intersection_pairs",
        lambda *_args, **_kwargs: next(results),
    )

    report = intersection_diagnostics.tube_bone_intersection_report(asset)

    assert report["introduced_face_pairs"] == 2
    assert report["material_fit_net_new_count"] == 0
    assert report["station_follow_net_new_count"] == 1
    assert report["positive_per_mesh_total_net_new_count"] == 1
    assert report["passed"] is False


def test_station_acceptance_rejects_only_intersection_regression(monkeypatch) -> None:
    class Fixture(SimpleNamespace):
        def validate(self) -> None:
            return None

    harmonic = np.zeros((6, 3), dtype=np.float32)
    final = harmonic.copy()
    final[:3, 0] = 0.25
    asset = Fixture(
        harmonic_reference_vertices=harmonic,
        vertices_rest=final,
        faces=np.asarray(((0, 1, 2), (3, 4, 5)), dtype=np.int32),
        source_mesh_names=["Artery", "Femur"],
        source_tissues=["vessel", "bone"],
        source_mesh_follow_modes=["station_translation", "final_bind_lbs"],
        source_vertex_ranges=np.asarray(((0, 3), (3, 6)), dtype=np.int32),
    )
    results = iter(({(0, 10)}, {(0, 10), (0, 11)}))
    monkeypatch.setattr(
        intersection_diagnostics,
        "_intersection_pairs",
        lambda *_args, **_kwargs: next(results),
    )

    accepted, report = (
        intersection_diagnostics.enforce_station_intersection_nonregression(asset)
    )

    np.testing.assert_array_equal(accepted.vertices_rest[:3], harmonic[:3])
    assert report["rejected_mesh_count"] == 1
    assert report["meshes"]["Artery"]["static_residual_accepted"] is False
