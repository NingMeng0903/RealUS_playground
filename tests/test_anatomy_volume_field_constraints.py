from __future__ import annotations

import numpy as np
import pytest

from projects.genesis_ue_sync.anatomy_retarget.shape_volume import (
    _beta_basis_digest,
    _internal_handle_report,
    _normalize_internal_handles,
    _solve_harmonic_field,
)


def _interior_node_cage() -> dict[str, np.ndarray]:
    nodes = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.2, 0.2, 0.2),
        ),
        dtype=np.float64,
    )
    return {
        "nodes": nodes,
        "elements": np.asarray(
            ((0, 1, 2, 4), (0, 1, 4, 3), (0, 4, 2, 3), (4, 1, 2, 3)),
            dtype=np.int32,
        ),
        "boundary": np.arange(4, dtype=np.int32),
        "source_triangles": np.asarray(
            ((0, 1, 2), (0, 1, 2), (0, 1, 2), (0, 1, 3)),
            dtype=np.int32,
        ),
        "source_bary": np.asarray(
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (0.0, 0.0, 1.0)),
            dtype=np.float64,
        ),
    }


@pytest.mark.parametrize(
    ("mode", "weight"),
    (("dirichlet", None), ("soft", 1.0e6)),
)
def test_internal_handle_influences_field_without_moving_outer_cage(
    mode: str,
    weight: float | None,
) -> None:
    cage = _interior_node_cage()
    handles: dict[str, object] = {
        "node_indices": np.asarray((4,), dtype=np.int32),
        "displacements": np.asarray(((0.02, 0.0, 0.0),), dtype=np.float64),
        "mode": mode,
    }
    if weight is not None:
        handles["weights"] = weight
    field = _solve_harmonic_field(
        cage,
        surface_displacement=np.zeros((4, 3), dtype=np.float64),
        internal_handles=handles,
    )
    np.testing.assert_allclose(field[cage["boundary"]], 0.0, atol=1.0e-12)
    assert field[4, 0] > 0.019
    report = _internal_handle_report(_normalize_internal_handles(cage, handles))
    assert report["harmonic_only"] is False
    assert report["internal_handle_count"] == 1


def test_no_internal_handles_reports_harmonic_only() -> None:
    cage = _interior_node_cage()
    report = _internal_handle_report(_normalize_internal_handles(cage, None))
    assert report["volume_field_mode"] == "harmonic_only"
    assert report["harmonic_only"] is True
    assert report["internal_handle_count"] == 0


def test_beta_cache_digest_covers_topology_surfaces_bary_map_and_handles() -> None:
    cage = _interior_node_cage()
    neutral = np.asarray(cage["nodes"][:4], dtype=np.float64)
    subject = neutral + np.asarray((0.001, 0.0, 0.0), dtype=np.float64)
    faces = np.asarray(((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)), dtype=np.int32)
    shapedirs = np.zeros((4, 3, 1), dtype=np.float32)

    def digest(
        candidate: dict[str, np.ndarray],
        *,
        subject_vertices: np.ndarray = subject,
        handles: dict[str, object] | None = None,
    ) -> str:
        return _beta_basis_digest(
            candidate,
            shapedirs,
            1,
            neutral_vertices=neutral,
            neutral_faces=faces,
            subject_vertices=subject_vertices,
            subject_faces=faces,
            internal_handles=handles,
        )

    baseline = digest(cage)
    changed_topology = dict(cage)
    changed_topology["elements"] = np.asarray(cage["elements"]).copy()
    changed_topology["elements"][0, [1, 2]] = changed_topology["elements"][0, [2, 1]]
    assert digest(changed_topology) != baseline

    changed_bary = dict(cage)
    changed_bary["source_bary"] = np.asarray(cage["source_bary"]).copy()
    changed_bary["source_bary"][0, 0] -= 1.0e-4
    changed_bary["source_bary"][0, 1] += 1.0e-4
    assert digest(changed_bary) != baseline

    changed_subject = subject.copy()
    changed_subject[0, 2] += 1.0e-4
    assert digest(cage, subject_vertices=changed_subject) != baseline

    handles = {
        "node_indices": np.asarray((4,), dtype=np.int32),
        "displacements": np.asarray(((0.01, 0.0, 0.0),), dtype=np.float64),
        "mode": "dirichlet",
    }
    handle_digest = digest(cage, handles=handles)
    assert handle_digest != baseline
    handles["displacements"] = np.asarray(((0.02, 0.0, 0.0),), dtype=np.float64)
    assert digest(cage, handles=handles) != handle_digest
