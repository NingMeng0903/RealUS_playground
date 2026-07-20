from __future__ import annotations

import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from projects.genesis_ue_sync.anatomy_retarget import source_skin_volume
from projects.genesis_ue_sync.anatomy_retarget.source_skin_volume import (
    _add_semantic_joint_handles,
    _build_source_cage,
    _incremental_harmonic_field,
    _topology_preserving_cage_registration,
    _transport_sampled_material,
)


def test_semantic_joint_handle_moves_pivot_in_shared_volume_field() -> None:
    nodes = np.asarray(
        ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (0.2, 0.2, 0.2)),
        dtype=np.float64,
    )
    elements = np.asarray(
        ((0, 1, 2, 4), (0, 1, 4, 3), (0, 4, 2, 3), (4, 1, 2, 3)),
        dtype=np.int32,
    )
    source_bind = np.eye(4, dtype=np.float64)[None]
    source_bind[0, :3, 3] = nodes[4]
    asset = SimpleNamespace(
        joint_names=["left_wrist"],
        source_bone_driver_types=["joint_local"],
        source_bone_smplx_a=np.asarray((0,), dtype=np.int32),
        source_bind_global=source_bind,
    )
    target = np.asarray(((0.22, 0.2, 0.2),), dtype=np.float64)
    field, report = _add_semantic_joint_handles(
        asset,
        cage={
            "nodes": nodes,
            "elements": elements,
            "boundary": np.asarray((0, 1, 2, 3), dtype=np.int32),
        },
        boundary_field=np.zeros_like(nodes),
        target_joints=target,
    )
    np.testing.assert_allclose(field[:4], 0.0, atol=1.0e-10)
    np.testing.assert_allclose(nodes[4] + field[4], target[0], atol=1.0e-7)
    assert report["groups"]["wrists"]["accepted_fraction"] == pytest.approx(1.0)
    assert report["minimum_jacobian_ratio"] >= 0.05


def test_incremental_harmonic_field_reaches_boundary_without_flips() -> None:
    nodes = np.asarray(
        ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (0.2, 0.2, 0.2)),
        dtype=np.float64,
    )
    elements = np.asarray(((0, 1, 2, 4), (0, 1, 4, 3), (0, 4, 2, 3), (4, 1, 2, 3)), dtype=np.int32)
    boundary = np.asarray((0, 1, 2, 3), dtype=np.int32)
    values = np.asarray(((0, 0, 0), (0.05, 0, 0), (0, 0.03, 0), (0, 0, -0.02)), dtype=np.float64)
    field, report = _incremental_harmonic_field(nodes, elements, boundary, values)
    np.testing.assert_allclose(field[boundary], values, atol=1.0e-7)
    assert report["inverted_tetrahedra"] == 0
    assert report["minimum_jacobian_ratio"] > 0.0
    assert report["minimum_incremental_step_jacobian_ratio"] >= 0.05


def test_incremental_harmonic_field_rejects_unreachable_inverted_target() -> None:
    nodes = np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)), dtype=np.float64)
    elements = np.asarray(((0, 1, 2, 3),), dtype=np.int32)
    boundary = np.arange(4, dtype=np.int32)
    values = np.zeros((4, 3), dtype=np.float64)
    values[3, 2] = -2.0
    with pytest.raises(RuntimeError, match="cannot avoid tetrahedron inversion"):
        _incremental_harmonic_field(nodes, elements, boundary, values)


def test_incremental_harmonic_field_rejects_positive_below_minimum_jacobian() -> None:
    nodes = np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)), dtype=np.float64)
    elements = np.asarray(((0, 1, 2, 3),), dtype=np.int32)
    boundary = np.arange(4, dtype=np.int32)
    values = np.zeros((4, 3), dtype=np.float64)
    values[3, 2] = -0.96
    with pytest.raises(RuntimeError, match="minimum Jacobian-ratio violation"):
        _incremental_harmonic_field(nodes, elements, boundary, values)


def test_sampled_volume_field_moves_soft_material_and_preserves_rigid_material() -> None:
    points = np.asarray(((0, 0, 0), (1, 0, 0), (2, 0, 0)), dtype=np.float64)
    delta = np.asarray(((0.2, 0, 0), (0, 0.3, 0), (0, 0, 0.4)), dtype=np.float64)
    protected = np.asarray((False, True, False))
    mapped = _transport_sampled_material(
        points,
        delta,
        protected=protected,
        outside=np.zeros(3, dtype=bool),
    )
    np.testing.assert_allclose(mapped[0], points[0] + delta[0])
    np.testing.assert_allclose(mapped[1], points[1])
    np.testing.assert_allclose(mapped[2], points[2] + delta[2])


def test_sampled_volume_field_rejects_outside_soft_material() -> None:
    with pytest.raises(ValueError, match=r"soft=1, protected=1"):
        _transport_sampled_material(
            np.zeros((2, 3), dtype=np.float64),
            np.zeros((2, 3), dtype=np.float64),
            protected=np.asarray((False, True)),
            outside=np.asarray((True, True)),
        )


def _tetra_surface() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nodes = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    elements = np.asarray(((0, 1, 2, 3),), dtype=np.int32)
    faces = np.asarray(((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)), dtype=np.int32)
    return nodes, elements, faces


def _install_corresponding_surface_distance(monkeypatch: pytest.MonkeyPatch) -> None:
    def point_mesh_squared_distance(
        query: np.ndarray,
        target: np.ndarray,
        _target_faces: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        closest = np.asarray(target, dtype=np.float64)
        squared = np.sum((np.asarray(query, dtype=np.float64) - closest) ** 2, axis=1)
        return squared, np.arange(len(query), dtype=np.int32), closest

    monkeypatch.setitem(
        sys.modules,
        "igl",
        SimpleNamespace(point_mesh_squared_distance=point_mesh_squared_distance),
    )


def test_production_surface_solver_moves_and_reports_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_corresponding_surface_distance(monkeypatch)
    nodes, elements, faces = _tetra_surface()
    target = nodes + np.asarray((0.005, -0.003, 0.002), dtype=np.float64)
    registered, report = _topology_preserving_cage_registration(
        nodes,
        elements,
        np.arange(4, dtype=np.int32),
        faces,
        target,
        faces,
    )
    assert report["accepted_surface_iterations"] > 0
    assert report["surface_rms_progress_m"] > 0.0
    assert report["boundary_displacement_rms_m"] > 0.0
    assert report["minimum_surface_jacobian_ratio"] > 0.0
    np.testing.assert_allclose(registered, target, atol=1.0e-6)


def test_production_surface_solver_rejects_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_corresponding_surface_distance(monkeypatch)
    nodes, elements, faces = _tetra_surface()
    with pytest.raises(RuntimeError, match="no measurable progress"):
        _topology_preserving_cage_registration(
            nodes,
            elements,
            np.arange(4, dtype=np.int32),
            faces,
            nodes,
            faces,
        )


def test_production_surface_solver_rejects_all_proposals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_corresponding_surface_distance(monkeypatch)
    nodes, elements, faces = _tetra_surface()

    def always_inverted(*_args: object, **_kwargs: object) -> np.ndarray:
        field = np.zeros_like(nodes)
        field[3, 2] = -2.0
        return field

    monkeypatch.setattr(
        source_skin_volume,
        "_harmonic_step",
        always_inverted,
    )
    with pytest.raises(RuntimeError, match="rejected all proposals"):
        _topology_preserving_cage_registration(
            nodes,
            elements,
            np.arange(4, dtype=np.int32),
            faces,
            nodes + 0.01,
            faces,
        )


def test_query_points_cannot_grow_source_cage(tmp_path: Path) -> None:
    assert tuple(inspect.signature(_build_source_cage).parameters) == (
        "vertices",
        "faces",
        "cache_path",
    )
    assert not (tmp_path / "must_not_exist.npz").exists()
