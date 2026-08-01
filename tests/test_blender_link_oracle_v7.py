from __future__ import annotations

from pathlib import Path

import numpy as np

from src.projects.genesis_ue_sync.anatomy_retarget.blender_link_oracle_v7 import (
    EXPECTED_OPERATOR_RIG_ARRAY_SHA256,
    _basis_fk_metrics,
    _frozen_operator_contract,
    _lbs_mesh_metrics,
    _matrix_metrics,
)
from src.projects.genesis_ue_sync.anatomy_retarget.v8_artifacts import (
    load_source_operator,
)


def _translation(x: float, y: float, z: float) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, 3] = (x, y, z)
    return result


def _rotation_z(angle: float) -> np.ndarray:
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    result = np.eye(4, dtype=np.float64)
    result[:2, :2] = ((cosine, -sine), (sine, cosine))
    return result


def test_basis_fk_rebuilds_global_action_without_derived_locals() -> None:
    parents = np.asarray((-1, 0), dtype=np.int32)
    rest_global = np.asarray(
        (_translation(0.2, -0.1, 0.0), _translation(1.2, -0.1, 0.0))
    )
    rest_local = rest_global.copy()
    rest_local[1] = np.linalg.inv(rest_global[0]) @ rest_global[1]
    basis = np.asarray(
        (
            (np.eye(4), np.eye(4)),
            (_translation(0.1, 0.0, 0.0) @ _rotation_z(0.2), _rotation_z(-0.4)),
        )
    )
    global_matrices = np.empty_like(basis)
    for frame in range(len(basis)):
        global_matrices[frame, 0] = rest_global[0] @ basis[frame, 0]
        global_matrices[frame, 1] = (
            global_matrices[frame, 0] @ rest_local[1] @ basis[frame, 1]
        )
    derived_local = global_matrices.copy()
    derived_local[:, 1] = (
        np.linalg.inv(global_matrices[:, 0]) @ global_matrices[:, 1]
    )

    serialization = _matrix_metrics(
        global_matrices=global_matrices,
        local_matrices=derived_local,
        parents=parents,
        unit_scale_m=0.01,
    )
    independent = _basis_fk_metrics(
        global_matrices=global_matrices,
        basis_matrices=basis,
        rest_global=rest_global,
        rest_local=rest_local,
        parents=parents,
        unit_scale_m=0.01,
    )

    assert serialization["pass"]
    assert independent["pass"]


def test_lbs_checker_reproduces_dynamic_non_root_mesh() -> None:
    parents = np.asarray((-1, 0), dtype=np.int32)
    rest_global = np.asarray((np.eye(4), _translation(1.0, 0.0, 0.0)))
    action_global = np.asarray(
        (
            rest_global,
            (np.eye(4), rest_global[1] @ _rotation_z(np.pi / 3.0)),
        )
    )
    rest = np.asarray(((1.0, 1.0, 0.0), (1.0, 2.0, 0.0)), dtype=np.float64)
    indices = np.zeros((len(rest), 14), dtype=np.int16)
    weights = np.zeros((len(rest), 14), dtype=np.float32)
    indices[:, 0] = 1
    weights[:, 0] = 1.0
    inverse_bind = np.linalg.inv(rest_global)
    evaluated = []
    for global_frame in action_global:
        transforms = global_frame @ inverse_bind
        selected = transforms[indices[:, 0]]
        evaluated.append(
            np.einsum("nij,nj->ni", selected[:, :3, :3], rest)
            + selected[:, :3, 3]
        )
    data = {
        "mesh__Synthetic__bind_vertices": rest.astype(np.float32),
        "mesh__Synthetic__vertices": np.asarray(evaluated, dtype=np.float32),
        "mesh__Synthetic__driver_indices": indices,
        "mesh__Synthetic__driver_weights": weights,
    }

    metrics = _lbs_mesh_metrics(
        data,
        name="Synthetic",
        rest_global=rest_global,
        action_global=action_global,
        action_frames=np.asarray((0, 1), dtype=np.int32),
        mesh_frames=np.asarray((0, 1), dtype=np.int32),
        parents=parents,
        unit_scale_m=0.01,
    )

    assert metrics["pass"]
    assert metrics["expected_non_root_dynamic"]
    assert metrics["non_root_dynamic"]
    assert not metrics["expected_dynamic_but_static"]


def test_frozen_142_operator_contract_includes_connect_and_inherit_scale() -> None:
    root = (
        Path(__file__).resolve().parents[1]
        / "outputs/anatomy_retarget/v8_candidates/rebuild_012/source_operator_v8"
    )
    if not root.is_dir():
        import pytest

        pytest.skip("frozen rebuild_012 operator is unavailable")
    operator = load_source_operator(root, mmap=True)
    report = _frozen_operator_contract(operator, root)
    assert report["pass"] is True
    assert report["bone_count"] == 235
    assert report["array_sha256"] == EXPECTED_OPERATOR_RIG_ARRAY_SHA256
    assert set(report["array_sha256"]) == {
        "parents",
        "rest_global",
        "rest_local",
        "use_connect",
        "inherit_scale",
    }
