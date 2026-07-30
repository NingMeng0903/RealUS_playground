from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from projects.genesis_ue_sync.anatomy_retarget import operator_bake_v8
from projects.genesis_ue_sync.anatomy_retarget.operator_bake_v8 import (
    _constraint_surface_v811,
    _has_anatomical_leg_guide_v810,
    _soft_volume_beta_basis_v811,
    sanitize_v8_runtime_metadata,
)


def test_runtime_metadata_removes_every_old_leg_and_patella_path() -> None:
    cleaned = sanitize_v8_runtime_metadata(
        {
            "source_full_local_fk_v2": True,
            "source_anatomical_guide_fk_v810": True,
            "source_leg_hinge_solve_v1": {"left": 1},
            "source_knee_hinge_splines_v7": {"left": 2},
            "nested": {
                "source_tibia_glide_splines_v7": {"left": 3},
                "source_patella_v71_response_v8": {"left": 4},
            },
            "unrelated": np.int64(5),
        }
    )
    encoded = repr(cleaned)
    for marker in (
        "source_leg_hinge_solve_v1",
        "source_knee_hinge_splines_v7",
        "source_tibia_glide_splines_v7",
        "source_patella_v71_response_v8",
    ):
        # The audit list records what was removed, but no executable key remains.
        assert marker not in cleaned
        assert marker not in cleaned["nested"]
    assert cleaned["unrelated"] == 5
    assert cleaned["source_fk_policy_v4"] == "selective_authority"
    assert cleaned["source_full_local_fk_v2"] is False
    assert cleaned["source_anatomical_guide_fk_v810"] is True
    assert cleaned["disable_soft_follow"] is True


def test_v811_uses_required_volume_shell_when_route_surface_is_omitted(
    monkeypatch,
) -> None:
    expected_vertices = np.asarray(((0.0, 0.0, 0.0),), dtype=np.float32)
    expected_faces = np.asarray(((0, 0, 0),), dtype=np.int32)
    seen: list[object] = []

    def load(path):
        seen.append(path)
        return expected_vertices, expected_faces

    monkeypatch.setattr(operator_bake_v8, "load_body_surface", load)
    vertices, faces = _constraint_surface_v811(
        None,
        None,
        source_skin_volume_dir="canonical-volume",
    )

    assert seen and str(seen[0]).endswith("smpl_canonical_tpose.obj")
    assert vertices is expected_vertices
    assert faces is expected_faces


def test_v811_soft_beta_basis_requires_every_soft_vertex(tmp_path) -> None:
    template = type(
        "Template",
        (),
        {
            "vertices_rest": np.zeros((4, 3), dtype=np.float32),
            "source_vertex_ranges": np.asarray(((0, 1), (1, 3), (3, 4))),
            "source_tissues": ("bone", "vessel", "nerve"),
        },
    )()
    np.savez(
        tmp_path / "source_skin_volume_beta_basis_v1.npz",
        vertex_ids=np.asarray((1,), dtype=np.int32),
        displacement_basis_m=np.zeros((10, 1, 3), dtype=np.float32),
    )

    with pytest.raises(ValueError, match="cover every V8.11 soft-tissue"):
        _soft_volume_beta_basis_v811(
            template=template,
            source_basis=np.zeros((10, 4, 3), dtype=np.float32),
            source_skin_volume_dir=tmp_path,
        )


def test_v811_soft_beta_basis_excludes_rigid_cranial_organs() -> None:
    template = SimpleNamespace(
        vertices_rest=np.zeros((3, 3), dtype=np.float32),
        source_vertex_ranges=np.asarray(((0, 1), (1, 2), (2, 3)), dtype=np.int32),
        source_tissues=("bone", "organ", "vessel"),
        source_mesh_names=("Skull", "Cerebrum", "Neck_Vessel"),
        source_bone_names=("Root", "Head_Bone", "Brain_Follow"),
        source_bone_parents=np.asarray((-1, 0, 1), dtype=np.int32),
        driver_indices=np.asarray(((1,), (2,), (0,)), dtype=np.int16),
        driver_weights=np.ones((3, 1), dtype=np.float32),
    )

    basis, report = _soft_volume_beta_basis_v811(
        template=template,
        source_basis=np.ones((10, 3, 3), dtype=np.float32),
        source_skin_volume_dir=None,
    )

    np.testing.assert_array_equal(basis[:, :2], 0.0)
    np.testing.assert_array_equal(basis[:, 2], 1.0)
    assert report["soft_vertex_count"] == 1


def test_v811_anatomical_leg_guide_requires_complete_non_degenerate_chains() -> None:
    names = [f"joint_{index}" for index in range(55)]
    for index, name in enumerate(
        (
            "left_hip",
            "left_knee",
            "left_ankle",
            "left_foot",
            "right_hip",
            "right_knee",
            "right_ankle",
            "right_foot",
        )
    ):
        names[index] = name
    stations = np.zeros((55, 3), dtype=np.float32)
    stations[:4, 1] = (0.0, -0.4, -0.8, -1.0)
    stations[4:8, 0] = 0.2
    stations[4:8, 1] = (0.0, -0.4, -0.8, -1.0)
    asset = SimpleNamespace(
        joint_names=names,
        source_driver_rest_joints=stations,
    )

    assert _has_anatomical_leg_guide_v810(asset) is True

    asset.source_driver_rest_joints[2] = asset.source_driver_rest_joints[1]
    assert _has_anatomical_leg_guide_v810(asset) is False
