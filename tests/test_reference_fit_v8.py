from types import SimpleNamespace

import numpy as np
import pytest

from projects.genesis_ue_sync.anatomy_retarget.reference_fit_v8 import (
    _hard_appendicular_bind_mask,
    _is_axial_product_bone,
    hard_appendicular_product_proof_v811,
    restore_unit_hard_product_v811,
)


@pytest.mark.parametrize(
    "mesh_name",
    [
        "Upper_Skull",
        "Mandible",
        "C1_Atlas",
        "C2_Axis",
        "C3",
        "C7",
        "T1",
        "T12",
        "L1",
        "L5",
        "Disc_C2_C3",
        "Disc_T12_L1",
        "Rib_1L",
        "Rib_12R",
        "Clavicle_L",
        "Scapula_R",
        "Sternum",
        "Central_Incisor",
        "Canine_4",
        "Premolar_2nd_3",
        "Molar_3rd_4",
    ],
)
def test_head_neck_thorax_compound_uses_one_continuous_authority(
    mesh_name: str,
) -> None:
    assert _is_axial_product_bone(mesh_name)


@pytest.mark.parametrize(
    "mesh_name",
    ["Femur_L", "Tibia_R", "Humerus_L", "Radius_R", "Ulna_L"],
)
def test_long_bones_stay_on_nonshrunk_authority(mesh_name: str) -> None:
    assert not _is_axial_product_bone(mesh_name)


def test_hard_bind_authority_includes_pelvis_root_and_limb_subtrees() -> None:
    names = [
        "Skeleton_SRT",
        "Hip_bone",
        "Femur_Rot_L",
        "Knee_Rotate_L",
        "Femur_Rot_R",
        "Knee_Rotate_R",
        "Spine_L5",
        "Shoulder_Rotate_L",
        "Elbow_Rot_L",
        "Shoulder_Rotate_R",
        "Elbow_Rot_R",
    ]
    parents = np.asarray((-1, 0, 1, 2, 1, 4, 1, 6, 7, 6, 9), dtype=np.int32)

    mask = _hard_appendicular_bind_mask(names, parents)

    selected = {name for name, keep in zip(names, mask, strict=True) if keep}
    assert "Hip_bone" in selected
    assert "Femur_Rot_L" in selected
    assert "Knee_Rotate_R" in selected
    assert "Elbow_Rot_L" in selected
    assert "Spine_L5" not in selected


def _hard_product(*, thin_femur: bool = False) -> SimpleNamespace:
    femur = np.asarray(
        (
            (-0.02, 0.00, 0.00),
            (0.02, 0.00, 0.00),
            (0.00, 0.00, -0.02),
            (0.00, 0.00, 0.02),
            (-0.02, -0.40, 0.00),
            (0.02, -0.40, 0.00),
            (0.00, -0.40, -0.02),
            (0.00, -0.40, 0.02),
        ),
        dtype=np.float32,
    )
    foot = np.asarray(
        (
            (-0.04, -0.42, -0.02),
            (0.04, -0.42, -0.02),
            (0.00, -0.42, 0.06),
            (0.00, -0.50, 0.00),
        ),
        dtype=np.float32,
    )
    source = np.concatenate((femur, foot), axis=0)
    target = source.copy()
    target[:, 0] += 0.12
    target[:, 2] -= 0.05
    if thin_femur:
        target[: len(femur), (0, 2)] = (
            target[: len(femur), (0, 2)] * np.float32(0.90)
        )
    return SimpleNamespace(
        validate=lambda: None,
        source_bind_vertices=source,
        vertices_rest=target,
        source_vertex_ranges=np.asarray(
            ((0, len(femur)), (len(femur), len(source))), dtype=np.int32
        ),
        source_mesh_names=("Femur_L", "Talus_L"),
        source_tissues=("bone", "bone"),
    )


def test_hard_product_proof_accepts_unit_scale_rigid_appendicular_meshes() -> None:
    proof = hard_appendicular_product_proof_v811(
        _hard_product(), product_label="synthetic"
    )

    assert proof["passed"] is True
    assert proof["failures"] == []
    assert proof["meshes"][0]["transverse_scale"] == pytest.approx(1.0)
    assert proof["meshes"][1]["similarity_scale"] == pytest.approx(1.0)


def test_hard_product_proof_rejects_thinned_long_bone_cross_section() -> None:
    proof = hard_appendicular_product_proof_v811(
        _hard_product(thin_femur=True), product_label="shrunk"
    )

    assert proof["passed"] is False
    assert proof["failures"] == ["Femur_L"]


def test_unit_hard_restoration_uses_source_shape_and_target_bind_se3() -> None:
    product = _hard_product(thin_femur=True)
    product.source_mesh_controller_bones = np.asarray((0, 1), dtype=np.int32)
    product.source_bone_names = ("Femur_Control", "Foot_Control")
    product.source_bind_global = np.tile(np.eye(4), (2, 1, 1))
    product.target_bind_global = product.source_bind_global.copy()
    product.target_bind_global[0, :3, 3] = (0.12, 0.03, -0.05)
    product.target_bind_global[1, :3, 3] = (-0.07, 0.02, 0.08)
    product.metadata = {}

    # SimpleNamespace is convenient for the proof-only fixture above; this
    # local dataclass adapter gives dataclasses.replace the production shape it
    # expects without constructing the unrelated full rig schema.
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class Product:
        source_bind_vertices: np.ndarray
        vertices_rest: np.ndarray
        source_vertex_ranges: np.ndarray
        source_mesh_names: tuple[str, ...]
        source_tissues: tuple[str, ...]
        source_mesh_controller_bones: np.ndarray
        source_bone_names: tuple[str, ...]
        source_bind_global: np.ndarray
        target_bind_global: np.ndarray
        joint_names: tuple[str, ...]
        rest_joints: np.ndarray
        metadata: dict

        def validate(self) -> None:
            return None

    fixture = Product(
        source_bind_vertices=product.source_bind_vertices,
        vertices_rest=product.vertices_rest,
        source_vertex_ranges=product.source_vertex_ranges,
        source_mesh_names=product.source_mesh_names,
        source_tissues=product.source_tissues,
        source_mesh_controller_bones=product.source_mesh_controller_bones,
        source_bone_names=product.source_bone_names,
        source_bind_global=product.source_bind_global,
        target_bind_global=product.target_bind_global,
        joint_names=("left_hip", "left_knee"),
        rest_joints=np.asarray(((0.0, 0.0, 0.0), (0.0, -0.40, 0.0))),
        metadata=product.metadata,
    )

    restored, report = restore_unit_hard_product_v811(fixture)

    femur_stop = int(fixture.source_vertex_ranges[0, 1])
    expected_femur = (
        fixture.source_bind_vertices[:femur_stop]
        + fixture.target_bind_global[0, :3, 3]
    )
    expected_femur[:, 1] = fixture.vertices_rest[:femur_stop, 1]
    np.testing.assert_allclose(
        restored.vertices_rest[:femur_stop],
        expected_femur,
        atol=1.0e-7,
    )
    assert report["hard_product_proof"]["passed"] is True
    assert report["scale"] == 1.0
