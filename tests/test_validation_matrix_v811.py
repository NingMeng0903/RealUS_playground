from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from projects.genesis_ue_sync.anatomy_retarget.leg_centerline_v810 import (
    _foot_chain_digest_v1,
)
from projects.genesis_ue_sync.anatomy_retarget import validation_matrix_v8
from projects.genesis_ue_sync.anatomy_retarget.source_skin_volume import (
    source_skinning_topology_digest_v811,
)
from projects.genesis_ue_sync.anatomy_retarget.validation_matrix_v8 import (
    MatrixBodySurfaceV811,
    _foot_chain_gate_v811,
    _hand_foot_bone_containment_gate_v811,
    _tube_containment_gate_v811,
    _v811_contract_gate,
)


_DIGEST = "a" * 64
_BODY_BETAS = np.zeros(10, dtype=np.float32)


def _foot_chain(schema_version: object = 1) -> dict[str, object]:
    mesh = {
        "mesh": "foot",
        "station_segment": "arch_forefoot",
        "rigid_rms_error_m": 0.0,
        "rigid_maximum_error_m": 0.0,
        "det_rotation": 1.0,
        "scale": 1.0,
    }
    stations = {
        name: {
            "source_m": [0.0, 0.0, 0.0],
            "target_m": [0.0, 0.0, 0.0],
            "mapped_geometry_m": [0.0, 0.0, 0.0],
            "residual_m": 0.0,
        }
        for name in ("ankle", "arch", "forefoot")
    }
    target_arch = {
        "method": "ankle_guided_unit_so3_source_arch_offset_v811",
        "authority": (
            "smplx_ankle_and_forefoot_guides_with_source_anatomical_arch_offset"
        ),
        "smplx_arch_joint_available": False,
    }
    chain = {
        "schema_version": schema_version,
        "method": "multi_station_rigid_foot_chain_v811",
        "sides": {
            "left": {
                "station_residual_m": 0.0,
                "stations": {name: dict(value) for name, value in stations.items()},
                "target_arch_construction": dict(target_arch),
                "per_mesh": [dict(mesh)],
            },
            "right": {
                "station_residual_m": 0.0,
                "stations": {name: dict(value) for name, value in stations.items()},
                "target_arch_construction": dict(target_arch),
                "per_mesh": [dict(mesh)],
            },
        },
    }
    chain["content_digest"] = _foot_chain_digest_v1(chain)
    return chain


def _subject(chain: dict[str, object]) -> SimpleNamespace:
    return SimpleNamespace(
        audit_report={"leg_centerline_v810": {"foot_chain_stations_v1": chain}},
        rigged_asset=SimpleNamespace(metadata={}),
    )


def _selective_asset() -> SimpleNamespace:
    return SimpleNamespace(
        source_bone_names=(),
        faces=np.empty((0, 3), dtype=np.int32),
        driver_indices=np.zeros((1, 14), dtype=np.int16),
        driver_weights=np.ones((1, 14), dtype=np.float32),
        source_vertex_ranges=np.asarray(((0, 1),), dtype=np.int32),
        source_mesh_names=("Vessel",),
        source_tissues=("vessel",),
        source_bind_vertices=np.zeros((1, 3), dtype=np.float32),
        metadata={
            "source_fk_policy_v4": "selective_authority",
            "source_full_local_fk_v2": False,
            "source_connected_local_fk_v3": False,
            "source_local_fk_bones_v3": [],
            "source_direct_driver_bones_v1": [],
            "source_leg_compound_roots_v1": {},
        },
    )


def _complete_correction(route: dict[str, object]) -> dict[str, object]:
    source_skinning_digest = source_skinning_topology_digest_v811(
        _selective_asset()
    )
    return {
        "source_skin_volume_v811": {
            "available": True,
            "passed": True,
            "schema_version": 1,
            "artifact_kind": "SourceSkinVolumeRegistrationV811",
            "content_digest": _DIGEST,
            "anatomy_transport": "soft_material_only_volume_field_v811",
            "soft_volume_tissues": [
                "vessel",
                "nerve",
                "organ",
                "heart",
                "connective_tissue",
            ],
            "topology_preserved": True,
            "source_weights_preserved": True,
            "source_skinning_topology_digest_before": source_skinning_digest,
            "source_skinning_topology_digest_after": source_skinning_digest,
            "source_skinning_topology_byte_identical": True,
            "source_vertex_order_preserved": True,
            "source_driver_slot_count": 14,
            "protected_material_preserved": True,
            "nonsoft_material_preserved": True,
            "rigid_hard_protection_preserved": True,
            "source_rig_rebind": {"rebound": False},
            "source_soft_prewrap": {
                "backend": "source_skin_local_normal_projection_laplacian_v811",
                "strict_passed": True,
                "topology_preserved": True,
                "source_weights_preserved": True,
                "protected_vertices_preserved": True,
            },
        },
        "source_skin_volume_beta_basis_v1": {
            "available": True,
            "passed": True,
            "content_digest": _DIGEST,
        },
        "head_compound_fit_v1": {
            "available": True,
            "passed": True,
            "content_digest": _DIGEST,
            "outside_count": 0,
            "center_drift_m": 0.0,
            "target_scale_loss": 0.0,
            "clearance_m": 0.0015,
            "uniform_scale": 1.0,
            "robust_target_scale": 1.0,
            "nonuniform_scale": False,
        },
        "tube_pose_corrective_v1": {
            "available": True,
            "passed": True,
            "content_digest": _DIGEST,
            "schema": "tube_pose_corrective_v1",
            "sample_count": 3,
            "vertex_count": 2,
            "driver_joint_count": 2,
            "runtime_spatial_query": False,
            "runtime_graph_solve": False,
            "runtime_collision": False,
        },
        "vessel_route_v8": route,
    }


def test_foot_chain_gate_turns_malformed_schema_into_failed_evidence() -> None:
    passed, report = _foot_chain_gate_v811(_subject(_foot_chain("not-an-integer")))

    assert passed is False
    assert "schema_version" in report["failures"]


def test_foot_chain_gate_rejects_a_named_station_over_two_millimetres() -> None:
    chain = _foot_chain()
    arch = chain["sides"]["left"]["stations"]["arch"]
    arch["mapped_geometry_m"] = [0.0021, 0.0, 0.0]
    arch["residual_m"] = 0.0021

    passed, report = _foot_chain_gate_v811(_subject(chain))

    assert passed is False
    assert "left.arch.residual" in report["failures"]


def test_foot_chain_gate_rejects_a_false_summary_residual() -> None:
    chain = _foot_chain()
    arch = chain["sides"]["left"]["stations"]["arch"]
    arch["mapped_geometry_m"] = [0.001, 0.0, 0.0]
    arch["residual_m"] = 0.001

    passed, report = _foot_chain_gate_v811(_subject(chain))

    assert passed is False
    assert "left.station_residual_consistency" in report["failures"]


def test_foot_chain_gate_recomputes_the_content_digest() -> None:
    chain = _foot_chain()
    chain["sides"]["left"]["per_mesh"][0]["mesh"] = "other-foot"

    passed, report = _foot_chain_gate_v811(_subject(chain))

    assert passed is False
    assert "content_digest" in report["failures"]


def test_v811_route_requires_vessel_and_nerve_and_never_raises_on_bad_counts() -> None:
    route = {
        "available": True,
        "passed": True,
        "tissues": ["vessel"],
        "skin_outside_count": "not-an-integer",
        "bone_clearance_violation_count": "not-an-integer",
        "edge_relative_change_q99": 0.0,
        "skin_margin_m": 0.00025,
        "bone_clearance_m": 0.00025,
        "source_reconstruction": {"skipped": True},
    }
    operator = SimpleNamespace(
        template_asset=_selective_asset(),
        correction_report=_complete_correction(route),
    )
    matrix_subject = SimpleNamespace(label="reference", subject=_subject(_foot_chain()))

    result = _v811_contract_gate(operator, [matrix_subject])

    assert result["pass"] is False
    assert result["checks"]["vessel_nerve_route"]["pass"] is False
    assert result["checks"]["vessel_nerve_route"]["tissues"] == ["vessel"]


def test_v811_volume_requires_rigid_hard_protection_evidence() -> None:
    route = {
        "available": True,
        "passed": True,
        "tissues": ["vessel", "nerve"],
        "skin_outside_count": 0,
        "bone_clearance_violation_count": 0,
        "edge_relative_change_q99": 0.0,
        "skin_margin_m": 0.00025,
        "bone_clearance_m": 0.00025,
        "source_reconstruction": {"skipped": True},
    }
    correction = _complete_correction(route)
    correction["source_skin_volume_v811"].pop(
        "rigid_hard_protection_preserved"
    )
    operator = SimpleNamespace(
        template_asset=_selective_asset(),
        correction_report=correction,
    )
    matrix_subject = SimpleNamespace(label="reference", subject=_subject(_foot_chain()))

    result = _v811_contract_gate(operator, [matrix_subject])

    assert result["pass"] is False
    assert result["checks"]["source_skin_volume_v811"]["pass"] is False


def test_v811_volume_rejects_unverified_source_skin_prewrap() -> None:
    route = {
        "available": True,
        "passed": True,
        "tissues": ["vessel", "nerve"],
        "skin_outside_count": 0,
        "bone_clearance_violation_count": 0,
        "edge_relative_change_q99": 0.0,
        "skin_margin_m": 0.00025,
        "bone_clearance_m": 0.00025,
        "source_reconstruction": {"skipped": True},
    }
    correction = _complete_correction(route)
    correction["source_skin_volume_v811"]["source_soft_prewrap"][
        "strict_passed"
    ] = False
    operator = SimpleNamespace(
        template_asset=_selective_asset(),
        correction_report=correction,
    )
    matrix_subject = SimpleNamespace(label="reference", subject=_subject(_foot_chain()))

    result = _v811_contract_gate(operator, [matrix_subject])

    assert result["pass"] is False
    assert result["checks"]["source_skin_volume_v811"]["source_skin_prewrap"] is False


def test_v811_volume_requires_byte_exact_source_skinning_evidence() -> None:
    route = {
        "available": True,
        "passed": True,
        "tissues": ["vessel", "nerve"],
        "skin_outside_count": 0,
        "bone_clearance_violation_count": 0,
        "edge_relative_change_q99": 0.0,
        "skin_margin_m": 0.00025,
        "bone_clearance_m": 0.00025,
        "source_reconstruction": {"skipped": True},
    }
    correction = _complete_correction(route)
    correction["source_skin_volume_v811"][
        "source_skinning_topology_digest_after"
    ] = "b" * 64
    operator = SimpleNamespace(
        template_asset=_selective_asset(),
        correction_report=correction,
    )
    matrix_subject = SimpleNamespace(label="reference", subject=_subject(_foot_chain()))

    result = _v811_contract_gate(operator, [matrix_subject])

    assert result["pass"] is False
    assert result["checks"]["source_skin_volume_v811"]["immutable_source_skinning"] is False


def _body_surface(
    *, canonical_betas: np.ndarray | None = _BODY_BETAS
) -> MatrixBodySurfaceV811:
    vertices = np.asarray(
        (
            (-1.0, -1.0, -1.0),
            (1.0, -1.0, -1.0),
            (1.0, 1.0, -1.0),
            (-1.0, 1.0, -1.0),
            (-1.0, -1.0, 1.0),
            (1.0, -1.0, 1.0),
            (1.0, 1.0, 1.0),
            (-1.0, 1.0, 1.0),
        ),
        dtype=np.float32,
    )
    faces = np.asarray(
        (
            (0, 2, 1),
            (0, 3, 2),
            (4, 5, 6),
            (4, 6, 7),
            (0, 1, 5),
            (0, 5, 4),
            (1, 2, 6),
            (1, 6, 5),
            (2, 3, 6),
            (2, 7, 6),
            (3, 0, 4),
            (3, 4, 7),
        ),
        dtype=np.int32,
    )
    weights = np.zeros((len(vertices), 55), dtype=np.float32)
    weights[:, 0] = 1.0
    inverse = np.tile(np.eye(4, dtype=np.float32), (55, 1, 1))
    return MatrixBodySurfaceV811(
        vertices=vertices,
        faces=faces,
        lbs_weights=weights,
        rest_joints=np.zeros((55, 3), dtype=np.float32),
        parents=np.asarray((-1,) + (0,) * 54, dtype=np.int32),
        inverse_bind=inverse,
        source="synthetic-cube",
        canonical_betas=canonical_betas,
        canonical_manifest_digest=_DIGEST,
        canonical_source_identity="synthetic-test-canonical",
    )


def test_v811_hard_hand_foot_gate_rejects_half_millimetre_surface_crossing() -> None:
    asset = SimpleNamespace(
        vertices_rest=np.zeros((2, 3), dtype=np.float32),
        source_vertex_ranges=np.asarray(((0, 1), (1, 2)), dtype=np.int32),
        source_tissues=("bone", "bone"),
        source_mesh_names=("Metacarpal_L", "Talus_R"),
        source_sides=("left", "right"),
        rest_joints=np.zeros((55, 3), dtype=np.float32),
        parents=np.asarray((-1,) + (0,) * 54, dtype=np.int32),
    )
    pose = np.zeros((55, 3), dtype=np.float32)

    rejected = _hand_foot_bone_containment_gate_v811(
        asset,
        np.asarray(((0.0, 0.0, 0.0), (1.0006, 0.0, 0.0)), dtype=np.float32),
        body_surface=_body_surface(),
        subject_betas=_BODY_BETAS,
        pose_axis_angle=pose,
        transl=np.zeros(3, dtype=np.float32),
    )

    assert rejected["available"] is True
    assert rejected["pass"] is False
    assert rejected["outside_count"] == 1
    assert set(rejected["regions"]) == {"left/hand", "right/foot"}

    accepted = _hand_foot_bone_containment_gate_v811(
        asset,
        np.asarray(((0.0, 0.0, 0.0), (0.9996, 0.0, 0.0)), dtype=np.float32),
        body_surface=_body_surface(),
        subject_betas=_BODY_BETAS,
        pose_axis_angle=pose,
        transl=np.zeros(3, dtype=np.float32),
    )

    assert accepted["pass"] is True
    assert accepted["beta_provenance"]["canonical_manifest_digest"] == _DIGEST


def test_v811_hard_hand_foot_gate_requires_matching_canonical_beta_provenance() -> None:
    asset = SimpleNamespace(
        vertices_rest=np.zeros((2, 3), dtype=np.float32),
        source_vertex_ranges=np.asarray(((0, 1), (1, 2)), dtype=np.int32),
        source_tissues=("bone", "bone"),
        source_mesh_names=("Metacarpal_L", "Talus_R"),
        source_sides=("left", "right"),
        rest_joints=np.zeros((55, 3), dtype=np.float32),
        parents=np.asarray((-1,) + (0,) * 54, dtype=np.int32),
    )
    posed = np.zeros((2, 3), dtype=np.float32)
    pose = np.zeros((55, 3), dtype=np.float32)

    missing = _hand_foot_bone_containment_gate_v811(
        asset,
        posed,
        body_surface=replace(_body_surface(), canonical_betas=None),
        subject_betas=_BODY_BETAS,
        pose_axis_angle=pose,
        transl=np.zeros(3, dtype=np.float32),
    )
    missing_subject = _hand_foot_bone_containment_gate_v811(
        asset,
        posed,
        body_surface=_body_surface(),
        pose_axis_angle=pose,
        transl=np.zeros(3, dtype=np.float32),
    )
    mismatched_betas = _BODY_BETAS.copy()
    mismatched_betas[0] = np.float32(0.1)
    mismatched = _hand_foot_bone_containment_gate_v811(
        asset,
        posed,
        body_surface=_body_surface(canonical_betas=mismatched_betas),
        subject_betas=_BODY_BETAS,
        pose_axis_angle=pose,
        transl=np.zeros(3, dtype=np.float32),
    )

    assert missing["available"] is False
    assert missing["pass"] is False
    assert "beta provenance" in missing["reason"]
    assert missing_subject["available"] is False
    assert missing_subject["pass"] is False
    assert "subject beta identity" in missing_subject["reason"]
    assert mismatched["available"] is False
    assert mismatched["pass"] is False
    assert "do not match" in mismatched["reason"]


def _tube_gate_asset() -> SimpleNamespace:
    """Minimal final-L1 layout: one tube point and one named bone surface."""

    return SimpleNamespace(
        vertices_rest=np.asarray(
            (
                (0.0, 0.0, 0.0),
                (-0.10, -0.10, -0.10),
                (0.10, -0.10, -0.10),
                (0.0, 0.10, -0.10),
                (0.0, 0.0, 0.10),
            ),
            dtype=np.float32,
        ),
        faces=np.asarray(
            (
                (1, 3, 2),
                (1, 2, 4),
                (1, 4, 3),
                (2, 3, 4),
            ),
            dtype=np.int32,
        ),
        source_vertex_ranges=np.asarray(((0, 1), (1, 5)), dtype=np.int32),
        source_tissues=("nerve", "bone"),
        source_mesh_names=("SyntheticNerve", "Femur_L"),
        source_sides=("left", "left"),
        rest_joints=np.zeros((55, 3), dtype=np.float32),
        parents=np.asarray((-1,) + (0,) * 54, dtype=np.int32),
    )


def test_v811_final_tube_gate_rejects_post_materialization_bone_collision(
    monkeypatch: object,
) -> None:
    """An L0 route report cannot hide a bone moved by the final leg chain."""

    def signed_distance(
        points: np.ndarray,
        surface_vertices: np.ndarray,
        _surface_faces: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # The canonical cube is always safe; the named Femur surface encloses
        # the nerve.  This deliberately models an L1-only collision.
        signed = np.full(
            len(points),
            -0.010 if len(surface_vertices) == 8 else -0.001,
            dtype=np.float64,
        )
        normals = np.tile(np.asarray((1.0, 0.0, 0.0)), (len(points), 1))
        return signed, np.asarray(points, dtype=np.float64), normals

    monkeypatch.setattr(validation_matrix_v8, "signed_distance", signed_distance)
    pose = np.zeros((55, 3), dtype=np.float32)
    report = _tube_containment_gate_v811(
        _tube_gate_asset(),
        _tube_gate_asset().vertices_rest,
        body_surface=_body_surface(),
        subject_betas=_BODY_BETAS,
        pose_axis_angle=pose,
        transl=np.zeros(3, dtype=np.float32),
    )

    assert report["available"] is True
    assert report["skin_outside_count"] == 0
    assert report["bone_clearance_violation_count"] == 1
    assert report["pass"] is False
    assert report["collision_violations"][0]["name"] == "Femur_L"


def test_v811_final_tube_gate_accepts_clear_final_geometry(
    monkeypatch: object,
) -> None:
    def signed_distance(
        points: np.ndarray,
        surface_vertices: np.ndarray,
        _surface_faces: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        signed = np.full(
            len(points),
            -0.010 if len(surface_vertices) == 8 else 0.001,
            dtype=np.float64,
        )
        normals = np.tile(np.asarray((1.0, 0.0, 0.0)), (len(points), 1))
        return signed, np.asarray(points, dtype=np.float64), normals

    monkeypatch.setattr(validation_matrix_v8, "signed_distance", signed_distance)
    asset = _tube_gate_asset()
    report = _tube_containment_gate_v811(
        asset,
        asset.vertices_rest,
        body_surface=_body_surface(),
        subject_betas=_BODY_BETAS,
        pose_axis_angle=np.zeros((55, 3), dtype=np.float32),
        transl=np.zeros(3, dtype=np.float32),
    )

    assert report["available"] is True
    assert report["skin_outside_count"] == 0
    assert report["skin_clearance_violation_count"] == 0
    assert report["bone_clearance_violation_count"] == 0
    assert report["pass"] is True
