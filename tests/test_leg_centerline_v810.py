from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import projects.genesis_ue_sync.anatomy_retarget.v8_artifacts as v8_artifacts
from projects.genesis_ue_sync.anatomy_retarget.leg_centerline_v810 import (
    LEG_CENTERLINE_SCHEMA_VERSION_V810,
    _CENTERLINE_EDGE_MAX_LIMIT,
    _CENTERLINE_EDGE_Q99_LIMIT,
    _apply_rigid_segment_rotation_v810,
    _apply_swept_segment_centerline_v810,
    _foot_arch_station_v811,
    _map_foot_stations_rigid_v811,
    _foot_station_v810,
    _proximal_mesh_cap_ids,
    _target_foot_arch_station_v811,
    has_leg_centerline_v810,
    reconstruct_leg_centerline_compounds_v810,
    transport_coupled_rbf_parent_frames_v810,
)
from projects.genesis_ue_sync.anatomy_retarget.fk_policy_v8 import (
    SELECTIVE_AUTHORITY_FK_POLICY_V4,
)
from projects.genesis_ue_sync.anatomy_retarget.mechanism_v8 import (
    fit_projected_station_rest_v810,
)
from projects.genesis_ue_sync.anatomy_retarget.version_v8 import (
    SOURCE_OPERATOR_ALGORITHM_VERSION,
    SOURCE_OPERATOR_CORRECTION_VERSION,
    SOURCE_OPERATOR_ORACLE_VERSION,
    SUBJECT_SOLVER_VERSION,
)


def _rotation_z(angle: float) -> np.ndarray:
    cosine = np.cos(angle)
    sine = np.sin(angle)
    return np.asarray(
        (
            (cosine, -sine, 0.0),
            (sine, cosine, 0.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )


def _cylinder(
    *,
    length: float,
    stations: int = 101,
    rings: int = 16,
    radius: float = 0.020,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fractions = np.linspace(0.0, 1.0, stations, dtype=np.float64)
    angles = np.linspace(0.0, 2.0 * np.pi, rings, endpoint=False)
    vertices = np.asarray(
        [
            (
                radius * np.cos(angle),
                -length * fraction,
                radius * np.sin(angle),
            )
            for fraction in fractions
            for angle in angles
        ],
        dtype=np.float64,
    )
    faces: list[tuple[int, int, int]] = []
    for station in range(stations - 1):
        first = station * rings
        second = (station + 1) * rings
        for ring in range(rings):
            nxt = (ring + 1) % rings
            faces.append((first + ring, second + ring, second + nxt))
            faces.append((first + ring, second + nxt, first + nxt))
    return vertices, np.asarray(faces, dtype=np.int32), fractions


def _driver_segment(*, length: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    start = np.asarray((0.080, 0.025, -0.035), dtype=np.float64)
    direction = np.asarray((0.18, -0.97, 0.16), dtype=np.float64)
    direction /= np.linalg.norm(direction)
    return start, start + float(length) * direction, direction


def _unique_edges(faces: np.ndarray) -> np.ndarray:
    triangles = np.asarray(faces, dtype=np.int64)
    edges = np.concatenate(
        (
            triangles[:, (0, 1)],
            triangles[:, (1, 2)],
            triangles[:, (2, 0)],
        ),
        axis=0,
    )
    return np.unique(np.sort(edges, axis=1), axis=0)


@pytest.mark.parametrize("anchor", ("proximal", "distal"))
def test_projected_station_preserves_anchor_length_and_all_edges(
    anchor: str,
) -> None:
    vertices, faces, _fractions = _cylinder(length=0.40)
    source_a = np.asarray((0.0, 0.0, 0.0))
    source_b = np.asarray((0.0, -0.40, 0.0))
    driver_a, driver_b, direction = _driver_segment(length=0.43)
    fit = fit_projected_station_rest_v810(
        source_a,
        source_b,
        driver_a,
        driver_b,
        anchor=anchor,
    )
    transformed = fit.apply(vertices)

    assert fit.scale == 1.0
    assert np.linalg.det(fit.rotation) == pytest.approx(1.0, abs=1.0e-12)
    np.testing.assert_allclose(
        fit.rotation.T @ fit.rotation,
        np.eye(3),
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        fit.apply(np.stack((source_a, source_b))),
        np.stack((fit.target_a, fit.target_b)),
        atol=1.0e-12,
    )
    assert np.linalg.norm(fit.target_b - fit.target_a) == pytest.approx(0.40)
    if anchor == "proximal":
        np.testing.assert_allclose(fit.target_a, driver_a, atol=0.0)
        np.testing.assert_allclose(
            fit.target_b,
            driver_a + 0.40 * direction,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            fit.free_endpoint_residual_m,
            -0.03 * direction,
            atol=1.0e-12,
        )
    else:
        np.testing.assert_allclose(fit.target_b, driver_b, atol=0.0)
        np.testing.assert_allclose(
            fit.target_a,
            driver_b - 0.40 * direction,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            fit.free_endpoint_residual_m,
            0.03 * direction,
            atol=1.0e-12,
        )

    edges = _unique_edges(faces)
    original_lengths = np.linalg.norm(
        vertices[edges[:, 0]] - vertices[edges[:, 1]],
        axis=1,
    )
    final_lengths = np.linalg.norm(
        transformed[edges[:, 0]] - transformed[edges[:, 1]],
        axis=1,
    )
    np.testing.assert_allclose(final_lengths, original_lengths, atol=1.0e-12)


def test_projected_station_reports_incompatible_length_without_scaling() -> None:
    source_a = np.asarray((0.0, 0.0, 0.0))
    source_b = np.asarray((0.0, -0.40, 0.0))
    driver_a, driver_b, _direction = _driver_segment(length=0.70)
    fit = fit_projected_station_rest_v810(
        source_a,
        source_b,
        driver_a,
        driver_b,
        anchor="proximal",
    )

    assert fit.source_length_m == pytest.approx(0.40)
    assert fit.driver_length_m == pytest.approx(0.70)
    assert fit.driver_length_residual_m == pytest.approx(0.30)
    assert fit.free_endpoint_residual_norm_m == pytest.approx(0.30)
    assert fit.scale == 1.0
    assert not fit.rotation.flags.writeable
    assert not fit.translation.flags.writeable
    assert not fit.affine.flags.writeable
    with pytest.raises(ValueError):
        fit.rotation[0, 0] = 0.0


def test_projected_station_rejects_invalid_anchor_and_degenerate_driver() -> None:
    with pytest.raises(ValueError, match="anchor"):
        fit_projected_station_rest_v810(
            (0.0, 0.0, 0.0),
            (0.0, -0.40, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, -0.43, 0.0),
            anchor="middle",
        )
    with pytest.raises(ValueError, match="non-degenerate"):
        fit_projected_station_rest_v810(
            (0.0, 0.0, 0.0),
            (0.0, -0.40, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            anchor="proximal",
        )


def test_whole_segment_rotation_keeps_every_edge_and_pivot_fixed() -> None:
    vertices, faces, _fractions = _cylinder(length=0.40)
    rings = 16
    pivot_ids = np.arange(
        (101 - 5) * rings,
        101 * rings,
        dtype=np.int64,
    )
    asset = SimpleNamespace(vertices_rest=vertices, faces=faces)
    delta, report = _apply_rigid_segment_rotation_v810(
        asset,
        side="left",
        rotvec=np.asarray((0.0, 0.0, np.deg2rad(11.0))),
        vertex_ids=np.arange(len(vertices), dtype=np.int64),
        pivot_ids=pivot_ids,
    )
    transformed = vertices + delta
    pivot = np.mean(vertices[pivot_ids], axis=0)
    affine = np.asarray(report["affine"], dtype=np.float64)

    np.testing.assert_allclose(
        pivot @ affine[:3, :3].T + affine[:3, 3],
        pivot,
        atol=1.0e-12,
    )
    edges = _unique_edges(faces)
    np.testing.assert_allclose(
        np.linalg.norm(
            transformed[edges[:, 0]] - transformed[edges[:, 1]],
            axis=1,
        ),
        np.linalg.norm(
            vertices[edges[:, 0]] - vertices[edges[:, 1]],
            axis=1,
        ),
        atol=1.0e-12,
    )
    assert report["pivot_translation_m"] < 1.0e-12
    assert report["edge_strain"]["all"]["maximum"] < 1.0e-12


def test_rbf_translation_coefficients_are_reexpressed_in_new_parent_frame() -> None:
    old_global = np.tile(np.eye(4), (2, 1, 1))
    new_global = old_global.copy()
    new_global[0, :3, :3] = _rotation_z(np.pi / 2.0)
    metadata = {
        "source_coupled_joint_response_v8": {
            "1": {
                "rbf_values_parent_local_m": [[1.0, 0.0, 0.0]],
                "rbf_zero_parent_local_m": [0.0, 1.0, 0.0],
                "rbf_weights_parent_local_m": [[0.0, 0.0, 1.0]],
            }
        }
    }
    transported, report = transport_coupled_rbf_parent_frames_v810(
        metadata,
        old_global=old_global,
        new_global=new_global,
        parents=np.asarray((-1, 0), dtype=np.int32),
    )
    response = transported["source_coupled_joint_response_v8"]["1"]
    np.testing.assert_allclose(
        response["rbf_values_parent_local_m"],
        ((0.0, -1.0, 0.0),),
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        response["rbf_zero_parent_local_m"],
        (1.0, 0.0, 0.0),
        atol=1.0e-12,
    )
    assert report["available"] is True
    assert report["transported_vector_count"] == 3


def test_schema_marker_selects_v810_without_requiring_full_coefficients() -> None:
    coefficients = {
        "leg_centerline_v810.schema_version": np.asarray(
            (LEG_CENTERLINE_SCHEMA_VERSION_V810,), dtype=np.int32
        )
    }
    assert has_leg_centerline_v810(coefficients) is True
    assert has_leg_centerline_v810({}) is False


def test_swept_segment_blend_fails_closed_at_the_strain_gate() -> None:
    vertices, faces, _fractions = _cylinder(length=0.40)
    rings = 16
    asset = SimpleNamespace(vertices_rest=vertices, faces=faces)
    ids = np.arange(len(vertices), dtype=np.int64)
    proximal_ids = np.arange(0, 5 * rings, dtype=np.int64)
    distal_ids = np.arange((101 - 5) * rings, 101 * rings, dtype=np.int64)
    stations = np.asarray((0.0, 0.15, 0.30, 0.50, 0.70, 0.85, 1.0))
    offsets = np.zeros((len(stations), 3), dtype=np.float64)
    offsets[1:-1, 0] = np.asarray((0.35, 0.45, 0.50, 0.45, 0.35))

    delta, report = _apply_swept_segment_centerline_v810(
        asset,
        side="left",
        segment="femur",
        vertex_ids=ids,
        proximal_ids=proximal_ids,
        distal_ids=distal_ids,
        station_fractions=stations,
        target_center_offsets_m=offsets,
    )

    assert report["blend"] < 1.0
    assert report["edge_strain"]["all"]["q99"] <= _CENTERLINE_EDGE_Q99_LIMIT
    assert report["edge_strain"]["all"]["maximum"] <= _CENTERLINE_EDGE_MAX_LIMIT
    np.testing.assert_allclose(delta[proximal_ids], 0.0, atol=1.0e-12)
    np.testing.assert_allclose(delta[distal_ids], 0.0, atol=1.0e-12)


def test_foot_station_uses_talus_calcaneus_and_forefoot_meshes() -> None:
    vertices = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.2),
            (0.0, -0.1, -0.1),
            (0.0, -0.1, 0.1),
            (0.1, -0.4, 0.0),
            (0.1, -0.6, 0.0),
            (-0.1, -0.4, 0.0),
            (-0.1, -0.6, 0.0),
        ),
        dtype=np.float64,
    )
    asset = SimpleNamespace(
        vertices_rest=vertices,
        source_mesh_names=[
            "Talus_L",
            "Calcaneus_L",
            "_1st_Metatarsal_L",
            "_2nd_Metatarsal_L",
        ],
        source_vertex_ranges=np.asarray(
            ((0, 2), (2, 4), (4, 6), (6, 8)), dtype=np.int32
        ),
    )

    station, report = _foot_station_v810(asset, suffix="L")

    np.testing.assert_allclose(station, (0.0, -0.5, 0.0), atol=1.0e-12)
    np.testing.assert_allclose(
        report["talus_center_m"], (0.0, 0.0, 0.1), atol=1.0e-12
    )
    np.testing.assert_allclose(
        report["calcaneus_center_m"], (0.0, -0.1, 0.0), atol=1.0e-12
    )
    assert report["forefoot_vertex_count"] == 4


def test_foot_arch_station_requires_all_anatomical_midfoot_domains() -> None:
    vertices = np.asarray(
        (
            (0.0, -0.08, 0.00),
            (0.0, -0.10, 0.00),
            (0.02, -0.11, 0.01),
            (0.02, -0.13, 0.01),
            (-0.02, -0.12, 0.02),
            (-0.02, -0.14, 0.02),
        ),
        dtype=np.float64,
    )
    asset = SimpleNamespace(
        vertices_rest=vertices,
        source_mesh_names=[
            "Navicular_L",
            "Cuboid_L",
            "Medial_Cuneiform_L",
        ],
        source_tissues=["bone", "bone", "bone"],
        source_vertex_ranges=np.asarray(((0, 2), (2, 4), (4, 6)), dtype=np.int32),
    )

    station, report = _foot_arch_station_v811(asset, suffix="L")

    np.testing.assert_allclose(station, np.mean(vertices, axis=0), atol=1.0e-12)
    assert report["required_domains"] == ["navicular", "cuboid", "cuneiform"]
    assert report["domain_meshes"]["navicular"] == ["Navicular_L"]
    assert report["domain_meshes"]["cuboid"] == ["Cuboid_L"]
    assert report["domain_meshes"]["cuneiform"] == ["Medial_Cuneiform_L"]

    missing = SimpleNamespace(
        vertices_rest=vertices[:4],
        source_mesh_names=["Navicular_L", "Medial_Cuneiform_L"],
        source_tissues=["bone", "bone"],
        source_vertex_ranges=np.asarray(((0, 2), (2, 4)), dtype=np.int32),
    )
    with pytest.raises(ValueError, match="missing L arch mesh domains"):
        _foot_arch_station_v811(missing, suffix="L")


def test_target_arch_is_ankle_guided_unit_so3_source_offset() -> None:
    source_ankle = np.asarray((0.0, 0.0, 0.0))
    source_arch = np.asarray((0.0, -0.10, 0.02))
    target_ankle = np.asarray((1.0, 2.0, 3.0))
    target_forefoot = np.asarray((1.35, 2.0, 3.0))
    rotation = _rotation_z(np.pi / 2.0)

    target_arch, report = _target_foot_arch_station_v811(
        source_ankle=source_ankle,
        source_arch=source_arch,
        target_ankle=target_ankle,
        target_forefoot=target_forefoot,
        rotation=rotation,
    )

    np.testing.assert_allclose(
        target_arch,
        target_ankle + (source_arch - source_ankle) @ rotation.T,
        atol=1.0e-12,
    )
    assert report["smplx_arch_joint_available"] is False
    assert report["rotation_determinant"] == pytest.approx(1.0, abs=1.0e-12)


def test_multi_station_foot_chain_keeps_each_mesh_strictly_rigid() -> None:
    source_ankle = np.asarray((0.0, 0.0, 0.0))
    source_arch = np.asarray((0.0, -0.10, 0.02))
    source_forefoot = np.asarray((0.0, -0.20, 0.0))
    target_ankle = np.asarray((1.0, 2.0, 3.0))
    target_arch = np.asarray((1.10, 2.0, 3.02))
    target_forefoot = np.asarray((1.35, 2.0, 3.0))
    rotation = _rotation_z(np.pi / 2.0)

    source_centers = np.asarray(
        (
            (0.015, -0.02, 0.004),
            (-0.010, -0.10, -0.006),
            (0.012, -0.18, 0.003),
        ),
        dtype=np.float64,
    )
    target_centers, station_parameters = _map_foot_stations_rigid_v811(
        source_centers,
        source_ankle=source_ankle,
        source_arch=source_arch,
        source_forefoot=source_forefoot,
        target_ankle=target_ankle,
        target_arch=target_arch,
        target_forefoot=target_forefoot,
        rotation=rotation,
    )

    assert np.all(np.diff(station_parameters) > 0.0)
    assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1.0e-12)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-12)

    mesh_offsets = np.asarray(
        (
            (-0.008, 0.0, 0.0),
            (0.008, 0.0, 0.0),
            (0.0, 0.005, 0.005),
            (0.0, -0.005, -0.005),
        ),
        dtype=np.float64,
    )
    for source_center, target_center in zip(
        source_centers, target_centers, strict=True
    ):
        source_mesh = source_center + mesh_offsets
        target_mesh = target_center + mesh_offsets @ rotation.T
        np.testing.assert_allclose(
            np.mean(target_mesh, axis=0), target_center, atol=1.0e-12
        )
        for first in range(len(mesh_offsets)):
            for second in range(first + 1, len(mesh_offsets)):
                assert np.linalg.norm(
                    target_mesh[first] - target_mesh[second]
                ) == pytest.approx(
                    np.linalg.norm(source_mesh[first] - source_mesh[second]),
                    abs=1.0e-12,
                )

    endpoints, endpoint_stations = _map_foot_stations_rigid_v811(
        np.stack((source_ankle, source_arch, source_forefoot)),
        source_ankle=source_ankle,
        source_arch=source_arch,
        source_forefoot=source_forefoot,
        target_ankle=target_ankle,
        target_arch=target_arch,
        target_forefoot=target_forefoot,
        rotation=rotation,
    )
    np.testing.assert_allclose(endpoints[0], target_ankle, atol=1.0e-12)
    np.testing.assert_allclose(endpoints[1], target_arch, atol=1.0e-12)
    np.testing.assert_allclose(endpoints[2], target_forefoot, atol=1.0e-12)
    source_first_length = np.linalg.norm(source_arch - source_ankle)
    source_total_length = source_first_length + np.linalg.norm(
        source_forefoot - source_arch
    )
    np.testing.assert_allclose(
        endpoint_stations,
        (0.0, source_first_length / source_total_length, 1.0),
        atol=1.0e-12,
    )


def test_multi_station_foot_chain_rejects_nonrigid_rotation() -> None:
    with pytest.raises(ValueError, match="proper unit-scale SO\\(3\\)"):
        _map_foot_stations_rigid_v811(
            np.asarray(((0.0, -0.10, 0.0),), dtype=np.float64),
            source_ankle=np.asarray((0.0, 0.0, 0.0)),
            source_arch=np.asarray((0.0, -0.10, 0.0)),
            source_forefoot=np.asarray((0.0, -0.20, 0.0)),
            target_ankle=np.asarray((1.0, 2.0, 3.0)),
            target_arch=np.asarray((1.0, 2.1, 3.0)),
            target_forefoot=np.asarray((1.0, 2.2, 3.0)),
            rotation=np.diag(np.asarray((1.0, 1.02, 1.0))),
        )


def test_proximal_fibula_cap_selects_only_the_near_end() -> None:
    y = np.linspace(0.0, -0.40, 100, dtype=np.float64)
    vertices = np.stack((np.zeros_like(y), y, np.zeros_like(y)), axis=1)
    asset = SimpleNamespace(
        vertices_rest=vertices,
        source_mesh_names=["Fibula_L"],
        source_vertex_ranges=np.asarray(((0, len(vertices)),), dtype=np.int32),
    )

    selected = _proximal_mesh_cap_ids(
        asset,
        mesh_name="Fibula_L",
        proximal=np.asarray((0.0, 0.0, 0.0)),
        distal=np.asarray((0.0, -0.40, 0.0)),
    )

    assert 12 <= len(selected) <= 13
    assert int(np.max(selected)) <= 12


def test_rebuild_013_materializes_one_contact_chain_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = Path(
        "outputs/anatomy_retarget/v8_candidates/rebuild_013"
    )
    operator_path = candidate / "source_operator_v8"
    subject_path = candidate / "subject_213328"
    if not operator_path.is_dir() or not subject_path.is_dir():
        pytest.skip("rebuild_013 integration assets are unavailable")

    # This integration fixture predates V8.11.  Do not deserialize it: legacy
    # full-FK/cache packs are only supported for read-only pose compatibility.
    subject_manifest = json.loads(
        (subject_path / "manifest.json").read_text(encoding="utf-8")
    )
    if subject_manifest.get("subject_solver_version") != SUBJECT_SOLVER_VERSION:
        pytest.skip("rebuild_013 is a legacy V8.10 subject cache fixture")

    operator_manifest = json.loads(
        (operator_path / "manifest.json").read_text(encoding="utf-8")
    )
    current_operator_versions = (
        operator_manifest.get("algorithm_version")
        == SOURCE_OPERATOR_ALGORITHM_VERSION
        and operator_manifest.get("oracle_version")
        == SOURCE_OPERATOR_ORACLE_VERSION
        and operator_manifest.get("correction_version")
        == SOURCE_OPERATOR_CORRECTION_VERSION
    )
    template_rig = operator_manifest.get("template_rig", {})
    template_json_fields = (
        template_rig.get("json_fields", {})
        if isinstance(template_rig, dict)
        else {}
    )
    metadata = (
        template_json_fields.get("metadata", {})
        if isinstance(template_json_fields, dict)
        else {}
    )
    selective_fk = (
        metadata.get("source_fk_policy_v4")
        == SELECTIVE_AUTHORITY_FK_POLICY_V4
        and metadata.get("source_full_local_fk_v2") is False
    )
    if not current_operator_versions or not selective_fk:
        pytest.skip("rebuild_013 is a legacy V8.10/full-FK cache fixture")

    operator = v8_artifacts.load_source_operator(operator_path)
    baseline = v8_artifacts.load_subject_runtime(subject_path)
    calls = 0

    def counted_reconstruct(asset, *, domains):
        nonlocal calls
        calls += 1
        return reconstruct_leg_centerline_compounds_v810(
            asset,
            domains=domains,
        )

    monkeypatch.setattr(
        v8_artifacts,
        "reconstruct_leg_centerline_compounds_v810",
        counted_reconstruct,
    )
    subject = v8_artifacts.materialize_subject(
        operator,
        betas=np.asarray(baseline.betas),
        gender=baseline.gender,
    )
    report = subject.audit_report["leg_centerline_v810"]

    assert calls == 1
    assert report["method"] == "single_pass_contact_first_joint_chain_v810"
    assert report["pelvis_correction"] == "identity"
    assert report["uniform_or_radial_bone_scale"] is False
    tube_transport = report["tube_rest_transport"]
    assert tube_transport["method"] == (
        "frozen_route_rebased_to_new_inverse_bind_v810"
    )
    assert tube_transport["maximum_rest_displacement_m"] == 0.0
    assert tube_transport["indices_or_weights_changed"] is False
    pelvis = report["pelvis_common_mode_diagnostic"]
    assert pelvis["pelvis_correction"] == "identity"
    assert pelvis["differential_mode_norm_m"] > 0.002

    guide = np.asarray(subject.rigged_asset.source_driver_rest_joints)
    for side in ("left", "right"):
        values = report["sides"][side]
        assert values["hip"]["head_socket_residual_m"] <= 0.002
        assert values["hip"]["head_socket_validation_residual_m"] <= 0.002
        assert values["femur"]["knee_smplx_residual_m"] <= 0.003
        assert values["shank"]["ankle_smplx_residual_m"] <= 0.003
        assert values["shank"]["edge_strain"]["all"]["maximum"] <= 0.03
        for segment in ("femur", "shank"):
            assert values[segment]["scale"] == 1.0
            assert values[segment]["proximal_det_rotation"] == pytest.approx(
                1.0,
                abs=1.0e-9,
            )
            assert values[segment]["distal_det_rotation"] == pytest.approx(
                1.0,
                abs=1.0e-9,
            )
        assert values["foot"]["scale"] == 1.0
        assert values["foot"]["det_rotation"] == pytest.approx(
            1.0,
            abs=1.0e-9,
        )
        foot_joint = values["joint_ids"]["foot"]
        np.testing.assert_allclose(
            guide[foot_joint],
            values["foot"]["mapped_geometry_station_m"],
            atol=1.0e-7,
        )

    authentication = subject.audit_report["tube_coupling"][
        "final_rest_authentication"
    ]
    assert authentication["available"] is True
    assert all(authentication["frozen_digest_match"].values())
