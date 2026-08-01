from __future__ import annotations

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
    _foot_station_v810,
    _proximal_mesh_cap_ids,
    has_leg_centerline_v810,
    reconstruct_leg_centerline_compounds_v810,
    transport_coupled_rbf_parent_frames_v810,
)
from projects.genesis_ue_sync.anatomy_retarget.mechanism_v8 import (
    fit_projected_station_rest_v810,
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
