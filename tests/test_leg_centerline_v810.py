from __future__ import annotations

import numpy as np
import pytest

from projects.genesis_ue_sync.anatomy_retarget.leg_centerline_v810 import (
    LEG_CENTERLINE_SCHEMA_VERSION_V810,
    has_leg_centerline_v810,
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
