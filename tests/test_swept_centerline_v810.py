from __future__ import annotations

import numpy as np
import pytest

from projects.genesis_ue_sync.anatomy_retarget.swept_centerline_v810 import (
    SweptCenterlineRestWarpV810,
    swept_centerline_rest_warp_v810,
)


def _cylinder(
    *,
    length: float = 0.42,
    station_count: int = 121,
    ring_count: int = 24,
    radius: float = 0.020,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fractions = np.linspace(0.0, 1.0, station_count, dtype=np.float64)
    angles = np.linspace(
        0.0,
        2.0 * np.pi,
        ring_count,
        endpoint=False,
        dtype=np.float64,
    )
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
    for station in range(station_count - 1):
        first = station * ring_count
        second = (station + 1) * ring_count
        for ring in range(ring_count):
            nxt = (ring + 1) % ring_count
            faces.append((first + ring, second + ring, second + nxt))
            faces.append((first + ring, second + nxt, first + nxt))
    return vertices, np.asarray(faces, dtype=np.int32), fractions


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


def _bow_warp(
    *,
    blend: float = 1.0,
    amplitude_m: float = 0.050,
) -> SweptCenterlineRestWarpV810:
    stations = np.linspace(0.0, 1.0, 9, dtype=np.float64)
    envelope = np.sin(np.pi * stations) ** 2
    offsets = np.stack(
        (
            float(amplitude_m) * envelope,
            np.zeros_like(stations),
            0.004 * np.sin(2.0 * np.pi * stations) * envelope,
        ),
        axis=1,
    )
    return SweptCenterlineRestWarpV810(
        proximal_m=np.asarray((0.0, 0.0, 0.0)),
        distal_m=np.asarray((0.0, -0.42, 0.0)),
        station_fractions=stations,
        target_center_offsets_m=offsets,
        blend=blend,
    )


@pytest.mark.parametrize("amplitude_m", (0.030, 0.050))
def test_smooth_bow_preserves_cross_sections_and_avoids_strain_spike(
    amplitude_m: float,
) -> None:
    vertices, faces, fractions = _cylinder()
    warp = _bow_warp(amplitude_m=amplitude_m)
    mapped = warp.apply(vertices)
    report = warp.report()

    ring_count = 24
    mapped_rings = mapped.reshape(len(fractions), ring_count, 3)
    centers = warp.center(fractions)
    radii = np.linalg.norm(mapped_rings - centers[:, None, :], axis=2)
    np.testing.assert_allclose(radii, 0.020, atol=1.0e-12, rtol=0.0)
    np.testing.assert_allclose(mapped_rings[0], vertices[:ring_count], atol=1.0e-12)
    np.testing.assert_allclose(
        mapped_rings[-1],
        vertices[-ring_count:],
        atol=1.0e-12,
    )

    edges = _unique_edges(faces)
    source_lengths = np.linalg.norm(
        vertices[edges[:, 0]] - vertices[edges[:, 1]],
        axis=1,
    )
    mapped_lengths = np.linalg.norm(
        mapped[edges[:, 0]] - mapped[edges[:, 1]],
        axis=1,
    )
    relative = np.abs(mapped_lengths / source_lengths - 1.0)
    assert float(np.quantile(relative, 0.99)) < 0.13
    assert float(np.max(relative)) < 0.15
    # The old endpoint-fixed rotation adapter exceeded 200% edge strain.
    assert float(np.max(relative)) < 0.20

    assert report["station_interpolation_error_max_m"] < 1.0e-12
    assert report["station_residual_max_m"] < 1.0e-12
    assert report["frame_determinant_min"] == pytest.approx(1.0, abs=1.0e-10)
    assert report["frame_determinant_max"] == pytest.approx(1.0, abs=1.0e-10)
    assert report["frame_scale_min"] == pytest.approx(1.0, abs=1.0e-10)
    assert report["frame_scale_max"] == pytest.approx(1.0, abs=1.0e-10)
    assert report["cross_section_scale"] == 1.0
    assert 0.0 < report["centerline_arc_strain"] < 0.10
    assert report["axial_jacobian_min"] >= 1.0 - 1.0e-10
    assert report["axial_jacobian_max"] < 1.20


def test_blend_reports_unapplied_station_residual_without_frame_scale() -> None:
    warp = _bow_warp(blend=0.4)
    report = warp.report()
    expected = -0.6 * warp.target_center_offsets_m
    np.testing.assert_allclose(
        report["station_residuals_m"],
        expected,
        atol=1.0e-12,
    )
    assert report["station_residual_max_m"] == pytest.approx(0.030)
    assert report["frame_scale_min"] == pytest.approx(1.0, abs=1.0e-10)
    assert report["frame_scale_max"] == pytest.approx(1.0, abs=1.0e-10)


def test_center_curve_is_c2_and_has_identity_endpoint_tangents() -> None:
    warp = _bow_warp()
    epsilon = 1.0e-7
    for station in warp.station_fractions[1:-1]:
        left = warp.center_derivative(station - epsilon, order=2)
        right = warp.center_derivative(station + epsilon, order=2)
        np.testing.assert_allclose(left, right, atol=2.0e-5, rtol=0.0)
    np.testing.assert_allclose(
        warp.center_derivative(0.0, order=1),
        warp.distal_m - warp.proximal_m,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        warp.center_derivative(1.0, order=1),
        warp.distal_m - warp.proximal_m,
        atol=1.0e-12,
    )


def test_points_beyond_the_axis_caps_remain_identity() -> None:
    warp = _bow_warp()
    points = np.asarray(
        (
            (0.015, 0.050, -0.004),
            (-0.011, -0.470, 0.008),
        ),
        dtype=np.float64,
    )
    np.testing.assert_allclose(warp.apply(points), points, atol=1.0e-12)


def test_one_shot_function_matches_reusable_warp() -> None:
    vertices, _faces, _fractions = _cylinder(station_count=17, ring_count=8)
    warp = _bow_warp(blend=0.65)
    expected = warp.apply(vertices)
    mapped, report = swept_centerline_rest_warp_v810(
        vertices,
        proximal_m=warp.proximal_m,
        distal_m=warp.distal_m,
        station_fractions=warp.station_fractions,
        target_center_offsets_m=warp.target_center_offsets_m,
        lambda_=0.65,
        report_sample_count=129,
    )
    np.testing.assert_allclose(mapped, expected, atol=1.0e-12)
    assert report["blend"] == pytest.approx(0.65)


@pytest.mark.parametrize(
    ("stations", "offsets", "blend", "message"),
    (
        (
            (0.0, 0.5, 1.0),
            ((0, 0, 0), (0.02, 0, 0), (0.01, 0, 0)),
            1.0,
            "offsets must be zero",
        ),
        (
            (0.0, 0.7, 0.6, 1.0),
            ((0, 0, 0), (0.02, 0, 0), (0.03, 0, 0), (0, 0, 0)),
            1.0,
            "strictly increasing",
        ),
        (
            (0.0, 0.5, 1.0),
            ((0, 0, 0), (0.02, 0, 0), (0, 0, 0)),
            1.1,
            "blend must be",
        ),
    ),
)
def test_invalid_sweep_contract_fails_closed(
    stations: tuple[float, ...],
    offsets: tuple[tuple[float, float, float], ...],
    blend: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        SweptCenterlineRestWarpV810(
            proximal_m=np.asarray((0.0, 0.0, 0.0)),
            distal_m=np.asarray((0.0, -0.42, 0.0)),
            station_fractions=np.asarray(stations),
            target_center_offsets_m=np.asarray(offsets),
            blend=blend,
        )
