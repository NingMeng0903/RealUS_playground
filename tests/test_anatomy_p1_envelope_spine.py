from __future__ import annotations

import numpy as np

from projects.genesis_ue_sync.anatomy_retarget.material_fit import (
    _anatomical_frame,
    _frame_coordinates,
    _from_frame_coordinates,
    _sample_spine_centerline,
    _uniform_envelope_fit,
    uniform_segment_similarity,
)


def test_uniform_envelope_median_scale_avoids_min_axis_shrink() -> None:
    source = np.asarray(
        [
            (-0.10, -0.05, -0.02),
            (0.10, -0.05, -0.02),
            (-0.10, 0.05, -0.02),
            (0.10, 0.05, -0.02),
            (-0.10, -0.05, 0.02),
            (0.10, -0.05, 0.02),
            (-0.10, 0.05, 0.02),
            (0.10, 0.05, 0.02),
        ],
        dtype=np.float64,
    )
    # Target is wider in X, similar in Y, slightly shorter in Z.
    target = source * np.asarray((1.20, 1.05, 0.90))
    _mapped_min, scale_min, report_min = _uniform_envelope_fit(
        source,
        target,
        scale_multiplier=1.0,
        center_offset=np.zeros(3),
        margin=1.0,
        scale_mode="min",
        source_center=np.zeros(3),
        target_center=np.zeros(3),
    )
    _mapped_med, scale_med, report_med = _uniform_envelope_fit(
        source,
        target,
        scale_multiplier=1.0,
        center_offset=np.zeros(3),
        margin=1.0,
        scale_mode="median",
        source_center=np.zeros(3),
        target_center=np.zeros(3),
    )
    assert scale_med > scale_min
    assert report_med["scale_mode"] == "median"
    assert report_min["saturated"] is False


def test_uniform_envelope_reports_saturation_without_silent_success() -> None:
    source = np.eye(3, dtype=np.float64)
    target = source * 3.0
    _mapped, scale, report = _uniform_envelope_fit(
        source,
        target,
        scale_multiplier=1.0,
        center_offset=np.zeros(3),
        margin=1.0,
        maximum_scale=1.35,
        minimum_scale=0.70,
        scale_mode="median",
        source_center=np.zeros(3),
        target_center=np.zeros(3),
    )
    assert scale == 1.35
    assert report["saturated"] is True
    assert float(report["raw_scale"]) > 1.35


def test_spine_centerline_is_monotonic_and_interpolates_endpoints() -> None:
    controls = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (0.0, 0.1, 0.02),
            (0.0, 0.25, 0.05),
            (0.0, 0.4, 0.04),
            (0.0, 0.55, 0.02),
            (0.0, 0.7, 0.0),
        ),
        dtype=np.float64,
    )
    fractions = np.linspace(0.0, 1.0, 24)
    samples = _sample_spine_centerline(controls, fractions)
    assert samples.shape == (24, 3)
    np.testing.assert_allclose(samples[0], controls[0], atol=1.0e-6)
    np.testing.assert_allclose(samples[-1], controls[-1], atol=1.0e-6)
    # Arc-length order along Y must remain strictly increasing for this chain.
    assert np.all(np.diff(samples[:, 1]) > 0.0)


def test_spine_centerline_preserves_nonuniform_anatomical_anchors() -> None:
    controls = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (0.01, 0.08, 0.02),
            (0.02, 0.22, 0.05),
            (0.01, 0.58, 0.03),
            (0.0, 0.75, 0.0),
        ),
        dtype=np.float64,
    )
    control_fractions = np.asarray((0.0, 0.08, 0.24, 0.72, 1.0), dtype=np.float64)
    samples = _sample_spine_centerline(
        controls,
        control_fractions,
        control_fractions=control_fractions,
    )
    np.testing.assert_allclose(samples, controls, atol=1.0e-8)
    assert np.all(np.diff(samples[:, 1]) > 0.0)


def test_anatomical_frame_roundtrip_is_world_axis_independent() -> None:
    frame = _anatomical_frame(
        origin=np.asarray((0.3, -0.2, 0.7)),
        lateral=np.asarray((1.0, 2.0, 0.5)),
        superior=np.asarray((-0.2, 0.4, 1.0)),
    )
    points = np.asarray(((0.1, 0.2, 0.3), (1.2, -0.5, 0.8)), dtype=np.float64)
    local = _frame_coordinates(points, frame)
    recovered = _from_frame_coordinates(local, frame)
    np.testing.assert_allclose(recovered, points, atol=1.0e-10)
    np.testing.assert_allclose(frame[:3, :3].T @ frame[:3, :3], np.eye(3), atol=1.0e-10)
    assert np.linalg.det(frame[:3, :3]) > 0.0


def test_uniform_segment_similarity_preserves_compound_shape() -> None:
    points = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.2, 0.0), (2.0, 0.0, 0.0)),
        dtype=np.float64,
    )
    mapped, scale, _rotation = uniform_segment_similarity(
        points,
        source_a=np.asarray((0.0, 0.0, 0.0)),
        source_b=np.asarray((2.0, 0.0, 0.0)),
        target_a=np.asarray((0.5, 1.0, -0.2)),
        target_b=np.asarray((0.5, 4.0, -0.2)),
    )
    assert scale == 1.5
    before = np.linalg.norm(points[:, None] - points[None, :], axis=2)
    after = np.linalg.norm(mapped[:, None] - mapped[None, :], axis=2)
    np.testing.assert_allclose(after, scale * before, atol=1.0e-10)
