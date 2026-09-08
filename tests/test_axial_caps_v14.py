from __future__ import annotations

import numpy as np
import pytest

from projects.genesis_ue_sync.anatomy_retarget.axial_caps_v14 import (
    AxialCapsFieldV14,
    apply_axial_caps_v14,
    axial_caps_field_from_array_dict_v14,
    axial_caps_field_to_array_dict_v14,
    build_axial_caps_field_v14,
    map_local_frame_v14,
    polar_rotation_v14,
)


def _tilted_segment(
    *,
    length: float = 0.40,
    stations: int = 9,
    rings: int = 16,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    origin = np.asarray((0.13, -0.08, 0.21), dtype=np.float64)
    axis = np.asarray((0.31, -0.82, 0.47), dtype=np.float64)
    axis /= np.linalg.norm(axis)
    radial_a = np.cross(axis, np.asarray((0.0, 0.0, 1.0)))
    radial_a /= np.linalg.norm(radial_a)
    radial_b = np.cross(axis, radial_a)
    t = np.repeat(np.linspace(0.0, 1.0, stations), rings)
    angle = np.tile(np.linspace(0.0, 2.0 * np.pi, rings, endpoint=False), stations)
    radius = 0.018 + 0.003 * t
    points = (
        origin[None, :]
        + length * t[:, None] * axis[None, :]
        + radius[:, None] * np.cos(angle)[:, None] * radial_a[None, :]
        + radius[:, None] * np.sin(angle)[:, None] * radial_b[None, :]
    )
    return points, t, origin, axis


def _field_from_ids(scale: float = 1.08) -> tuple[np.ndarray, AxialCapsFieldV14, np.ndarray]:
    points, _t, origin, axis = _tilted_segment()
    rings = 16
    field = build_axial_caps_field_v14(
        points,
        axis_origin=origin,
        axis=axis,
        source_length_m=0.40,
        total_length_scale=scale,
        proximal_cap_ids=np.arange(rings, dtype=np.int64),
        distal_cap_ids=np.arange(len(points) - rings, len(points), dtype=np.int64),
    )
    return points, field, axis


def test_caps_are_rigid_and_each_cross_section_keeps_its_radius() -> None:
    points, field, axis = _field_from_ids()
    mapped = field.map_points(points)
    rings = 16
    delta = field.delta_m * axis

    # Both frozen endpoint domains retain their shape exactly.  The distal
    # cap is allowed to move only as one rigid axial translation.
    np.testing.assert_array_equal(mapped[:rings], points[:rings])
    np.testing.assert_allclose(
        mapped[-rings:] - points[-rings:],
        np.broadcast_to(delta, (rings, 3)),
        atol=3.0e-16,
        rtol=0.0,
    )
    before_pairwise = np.linalg.norm(
        points[-rings:, None, :] - points[None, -rings:, :],
        axis=-1,
    )
    after_pairwise = np.linalg.norm(
        mapped[-rings:, None, :] - mapped[None, -rings:, :],
        axis=-1,
    )
    np.testing.assert_allclose(after_pairwise, before_pairwise, atol=1.0e-14, rtol=0.0)

    # Radial vectors about the source axis are unchanged at every station;
    # this catches accidental isotropic sR or transverse scale.
    scalar = (points - field.axis_origin) @ axis
    mapped_scalar = (mapped - field.axis_origin) @ axis
    radial_before = points - field.axis_origin - scalar[:, None] * axis[None, :]
    radial_after = mapped - field.axis_origin - mapped_scalar[:, None] * axis[None, :]
    np.testing.assert_allclose(radial_after, radial_before, atol=2.0e-15, rtol=0.0)
    radius_before = np.linalg.norm(radial_before, axis=1)
    radius_after = np.linalg.norm(radial_after, axis=1)
    np.testing.assert_allclose(radius_after, radius_before, atol=2.0e-15, rtol=0.0)


def test_shared_field_maps_controller_and_soft_probes_identically() -> None:
    points, field, _axis = _field_from_ids(scale=0.94)
    controllers = points[[4, 42, 95]] + np.asarray((0.004, -0.003, 0.002))
    soft = points[[8, 52, 120]] + np.asarray((-0.002, 0.001, 0.003))
    separate = np.concatenate((field.map_points(controllers), field.map_points(soft)))
    joined = field.sample(np.concatenate((controllers, soft)))
    np.testing.assert_array_equal(joined, separate)
    assert field.analytic_jacobian_minimum > 0.0
    assert field.report()["radial_scale"] == 1.0


@pytest.mark.parametrize("scale", (0.90, 1.10))
def test_ten_percent_bounds_and_full_domain_jacobian(scale: float) -> None:
    points, _t, origin, axis = _tilted_segment()
    field = build_axial_caps_field_v14(
        points,
        axis_origin=origin,
        normalized_axis=axis,
        source_length_m=0.40,
        total_length_scale=scale,
        cap_scalar_bounds=(0.06, 0.34),
    )
    expected = 1.0 + min(0.0, field.delta_m * 1.5 / field.transition_width_m)
    assert field.analytic_jacobian_minimum == pytest.approx(expected)
    assert field.analytic_jacobian_minimum > 0.0
    jacobian = field.jacobian(points)
    determinants = np.linalg.det(jacobian)
    assert float(np.min(determinants)) > 0.0
    assert field.target_length_m == pytest.approx(0.40 * scale)


def test_analytic_fold_is_rejected_even_when_no_sample_hits_mid_transition() -> None:
    points, _t, origin, axis = _tilted_segment(stations=3)
    with pytest.raises(ValueError, match="non-positive analytic Jacobian"):
        build_axial_caps_field_v14(
            points,
            axis_origin=origin,
            axis=axis,
            source_length_m=0.40,
            total_length_scale=0.90,
            cap_scalar_bounds=(0.195, 0.205),
        )


def test_identity_is_bit_exact_and_serialization_is_array_only() -> None:
    points, field, _axis = _field_from_ids(scale=1.0)
    mapped = field.map_points(points)
    assert mapped.tobytes() == points.tobytes()
    points_f32 = points.astype(np.float32)
    mapped_f32 = field.map_points(points_f32)
    assert mapped_f32.dtype == points_f32.dtype
    assert mapped_f32.tobytes() == points_f32.tobytes()
    assert np.array_equal(field.sample(points), points)
    arrays = axial_caps_field_to_array_dict_v14(field)
    assert all(isinstance(value, np.ndarray) for value in arrays.values())
    restored = axial_caps_field_from_array_dict_v14(arrays)
    assert restored.report() == field.report()
    assert restored.map_points(points).tobytes() == points.tobytes()


def test_polar_frame_mapping_discards_axial_scale_from_bind_rotation() -> None:
    linear = np.asarray(((1.8, 0.2, 0.0), (0.0, 0.7, 0.1), (0.0, 0.0, 1.2)))
    rotation = polar_rotation_v14(linear)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0)

    points, field, _axis = _field_from_ids(scale=1.08)
    frame = np.eye(4, dtype=np.float64)
    frame[:3, :3] = linear
    frame[:3, 3] = points[55]
    mapped_frame = map_local_frame_v14(field, frame)
    np.testing.assert_allclose(
        mapped_frame[:3, :3].T @ mapped_frame[:3, :3],
        np.eye(3),
        atol=1.0e-12,
    )
    assert np.linalg.det(mapped_frame[:3, :3]) == pytest.approx(1.0)
    np.testing.assert_allclose(mapped_frame[:3, 3], field.map_points(frame[None, :3, 3])[0])


@pytest.mark.parametrize(
    ("kwargs", "match"),
    (
        ({"total_length_scale": 0.89}, "total_length_scale"),
        ({"total_length_scale": 1.11}, "total_length_scale"),
        ({"axis": (0.0, 0.0, 0.0)}, "degenerate"),
        ({"cap_scalar_bounds": (0.3, 0.2)}, "overlap"),
        ({"proximal_cap_ids": np.asarray([0.0]), "distal_cap_ids": np.asarray([1])}, "integer"),
        ({"proximal_cap_ids": np.asarray([0, 0]), "distal_cap_ids": np.asarray([1])}, "duplicate"),
    ),
)
def test_invalid_cap_contracts_fail_closed(kwargs: dict[str, object], match: str) -> None:
    points, _t, origin, axis = _tilted_segment()
    arguments: dict[str, object] = {
        "axis_origin": origin,
        "axis": axis,
        "source_length_m": 0.40,
        "total_length_scale": 1.0,
        "cap_scalar_bounds": (0.06, 0.34),
    }
    arguments.update(kwargs)
    if "proximal_cap_ids" in kwargs:
        arguments.pop("cap_scalar_bounds")
        arguments["distal_cap_ids"] = kwargs.get("distal_cap_ids", np.asarray([1]))
    with pytest.raises(ValueError, match=match):
        build_axial_caps_field_v14(points, **arguments)


def test_apply_result_exposes_mapped_points_and_reusable_field() -> None:
    points, _t, origin, axis = _tilted_segment()
    result = apply_axial_caps_v14(
        points,
        axis_origin=origin,
        endpoints=np.stack((origin, origin + 0.40 * axis)),
        total_length_scale=1.02,
        cap_scalar_bounds=(0.06, 0.34),
    )
    assert result.mapped_points is result.points
    np.testing.assert_array_equal(result.points, result.sample.points)
    np.testing.assert_array_equal(result.field.sample(points), result.points)
