from __future__ import annotations

import numpy as np
import pytest

from projects.genesis_ue_sync.anatomy_retarget.mechanism_v8 import (
    apply_cap_preserving_axial_rest_v810,
)


def _tilted_cylinder(
    *,
    length: float = 0.40,
    stations: int = 101,
    rings: int = 12,
    radius: float = 0.02,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    proximal = np.asarray((0.03, -0.02, 0.07), dtype=np.float64)
    direction = np.asarray((0.17, -0.96, 0.22), dtype=np.float64)
    direction /= np.linalg.norm(direction)
    radial_a = np.cross(direction, (0.0, 0.0, 1.0))
    radial_a /= np.linalg.norm(radial_a)
    radial_b = np.cross(direction, radial_a)
    axial_parameter = np.repeat(
        np.linspace(0.0, 1.0, stations, dtype=np.float64),
        rings,
    )
    angles = np.tile(
        np.linspace(0.0, 2.0 * np.pi, rings, endpoint=False),
        stations,
    )
    vertices = (
        proximal[None, :]
        + length * axial_parameter[:, None] * direction[None, :]
        + radius * np.cos(angles)[:, None] * radial_a[None, :]
        + radius * np.sin(angles)[:, None] * radial_b[None, :]
    )
    return (
        vertices,
        axial_parameter,
        proximal,
        proximal + length * direction,
    )


@pytest.mark.parametrize(
    ("proximal_cap_fraction", "distal_cap_fraction"),
    ((0.08, 0.12), (0.15, 0.05)),
)
def test_cap_preserving_adapter_moves_only_along_axis_without_radial_scale(
    proximal_cap_fraction: float,
    distal_cap_fraction: float,
) -> None:
    vertices, axial_parameter, proximal, distal = _tilted_cylinder()
    result = apply_cap_preserving_axial_rest_v810(
        vertices,
        proximal=proximal,
        distal=distal,
        target_length_delta_m=0.024,
        axial_parameter=axial_parameter,
        proximal_cap_fraction=proximal_cap_fraction,
        distal_cap_fraction=distal_cap_fraction,
    )
    rings = 12

    np.testing.assert_allclose(result.phi[:rings], 0.0, atol=0.0)
    np.testing.assert_allclose(result.phi[-rings:], 1.0, atol=0.0)
    np.testing.assert_allclose(
        result.vertices[:rings],
        vertices[:rings],
        atol=0.0,
    )
    np.testing.assert_allclose(
        result.displacement[-rings:],
        np.broadcast_to(0.024 * result.axis_direction, (rings, 3)),
        atol=1.0e-15,
    )
    axial_component = (
        result.displacement @ result.axis_direction
    )[:, None] * result.axis_direction[None, :]
    np.testing.assert_allclose(
        result.displacement,
        axial_component,
        atol=1.0e-15,
    )

    before_cross_sections = vertices.reshape(-1, rings, 3)
    after_cross_sections = result.vertices.reshape(-1, rings, 3)
    before_centered = before_cross_sections - np.mean(
        before_cross_sections,
        axis=1,
        keepdims=True,
    )
    after_centered = after_cross_sections - np.mean(
        after_cross_sections,
        axis=1,
        keepdims=True,
    )
    np.testing.assert_allclose(after_centered, before_centered, atol=1.0e-14)
    assert result.applied_delta_m == pytest.approx(0.024)
    assert result.remaining_residual_m == pytest.approx(0.0)
    assert result.profile_peak_slope == pytest.approx(
        1.0 / (1.0 - 1.5 * (proximal_cap_fraction + distal_cap_fraction))
    )
    assert result.cross_section_scale == 1.0
    assert np.all(np.diff(result.phi.reshape(-1, rings)[:, 0]) >= 0.0)
    assert np.all(result.profile_derivative >= 0.0)
    np.testing.assert_allclose(
        result.axial_jacobian,
        1.0 + result.axial_strain,
        atol=0.0,
    )


@pytest.mark.parametrize("target_delta", (-0.080, 0.080))
def test_axial_strain_budget_clips_delta_and_reports_residual(
    target_delta: float,
) -> None:
    vertices, axial_parameter, proximal, distal = _tilted_cylinder()
    result = apply_cap_preserving_axial_rest_v810(
        vertices,
        proximal=proximal,
        distal=distal,
        target_length_delta_m=target_delta,
        axial_parameter=axial_parameter,
        max_abs_axial_strain=0.04,
    )
    expected_limit = 0.04 * 0.40 / result.profile_peak_slope

    assert result.applied_delta_m == pytest.approx(
        np.copysign(expected_limit, target_delta)
    )
    assert result.remaining_residual_m == pytest.approx(
        target_delta - result.applied_delta_m
    )
    assert result.maximum_abs_applied_strain == pytest.approx(0.04)
    if target_delta < 0.0:
        assert result.minimum_axial_jacobian == pytest.approx(0.96)
        assert result.maximum_axial_jacobian == pytest.approx(1.0)
    else:
        assert result.minimum_axial_jacobian == pytest.approx(1.0)
        assert result.maximum_axial_jacobian == pytest.approx(1.04)
    assert np.min(result.axial_jacobian) > 0.0


@pytest.mark.parametrize(
    ("segment", "expected_strain_limit"),
    (("femur", 0.12), ("shank", 0.08)),
)
def test_segment_defaults_clip_against_analytic_peak_strain(
    segment: str,
    expected_strain_limit: float,
) -> None:
    vertices, axial_parameter, proximal, distal = _tilted_cylinder()
    result = apply_cap_preserving_axial_rest_v810(
        vertices,
        proximal=proximal,
        distal=distal,
        target_length_delta_m=0.20,
        axial_parameter=axial_parameter,
        segment=segment,
    )
    expected_delta = (
        expected_strain_limit * 0.40 / result.profile_peak_slope
    )

    assert result.segment == segment
    assert result.max_abs_axial_strain == pytest.approx(expected_strain_limit)
    assert result.applied_delta_m == pytest.approx(expected_delta)
    assert result.remaining_residual_m == pytest.approx(0.20 - expected_delta)
    assert result.maximum_abs_applied_strain == pytest.approx(
        expected_strain_limit
    )


def test_c2_profile_has_rigid_caps_and_zero_cap_strain() -> None:
    vertices, _parameter, proximal, distal = _tilted_cylinder(
        stations=11,
        rings=1,
        radius=0.0,
    )
    parameter = np.asarray(
        (0.0, 0.025, 0.05, 0.10, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95, 1.0)
    )
    result = apply_cap_preserving_axial_rest_v810(
        vertices,
        proximal=proximal,
        distal=distal,
        target_length_delta_m=0.01,
        axial_parameter=parameter,
        proximal_cap_fraction=0.10,
        distal_cap_fraction=0.20,
    )

    assert np.all(result.phi[parameter <= 0.10] == 0.0)
    assert np.all(result.phi[parameter >= 0.80] == 1.0)
    assert np.all(result.profile_derivative[parameter <= 0.10] == 0.0)
    assert np.all(result.profile_derivative[parameter >= 0.80] == 0.0)
    assert np.all(np.diff(result.phi) >= 0.0)
    assert result.profile_derivative[4] == pytest.approx(
        result.profile_peak_slope
    )
    assert result.profile_derivative[5] == pytest.approx(
        result.profile_peak_slope
    )
    assert np.max(result.profile_derivative) == pytest.approx(
        1.0 / (1.0 - 1.5 * (0.10 + 0.20))
    )


@pytest.mark.parametrize(
    ("override", "match"),
    (
        ({"target_length_delta_m": np.inf}, "finite"),
        ({"axial_parameter": (0.0, np.nan)}, "finite"),
        ({"axial_parameter": (0.0, 1.01)}, r"\[0, 1\]"),
        ({"proximal": (0.0, 0.0, 0.0), "distal": (0.0, 0.0, 0.0)}, "non-degenerate"),
        ({"segment": "foot"}, "segment"),
        ({"proximal_cap_fraction": 0.0}, "cap fractions"),
        (
            {
                "proximal_cap_fraction": 0.60,
                "distal_cap_fraction": 0.50,
            },
            "cap fractions",
        ),
        ({"max_abs_axial_strain": 1.0}, "max_abs_axial_strain"),
    ),
)
def test_cap_preserving_adapter_rejects_invalid_inputs(
    override: dict[str, object],
    match: str,
) -> None:
    arguments: dict[str, object] = {
        "vertices": ((0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        "proximal": (0.0, 0.0, 0.0),
        "distal": (0.0, 1.0, 0.0),
        "target_length_delta_m": 0.01,
        "axial_parameter": (0.0, 1.0),
    }
    arguments.update(override)
    with pytest.raises(ValueError, match=match):
        apply_cap_preserving_axial_rest_v810(**arguments)


def test_zero_strain_budget_applies_nothing_and_reports_full_residual() -> None:
    result = apply_cap_preserving_axial_rest_v810(
        ((0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        proximal=(0.0, 0.0, 0.0),
        distal=(0.0, 1.0, 0.0),
        target_length_delta_m=-0.95,
        axial_parameter=(0.0, 1.0),
        max_abs_axial_strain=0.0,
    )
    assert result.applied_delta_m == 0.0
    assert result.remaining_residual_m == pytest.approx(-0.95)
    np.testing.assert_allclose(result.axial_jacobian, 1.0, atol=0.0)
