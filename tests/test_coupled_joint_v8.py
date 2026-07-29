from __future__ import annotations

import numpy as np
import pytest

from projects.genesis_ue_sync.anatomy_retarget.coupled_joint_v8 import (
    bake_coupled_rbf_response_v8,
    coupled_state_centers_v8,
    evaluate_coupled_rbf_response_v8,
)


def _coupled_translation(states: np.ndarray) -> np.ndarray:
    """A smooth field with terms no sum of independent axes can represent."""

    x, y, z = np.asarray(states, dtype=np.float64).T
    return np.stack(
        (
            0.0020 * x + 0.0015 * x * y,
            -0.0012 * y + 0.0010 * y * z,
            0.0008 * z + 0.0013 * x * z,
        ),
        axis=1,
    )


def test_coupled_state_design_contains_mixed_axis_samples() -> None:
    centers = coupled_state_centers_v8(support_radius_rad=np.radians(130.0))

    assert centers.shape == (53, 3)
    assert np.allclose(centers[0], 0.0)
    assert np.count_nonzero(np.count_nonzero(centers, axis=1) >= 2) == 40
    assert np.max(np.linalg.norm(centers, axis=1)) == pytest.approx(
        np.radians(130.0)
    )


@pytest.mark.parametrize(
    "capture_state_deg",
    (
        (34.19, 31.61, -82.09),
        (-2.88, -8.42, 3.00),
        (11.01, 4.57, -1.54),
        (19.26, 16.18, 10.39),
        (40.77, 21.23, 17.04),
        (24.55, -5.50, 11.98),
        (33.25, -14.14, -32.43),
        (38.92, 22.80, -5.31),
    ),
)
def test_coupled_rbf_supports_large_composite_capture_states(
    capture_state_deg: tuple[float, float, float],
) -> None:
    radius = np.radians(130.0)
    centers = coupled_state_centers_v8(support_radius_rad=radius)
    response = bake_coupled_rbf_response_v8(
        states_rotvec_rad=centers,
        translations_parent_local_m=_coupled_translation(centers),
        smplx_joint=4,
        joint_kind="knee",
        support_radius_rad=radius,
        maximum_translation_m=0.020,
    )

    state = np.radians(np.asarray(capture_state_deg))
    actual = evaluate_coupled_rbf_response_v8(response, state)

    assert actual.shape == (3,)
    assert np.all(np.isfinite(actual))
    assert np.linalg.norm(actual) <= 0.020
    assert response["independent_axis_sum"] is False


def test_coupled_rbf_has_true_cross_axis_interaction_and_zero_bind() -> None:
    radius = np.radians(130.0)
    centers = coupled_state_centers_v8(support_radius_rad=radius)
    response = bake_coupled_rbf_response_v8(
        states_rotvec_rad=centers,
        translations_parent_local_m=_coupled_translation(centers),
        smplx_joint=4,
        joint_kind="knee",
        support_radius_rad=radius,
        maximum_translation_m=0.020,
    )
    mixed = np.radians(np.asarray((42.0, -37.0, 26.0)))
    x_only = np.asarray((mixed[0], 0.0, 0.0))
    y_only = np.asarray((0.0, mixed[1], 0.0))
    xy_only = np.asarray((mixed[0], mixed[1], 0.0))
    zero = evaluate_coupled_rbf_response_v8(response, np.zeros(3))
    interaction = (
        evaluate_coupled_rbf_response_v8(response, xy_only)
        - evaluate_coupled_rbf_response_v8(response, x_only)
        - evaluate_coupled_rbf_response_v8(response, y_only)
        + zero
    )

    np.testing.assert_array_equal(zero, np.zeros(3))
    assert np.linalg.norm(interaction) > 1.0e-5


def test_coupled_rbf_fails_closed_outside_baked_state_ball() -> None:
    radius = np.radians(75.0)
    centers = coupled_state_centers_v8(support_radius_rad=radius)
    response = bake_coupled_rbf_response_v8(
        states_rotvec_rad=centers,
        translations_parent_local_m=np.zeros_like(centers),
        smplx_joint=7,
        joint_kind="ankle",
        support_radius_rad=radius,
        maximum_translation_m=0.020,
    )

    with pytest.raises(ValueError, match="exceeds"):
        evaluate_coupled_rbf_response_v8(
            response,
            np.radians(np.asarray((76.0, 0.0, 0.0))),
        )
