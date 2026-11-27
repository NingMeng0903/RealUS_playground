from __future__ import annotations

import json

import numpy as np

from perception.benchmark.observation_geometry import (
    BODY_WRIST_INDEX,
    HAND_ROOT_INDEX,
    filter_hand_observations,
)


def _rest_model() -> np.ndarray:
    """Small, finite Body25 + hand21 + hand21 rest landmark model."""

    rest = np.zeros((67, 3), dtype=np.float64)
    rest[:25] = np.arange(25, dtype=np.float64)[:, None] * np.array([0.01, 0.0, 0.0])
    for side, body_index, root_index, x in (
        ("left", 7, 25, 0.12),
        ("right", 4, 46, -0.12),
    ):
        del side
        rest[body_index] = (0.0, 0.0, 0.0)
        rest[root_index] = (x, 0.0, 0.0)
        for local_chain, chain in enumerate(((1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20))):
            del local_chain
            previous = root_index
            for local_index in chain:
                rest[root_index + local_index] = rest[previous] + np.array([0.0, 0.03, 0.0])
                previous = root_index + local_index
    return rest


def _valid_frame(rest: np.ndarray) -> np.ndarray:
    points = np.column_stack((rest.copy(), np.full(67, 0.9, dtype=np.float64)))
    # Give the body landmarks a non-degenerate but irrelevant current pose.
    points[:, :3] += np.array([1.3, -0.4, 2.2])
    return points


def _rotate_z(points: np.ndarray, angle: float) -> np.ndarray:
    cosine, sine = np.cos(angle), np.sin(angle)
    rotation = np.array(((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)))
    result = points.copy()
    result[:, :3] = points[:, :3] @ rotation.T
    return result


def test_large_current_translation_and_rotation_are_retained_and_input_is_unchanged():
    rest = _rest_model()
    original = _rotate_z(_valid_frame(rest), 1.1)
    original[:, :3] += np.array([-3.5, 1.7, 0.8])
    before = original.copy()

    filtered, metadata = filter_hand_observations(original, rest)

    np.testing.assert_array_equal(original, before)
    np.testing.assert_allclose(filtered[:, :3], original[:, :3])
    assert np.all(filtered[:, 3] > 0.0)
    assert metadata["rejected_count"] == 0


def test_gross_mcp_outlier_is_removed_while_distal_fingers_survive_from_root():
    rest = _rest_model()
    points = _valid_frame(rest)
    points[30, :3] += np.array([0.5, 0.0, 0.0])  # left hand local MCP 5

    filtered, metadata = filter_hand_observations(points, rest)

    assert filtered[30, 3] == 0.0
    assert metadata["reasons"][30] == "outside_anchor_reach"
    assert np.all(filtered[[31, 32, 33], 3] > 0.0)
    assert np.all(filtered[[26, 27, 28, 29], 3] > 0.0)


def test_flexed_chain_uses_rest_path_length_instead_of_rest_endpoint_distance():
    rest = _rest_model()
    # Make left middle-finger rest geometry a bent path whose endpoint returns
    # close to the root.  The current pose straightens the same four edges.
    root = rest[25].copy()
    rest[34] = root + np.array([0.03, 0.0, 0.0])
    rest[35] = rest[34] + np.array([0.0, 0.03, 0.0])
    rest[36] = rest[35] + np.array([-0.03, 0.0, 0.0])
    rest[37] = rest[36] + np.array([0.0, -0.03, 0.0])
    points = _valid_frame(rest)
    current_root = points[25, :3].copy()
    for offset, global_index in enumerate((34, 35, 36, 37), start=1):
        points[global_index, :3] = current_root + np.array([0.03 * offset, 0.0, 0.0])

    filtered, metadata = filter_hand_observations(points, rest)

    assert np.all(filtered[[34, 35, 36, 37], 3] > 0.0), metadata


def test_wrong_body_wrist_prefers_coherent_hand_root_and_masks_body_wrist():
    rest = _rest_model()
    points = _valid_frame(rest)
    points[7, :3] = points[25, :3] + np.array([0.206, 0.0, 0.0])
    points[7, 3] = 0.27
    points[25, 3] = 0.37

    filtered, metadata = filter_hand_observations(points, rest)
    hand = metadata["hands"]["left"]

    assert hand["anchor_choice"] == "hand_root"
    assert hand["hand_root_mcp_support"] >= 3
    assert filtered[7, 3] == 0.0
    assert filtered[25, 3] > 0.0
    assert np.all(filtered[26:46, 3] > 0.0)


def test_zero_support_solitary_hand_root_is_masked_and_body_anchor_survives():
    rest = _rest_model()
    points = _valid_frame(rest)
    # The Body25 wrist and fingers remain in the model-consistent cluster, but
    # the observed hand root is a detached low-confidence point.  All five
    # MCPs are still valid, so the root must not pull the hand toward itself.
    points[25, :3] += np.array([0.5, 0.0, 0.0])
    points[25, 3] = 0.19

    filtered, metadata = filter_hand_observations(points, rest)
    hand = metadata["hands"]["left"]

    assert hand["anchor_choice"] == "body_wrist"
    assert hand["hand_root_anchor_unusable"] is True
    assert hand["hand_root_valid"] is False
    assert hand["hand_root_input_valid"] is True
    assert hand["hand_root_raw_confidence"] == 0.19
    assert metadata["reasons"][25] == "unsupported_wrist_anchor"
    assert filtered[25, 3] == 0.0
    assert metadata["rejected_mask"][25] is True
    assert np.all(filtered[26:46, 3] > 0.0)


def test_root_with_one_supporting_finger_is_not_marked_unsupported():
    rest = _rest_model()
    points = _valid_frame(rest)
    # Keep the first finger attached and throw the other four chains away.
    # One in-reach chain gives the root positive support, so the zero-support
    # guard must not fire even though at least three MCPs remain observed.
    for chain in ((5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)):
        for local_index in chain:
            points[25 + local_index, :3] += np.array([0.5, 0.0, 0.0])

    filtered, metadata = filter_hand_observations(points, rest)
    hand = metadata["hands"]["left"]

    assert hand["hand_root_anchor_unusable"] is False
    assert hand["hand_root_valid"] is True
    assert hand["hand_root_score"] > 0.0
    assert metadata["reasons"][25] == "accepted"
    assert filtered[25, 3] > 0.0
    assert np.all(filtered[26:30, 3] > 0.0)
    assert metadata["policy_version"] == 3


def test_rootless_weak_cloud_is_masked_but_strong_points_are_retained_without_inventing_root():
    rest = _rest_model()
    points = _valid_frame(rest)
    points[7, 3] = 0.0
    points[25, 3] = 0.0
    points[26:46, 3] = 0.4
    points[35, 3] = 0.6

    filtered, metadata = filter_hand_observations(points, rest)
    hand = metadata["hands"]["left"]

    assert hand["anchor_choice"] == "unanchored"
    assert hand["unanchored"] is True
    assert filtered[25, 3] == 0.0
    assert filtered[35, 3] == 0.6
    assert filtered[26, 3] == 0.0
    assert metadata["reasons"][26] == "unanchored_low_confidence_suppressed"


def test_all_nan_current_frame_is_explicitly_missing_and_metadata_is_json_safe():
    rest = _rest_model()
    points = np.full((67, 4), np.nan, dtype=np.float64)

    filtered, metadata = filter_hand_observations(points, rest)

    assert np.all(filtered[:, 3] == 0.0)
    assert metadata["rejected_count"] == 0  # Missing input is not a new geometric rejection.
    assert not any(metadata["input_valid_mask"])
    assert not any(metadata["rejected_mask"])
    assert all(reason == "input_missing_or_low_confidence" for reason in metadata["reasons"])
    json.dumps(metadata, allow_nan=False)


def test_layout_constants_match_body25_hand21_contract():
    assert BODY_WRIST_INDEX == {"left": 7, "right": 4}
    assert HAND_ROOT_INDEX == {"left": 25, "right": 46}
