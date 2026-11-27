"""Focused tests for the finite categorical sensor contract."""
from dataclasses import replace
import sys

import numpy as np

sys.path.insert(0, ".")
import sensor
from observation import environment as env


POSES = np.array([[0.0, 0.0, 0.0], [0.008, -0.004, 0.15]])


def _passing_world():
    # Cancel the renderer's nominal transverse slope for source 0.  Source 0
    # is a long in-plane target; source 1 remains outside the plane.
    yc = np.zeros((2, 5), dtype=float)
    yc[0] = -0.25 * env.CONTROL_X
    return env._world(
        np.array([0.0, -0.014]),
        np.array([[1.0, 1.0], [-1.0, 1.0]]),
        yc,
        np.zeros((2, 5)),
        np.array([0.035, 0.035]),
        np.zeros(2),
        np.zeros(2),
    )


def _off_frame_world():
    return env._world(
        np.array([0.0, 0.30]),
        np.array([[1.0, 1.0], [-1.0, 1.0]]),
        np.zeros((2, 5)),
        np.zeros((2, 5)),
        np.array([0.035, 0.035]),
        np.zeros(2),
        np.zeros(2),
    )


def _swap_sources(world):
    return replace(
        world,
        modes=world.modes[::-1].copy(),
        y_controls=world.y_controls[::-1].copy(),
        z_controls=world.z_controls[::-1].copy(),
        depths=world.depths[::-1].copy(),
        identity_z=world.identity_z[::-1].copy(),
        detect_z=world.detect_z[::-1].copy(),
        points=world.points[::-1].copy(),
    )


def test_shape_normalization_and_public_encoding():
    worlds = env.sample_worlds(31, 3)
    result = sensor.channel(worlds, POSES)
    assert len(sensor.OUTCOMES) == 61
    assert result.joint.shape == (3, 2, 2, 4, 61)
    assert result.likelihood.shape == (3, 2, 2, 61)
    assert result.correct_joint.shape == (3, 2, 2, 61)
    assert result.geometric_target_visible.shape == (3, 2)
    assert np.allclose(result.joint.sum(axis=(3, 4)), 1.0)
    assert np.allclose(result.likelihood.sum(axis=-1), 1.0)
    assert np.all(result.joint >= 0.0)
    assert np.all(result.correct_joint >= 0.0)
    assert np.all(result.correct_joint <= result.joint[:, :, :, 0, :])
    assert sensor.PUBLIC_GATE.sum() == 3
    assert sensor.OUTCOMES[0]["visible"] is False
    for key in ("image_u_m", "image_depth_m", "visible_length_m", "image_axis_rad", "identity_score"):
        assert sensor.OUTCOMES[0][key] is None


def test_off_frame_has_clutter_support_but_no_correct_mass():
    result = sensor.channel([_off_frame_world()], [[0.0, 0.0, 0.0]])
    assert not bool(result.geometric_target_visible[0, 0])
    # No source has geometric return, but bad-mode clutter remains a positive
    # alternative and therefore the public likelihood has open support.
    assert np.sum(result.joint[0, 1, 0, 2, 1:]) > 0.0
    assert np.sum(result.joint[0, 0, 0, 2, 1:]) > 0.0
    assert np.allclose(result.correct_joint, 0.0)


def test_mode_transitions_share_stationary_bad_marginal():
    stationary = np.array([0.7, 0.3])
    for name, transition in sensor.TRANSITIONS.items():
        assert np.allclose(transition.sum(axis=1), 1.0)
        if name != "clean":
            assert np.allclose(stationary @ transition, stationary)
            assert np.allclose(sensor.INITIAL_MODE[name], stationary)
    assert np.allclose(sensor.INITIAL_MODE["clean"], [1.0, 0.0])


def test_identity_blind_source_swap_has_same_public_mixture():
    world = _passing_world()
    swapped = _swap_sources(world)
    original = sensor.channel([world], [[0.0, 0.0, 0.0]], identity_blind=True)
    exchanged = sensor.channel([swapped], [[0.0, 0.0, 0.0]], identity_blind=True)
    assert np.allclose(original.likelihood, exchanged.likelihood, atol=2e-12)
    assert np.sum(original.correct_joint) > 0.0
    assert np.allclose(exchanged.correct_joint, 0.0)


def test_channel_is_deterministic_and_target_quality_is_geometry_only():
    world = _passing_world()
    first = sensor.channel([world], [[0.0, 0.0, 0.0]])
    second = sensor.channel([world], [[0.0, 0.0, 0.0]])
    assert np.array_equal(first.joint, second.joint)
    assert np.array_equal(first.correct_joint, second.correct_joint)
    # A correct evaluator mass can only occur in the center/long/high-ID
    # public gate and only when the private target geometry is actually good.
    assert np.sum(first.correct_joint[0, 0, 0]) > 0.0
    assert np.all(first.correct_joint[0, 0, 0, ~sensor.PUBLIC_GATE] == 0.0)
