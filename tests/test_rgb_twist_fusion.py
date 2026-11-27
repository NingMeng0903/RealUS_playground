import numpy as np
import pytest

from perception.benchmark.rgb_twist_fusion import fuse_twist_predictions


def _phis(degrees):
    a = np.deg2rad(degrees)
    return np.tile(np.stack([np.cos(a), np.sin(a)], axis=-1)[:, None, :], (1, 54, 1))


def test_circular_consensus_crosses_pi_without_averaging_to_zero_angle():
    fallback = _phis([0])[0]
    fused, accepted, strength = fuse_twist_predictions(_phis([179, -179]), fallback)
    assert accepted.all()
    np.testing.assert_allclose(fused, _phis([180])[0], atol=1e-6)
    assert (strength > .99).all()


def test_disagreement_and_single_view_preserve_fallback():
    fallback = _phis([37])[0]
    for observations in (_phis([0, 180]), _phis([50])):
        fused, accepted, _ = fuse_twist_predictions(observations, fallback)
        assert not accepted.any()
        np.testing.assert_allclose(fused, fallback, atol=1e-6)


def test_invalid_camera_is_excluded_and_magnitude_is_not_confidence():
    observations = _phis([25, 25, 100])
    observations[0] *= 8
    observations[2] = np.nan
    fused, accepted, _ = fuse_twist_predictions(observations, _phis([0])[0])
    assert accepted.all()
    np.testing.assert_allclose(fused, _phis([25])[0], atol=1e-6)


def test_zero_fallback_is_rejected():
    with pytest.raises(ValueError, match='nonzero'):
        fuse_twist_predictions(_phis([0, 0]), np.zeros((54, 2)))
