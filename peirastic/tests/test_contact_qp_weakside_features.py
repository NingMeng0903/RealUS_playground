"""Weak-side q25 evidence retains legacy means, registration and unknown flags."""
import json

import numpy as np
import pytest

from peirastic.contact_qp.features import FeatureExtractor, confidence_features, weakside_quality, welleweerd_config
from peirastic.contact_qp.types import ContactObservation, WEAK_SIDE_FEATURE_VERSION


def test_fixed_boundary_is_excluded_and_fragmented_weak_side_is_detected():
    cfg = welleweerd_config()
    c = np.full((100, 145), .9)
    c[0] = 1.
    c[1:22, 5:22] = .7
    image = np.tile(np.arange(100)[:, None], (1, 145))
    features = confidence_features(image, c, cfg)
    assert features['weakside_roi_rows'] == [1, 22]
    assert features['confidence_lr'][0] == pytest.approx(.7)
    assert features['quality_lcr'][0] > .8
    assert features['confidence_lr'][1] == pytest.approx(.9)
    c[0] = 0.
    np.testing.assert_allclose(weakside_quality(c, cfg)['confidence_lr'], [.7, .9])


def test_symmetric_registered_windows_mirror_weak_side():
    cfg = welleweerd_config(width=100)
    c = np.tile(np.linspace(.2, .95, 100), (100, 1))
    np.testing.assert_allclose(weakside_quality(c, cfg)['confidence_lr'],
                               weakside_quality(c[:, ::-1], cfg)['confidence_lr'][::-1])


def test_unknown_does_not_zero_quality_and_transport_roundtrips():
    cfg = welleweerd_config()
    image = np.tile(np.arange(100)[:, None], (1, 145)).astype(float)
    image[:, 80:] = 0.
    obs, c = FeatureExtractor(cfg).extract(image, frame_seq=2, source_id='camera',
                                         capture_time_s=2., received_time_s=2.1)
    assert obs.weakside_feature_version == WEAK_SIDE_FEATURE_VERSION
    assert obs.timestamp_semantics == 'effective_image_time'
    np.testing.assert_array_equal(obs.confidence_lr_valid, [True, False])
    assert obs.confidence_lr[1] > 0
    np.testing.assert_allclose(obs.confidence_lr, weakside_quality(c, cfg)['confidence_lr'])
    restored = ContactObservation.from_dict(json.loads(json.dumps(obs.to_dict(), allow_nan=False)))
    assert restored.to_dict() == obs.to_dict()
    with pytest.raises(ValueError):
        obs.confidence_lr[0] = 0.
