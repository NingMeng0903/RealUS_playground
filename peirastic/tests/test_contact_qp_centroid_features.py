"""Full-column image centroid, additive evidence transport, and image convention."""
from dataclasses import replace
import json

import numpy as np
import pytest

from peirastic.apps.contact_qp_features import encode_feature_payload
from peirastic.contact_qp.features import (
    CONFIDENCE_FEATURE_VERSION, FeatureConfig, FeatureExtractor,
    confidence_centroid_from_columns, confidence_features,
)
from peirastic.contact_qp.types import ContactObservation


def observation(**kwargs):
    return ContactObservation(1, 'camera', 1., 1.1, [.8, .4, .8], [True]*3,
                              'registration', 'windows', **kwargs)


@pytest.mark.parametrize('columns,expected', [
    ([1, 1, 1, 1], 0.), ([3, 1, 1, 3], 0.), ([1], 0.),
    ([1, 0, 0, 0], -.75), ([0, 0, 0, 1], .75),
    ([1, 0, 0, 3], .375),
])
def test_centroid_image_direction_symmetry_and_pixel_centers(columns, expected):
    evidence = confidence_centroid_from_columns(columns)
    assert evidence['confidence_feature_version'] == CONFIDENCE_FEATURE_VERSION
    assert evidence['confidence_centroid_valid']
    assert evidence['confidence_centroid_x'] == pytest.approx(expected)
    assert confidence_centroid_from_columns(columns[::-1])['confidence_centroid_x'] == pytest.approx(-expected)


@pytest.mark.parametrize('columns', [[], [0, 0], [np.nan, 1], [np.inf, 1],
                                     [-1, 2], [1e308, 1e308]])
def test_empty_or_invalid_mass_has_no_centroid(columns):
    evidence = confidence_centroid_from_columns(columns)
    assert evidence == dict(confidence_centroid_x=None, confidence_centroid_valid=False,
                            confidence_feature_version=CONFIDENCE_FEATURE_VERSION)
    assert json.loads(json.dumps(observation(**evidence).to_dict(), allow_nan=False))['confidence_centroid_x'] is None


def test_column_shape_must_not_silently_collapse_an_image():
    with pytest.raises(ValueError, match='one-dimensional'):
        confidence_centroid_from_columns([[1, 2], [3, 4]])


def test_old_payload_roundtrip_preserves_absence_and_schema():
    old = observation().to_dict()
    assert old['schema_version'] == 1
    assert not any(key.startswith('confidence_') for key in old)
    restored = ContactObservation.from_dict(old)
    assert restored.confidence_centroid_x is None
    assert not restored.confidence_centroid_valid
    assert restored.confidence_feature_version is None
    assert restored.to_dict() == old


def test_new_payload_roundtrip_keeps_versioned_evidence_bounded():
    evidence = confidence_centroid_from_columns([1, 2, 4, 8])
    obs = observation(**evidence)
    blob = encode_feature_payload(obs, None, .035)
    restored = ContactObservation.from_dict(json.loads(blob))
    assert restored.to_dict() == obs.to_dict()
    assert restored.version == observation().version  # Existing LCR contract remains stable.
    assert len(blob) < 8192
    assert json.loads(blob)['confidence_centroid_x'] > 0


@pytest.mark.parametrize('overrides', [
    dict(confidence_centroid_valid=True),
    dict(confidence_centroid_x=.2),
    dict(confidence_feature_version=''),
    dict(confidence_centroid_valid='false'),
    *[dict(confidence_centroid_x=x, confidence_centroid_valid=True,
           confidence_feature_version=CONFIDENCE_FEATURE_VERSION)
      for x in (None, np.nan, np.inf, -1.01, 1.01, True)],
])
def test_inconsistent_or_invalid_transport_is_rejected(overrides):
    with pytest.raises(ValueError):
        observation(**overrides)


def test_full_columns_include_mass_outside_lcr_windows_and_match_pixel_barycentre():
    cfg = FeatureConfig(width=8, height=8, region_count=8, near_depth=(0., .5),
                        lateral_windows=((.25, .375), (.375, .5), (.5, .75)))
    image = np.broadcast_to(np.arange(8)[:, None], (8, 8))
    c = np.zeros((8, 8))
    c[:, -1] = 1.
    features = confidence_features(image, c, cfg)
    assert features['quality_lcr'] == [0., 0., 0.]
    assert features['confidence_centroid_x'] == .875
    x_pixel = features['barycenter_row_col_1based'][1]
    assert features['confidence_centroid_x'] == (2*x_pixel - 1 - cfg.width)/cfg.width


@pytest.mark.parametrize('algorithm', ['randomwalk_thesis_v1', 'randomwalk_camp_bmode_v2',
                                       'randomwalk_welleweerd2020_v3'])
def test_extraction_centroid_matches_columns_and_hflip_is_already_in_image(algorithm):
    cfg = FeatureConfig(width=24, height=32, algorithm_version=algorithm)
    image = np.random.default_rng(44).integers(30, 200, (32, 24)).astype(float)
    image[2:, :8] = 0.
    image[:2, :8] = 200.
    extractor = FeatureExtractor(cfg)
    kwargs = dict(frame_seq=1, source_id='camera', capture_time_s=2., received_time_s=2.1)
    obs, confidence = extractor.extract(image, **kwargs)
    flipped, _ = extractor.extract(image[:, ::-1], hflip=True, **kwargs)
    metadata_only, _ = extractor.extract(image, hflip=True, **kwargs)
    y0, y1 = (int(cfg.height*f) for f in cfg.near_depth)
    expected = confidence_centroid_from_columns(confidence[y0:max(y0+1, y1)].mean(axis=0))
    assert obs.confidence_centroid_x == expected['confidence_centroid_x']
    assert obs.confidence_centroid_valid and obs.confidence_centroid_x > 0
    assert flipped.confidence_centroid_x == pytest.approx(-obs.confidence_centroid_x, abs=1e-9)
    assert metadata_only.confidence_centroid_x == obs.confidence_centroid_x
    assert metadata_only.registration_version != obs.registration_version
    signed, _ = FeatureExtractor(replace(cfg, image_x_sign=-1)).extract(image, **kwargs)
    assert signed.confidence_centroid_x == obs.confidence_centroid_x


def test_mass_validity_does_not_override_image_validity():
    cfg = FeatureConfig(width=8, height=8, region_count=8)
    obs, _ = FeatureExtractor(cfg).extract(np.zeros((8, 8)), frame_seq=1, source_id='camera',
                                          capture_time_s=2., received_time_s=2.1)
    assert obs.confidence_centroid_valid  # Harmonic confidence still has positive mass.
    assert not obs.valid.any()
    assert not obs.fresh(2.1, .3)
