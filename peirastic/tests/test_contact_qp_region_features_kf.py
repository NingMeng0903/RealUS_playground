"""Spatial localization and delayed fusion of independently observed regions."""
from dataclasses import replace
import json

import numpy as np
import pytest

from peirastic.contact_qp.features import FeatureConfig, region_quality, welleweerd_config
from peirastic.contact_qp.delayed_kf import DelayConfidenceKF, KfConfig
from peirastic.contact_qp.types import ContactObservation, REGION_FEATURE_VERSION
from peirastic.apps.contact_qp_features import encode_feature_payload


def observed(seq=1, stamp=1., count=10, **kwargs):
    cfg = welleweerd_config(region_count=count)
    values = dict(frame_seq=seq, source_id='camera', effective_time_s=stamp,
                  received_time_s=1.2, quality=[.99]*3, valid=[False]*3,
                  registration_version='registered', window_version=cfg.window_version,
                  calibration_version=cfg.calibration_version,
                  region_confidence=np.linspace(.5, .95, count), region_valid=[True]*count,
                  region_edges=cfg.region_edges, region_layout_version=cfg.region_layout_version,
                  region_feature_version=REGION_FEATURE_VERSION, timestamp_semantics='effective_image_time')
    return ContactObservation(**(values | kwargs))


def regional_filter(count=10):
    return DelayConfidenceKF(KfConfig(q=.004, r=.001, channels=count, observation_kind='regions'))


def test_ten_bins_localize_center_and_disconnected_poor_regions_without_lcr():
    cfg = welleweerd_config(width=100, region_count=10)
    confidence = np.full((100, 100), .95)
    confidence[1:, 10:20] = .4
    confidence[1:, 40:60] = .5
    confidence[1:, 80:90] = .6
    confidence[0] = 1.
    feature = region_quality(confidence, cfg)
    assert feature['region_bad_indices'] == [1, 4, 5, 8]
    np.testing.assert_allclose(np.diff(feature['region_edges']), .1)
    assert feature['region_roi_rows'] == [1, 22]
    np.testing.assert_allclose(np.asarray(feature['region_confidence'])[[1, 4, 5, 8]], [.4, .5, .5, .6])


@pytest.mark.parametrize('count', [2, 7, 10, 16, 20])
def test_fractional_pixel_boundaries_preserve_full_width_and_mirror(count):
    cfg = welleweerd_config(region_count=count)
    rng = np.random.default_rng(38)
    confidence = np.tile(rng.uniform(.3, 1., cfg.width), (cfg.height, 1))
    confidence[0] = 1.
    original = region_quality(confidence, cfg)
    mirrored = region_quality(confidence[:, ::-1], cfg)
    np.testing.assert_allclose(original['region_confidence'], mirrored['region_confidence'][::-1], atol=1e-12)
    assert original['region_edges'][0] == 0. and original['region_edges'][-1] == 1.


def test_unknown_regions_preserve_raw_values_and_payload_is_versioned():
    cfg = welleweerd_config()
    unknown = np.zeros(cfg.width, dtype=bool); unknown[70] = True
    feature = region_quality(np.full((cfg.height, cfg.width), .6), cfg, unknown)
    assert not feature['region_valid'][4]
    assert sum(feature['region_valid']) == 9
    assert feature['region_confidence'][4] == pytest.approx(.6)
    raw = {k: feature[k] for k in ('region_confidence', 'region_valid', 'region_edges',
                                  'region_feature_version', 'region_layout_version')}
    obs = observed(**raw)
    blob = encode_feature_payload(obs, feature, .005)
    restored = ContactObservation.from_dict(json.loads(blob))
    np.testing.assert_array_equal(restored.region_valid, obs.region_valid)
    assert restored.to_dict() == obs.to_dict()


def test_regional_kf_has_twenty_states_and_ignores_legacy_three_window_validity():
    kf = regional_filter()
    obs = observed()
    assert not obs.valid.any()  # Legacy diagnostics cannot veto regional evidence.
    assert kf.ingest(obs, 1.2)
    prediction = kf.predict(1.2)
    assert prediction.valid and prediction.state.shape == (20,)
    assert prediction.covariance.shape == (20, 20)
    np.testing.assert_array_equal(prediction.quality_raw, obs.region_confidence)
    assert prediction.age_s == pytest.approx(.2)
    assert not kf.ingest(obs, 1.2)
    np.testing.assert_array_equal(kf.predict(1.2).state, prediction.state)
    assert np.linalg.eigvalsh(prediction.covariance).min() > 0


def test_regional_oos_replay_matches_causal_order():
    frames = [observed(seq=i, stamp=1.+.03*i,
                       region_confidence=np.linspace(.4, .8, 10)+i*.01) for i in range(6)]
    ordered, shuffled = regional_filter(), regional_filter()
    for obs in frames:
        assert ordered.ingest(obs, 1.2)
    for i in [3, 1, 4, 0, 5, 2]:
        assert shuffled.ingest(frames[i], 1.2)
    a, b = ordered.predict(1.24), shuffled.predict(1.24)
    np.testing.assert_allclose(a.state, b.state, atol=1e-14)
    np.testing.assert_allclose(a.covariance, b.covariance, atol=1e-14)


def test_unknown_or_wrong_count_pauses_without_retiring_healthy_regional_stream():
    kf = regional_filter()
    assert kf.ingest(observed(), 1.2)
    assert not kf.ingest(observed(2, 1.03, count=8), 1.2)
    assert not kf.predict(1.2).valid
    assert kf.ingest(observed(3, 1.06), 1.2)
    assert kf.predict(1.2).valid
    assert not kf.ingest(observed(4, 1.09, region_valid=[True]*5+[False]+[True]*4), 1.2)
    assert not kf.predict(1.2).valid
    assert kf.ingest(observed(5, 1.12), 1.2)
    assert kf.predict(1.2).valid and kf.generation == 0


def test_layout_changes_reset_and_retire_old_layout_independently_of_lcr_hash():
    cfg = welleweerd_config()
    other = replace(cfg, near_depth=(0., .25))
    kf = regional_filter()
    assert kf.ingest(observed(), 1.2)
    assert kf.ingest(observed(2, 1.03, region_layout_version=other.region_layout_version), 1.2)
    assert kf.generation == 1
    assert not kf.ingest(observed(3, 1.06), 1.2)
    assert kf.last_reason == 'retired_identity'
    assert replace(cfg, region_count=16).window_version == cfg.window_version
    assert replace(cfg, region_count=16).region_layout_version != cfg.region_layout_version


def test_two_region_mode_cannot_silently_accept_legacy_lr_frame():
    kf = regional_filter(2)
    assert kf.ingest(observed(count=2), 1.2)
    old = ContactObservation(2, 'camera', 1.03, 1.2, [.9]*3, [True]*3, 'registered', 'old')
    assert not kf.ingest(old, 1.2)
    assert not kf.predict(1.2).valid
    assert kf.ingest(observed(3, 1.06, count=2), 1.2)
    assert kf.predict(1.2).valid


@pytest.mark.parametrize('count', [True, 0, 1, 10.5, 65, 146])
def test_invalid_region_count_rejected(count):
    with pytest.raises(ValueError, match='region_count'):
        welleweerd_config(region_count=count)
