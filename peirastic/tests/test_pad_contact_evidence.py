"""Raw near-field evidence and delayed frame control, using synthetic images."""
from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

from peirastic.apps.contact_qp_features import encode_feature_payload
from peirastic.contact_qp.features import FeatureConfig, FeatureExtractor, nearfield_echo_features, window_quality
from peirastic.contact_qp.types import ContactObservation, ECHO_FEATURE_VERSION, REGION_FEATURE_VERSION
from peirastic.realman8dof.force.contact_observer import ConfidenceSubscriber
from peirastic.realman8dof.modes.pad_contact_config import load_pad_contact_config
from peirastic.realman8dof.modes.pad_contact_evidence import PadContactEvidence


def fixture():
    cfg = load_pad_contact_config()
    cfg['pad']['evidence']['baseline_hold_s'] = .1
    receiver = SimpleNamespace(observation=None, close=lambda **kwargs: None)
    guide = PadContactEvidence(cfg, receiver)
    counter = [0]
    def push(echo=None, t=1., *, seq=None, source='camera:one', regions=None, **kwargs):
        counter[0] += 1
        n = cfg['feature']['config']['width']
        echo = np.full(n, .5) if echo is None else np.asarray(echo)
        regions = np.full(10, .995) if regions is None else np.asarray(regions)
        obs = ContactObservation(counter[0] if seq is None else seq, source, t, t+.01,
            np.full(3, .995), np.ones(3, dtype=bool), cfg['feature']['registration_version'],
            cfg['feature']['window_version'], calibration_version=cfg['geometry']['calibration_version'],
            timestamp_semantics='effective_image_time', region_feature_version=REGION_FEATURE_VERSION,
            region_layout_version=cfg['feature']['region_layout_version'], region_confidence=regions,
            region_valid=np.ones(10, dtype=bool), region_edges=np.linspace(0, 1, 11),
            echo_feature_version=ECHO_FEATURE_VERSION, nearfield_echo=echo,
            nearfield_echo_valid=np.ones(n, dtype=bool), nearfield_echo_depth=[.04, .22])
        receiver.observation = obs
        return guide.update(now_s=t+.02, dt_s=.005, contact_present=True,
                            force_n=kwargs.get('force_n', 4.), stick_speed_m_s=.04)
    return cfg, guide, receiver, push


def warm(push):
    push(t=1.)
    push(t=1.05)
    return push(t=1.1)


def hole(width=2, side='left'):
    values = np.full(145, .5)
    if side == 'left': values[:width] = .01
    elif side == 'right': values[-width:] = .01
    elif side == 'both': values[:width] = values[-width:] = .01
    elif side == 'center': values[70:70+width] = .01
    return values


def test_nearfield_full_aperture_sees_small_side_hole_and_ignores_deep_shadow():
    cfg = FeatureConfig(width=145, height=100)
    image = np.tile(np.linspace(80, 180, 100)[:, None], (1, 145))
    raw = nearfield_echo_features(image, cfg)
    deep = image.copy(); deep[40:, :40] = 0
    np.testing.assert_array_equal(nearfield_echo_features(deep, cfg)['nearfield_echo'], raw['nearfield_echo'])
    missing = image.copy(); missing[4:22, :2] = 0
    actual = nearfield_echo_features(missing, cfg)
    assert actual['nearfield_echo'][:2] == [0., 0.]
    assert all(actual['nearfield_echo_valid'])
    assert not any(nearfield_echo_features(np.zeros_like(image), cfg)['nearfield_echo_valid'])


def test_echo_extension_keeps_confidence_and_versions_and_fits_receiver_envelope():
    config = load_pad_contact_config()
    cfg = FeatureConfig(**config['feature']['config'])
    im = np.random.default_rng(2).integers(0, 256, (100, 145))
    extractor = FeatureExtractor(cfg)
    obs, confidence = extractor.extract(im, frame_seq=900, source_id='realus.us_framegrab:camera:session',
        capture_time_s=123456.1, received_time_s=123456.12)
    np.testing.assert_array_equal(obs.quality, window_quality(confidence, cfg))
    assert obs.window_version == config['feature']['window_version']
    assert obs.region_layout_version == config['feature']['region_layout_version']
    payload = encode_feature_payload(obs, extractor.last_features, .012)
    assert len(payload) <= 8192
    pending = [[b'contact_qp_features_v1', payload]]
    socket = SimpleNamespace(poll=lambda _: bool(pending), recv_multipart=lambda: pending.pop(0))
    subscriber = ConfidenceSubscriber(socket=socket)
    restored = subscriber.snapshot()
    assert subscriber.rejected == 0
    assert restored.to_dict() == obs.to_dict()
    assert json.loads(payload)['nearfield_echo_depth'] == [.04, .22]


def test_optional_echo_schema_rejects_unversioned_or_invalid_vectors():
    _, _, receiver, push = fixture()
    push()
    obs = receiver.observation
    assert ContactObservation.from_dict(obs.to_dict()).to_dict() == obs.to_dict()
    with pytest.raises(ValueError): replace(obs, echo_feature_version=None)
    with pytest.raises(ValueError): replace(obs, nearfield_echo=[2.]*145)
    with pytest.raises(ValueError): replace(obs, nearfield_echo_valid=[True]*144)


def test_invalid_image_records_why_visual_is_disabled():
    _, guide, receiver, push = fixture()
    push()
    receiver.observation = replace(receiver.observation, frame_seq=2,
        effective_time_s=1.01, received_time_s=1.02, echo_feature_version=None,
        nearfield_echo=None, nearfield_echo_valid=None, nearfield_echo_depth=None)
    result = guide.update(now_s=1.03, dt_s=.005, contact_present=True,
                          force_n=4., stick_speed_m_s=.01)
    assert result.reason == 'image_invalid'
    assert result.facts['image_compatibility_reason'] == 'nearfield_echo_feature_missing_or_incompatible'
    assert not result.valid and result.omega_request_rad_s == result.alpha_target == 0.


def test_tiny_two_column_hole_with_q_above_point99_gives_small_continuous_request():
    _, guide, _, push = fixture()
    assert warm(push).reason == 'healthy'
    first = push(hole(), 1.2)
    assert first.omega_request_rad_s == 0 and first.alpha_target == pytest.approx(.25)
    result = push(hole(), 1.25)
    assert result.valid and -0.02 < result.omega_request_rad_s < 0
    assert min(result.facts['quality_raw']) > .99
    assert result.facts['dark_runs'][0]['width_columns'] == 2
    before = guide.baseline.copy()
    push(hole(), 1.3)
    np.testing.assert_array_equal(guide.baseline, before)


def test_duplicate_frames_cannot_confirm_direction_refresh_age_or_complete_recovery():
    _, guide, receiver, push = fixture()
    warm(push)
    push(hole(), 1.2)
    for now in (1.23, 1.24, 1.25):
        result = guide.update(now_s=now, dt_s=.005, contact_present=True, force_n=4., stick_speed_m_s=.04)
        assert result.omega_request_rad_s == 0 and result.facts['direction_frames'] == 1
    old = receiver.observation
    receiver.observation = replace(old, received_time_s=1.35)
    result = guide.update(now_s=1.36, dt_s=.005, contact_present=True, force_n=4., stick_speed_m_s=.04)
    assert result.facts['received_time_s'] == old.received_time_s
    stale = guide.update(now_s=1.501, dt_s=.005, contact_present=True, force_n=4., stick_speed_m_s=.04)
    assert stale.reason == 'image_stale' and stale.alpha_target == 0


def test_reverse_requires_two_new_images_and_drops_old_direction_immediately():
    _, _, _, push = fixture()
    warm(push)
    push(hole(), 1.2)
    assert push(hole(), 1.25).omega_request_rad_s < 0
    assert push(hole(side='right'), 1.3).omega_request_rad_s == 0
    assert push(hole(side='right'), 1.35).omega_request_rad_s > 0


def test_persistent_defect_pauses_at_half_second():
    _, _, _, push = fixture()
    warm(push)
    push(hole(), 1.2)
    for t in (1.3, 1.4, 1.5, 1.6): assert push(hole(), t).alpha_target == pytest.approx(.25)
    assert push(hole(), 1.71).alpha_target == 0


def test_recovery_requires_new_raw_observations_for_quarter_second():
    _, guide, receiver, push = fixture()
    warm(push)
    push(hole(), 1.2); push(hole(), 1.25)
    recovered = push(t=1.3)
    assert recovered.reason == 'recovery_confirming' and recovered.alpha_target == 0
    # A forecast with excellent confidence cannot advance raw recovery evidence.
    guide.kf.predict = lambda _: SimpleNamespace(reason='valid', age_s=.1, quality_raw=np.ones(10))
    held = guide.update(now_s=1.57, dt_s=.005, contact_present=True, force_n=4., stick_speed_m_s=.04)
    assert held.alpha_target == 0 and not held.facts['observation_fresh']
    assert push(t=1.56).reason == 'healthy'


@pytest.mark.parametrize('echo,reason', [
    (np.full(145, .01), 'uniform_low_echo'),
    (hole(side='both'), 'bilateral_missing_echo'),
    (hole(side='center'), 'ambiguous_missing_echo'),
])
def test_ambiguous_missing_echo_pauses_without_inventing_direction(echo, reason):
    _, _, _, push = fixture()
    warm(push)
    result = push(echo, 1.2)
    assert result.reason == reason and result.omega_request_rad_s == result.alpha_target == 0


def test_deep_only_confidence_shadow_cannot_trigger_and_conflict_pauses():
    _, _, _, push = fixture()
    warm(push)
    shadow = np.r_[np.full(5, .995), np.full(5, .4)]
    assert push(t=1.2, regions=shadow).reason == 'healthy'
    conflict = push(hole(), 1.25, regions=shadow)
    assert conflict.reason == 'conflicting_spatial_evidence'
    assert conflict.alpha_target == conflict.omega_request_rad_s == 0


def test_contact_force_and_dropout_freeze_baseline():
    _, guide, _, push = fixture()
    warm(push)
    baseline = guide.baseline.copy()
    unstable = push(np.full(145, .8), 1.2, force_n=3.)
    assert unstable.reason == 'force_unstable'
    np.testing.assert_array_equal(guide.baseline, baseline)
    absent = guide.update(now_s=1.23, dt_s=.005, contact_present=False, force_n=3., stick_speed_m_s=.04)
    assert absent.reason == 'contact_unavailable'
    np.testing.assert_array_equal(guide.baseline, baseline)


def test_stream_restart_resets_baseline_and_old_source_cannot_replace_new_source():
    _, guide, _, push = fixture()
    warm(push)
    result = push(t=1.2, seq=0, source='camera:two')
    assert result.reason == 'baseline_warming' and guide.baseline is None
    result = push(hole(), 1.25, seq=999, source='camera:one')
    assert result.facts['source_id'] == 'camera:two'
    assert not result.facts['observation_fresh']
    assert result.omega_request_rad_s == 0


def test_missing_or_mismatched_echo_is_explicit_pause():
    _, guide, receiver, push = fixture()
    push()
    receiver.observation = replace(receiver.observation, frame_seq=2, effective_time_s=1.02, received_time_s=1.03,
        echo_feature_version=None, nearfield_echo=None, nearfield_echo_valid=None, nearfield_echo_depth=None)
    result = guide.update(now_s=1.04, dt_s=.005, contact_present=True, force_n=4., stick_speed_m_s=.04)
    assert result.reason == 'image_invalid' and result.alpha_target == 0


def test_gradual_force_swing_freezes_baseline_even_when_single_steps_are_small():
    _, guide, _, push = fixture()
    warm(push)
    for i, force in enumerate((3.8, 3.6, 3.8, 4., 4.2)):
        push(np.full(145, .6), 1.12+i*.02, force_n=force)
    before = guide.baseline.copy()
    push(np.full(145, .7), 1.22, force_n=4.4)
    np.testing.assert_array_equal(guide.baseline, before)
