"""Effective-time fusion, bounded replay, expiry, and fixed-noise invariants."""
from dataclasses import replace
import json

import numpy as np
import pytest

from peirastic.contact_qp.delayed_kf import (DelayConfidenceKF, KfConfig,
                                           measurement_update, transition)
from peirastic.contact_qp.types import ContactObservation, WEAK_SIDE_FEATURE_VERSION


def observation(seq=1, stamp=1., values=(.8, .9), **changes):
    raw = dict(frame_seq=seq, source_id='camera', effective_time_s=stamp,
               received_time_s=1.2, quality=[.85, .8, .95], valid=[True]*3,
               registration_version='registered', window_version='windows',
               confidence_lr=values, confidence_lr_valid=[True, True],
               weakside_feature_version=WEAK_SIDE_FEATURE_VERSION,
               timestamp_semantics='effective_image_time')
    return ContactObservation(**(raw | changes))


def filter_():
    return DelayConfidenceKF(KfConfig(q=.004, r=.001))


def test_white_acceleration_covariance_and_joseph_update():
    f, q = transition(.2, .3)
    np.testing.assert_allclose(f, [[1, .2, 0, 0], [0, 1, 0, 0],
                                   [0, 0, 1, .2], [0, 0, 0, 1]])
    np.testing.assert_allclose(q[:2, :2], .3*np.array([[.2**3/3, .2**2/2], [.2**2/2, .2]]))
    np.testing.assert_array_equal(q[:2, 2:], 0)
    p = np.diag([2., 1., 2., 1.])
    x, post, innovation, s = measurement_update(np.zeros(4), p, [.5, .8], .1)
    np.testing.assert_allclose(x, [2/2.1*.5, 0, 2/2.1*.8, 0])
    np.testing.assert_allclose(np.diag(post), [2*.1/2.1, 1., 2*.1/2.1, 1.])
    assert np.linalg.eigvalsh(post).min() > 0


def test_every_frame_fused_once_predictions_do_not_refuse_measurement():
    kf = filter_()
    obs = observation()
    assert kf.ingest(obs, 1.2)
    before = kf.predict(1.2)
    assert before.age_s == pytest.approx(.2)  # no second subtraction of .152
    assert not kf.ingest(obs, 1.2)
    assert kf.last_reason == 'duplicate_frame'
    np.testing.assert_array_equal(kf.predict(1.2).state, before.state)
    np.testing.assert_array_equal(kf.predict(1.2).covariance, before.covariance)
    assert kf.predict(1.25).covariance[0, 0] > before.covariance[0, 0]
    assert json.loads(json.dumps(before.to_dict(), allow_nan=False))['valid']


def test_out_of_order_replay_equals_sorted_input_including_earlier_first_frame():
    frames = [observation(i, 1.+i*.04, (.5+i*.05, .9-i*.02)) for i in range(5)]
    sorted_kf, out_of_order = filter_(), filter_()
    for obs in frames:
        assert sorted_kf.ingest(obs, 1.2)
    for i in [2, 0, 4, 1, 3]:
        assert out_of_order.ingest(frames[i], 1.2)
    np.testing.assert_allclose(out_of_order.predict(1.24).state, sorted_kf.predict(1.24).state, atol=1e-14)
    np.testing.assert_allclose(out_of_order.predict(1.24).covariance, sorted_kf.predict(1.24).covariance, atol=1e-14)


def test_replay_after_pruning_preserves_anchor_posterior():
    frames = [observation(i, 1.+i*.05, (.5+i*.01, .8), received_time_s=1.+i*.05)
              for i in range(14)]
    a, b = filter_(), filter_()
    for obs in frames[:8]:
        a.ingest(obs, obs.received_time_s)
        b.ingest(obs, obs.received_time_s)
    for obs in frames[8:]:
        a.ingest(obs, 1.65)
    for i in [10, 9, 8, 13, 11, 12]:
        b.ingest(frames[i], 1.65)
    np.testing.assert_allclose(a.predict(1.66).state, b.predict(1.66).state, atol=1e-14)
    np.testing.assert_allclose(a.predict(1.66).covariance, b.predict(1.66).covariance, atol=1e-14)


def test_unknown_newest_frame_disables_visual_until_new_valid_frame():
    kf = filter_()
    kf.ingest(observation(), 1.2)
    assert not kf.ingest(observation(2, 1.1, confidence_lr_valid=[True, False]), 1.2)
    out = kf.predict(1.2)
    assert not out.valid and out.reason == 'invalid_windows'
    assert out.quality_task is None
    np.testing.assert_allclose(out.quality_raw, [.8, .9])
    # Old valid arrival must not turn an invalid newer frame back into evidence.
    assert kf.ingest(observation(0, 1.05), 1.2)
    assert not kf.predict(1.2).valid
    assert kf.ingest(observation(3, 1.15), 1.2)
    assert kf.predict(1.2).valid


def test_unsupported_version_pauses_without_retiring_healthy_identity():
    kf = filter_()
    assert kf.ingest(observation(), 1.2)
    assert not kf.ingest(observation(2, 1.05, weakside_feature_version='other'), 1.2)
    paused = kf.predict(1.2)
    assert not paused.valid
    assert paused.reason == 'feature_version_or_timestamp_semantics'
    np.testing.assert_allclose(paused.quality_raw, [.8, .9])
    assert kf.ingest(observation(3, 1.1), 1.2)
    assert kf.predict(1.2).valid
    assert kf.generation == 0


def test_expiry_future_old_and_backwards_timestamps():
    kf = filter_()
    assert not kf.ingest(observation(received_time_s=1.3), 1.2)
    assert kf.last_reason == 'future_timestamp'
    assert not kf.ingest(observation(stamp=.8), 1.2)
    assert kf.last_reason == 'too_old'
    assert kf.ingest(observation(), 1.2)
    assert not kf.predict(1.31).valid
    assert kf.predict(1.32).quality_task is None
    assert not kf.ingest(observation(2, 1.1), 1.2)
    assert kf.last_reason == 'bad_or_backwards_now'
    assert not kf.predict(float('nan')).valid


def test_task_clips_prediction_but_keeps_raw_state_and_uncertainty():
    kf = filter_()
    kf.ingest(observation(0, 1., (.8, .2)), 1.2)
    kf.ingest(observation(1, 1.1, (1., 0.)), 1.2)
    out = kf.predict(1.29)
    assert out.valid
    assert out.quality_raw[0] > 1 and out.quality_raw[1] < 0
    np.testing.assert_array_equal(out.quality_task, [1., 0.])
    assert np.isfinite(out.covariance).all()


@pytest.mark.parametrize('change', [dict(source_id='new-camera'),
                                    dict(registration_version='new-registration'),
                                    dict(window_version='new-windows')])
def test_identity_change_resets_rate_and_history(change):
    kf = filter_()
    kf.ingest(observation(0, 1., (.5, .8)), 1.2)
    kf.ingest(observation(1, 1.1, (.8, .5)), 1.2)
    assert kf.ingest(observation(0, 1.12, (.7, .7), **change), 1.2)
    assert kf.generation == 1 and kf.last_reason == 'reset_updated'
    np.testing.assert_array_equal(kf.predict(1.2).state[[1, 3]], 0)


def test_legacy_or_wrong_version_cannot_drive_new_filter():
    kf = filter_()
    assert not kf.ingest(observation(weakside_feature_version='other'), 1.2)
    assert kf.last_reason == 'feature_version_or_timestamp_semantics'
    old = observation().to_dict()
    for key in ('confidence_lr', 'confidence_lr_valid', 'weakside_feature_version', 'timestamp_semantics'):
        old.pop(key)
    assert not kf.ingest(ContactObservation.from_dict(old), 1.2)
    assert kf.predict(1.2).quality_task is None


def test_config_requires_offline_noise_and_valid_history():
    with pytest.raises(TypeError):
        KfConfig()
    for change in (dict(q=-1.), dict(r=0.), dict(r=float('nan')), dict(history_s=.2)):
        with pytest.raises(ValueError):
            KfConfig(**(dict(q=.1, r=.1) | change))


def test_retired_registration_cannot_reset_new_history_backwards():
    kf = filter_()
    assert kf.ingest(observation(0, 1.), 1.2)
    assert kf.ingest(observation(1, 1.1, registration_version='new-registration'), 1.2)
    assert not kf.ingest(observation(2, 1.15), 1.2)
    assert kf.last_reason == 'retired_identity'
    assert kf.generation == 1
    assert kf.predict(1.2).effective_time_s == 1.1


def test_expired_diagnostics_survive_pruning_and_recover_on_valid_frame():
    kf = filter_()
    assert kf.ingest(observation(), 1.2)
    assert not kf.ingest(observation(), 1.4)
    out = kf.predict(1.4)
    assert not out.valid and out.reason == 'image_expired'
    assert out.quality_raw is not None and out.covariance is not None
    assert kf.ingest(observation(3, 1.3, received_time_s=1.4), 1.4)
    assert kf.predict(1.4).valid
