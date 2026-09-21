"""Offline alignment and original-image replay contracts; no transport."""
import hashlib
import json

import numpy as np
import pytest

from peirastic.apps.pad_contact_replay import (
    extract_original, nearest_control, observation_for_layout, temporal_split,
)
from peirastic.contact_qp.features import FeatureExtractor, region_quality, welleweerd_config


def test_nearest_time_is_explicitly_approximate_and_does_not_invent_publication():
    rows = [dict(record_monotonic_s=1., omega_cmd=.2), dict(record_monotonic_s=2., omega_cmd=.4)]
    row, delta = nearest_control(rows, np.array([1., 2.]), 1.8)
    assert row is rows[1]
    assert delta == pytest.approx(.2)
    assert 'published_omega_tool_y_rad_s' not in row
    assert nearest_control([], np.array([]), 1.) == (None, None)


def test_temporal_split_is_contiguous_disjoint_and_covers_original_frames():
    split = temporal_split(np.linspace(10., 20., 100))
    assert split['calibration']['frame_start'] == 0
    assert split['calibration']['frame_stop_exclusive'] == split['validation']['frame_start'] == 60
    assert split['validation']['frame_stop_exclusive'] == 100
    assert split['calibration']['capture_time_range_s'][1] < split['validation']['capture_time_range_s'][0]


def test_twenty_regions_come_from_original_jpeg_not_ten_value_interpolation():
    cv2 = pytest.importorskip('cv2')
    config = welleweerd_config(width=145, height=128, region_count=20)
    extractor = FeatureExtractor(config)
    image = np.tile(np.linspace(40, 220, 290, dtype=np.uint8), (256, 1))
    image[:, 24:32] = 1
    ok, jpeg = cv2.imencode('.jpg', image)
    assert ok
    metadata = dict(capture_monotonic_ns=10_000_000_000, frame_index=7,
                    source_id='test.camera', publisher_instance_id='episode', hflip=False)
    decoded, obs, confidence = extract_original(jpeg.tobytes(), metadata, extractor, 10.02)
    assert decoded.shape == image.shape
    assert obs.frame_seq == 7
    assert obs.effective_time_s == pytest.approx(10.-config.effective_delay_s)
    assert obs.received_time_s == 10.02
    assert len(obs.region_confidence) == 20
    ten, _ = observation_for_layout(obs, confidence, extractor, 10)
    original_twenty = region_quality(confidence, config, np.asarray(extractor.last_features['unknown_columns'], dtype=bool))
    np.testing.assert_allclose(obs.region_confidence, original_twenty['region_confidence'])
    assert len(ten.region_confidence) == 10
    assert ten.region_layout_version != obs.region_layout_version
    assert len(obs.nearfield_echo) == len(ten.nearfield_echo) == 145
    # The thin original-image notch is not equivalent to interpolating 10 Qs.
    interpolated = np.interp(np.linspace(.025, .975, 20), np.linspace(.05, .95, 10), ten.region_confidence)
    assert not np.allclose(obs.region_confidence, interpolated)


def test_original_capture_timestamp_is_required():
    cv2 = pytest.importorskip('cv2')
    _, jpeg = cv2.imencode('.jpg', np.ones((20, 20), np.uint8))
    with pytest.raises(ValueError, match='original monotonic'):
        extract_original(jpeg.tobytes(), {}, FeatureExtractor(), 1.)


def test_small_recording_replay_preserves_source_and_marks_unknown_publication(tmp_path, monkeypatch):
    cv2 = pytest.importorskip('cv2')
    h5py = pytest.importorskip('h5py')
    pytest.importorskip('matplotlib')
    from peirastic.apps.pad_contact_replay import replay_session
    from peirastic.realman8dof.modes.pad_scan_config import load_pad_scan_config
    import peirastic.realman8dof.modes.pad_contact_evidence as evidence_module
    def forbidden_receiver(*args, **kwargs):
        raise AssertionError('offline replay must not create a network receiver')
    monkeypatch.setattr(evidence_module, 'FeatureReceiver', forbidden_receiver)
    session = tmp_path/'recording'; session.mkdir()
    raw = session/'raw.h5'
    _, jpeg = cv2.imencode('.jpg', np.tile(np.linspace(40, 220, 145, dtype=np.uint8), (128, 1)))
    with h5py.File(raw, 'w') as file:
        group = file.create_group('ultrasound')
        data = group.create_dataset('jpeg', (2,), dtype=h5py.vlen_dtype(np.uint8))
        data[0] = jpeg.ravel(); data[1] = jpeg.ravel()
        group.create_dataset('metadata_json', data=[json.dumps(dict(
            capture_monotonic_ns=10_000_000_000+i*40_000_000, frame_index=i,
            source_id='realus.us_framegrab', publisher_instance_id='offline-test',
            crop_box=[753, 1417, 154, 880], hflip=False)).encode() for i in range(2)])
    controls = [dict(event='control_sample', record_monotonic_s=10.+i*.04,
                     fz=3., contact=True, omega_cmd=.1, alpha=1., frame_seq=None) for i in range(2)]
    (session/'pad.jsonl').write_text('\n'.join(json.dumps(r) for r in controls))
    before = hashlib.sha256(raw.read_bytes()).hexdigest()
    output = tmp_path/'analysis'
    metrics = replay_session(session, output, load_pad_scan_config())
    assert metrics['replayed_frame_count'] == 2
    assert metrics['closed_loop_proof'] is False
    assert hashlib.sha256(raw.read_bytes()).hexdigest() == before
    rows = [json.loads(line) for line in (output/'frames.jsonl').read_text().splitlines()]
    assert all(len(r['q_raw']) == 20 and len(r['q_raw_10']) == 10 for r in rows)
    assert all(r['publication_status'] == 'unknown' and r['published_omega_tool_y_rad_s'] is None for r in rows)
    assert (output/'timeline.png').is_file()
