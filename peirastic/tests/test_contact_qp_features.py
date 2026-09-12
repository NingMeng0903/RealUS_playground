"""Independent graph/clock/version tests; no robot or image transport required."""
from dataclasses import replace

import numpy as np
import pytest

from peirastic.contact_qp.features import (
    FeatureConfig, FeatureExtractor, LatestObservation, effective_image_time,
    random_walk_confidence, window_quality,
)
from peirastic.contact_qp.types import ContactObservation


def observation(**kwargs):
    base = dict(frame_seq=1, source_id="camera:instance1", effective_time_s=1.,
                received_time_s=1.16, quality=[.8, .1, .8], valid=[True]*3,
                registration_version="reg1", window_version="win1")
    base.update(kwargs)
    return ContactObservation(**base)


def test_grid_factorization_order_matches_default_superlu():
    from scipy.sparse.linalg import spsolve
    from peirastic.contact_qp.features import random_walk_confidence
    cfg = FeatureConfig(width=24, height=32, algorithm_version='randomwalk_welleweerd2020_v3')
    im = np.random.default_rng(11).integers(20, 220, (32, 24)).astype(float)
    a = random_walk_confidence(im, cfg)
    # The public helper now uses MMD_AT_PLUS_A; COLAMD must stay interchangeable.
    from peirastic.contact_qp import features as feat
    old = feat.spsolve
    calls = []
    def wrapped(A, b, **kwargs):
        calls.append(kwargs.get('permc_spec'))
        colamd = old(A, b, permc_spec='COLAMD')
        chosen = old(A, b, **kwargs)
        np.testing.assert_allclose(chosen, colamd, atol=1e-11)
        return chosen
    feat.spsolve = wrapped
    try:
        b = random_walk_confidence(im, cfg)
    finally:
        feat.spsolve = old
    np.testing.assert_allclose(a, b, atol=1e-11)
    assert calls and calls[-1] == 'MMD_AT_PLUS_A'


def test_uniform_graph_has_analytic_harmonic_solution():
    cfg = FeatureConfig(width=24, height=32, attenuation=0, contrast=0)
    c = random_walk_confidence(np.full((32, 24), 80.), cfg)
    np.testing.assert_allclose(c, np.broadcast_to(np.linspace(1, 0, 32)[:, None], c.shape), atol=1e-12)


@pytest.mark.parametrize('version', ['randomwalk_thesis_v1', 'randomwalk_camp_bmode_v2'])
def test_shadow_is_directional_and_mirrors_without_training(version):
    cfg = FeatureConfig(width=48, height=64, algorithm_version=version)
    rng = np.random.default_rng(44)
    im = np.clip(90+20*rng.standard_normal((64, 48)), 0, 255)
    im[2:, :16] = 0; im[:2, :16] = 200
    a = random_walk_confidence(im, cfg)
    b = random_walk_confidence(im[:, ::-1], cfg)
    np.testing.assert_allclose(a, b[:, ::-1], atol=2e-9)
    q = window_quality(a, cfg)
    assert q[0] < q[2]
    if version == 'randomwalk_thesis_v1':
        assert q[0] < .05 and q[2] > .3
    assert np.all(a[0] == 1) and np.all(a[-1] == 0)


def _matlab_dense_oracle(image, cfg):
    """Independent all-pairs graph, MATLAB zero-diagonal normalization included."""
    h, w = image.shape
    im = (image-image.min())/(np.ptp(image)+np.finfo(float).eps)
    im *= (1-np.exp(-cfg.attenuation*np.linspace(0, 1, h)))[:, None]
    edges = []
    # Enumerate every directed neighbour, including zero diagonal placeholders.
    for a in range(h*w):
        ya, xa = divmod(a, w)
        for b in range(h*w):
            yb, xb = divmod(b, w)
            if max(abs(ya-yb), abs(xa-xb)) <= 1:
                edges.append((a, b, abs(im[ya, xa]-im[yb, xb]), xa != xb))
    values = np.array([e[2] for e in edges])
    values = (values-values.min())/(np.ptp(values)+np.finfo(float).eps)
    values += np.array([e[3] for e in edges])*cfg.lateral_penalty
    values = (values-values.min())/(np.ptp(values)+np.finfo(float).eps)
    lap = np.zeros((h*w, h*w))
    for (a, b, _, _), cost in zip(edges, values):
        if a != b:
            lap[a, b] = -(np.exp(-cfg.contrast*cost)+cfg.camp_weight_epsilon)
    np.fill_diagonal(lap, -lap.sum(axis=1))
    result = np.zeros(h*w)
    result[:w] = 1
    result[w:-w] = np.linalg.solve(lap[w:-w, w:-w], -lap[w:-w, :w].sum(axis=1))
    return result.reshape(h, w)


@pytest.mark.parametrize('gamma', [0., .03, .05])
def test_camp_bmode_matches_independent_dense_matlab_oracle(gamma):
    cfg = FeatureConfig(width=9, height=8, lateral_penalty=gamma)
    im = np.random.default_rng(29).integers(30, 200, (8, 9)).astype(float)
    np.testing.assert_allclose(random_walk_confidence(im, cfg),
                               _matlab_dense_oracle(im, cfg), atol=2e-11)


def test_versions_are_explicit_and_old_black_region_counterexample_replays():
    cfg = FeatureConfig()
    old = replace(cfg, algorithm_version='randomwalk_thesis_v1')
    assert cfg.window_version != old.window_version
    assert old.window_version == 'fecfe5c107ef36f7ce31'
    with pytest.raises(ValueError, match='algorithm_version'):
        replace(cfg, algorithm_version='unknown')
    im = np.zeros((cfg.height, cfg.width))
    im[:, :cfg.width//2] = 80
    np.testing.assert_allclose(window_quality(random_walk_confidence(im, old), old),
                               [.849080663795196, .8632548346501497, .8783688666743188], atol=1e-9)
    # A black column can still have high propagation confidence. No dark mask
    # is silently multiplied into either algorithm's probability map.
    assert window_quality(random_walk_confidence(im, cfg), cfg)[2] > .5


def test_invalid_frames_not_confused_with_reliable_low_quality():
    cfg = FeatureConfig(width=24, height=32)
    obs, _ = FeatureExtractor(cfg).extract(np.zeros((32, 24)), frame_seq=1, source_id="test",
                                          capture_time_s=2., received_time_s=2.01)
    assert not obs.valid.any()
    low = observation(quality=[0., 0., 0.])
    assert low.fresh(1.2, .3)


def test_delay_is_applied_once_for_raw_and_aligned_inputs():
    assert effective_image_time(2., .152) == pytest.approx(1.848)
    assert effective_image_time(1.848, .152, already_aligned=True) == pytest.approx(1.848)
    with pytest.raises(ValueError):
        observation(effective_time_s=3.)


def test_latest_duplicate_late_and_version_change():
    store = LatestObservation()
    assert store.accept(observation())
    assert not store.accept(observation())
    assert not store.accept(observation(frame_seq=2, effective_time_s=.9))
    assert store.accept(observation(frame_seq=2, effective_time_s=1.01, window_version="win2"))
    assert store.generation == 1
    assert store.accept(observation(frame_seq=0, source_id="camera:instance2", effective_time_s=1.02))
    assert store.generation == 2


def test_registration_tracks_crop_flip_delay_and_window_configuration():
    cfg = FeatureConfig(width=24, height=32)
    extractor = FeatureExtractor(cfg)
    im = np.arange(32*24).reshape(32, 24) % 255
    kw = dict(frame_seq=1, source_id="test", capture_time_s=2., received_time_s=2.01)
    a, _ = extractor.extract(im, **kw, crop_box=[0, 24, 0, 32])
    b, _ = extractor.extract(im, **kw, crop_box=[1, 25, 0, 32])
    c, _ = extractor.extract(im, **kw, hflip=True)
    assert len({a.registration_version, b.registration_version, c.registration_version}) == 3
    assert replace(cfg, effective_delay_s=.16).window_version != cfg.window_version
    assert replace(cfg, near_depth=(.05, .25)).window_version != cfg.window_version


def test_changed_delay_invalidates_older_evidence_before_effective_time_ordering():
    store = LatestObservation()
    store.accept(observation())
    assert store.accept(observation(frame_seq=2, effective_time_s=.99, registration_version="reg2"))
    assert store.generation == 1 and store.observation.registration_version == "reg2"
    assert not store.accept(observation())
    assert store.accept(observation(frame_seq=0, source_id="new-camera", effective_time_s=.9))
    assert not store.accept(observation(frame_seq=100, effective_time_s=1.1))
    assert not observation().fresh(1.05, .3)


def test_worker_rejects_missing_provenance_before_image_decode():
    import json
    from peirastic.apps.contact_qp_features import process_parts
    for meta in ({}, {"source_id": "camera"}, {"source_id": ":", "publisher_instance_id": ""}):
        with pytest.raises(ValueError, match="source_id"):
            process_parts([b"topic", json.dumps(meta).encode(), b"badjpeg"], None, 2.)


@pytest.mark.parametrize('meta',[[],None,{'source_id':'realus.us_framegrab','publisher_instance_id':'A',
    'frame_index':True},{'source_id':'realus.us_framegrab','publisher_instance_id':'A','frame_index':1.5},
    {'source_id':'realus.us_framegrab','publisher_instance_id':'A','frame_index':1,'schema_version':1,
     'encoding':'jpeg','timestamp_source':'host_frame_read_complete','clock_domain':'realus_shared','header':None},
    {'source_id':'realus.us_framegrab','publisher_instance_id':'A','frame_index':1,'schema_version':1,
     'encoding':'jpeg','timestamp_source':'host_frame_read_complete','clock_domain':'realus_shared','header':{'stamp':[]}}])
def test_worker_bad_json_shapes_are_recoverable_value_errors(meta):
    import json
    from peirastic.apps.contact_qp_features import process_parts
    with pytest.raises(ValueError):process_parts([b'topic',json.dumps(meta).encode(),b'jpeg'],None,2.)


def test_real_shared_envelope_preserves_raw_capture_clock_and_applies_delay_once(monkeypatch):
    import json
    import sys
    from types import SimpleNamespace
    from peirastic.apps.contact_qp_features import process_parts
    monkeypatch.setitem(sys.modules,'cv2',SimpleNamespace(IMREAD_GRAYSCALE=0,
                        imdecode=lambda *a:np.zeros((8,8))))
    captured={}
    class Extractor:
        def extract(self,image,**kwargs):
            captured.update(kwargs)
            return kwargs['capture_time_s']-.152,None
    metadata=dict(source_id='realus.us_framegrab',publisher_instance_id='publisher-A',frame_index=12,
        schema_version=1,encoding='jpeg',timestamp_source='host_frame_read_complete',
        clock_domain='realus_shared',clock_id='shared-clock-A',capture_monotonic_ns=2_000_000_000,
        source_time_ns=1_788_954_131_312_189_669,time_offset_ns=-152_000_000,
        timestamp_ns=1_788_954_131_312_189_669,header={'stamp':{'sec':1_788_954_131,'nanosec':312_189_669}})
    assert process_parts([b'topic',json.dumps(metadata).encode(),b'jpeg'],Extractor(),2.01)==1.848
    assert captured['capture_time_s']==2.
    assert captured['source_id']=='realus.us_framegrab:publisher-A'
    for change in ({'capture_monotonic_ns':True},{'capture_monotonic_ns':0},
                   {'timestamp_source':'unknown'},{'clock_domain':'unknown'},
                   {'capture_clock_domain':'realus_shared'},{'clock_id':None}):
        with pytest.raises(ValueError,match='monotonic capture'):
            process_parts([b'topic',json.dumps(metadata|change).encode(),b'jpeg'],Extractor(),2.01)


def test_observation_copies_arrays_and_roundtrips_versions():
    quality = np.array([.8, .2, .7])
    obs = observation(quality=quality)
    quality[:] = 0
    np.testing.assert_allclose(obs.quality, [.8, .2, .7])
    with pytest.raises(ValueError):
        obs.quality[0] = 0
    restored = ContactObservation.from_dict(obs.to_dict())
    assert restored.version == obs.version
    np.testing.assert_array_equal(restored.valid, obs.valid)
    assert not obs.fresh(1.5, .3)


def test_welleweerd_profile_preserves_solver_and_legacy_hashes():
    from peirastic.contact_qp.features import welleweerd_config, revision
    from dataclasses import asdict
    cfg = FeatureConfig()
    old_values = asdict(cfg)
    old_values.pop('low_confidence_threshold')
    old_values.pop('unknown_scanline_range')
    assert cfg.window_version == revision(old_values)
    paper = welleweerd_config(width=16, height=20)
    image = np.random.default_rng(31).integers(0, 255, (20, 16))
    np.testing.assert_array_equal(random_walk_confidence(image, paper),
        random_walk_confidence(image, replace(paper, algorithm_version='randomwalk_camp_bmode_v2')))
    assert replace(paper, low_confidence_threshold=.7).window_version != paper.window_version


def test_paper_statistics_and_explicit_regions():
    from peirastic.contact_qp.features import welleweerd_config, confidence_features
    cfg = welleweerd_config(width=8, height=8, near_depth=(0., .5))
    c = np.broadcast_to(np.array([.2]*4 + [1.]*4), (8, 8)).copy()
    image = np.broadcast_to(np.arange(8)[:, None], (8, 8)).copy()
    out = confidence_features(image, c, cfg)
    assert out['roi_mean'] == pytest.approx(.6)
    assert out['paper_eq3_fullarea_mean'] == pytest.approx(.3)
    assert out['barycenter_row_col_1based'][0] == pytest.approx(2.5)
    assert out['barycenter_row_col_1based'][1] == pytest.approx(35/6)
    assert out['regions'] == [dict(label='low_confidence', x_start=0, x_stop=4)]
    np.testing.assert_array_equal(c[:, :4], .2)


def test_paper_black_half_is_unknown_without_modifying_raw_confidence():
    from peirastic.contact_qp.features import welleweerd_config, confidence_features
    cfg = welleweerd_config(width=24, height=32)
    image = np.broadcast_to(np.arange(32)[:, None]+30., (32, 24)).copy()
    image[:, 12:] = 0
    obs, raw = FeatureExtractor(cfg).extract(image, frame_seq=1, source_id='test',
                                            capture_time_s=2., received_time_s=2.01)
    assert not obs.valid[2] and obs.valid[0]
    assert raw[0, -1] == 1  # validity never overwrites the Dirichlet map
    out = confidence_features(image, raw, cfg)
    assert all(out['unknown_columns'][12:])
    assert not any(out['low_confidence_columns'][12:])
    assert out['frame_status'] == 'partly_unknown'
    for value in (0., 80.):
        flat = np.full((32, 24), value)
        obs, raw = FeatureExtractor(cfg).extract(flat, frame_seq=2, source_id='test',
                                                capture_time_s=2., received_time_s=2.01)
        assert not obs.valid.any()
        assert confidence_features(flat, raw, cfg)['frame_status'] == 'unknown'


def test_load_shared_paper_config(tmp_path):
    import json
    from dataclasses import asdict
    from peirastic.contact_qp.features import load_feature_config, welleweerd_config
    cfg = welleweerd_config()
    path = tmp_path/'controller.json'
    path.write_text(json.dumps({'feature': {'config': asdict(cfg)}}))
    assert load_feature_config(path).window_version == cfg.window_version


def test_fragmented_confidence_feature_payload_is_bounded_and_readable():
    import json
    from peirastic.apps.contact_qp_features import encode_feature_payload
    from peirastic.contact_qp.features import welleweerd_config,confidence_features
    cfg=welleweerd_config()
    image=np.tile(np.linspace(0,255,cfg.height)[:,None],(1,cfg.width))
    confidence=np.tile(np.where(np.arange(cfg.width)%2,.1,.9),(cfg.height,1))
    full=confidence_features(image,confidence,cfg)
    assert len(full['regions'])>=70
    original=observation()
    blob=encode_feature_payload(original,full,.012)
    assert len(blob)<=8192
    decoded=json.loads(blob)
    restored=ContactObservation.from_dict(decoded)
    assert restored.to_dict()==original.to_dict()
    summary=decoded['confidence_features']
    assert 'regions' not in summary and 'column_confidence' not in summary
    assert summary['low_confidence_column_count']==int(np.count_nonzero(full['low_confidence_columns']))
    assert summary['threshold']==cfg.low_confidence_threshold
    assert len(full['column_confidence'])==cfg.width  # preview data retained
