"""The handheld feature entry must not silently reuse the scan stream."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from peirastic.apps.contact_qp_check import observation_problem
from peirastic.apps.contact_qp_features import encode_feature_payload
from peirastic.apps.pad_contact_features import (
    ensure_pad_feature_publisher,
    feature_identity,
    feature_publisher_command,
    main,
    publisher_arguments,
)
from peirastic.contact_qp.features import FeatureConfig, FeatureExtractor, region_quality
from peirastic.contact_qp.runtime_config import load_study_config
from peirastic.realman8dof.modes.pad_scan_config import load_pad_scan_config


def test_default_handheld_layout_is_independent_of_scan():
    pad = load_pad_scan_config()
    scan = load_study_config('peirastic/config/contact_qp/active_probe50_delay_kf_cop.yaml')
    assert feature_identity(pad)['region_count'] == 20
    assert pad['image_kf']['channels'] == pad['feature']['config']['region_count']
    assert scan['feature']['config']['region_count'] == 10
    assert pad['feature_endpoint'] == 'tcp://127.0.0.1:17362'
    assert pad['feature_endpoint'] != scan['feature_endpoint']


def test_allocation_reserve_does_not_replace_final_speed_limit():
    config = load_pad_scan_config()
    assert config['pad']['normal_max_m_s'] == .015
    assert config['pad']['normal_allocation_margin_m_s'] == .0001
    assert config['pad']['control_hold_s'] == .005
    assert 'minimum_publication_period_s' not in config['pad']
    config['pad']['normal_allocation_margin_m_s'] = config['pad']['normal_max_m_s']
    with pytest.raises(ValueError, match='inconsistent'):
        load_pad_scan_config(config)


def test_ultrapoc_publishes_stick_before_image_without_lifecycle():
    from peirastic.core.modes import Mode, ModeRequest
    from peirastic.realman8dof.modes.pad_visual import pad_hybrid_payload
    from peirastic.realman8dof.session import compile_request
    from peirastic.tests.test_api_facade import _SEED, _ctx

    raw, ctx = _ctx()
    payload = pad_hybrid_payload('ultrapoc', desired_z=4.0)
    payload['v_cmd'] = [0.02, 0.0, 0.0, 0.0, 0.0, 0.0]
    phase = compile_request(ctx, ModeRequest(Mode.TRACK_HYBRID, payload), raw=raw)
    try:
        assert getattr(phase.outer, 'command_lifecycle', None) is None
        pose = ctx.kin.fk_pose(_SEED)
        phase.outer.set_origin(pose)
        command = phase.outer.sample(
            0., pose, np.array([0., 0., 4., 0., 0.2, 0.]),
            contact=True, dt_actual=.005, sensor_age_s=.001, feedback_age_s=.001)
        assert command[0] == pytest.approx(0.02, abs=0.003)
        assert phase.outer.last_pad_alpha == pytest.approx(1.0)
    finally:
        if phase.on_exit is not None:
            phase.on_exit()


def test_resolved_handheld_request_fits_controller_mailbox():
    import json
    from peirastic.core.ipc import PAYLOAD_MAX
    from peirastic.core.modes import Mode, ModeRequest
    from peirastic.realman8dof.modes.pad_visual import pad_hybrid_payload
    payload = pad_hybrid_payload('ultrapoc', desired_z=4.,
        log_path='/tmp/pad-analysis/session/pad.jsonl', contact_config=load_pad_scan_config())
    encoded = json.dumps(ModeRequest(Mode.TRACK_HYBRID, payload).to_json(), separators=(',', ':')).encode()
    assert len(encoded) <= PAYLOAD_MAX


def test_ensure_publisher_starts_only_when_the_stream_is_missing():
    config = load_pad_scan_config()
    started = []

    def already_live(*_a, **_k):
        return {'status': 'ready'}

    child, state = ensure_pad_feature_publisher(
        config, check=already_live, spawn=lambda cmd: started.append(cmd) or object())
    assert child is None and state == 'already_live' and started == []

    def missing(*_a, **_k):
        raise RuntimeError('no confidence messages')

    child, state = ensure_pad_feature_publisher(
        config, repo='/tmp/repo', check=missing, spawn=lambda cmd: started.append(cmd) or object())
    assert state == 'started_no_frames'
    assert started[0] == feature_publisher_command(repo='/tmp/repo')


def test_publisher_explicitly_uses_resolved_config_and_endpoint():
    config = deepcopy(load_pad_scan_config())
    config['feature_endpoint'] = 'tcp://127.0.0.1:19462'
    args = publisher_arguments(config, '/tmp/custom pad.yaml', input_endpoint='inproc://images',
                               max_hz=20., max_frames=3)
    assert args[args.index('--feature-config') + 1] == '/tmp/custom pad.yaml'
    assert args[args.index('--output') + 1] == config['feature_endpoint']
    assert args[args.index('--input') + 1] == 'inproc://images'


def test_validate_only_never_opens_transport(monkeypatch, capsys):
    import peirastic.apps.contact_qp_features as worker
    monkeypatch.setattr(worker, 'main', lambda *a, **kw: pytest.fail('opened feature publisher'))
    assert main(['--validate-only']) == 0
    output = capsys.readouterr().out
    assert '"region_count": 20' in output
    assert '17362' in output


def test_twenty_regions_expose_a_narrow_column_defect_and_preserve_symmetry():
    confidence = np.full((100, 145), .995)
    confidence[:, :2] = .70
    ten = FeatureConfig(width=145, height=100, region_count=10)
    twenty = replace(ten, region_count=20)
    assert region_quality(confidence, ten)['region_confidence'][0] == pytest.approx(.995)
    left = region_quality(confidence, twenty)['region_confidence']
    right = region_quality(confidence[:, ::-1], twenty)['region_confidence']
    assert left[0] == pytest.approx(.70)
    np.testing.assert_allclose(left, right[::-1])
    assert len(left) == 20


@pytest.mark.parametrize('kind', ['texture', 'left_missing', 'flat'])
def test_complete_twenty_region_payload_fits_existing_transport(kind):
    config = load_pad_scan_config()
    fc = FeatureConfig(**config['feature']['config'])
    image = np.random.default_rng(21).integers(0, 256, (100, 145), dtype=np.uint8)
    if kind == 'left_missing':
        image[:, :12] = 0
    elif kind == 'flat':
        image[:] = 0
    extractor = FeatureExtractor(fc)
    reg = config['feature']['registration']
    obs, _ = extractor.extract(image, frame_seq=1, source_id=reg['source_id'],
                               capture_time_s=10., received_time_s=10.05,
                               crop_box=reg['crop_box'], hflip=reg['hflip'],
                               already_aligned=reg['already_aligned'], clock_domain=reg['clock_domain'])
    payload = encode_feature_payload(obs, extractor.last_features, .05)
    assert len(payload) <= 8192
    assert len(obs.region_confidence) == fc.region_count
    assert len(obs.nearfield_echo) == fc.width
    assert observation_problem(obs, config, now_s=10.06) is None
    wrong = replace(obs, region_layout_version='old-ten-layout')
    assert 'layout' in observation_problem(wrong, config, now_s=10.06)
    no_echo = replace(obs, echo_feature_version=None, nearfield_echo=None,
                      nearfield_echo_valid=None, nearfield_echo_depth=None)
    assert 'near-field' in observation_problem(no_echo, config, now_s=10.06)
