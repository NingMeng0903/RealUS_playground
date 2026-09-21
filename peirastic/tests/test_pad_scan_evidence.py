"""Handheld evidence uses raw 145-column echo and strict 20-region metadata."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from peirastic.contact_qp.features import FeatureConfig, FeatureExtractor
from peirastic.contact_qp.types import ContactObservation, ECHO_FEATURE_VERSION, REGION_FEATURE_VERSION
from peirastic.apps.contact_qp_features import encode_feature_payload
from peirastic.realman8dof.force.torque_tilt import TorqueTilt, TorqueTiltConfig
from peirastic.realman8dof.modes.pad_contact_evidence import PadContactEvidence
from peirastic.realman8dof.modes.pad_scan_config import load_pad_scan_config, default_pad_scan_config_path
from peirastic.realman8dof.modes.pad_visual import PadVisualGuide, pad_hybrid_payload


def setup():
    config = load_pad_scan_config()
    config['pad']['evidence']['baseline_hold_s'] = .1
    receiver = SimpleNamespace(observation=None)
    fc = FeatureConfig(**config['feature']['config'])
    def frame(t=1., echo=None):
        receiver.observation = ContactObservation(
            frame_seq=round(t*1000), source_id='camera:test', effective_time_s=t, received_time_s=t+.01,
            quality=np.full(3, .995), valid=np.ones(3, dtype=bool),
            registration_version=config['feature']['registration_version'], window_version=fc.window_version,
            calibration_version=fc.calibration_version, timestamp_semantics='effective_image_time',
            region_confidence=np.full(20, .995), region_valid=np.ones(20, dtype=bool),
            region_edges=fc.region_edges, region_layout_version=fc.region_layout_version,
            region_feature_version=REGION_FEATURE_VERSION, echo_feature_version=ECHO_FEATURE_VERSION,
            nearfield_echo=np.full(145, .5) if echo is None else echo,
            nearfield_echo_valid=np.ones(145, dtype=bool), nearfield_echo_depth=[.04, .22])
        return receiver.observation
    return config, receiver, frame


def update(evidence, t):
    return evidence.update(now_s=t+.02, dt_s=.005, contact_present=True, force_n=4., stick_speed_m_s=.04)


def test_config_is_independent_resolved_and_validated():
    config = load_pad_scan_config()
    assert default_pad_scan_config_path().name == 'pad_scan.yaml'
    assert config['feature']['config']['region_count'] == config['image_kf']['channels'] == 20
    assert config['feature_endpoint'].endswith(':17362')
    payload = pad_hybrid_payload('ultrapoc', contact_config=config)
    assert payload['pad_visual']['pad']['policy'] == 'pad_scan_v1'
    config['image_kf']['channels'] = 10
    assert payload['pad_visual']['image_kf']['channels'] == 20
    with pytest.raises(ValueError, match='channels'):
        load_pad_scan_config(config)


@pytest.mark.parametrize('mutation,reason', [
    ({'region_confidence': np.full(10, .99), 'region_valid': np.ones(10, dtype=bool),
      'region_edges': np.linspace(0, 1, 11)}, 'region_count_mismatch'),
    ({'region_layout_version': 'wrong_layout'}, 'region_layout_mismatch'),
    ({'region_edges': np.r_[0., .04, np.linspace(.1, 1., 19)]}, 'region_edges_mismatch'),
    ({'timestamp_semantics': 'capture_time'}, 'timestamp_semantics_mismatch'),
])
def test_reject_before_kf_and_raw_fallback(mutation, reason):
    config, receiver, frame = setup()
    evidence = PadContactEvidence(config, receiver)
    # Exercise runtime validation as well as the stricter wire dataclass:
    # a custom/in-process receiver can supply metadata without its constructor.
    receiver.observation = frame()
    for key, value in mutation.items():
        object.__setattr__(receiver.observation, key, value)
    ingested = []
    evidence.kf.ingest = lambda *args: ingested.append(args)
    result = update(evidence, 1.)
    assert not ingested and not result.valid
    assert result.alpha_target == result.omega_request_rad_s == 0.
    assert result.facts['image_compatibility_reason'] == reason


def test_layout_change_resets_baseline_and_kf():
    config, receiver, frame = setup()
    evidence = PadContactEvidence(config, receiver)
    for t in (1., 1.05, 1.1): frame(t); update(evidence, t)
    assert evidence.baseline is not None
    receiver.observation = replace(frame(1.2), region_layout_version='other')
    result = update(evidence, 1.2)
    assert evidence.baseline is None and result.alpha_target == 0.
    assert result.facts['quality_predicted'] is None


def test_guide_current_speed_pause_and_raw_reversal_confirmation():
    config, receiver, frame = setup()
    guide = PadVisualGuide(config, TorqueTiltConfig(), receiver=receiver, progress=True)
    def step(t, echo=None, speed=.04):
        frame(t, echo)
        guide._preview_scan_evidence(now_s=t+.02, dt_s=.05, contact_present=True,
                                    force_n=4., stick_speed_m_s=speed)
    for t in (1., 1.05, 1.1, 1.15, 1.2, 1.25): step(t)
    echo = np.full(145, .5); echo[:2] = .01
    step(1.3, echo)
    assert guide.last_alpha == guide.omega_target == 0.
    step(1.35, echo)
    assert guide.evidence_valid and -0.10 <= guide.omega_target < 0.
    step(1.4, echo, speed=.08)
    assert guide.last_alpha <= config['pad']['repair_speed_cap_m_s'] / .08 + 1e-12
    for t in (1.5, 1.6, 1.7, 1.81): step(t, echo)
    assert guide.last_alpha == 0.
    step(1.85, echo[::-1])
    assert guide.omega_target == guide.last_visual_omega == 0.
    step(1.9, echo[::-1])
    assert guide.omega_target > 0.
    guide._preview_scan_evidence(now_s=2.3, dt_s=.005, contact_present=True,
                                force_n=4., stick_speed_m_s=.04)
    assert guide.last_alpha == guide.last_visual_omega == 0.


def test_high_q25_left_wedge_still_requests_echo_omega_without_parking_alpha():
    config, receiver, frame = setup()
    guide = PadVisualGuide(config, TorqueTiltConfig(), receiver=receiver, progress=True)
    tilt = TorqueTilt(TorqueTiltConfig())
    echo = np.full(145, .5)
    echo[:16] = .01
    pose = np.zeros(6)
    for t in (1.0, 1.05):
        obs = frame(t, echo)
        object.__setattr__(obs, 'region_confidence', np.full(20, .99))
        guide.preview(pose=pose, dt_s=.005, contact_present=True, tilt=tilt,
                      now_s=t + .02, force_n=4., stick_speed_m_s=0.)
    assert guide.last_facts.get('echo_reason') == 'unilateral_missing_echo'
    assert abs(guide.omega_target) > 0.
    assert abs(guide.omega_target) <= config['pad']['static_visual_omega_rad_s'] + 1e-12
    assert guide.last_alpha > 0.9
    assert guide._hold_until_s is None


def test_moving_echo_keeps_full_omega_and_parked_echo_does_not_hold():
    config, receiver, frame = setup()
    guide = PadVisualGuide(config, TorqueTiltConfig(), receiver=receiver, progress=True)
    tilt = TorqueTilt(TorqueTiltConfig())
    echo = np.full(145, .5)
    echo[:16] = .01
    pose = np.zeros(6)
    for t in (1.0, 1.05):
        obs = frame(t, echo)
        object.__setattr__(obs, 'region_confidence', np.full(20, .99))
        guide.preview(pose=pose, dt_s=.005, contact_present=True, tilt=tilt,
                      now_s=t + .02, force_n=4., stick_speed_m_s=.05)
    assert abs(guide.omega_target) > config['pad']['static_visual_omega_rad_s']
    assert guide._hold_until_s is not None
    guide.preview(pose=pose, dt_s=.005, contact_present=True, tilt=tilt,
                  now_s=1.12, force_n=4., stick_speed_m_s=0.)
    assert abs(guide.omega_target) <= config['pad']['static_visual_omega_rad_s'] + 1e-12
    assert guide._hold_until_s is None


def test_parked_stable_wedge_raises_omega_after_commit_but_flips_stay_capped():
    config, receiver, frame = setup()
    guide = PadVisualGuide(config, TorqueTiltConfig(), receiver=receiver, progress=True)
    tilt = TorqueTilt(TorqueTiltConfig())
    left = np.full(145, .5)
    left[:16] = .01
    right = left[::-1]
    pose = np.zeros(6)
    def tick(t, echo, speed=0.):
        obs = frame(t, echo)
        object.__setattr__(obs, 'region_confidence', np.full(20, .99))
        guide.preview(pose=pose, dt_s=.005, contact_present=True, tilt=tilt,
                      now_s=t + .02, force_n=4., stick_speed_m_s=speed)
    for t in np.arange(1.0, 1.40, .05):
        tick(t, left)
    assert abs(guide.omega_target) <= config['pad']['static_visual_omega_rad_s'] + 1e-12
    for t in np.arange(1.40, 1.95, .05):
        tick(t, left)
    assert abs(guide.omega_target) > config['pad']['static_visual_omega_rad_s']
    assert abs(guide.omega_target) <= config['pad']['static_committed_omega_rad_s'] + 1e-12
    assert guide._hold_until_s is None
    flipper = PadVisualGuide(config, TorqueTiltConfig(), receiver=receiver, progress=True)
    def flip_tick(t, echo):
        obs = frame(t, echo)
        object.__setattr__(obs, 'region_confidence', np.full(20, .99))
        flipper.preview(pose=pose, dt_s=.005, contact_present=True, tilt=tilt,
                        now_s=t + .02, force_n=4., stick_speed_m_s=0.)
    for i, t in enumerate(np.arange(2.0, 3.6, .05)):
        flip_tick(t, left if (i // 6) % 2 == 0 else right)
    assert abs(flipper.omega_target) <= config['pad']['static_visual_omega_rad_s'] + 1e-12
    assert flipper._hold_until_s is None


def test_publisher_twenty_regions_keeps_full_echo_under_envelope():
    config = load_pad_scan_config()
    extractor = FeatureExtractor(FeatureConfig(**config['feature']['config']))
    obs, _ = extractor.extract(np.random.default_rng(42).integers(0, 256, (100, 145)),
        frame_seq=1, source_id='camera:test', capture_time_s=10., received_time_s=10.01)
    assert len(obs.region_confidence) == 20 and len(obs.nearfield_echo) == 145
    assert len(encode_feature_payload(obs, extractor.last_features, .01)) <= 8192


def test_live_stick_jump_cannot_exceed_repair_cap_during_five_ms_ramp():
    config, receiver, frame = setup()
    guide = PadVisualGuide(config, TorqueTiltConfig(), receiver=receiver, progress=True)
    echo = np.full(145, .5)
    echo[:8] = .01
    def tick(t, speed):
        frame(t, echo)
        guide._preview_scan_evidence(now_s=t+.02, dt_s=.005, contact_present=True,
                                    force_n=4., stick_speed_m_s=speed)
    tick(1., .04)
    tick(1.05, .04)
    for i in range(80):
        tick(1.055+i*.005, .04)
    assert guide.last_alpha == pytest.approx(.5, abs=.03)
    previous = guide.last_alpha
    tick(1.46, .16)
    cap = config['pad']['repair_speed_cap_m_s']
    step = .005 / config['pad']['alpha_ramp_down_s']
    assert previous - guide.last_alpha <= step + 1e-12
    assert guide.last_alpha <= previous
    for i in range(80):
        tick(1.465+i*.005, .16)
    assert guide.last_alpha*.16 <= cap + 1e-12
    assert guide.last_facts['repair_speed_cap_limited']
    previous = guide.last_alpha
    tick(1.87, .04)
    assert guide.last_alpha-previous <= .005/config['pad']['alpha_ramp_up_s']+1e-12


def test_raw_healthy_recovery_does_not_require_force_stable_baseline():
    config, receiver, frame = setup()
    config['pad']['evidence']['baseline_hold_s'] = 1.
    evidence = PadContactEvidence(config, receiver)
    # Each force step is valid, but the rolling range repeatedly exceeds the
    # 1.4 N window, so this stream cannot supply the 1 s baseline learning hold.
    forces = (3., 3.5, 4., 4.5, 4., 3.5)
    for i in range(31):
        t = 1.+i*.05
        frame(t)
        result = evidence.update(now_s=t+.02, dt_s=.05, contact_present=True,
                                 force_n=forces[i % len(forces)], stick_speed_m_s=.04)
        assert evidence.baseline is None
        assert result.alpha_target == (0. if i < 5 else 1.)
        assert result.reason == ('recovery_confirming' if i < 5 else 'healthy')
    assert not result.facts['recovery_required']


def test_initial_healthy_confirmation_requires_fresh_frames():
    config, receiver, frame = setup()
    evidence = PadContactEvidence(config, receiver)
    frame(1.)
    first = update(evidence, 1.)
    assert first.reason == 'recovery_confirming' and first.alpha_target == 0.
    for t in (1.1, 1.2, 1.26):
        result = update(evidence, t)
        assert result.alpha_target == 0.
        assert not result.facts['observation_fresh']
    frame(1.27)
    assert update(evidence, 1.27).alpha_target == 1.


def test_confirmed_restore_starts_a_new_half_second_repair_episode():
    config, receiver, frame = setup()
    config['pad']['evidence']['baseline_hold_s'] = 1.
    evidence = PadContactEvidence(config, receiver)
    echo = np.full(145, .5)
    echo[:2] = .01
    for t in (1., 1.05, 1.2, 1.4, 1.51):
        frame(t, echo)
        result = update(evidence, t)
    assert result.alpha_target == 0.
    for t in (1.55, 1.65, 1.75):
        frame(t)
        result = update(evidence, t)
        assert result.alpha_target == 0.
        assert evidence._repair_since is not None
    frame(1.8)
    result = update(evidence, 1.8)
    assert result.alpha_target == 1. and evidence._repair_since is None
    assert evidence.baseline is None
    frame(1.85, echo)
    assert update(evidence, 1.85).alpha_target == 0.
    for t in (1.9, 2.1, 2.3):
        frame(t, echo)
        result = update(evidence, t)
        assert result.alpha_target == pytest.approx(.5)
    frame(2.36, echo)
    assert update(evidence, 2.36).alpha_target == 0.


@pytest.mark.parametrize('interruption', ['stale', 'invalid', 'blank'])
def test_invalid_or_blank_frames_cannot_complete_raw_recovery(interruption):
    config, receiver, frame = setup()
    evidence = PadContactEvidence(config, receiver)
    frame(1.)
    update(evidence, 1.)
    if interruption == 'stale':
        result = update(evidence, 1.31)
    elif interruption == 'invalid':
        receiver.observation = replace(frame(1.31), nearfield_echo_valid=np.zeros(145, dtype=bool))
        result = update(evidence, 1.31)
    else:
        frame(1.31, np.zeros(145))
        result = update(evidence, 1.31)
    assert result.alpha_target == 0.
    assert result.reason == {'stale': 'image_stale', 'invalid': 'image_invalid',
                             'blank': 'uniform_low_echo'}[interruption]
    for t in (1.4, 1.5, 1.6):
        frame(t)
        result = update(evidence, t)
        assert result.reason == 'recovery_confirming' and result.alpha_target == 0.
    frame(1.65)
    assert update(evidence, 1.65).alpha_target == 1.


def test_loader_keeps_visual_hold_and_pad_force_band():
    config = load_pad_scan_config()
    assert config['qp']['pad_visual_hold_s'] == pytest.approx(.25)
    assert config['pad']['alpha_min'] == pytest.approx(.12)
    assert config['qp']['alpha_min'] == config['qp']['pad_alpha_min'] == pytest.approx(.12)
    assert config['pad']['max_visual_omega_rad_s'] == pytest.approx(.16)
    assert config['pad']['static_stick_m_s'] == pytest.approx(.01)
    assert config['pad']['static_visual_omega_rad_s'] == pytest.approx(.04)
    assert config['pad']['static_commit_s'] == pytest.approx(.80)
    assert config['pad']['static_committed_omega_rad_s'] == pytest.approx(.12)
    assert config['pad']['relative_spread_floor'] == pytest.approx(.12)
    assert config['pad']['evidence']['echo_moment_cubic_knee'] == pytest.approx(.25)
    assert config['pad']['evidence']['echo_moment_compress'] == pytest.approx(0.)
    assert config['pad']['force_gate_end_n'] == pytest.approx(5.5)
    assert config['qp']['differential_repair']['soft_force_n'] == 4.5
    from peirastic.realman8dof.modes.pad_scan_config import pad_visual_force_gate
    assert pad_visual_force_gate(4., 4., 5.5) == pytest.approx(1.)
    assert pad_visual_force_gate(5.5, 4., 5.5) == pytest.approx(0.)
    assert pad_visual_force_gate(4.75, 4., 5.5) == pytest.approx(.5)


def test_two_column_wedge_is_cubic_and_wide_edge_keeps_linear_omega():
    config, receiver, frame = setup()
    linear = dict(config)
    linear['pad'] = dict(config['pad'])
    linear['pad']['evidence'] = dict(config['pad']['evidence'], echo_moment_cubic_knee=0.)
    narrow = np.full(145, .5)
    narrow[:2] = .01
    wide = np.full(145, .5)
    wide[:40] = .01
    shaped = PadContactEvidence(config, receiver)
    raw = PadContactEvidence(linear, receiver)
    for t in (1., 1.05, 1.1):
        frame(t, narrow)
        update(shaped, t)
        update(raw, t)
    frame(1.15, narrow)
    small = update(shaped, 1.15)
    plain = update(raw, 1.15)
    assert small.reason == plain.reason == 'unilateral_missing_echo'
    assert abs(small.omega_request_rad_s) < abs(plain.omega_request_rad_s)
    for t in (1.2, 1.25):
        frame(t, wide)
        update(shaped, t)
        update(raw, t)
    frame(1.3, wide)
    deep = update(shaped, 1.3)
    deep_plain = update(raw, 1.3)
    assert abs(deep.omega_request_rad_s) == pytest.approx(abs(deep_plain.omega_request_rad_s), rel=.05)
    assert abs(deep.omega_request_rad_s) > abs(small.omega_request_rad_s)
