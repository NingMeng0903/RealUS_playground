"""Pad visual ωy injection: image-ready opens the visual branch, dropout returns to moment."""
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from peirastic.contact_qp.confidence_fusion import FEATURE_VERSION
from peirastic.contact_qp.runtime_config import load_study_config
from peirastic.realman8dof.force.torque_tilt import TorqueTilt, TorqueTiltConfig
from peirastic.contact_qp.features import FeatureConfig
from peirastic.contact_qp.types import ContactObservation, REGION_FEATURE_VERSION
from peirastic.realman8dof.force.protocol import ForceOutput
from peirastic.realman8dof.modes.contact_recording import drain_study_records
from peirastic.realman8dof.modes.pad_visual import (
    PadVisualGuide,
    allocate_pad_log_path,
    attach_pad_visual,
    pad_hybrid_payload,
    resolve_pad_visual_config,
    wrap_pad_hybrid_log,
)
from peirastic.realman8dof.modes.track import HybridTffOuter
from peirastic.tests.test_contact_qp_solver import observation

CONFIG = Path('peirastic/config/contact_qp/active_probe50_delay_kf_cop.yaml')


def _guide(receiver):
    return PadVisualGuide(load_study_config(CONFIG), TorqueTiltConfig(), receiver=receiver)


def _frame(now=10., centroid=.4, quality=(.2, .8, .9), **kwargs):
    kwargs.setdefault('confidence_feature_version', FEATURE_VERSION)
    return replace(observation(quality, now=now),
                   confidence_centroid_x=centroid, confidence_centroid_valid=True, **kwargs)


def test_pad_visual_uses_image_and_falls_back_on_dropout():
    receiver = SimpleNamespace(observation=None)
    guide = _guide(receiver)
    tilt = TorqueTilt(TorqueTiltConfig())
    tilt._w = tilt.omega_y = 0.05
    pose = np.zeros(6)
    assert guide.preview(pose=pose, dt_s=.005, contact_present=True, tilt=tilt, now_s=10.) is None
    receiver.observation = _frame()
    omega = guide.preview(pose=pose, dt_s=.005, contact_present=True, tilt=tilt, now_s=10.)
    assert omega is not None
    assert abs(omega - 0.05) < 0.03
    published = np.zeros(6)
    published[4] = omega
    guide.commit(published, np.eye(3))
    receiver.observation = None
    assert guide.preview(pose=pose, dt_s=.005, contact_present=True, tilt=tilt, now_s=10.005) is None
    assert tilt._w == pytest.approx(0.05)


def test_pad_visual_version_mismatch_returns_to_moment():
    receiver = SimpleNamespace(observation=_frame(confidence_feature_version='not_a_centroid'))
    guide = _guide(receiver)
    tilt = TorqueTilt(TorqueTiltConfig())
    assert guide.preview(pose=np.zeros(6), dt_s=.005, contact_present=True, tilt=tilt, now_s=10.) is None
    assert guide.last_facts['visual_reason'] == 'version_mismatch'


def test_explicit_pad_visual_rejects_position_owned_tool_y():
    with pytest.raises(ValueError, match='injection point'):
        attach_pad_visual(
            {'pad_visual': True, 'selection': [1, 1, 0, 1, 1, 1]},
            TorqueTiltConfig(enabled=False),
        )


def test_disabled_pad_visual_skips_even_when_tilt_is_off():
    assert attach_pad_visual({'pad_visual': False, 'reference': 'pad'},
                             TorqueTiltConfig(enabled=False)) is None


def test_pad_payload_without_visual_keeps_moment_tilt(monkeypatch):
    monkeypatch.setenv('CONTACT_QP_CONFIG', str(CONFIG.resolve()))
    payload = {'reference': 'pad', 'use_tff_split': True, 'contact_qp': {'mode': 'active'}}
    assert resolve_pad_visual_config(payload) is None
    assert attach_pad_visual(payload, TorqueTiltConfig()) is None
    tilt = TorqueTilt(TorqueTiltConfig())
    omega = tilt.update(np.array([0., 0., 4., 0., 0.2, 0.]), np.zeros(6),
                        dt_s=.005, contact=True, pose=np.zeros(6))
    assert abs(omega) > 1e-4


def test_explicit_pad_visual_attaches_guide():
    guide = attach_pad_visual({'pad_visual': True}, TorqueTiltConfig())
    try:
        assert guide is not None
    finally:
        if guide is not None:
            guide.close(wait=False)


def test_pad_hybrid_phase_is_moment_unless_visual_is_requested():
    from peirastic.core.modes import Mode, ModeRequest
    from peirastic.realman8dof.session import compile_request
    from peirastic.tests.test_api_facade import _SEED, _ctx

    raw, ctx = _ctx()
    payload = {'reference': 'pad', 'use_tff_split': True, 'v_cmd': [0.]*6, 'desired_z': 4.}
    phase = compile_request(ctx, ModeRequest(Mode.TRACK_HYBRID, payload), raw=raw,
                            twist_read=lambda: np.zeros(6))
    assert getattr(phase.outer, 'pad_visual', None) is None
    pose = ctx.kin.fk_pose(_SEED)
    phase.outer.set_origin(pose)
    wrench = np.array([0., 0., 4., 0., 0.2, 0.])
    command = phase.outer.sample(0., pose, wrench, contact=True, dt_actual=.005)
    assert abs(command[4]) > 1e-4
    visual = compile_request(
        ctx, ModeRequest(Mode.TRACK_HYBRID, dict(payload, pad_visual=True)), raw=raw,
        twist_read=lambda: np.zeros(6))
    assert visual.outer.pad_visual is not None
    if visual.on_exit is not None:
        visual.on_exit()


def _region_frame(quality, *, now=10., seq=None):
    study = load_study_config(CONFIG)
    feature = FeatureConfig(**study['feature']['config'])
    values = np.asarray(quality, dtype=float)
    return ContactObservation(
        frame_seq=int(seq if seq is not None else round(float(now) * 2000)),
        source_id='synthetic', effective_time_s=now, received_time_s=now,
        quality=[0.9, 0.9, 0.9], valid=[True, True, True],
        registration_version='registration_v1', window_version=feature.window_version,
        calibration_version=feature.calibration_version,
        region_confidence=values, region_valid=np.ones(len(values), dtype=bool),
        region_edges=feature.region_edges, region_layout_version=feature.region_layout_version,
        region_feature_version=REGION_FEATURE_VERSION, timestamp_semantics='effective_image_time',
    )


def test_pad_hybrid_payload_legacy_is_tff_tilt():
    from peirastic.scan_path import force_profile

    payload = pad_hybrid_payload(desired_z=4.0)
    assert payload.get('law') is None
    assert 'force_axes' not in payload
    assert 'pad_visual' not in payload
    assert payload['label'] == 'track_hybrid_pad'
    assert payload['hybrid_motion'] == force_profile('icra')['hybrid_motion']
    assert payload['desired_z'] == pytest.approx(4.0)
    baseline = pad_hybrid_payload('legacy', desired_z=4.0, force_profile='baseline')
    assert 'hybrid_motion' not in baseline


def test_pad_hybrid_payload_ac2d_owns_z_and_omega():
    from peirastic.scan_path import SCAN_FORCE_AXES

    payload = pad_hybrid_payload('ac2d', desired_z=4.0)
    assert payload['law'] == 'ac2d'
    assert payload['force_axes'] == list(SCAN_FORCE_AXES)
    assert 'pad_visual' not in payload
    with pytest.raises(ValueError, match='pad hybrid law'):
        pad_hybrid_payload('contact_qp')


def test_pad_hybrid_payload_ultrapoc_requests_progress():
    from peirastic.scan_path import SCAN_FORCE_AXES, force_profile

    payload = pad_hybrid_payload('ultrapoc', desired_z=4.0)
    assert payload['pad_visual'] == {'progress': True}
    assert 'pad_contact' not in payload
    assert payload['force_axes'] == list(SCAN_FORCE_AXES)
    assert payload['hybrid_motion'] == force_profile('icra')['hybrid_motion']
    assert payload.get('law') != 'ac2d'


def test_pad_progress_uses_region_alpha_and_falls_back_on_dropout():
    receiver = SimpleNamespace(observation=None)
    guide = PadVisualGuide(load_study_config(CONFIG), TorqueTiltConfig(),
                           receiver=receiver, progress=True)
    tilt = TorqueTilt(TorqueTiltConfig())
    pose = np.zeros(6)
    assert guide.preview(pose=pose, dt_s=.005, contact_present=True, tilt=tilt, now_s=10.) is None
    assert guide.last_alpha == pytest.approx(1.0)
    assert guide.last_visual_omega == pytest.approx(0.0)
    n = FeatureConfig(**load_study_config(CONFIG)['feature']['config']).region_count
    hole = np.ones(n)
    hole[0] = 0.5
    receiver.observation = _region_frame(hole, now=10.)
    assert guide.preview(pose=pose, dt_s=.005, contact_present=True, tilt=tilt, now_s=10.) is None
    assert 0.25 < guide.last_alpha < 1.0
    assert abs(guide.last_visual_omega) > 0.0
    assert abs(guide.last_visual_omega) < abs(guide.omega_target)
    for step in range(1, 120):
        now = 10. + 0.005 * step
        receiver.observation = _region_frame(hole, now=now)
        assert guide.preview(pose=pose, dt_s=.005, contact_present=True, tilt=tilt, now_s=now) is None
    assert guide.last_alpha == pytest.approx(0.25, abs=0.02)
    receiver.observation = None
    assert guide.preview(pose=pose, dt_s=.005, contact_present=True, tilt=tilt, now_s=10.6) is None
    assert guide.last_alpha == pytest.approx(0.25, abs=0.03)
    for step in range(120):
        guide.preview(pose=pose, dt_s=.005, contact_present=True, tilt=tilt,
                      now_s=11.0 + 0.005 * step)
    assert guide.last_alpha == pytest.approx(1.0, abs=0.02)
    assert guide.last_visual_omega == pytest.approx(0.0, abs=0.01)


def test_pad_progress_gates_then_steps_when_a_region_crosses_cmin():
    receiver = SimpleNamespace(observation=None)
    guide = PadVisualGuide(load_study_config(CONFIG), TorqueTiltConfig(),
                           receiver=receiver, progress=True)
    tilt = TorqueTilt(TorqueTiltConfig())
    pose = np.zeros(6)
    n = FeatureConfig(**load_study_config(CONFIG)['feature']['config']).region_count
    assert guide.rocking_reference is None
    assert guide.deficit_mode == 'relative'
    assert guide.brake_on_sided is False
    assert guide.max_visual_omega_rad_s == pytest.approx(0.16)
    assert guide.ramp_s == pytest.approx(0.12)
    assert guide.omega_step_rad_s == pytest.approx(0.06)
    assert guide.visual_hold_s == pytest.approx(0.0)
    above = np.ones(n)
    above[0] = 0.98
    receiver.observation = _region_frame(above, now=10.)
    assert guide.preview(pose=pose, dt_s=.005, contact_present=True, tilt=tilt, now_s=10.) is None
    assert guide.omega_target == pytest.approx(0.0)
    assert guide.alpha_target == pytest.approx(1.0)
    just_under = np.ones(n)
    just_under[0] = 0.85
    receiver.observation = _region_frame(just_under, now=10.005)
    assert guide.preview(pose=pose, dt_s=.005, contact_present=True, tilt=tilt, now_s=10.005) is None
    assert abs(guide.omega_target) > 0.0
    assert guide.alpha_target < 1.0
    assert guide.last_facts['deficit_mode'] == 'relative'
    assert guide.last_facts['quality_reference'] == pytest.approx(1.0)
    deeper = np.ones(n)
    deeper[0] = 0.70
    receiver.observation = _region_frame(deeper, now=10.01)
    assert guide.preview(pose=pose, dt_s=.005, contact_present=True, tilt=tilt, now_s=10.01) is None
    assert abs(guide.omega_target) >= abs(0.06)
    assert abs(guide.omega_target) <= guide.max_visual_omega_rad_s + 1e-12
    assert guide.alpha_target == pytest.approx(guide.alpha_min)


def test_pad_relative_keeps_human_on_moment_and_still_publishes_omega_alpha():
    receiver = SimpleNamespace(observation=None)
    guide = PadVisualGuide(load_study_config(CONFIG), TorqueTiltConfig(),
                           receiver=receiver, progress=True)
    tilt = TorqueTilt(TorqueTiltConfig())
    pose = np.zeros(6)
    n = FeatureConfig(**load_study_config(CONFIG)['feature']['config']).region_count
    human = np.linspace(0.867, 0.901, n)
    receiver.observation = _region_frame(human, now=10.)
    assert guide.preview(pose=pose, dt_s=.005, contact_present=True, tilt=tilt, now_s=10.) is None
    assert guide.omega_target == pytest.approx(0.0)
    assert guide.alpha_target == pytest.approx(1.0)
    assert guide.last_visual_omega == pytest.approx(0.0)
    assert guide.last_alpha == pytest.approx(1.0)
    assert guide.last_facts['deficit_mode'] == 'relative'
    phantom = np.ones(n)
    phantom[0] = 0.70
    receiver.observation = _region_frame(phantom, now=10.01)
    assert guide.preview(pose=pose, dt_s=.005, contact_present=True, tilt=tilt, now_s=10.01) is None
    assert abs(guide.omega_target) > 0.0
    assert guide.alpha_target < 1.0
    assert guide.last_facts['quality_reference'] > 0.9


def test_pad_progress_kf_predicts_sided_hole_before_raw_gate():
    receiver = SimpleNamespace(observation=None)
    guide = PadVisualGuide(load_study_config(CONFIG), TorqueTiltConfig(),
                           receiver=receiver, progress=True)
    assert guide._image_kf is not None
    tilt = TorqueTilt(TorqueTiltConfig())
    pose = np.zeros(6)
    n = FeatureConfig(**load_study_config(CONFIG)['feature']['config']).region_count
    stamps = (9.80, 9.90, 10.00)
    left = (1.00, 0.88, 0.76)
    for seq, (stamp, q0) in enumerate(zip(stamps, left), start=1):
        quality = np.ones(n)
        quality[0] = q0
        receiver.observation = _region_frame(quality, now=stamp, seq=seq)
        assert guide.preview(pose=pose, dt_s=.005, contact_present=True, tilt=tilt, now_s=stamp) is None
    receiver.observation = _region_frame(np.concatenate(([0.76], np.ones(n - 1))), now=10.00, seq=3)
    assert guide.preview(pose=pose, dt_s=.005, contact_present=True, tilt=tilt, now_s=10.23) is None
    assert guide.last_facts['kf_valid'] is True
    predicted = np.asarray(guide.last_facts['kf_quality'], dtype=float)
    assert predicted[0] < 0.70
    assert abs(guide.omega_target) >= guide.omega_step_rad_s - 1e-12
    assert guide.alpha_target == pytest.approx(guide.alpha_min)


def test_pad_progress_releases_visual_omega_when_image_recovers():
    receiver = SimpleNamespace(observation=None)
    guide = PadVisualGuide(load_study_config(CONFIG), TorqueTiltConfig(),
                           receiver=receiver, progress=True)
    tilt = TorqueTilt(TorqueTiltConfig())
    pose = np.zeros(6)
    n = FeatureConfig(**load_study_config(CONFIG)['feature']['config']).region_count
    hole = np.ones(n)
    hole[0] = 0.70
    receiver.observation = _region_frame(hole, now=10.)
    guide.preview(pose=pose, dt_s=.005, contact_present=True, tilt=tilt, now_s=10.)
    assert abs(guide.omega_target) > 0.03
    receiver.observation = _region_frame(np.ones(n), now=10.2)
    guide.preview(pose=pose, dt_s=.005, contact_present=True, tilt=tilt, now_s=10.2)
    assert guide.last_facts['visual_reason'] != 'visual_hold'
    assert abs(guide.omega_target) < 0.02


def test_hybrid_scales_pad_velocity_by_published_alpha():
    class _Position:
        def __init__(self):
            self.last_path_twist = np.array([0.02, 0.0, 0.0, 0.0, 0.0, 0.0])
            self.last_feedback_twist = np.zeros(6)
            self.last_vel_ff = np.zeros(6)
            self.last_pose_d = np.zeros(6)
            self.last_err_mm = 0.0
            self.cfg = SimpleNamespace(euler_order='xyz', track_axes=np.ones(6))

        def sample(self, t_s, current_pose, f_ext):
            del t_s, current_pose, f_ext
            return self.last_path_twist.copy()

    class _Law:
        controller = SimpleNamespace(contact_present=True)
        tilt = None

        def reset(self, **_kwargs):
            return None

        def update(self, **_kwargs):
            return ForceOutput(v_force=np.zeros(6), v_force_z=0.0, contact_active=True, f_des_z=4.0)

    class _Guide:
        last_alpha = 0.25

        def preview(self, **_kwargs):
            return None

        def commit(self, *_args):
            return None

    outer = HybridTffOuter(
        _Position(), _Law(), desired_force=np.array([0., 0., 4., 0., 0., 0.]),
        selection=np.array([1., 1., 0., 1., 0., 1.]),
    )
    outer.pad_visual = _Guide()
    command = outer.sample(0., np.zeros(6), np.array([0., 0., 4., 0., 0., 0.]),
                           contact=True, dt_actual=.005)
    assert command[0] == pytest.approx(0.005)
    assert outer.last_pad_alpha == pytest.approx(0.25)
    _Guide.last_alpha = None
    command = outer.sample(0.005, np.zeros(6), np.array([0., 0., 4., 0., 0., 0.]),
                           contact=True, dt_actual=.005)
    assert command[0] == pytest.approx(0.02)


def test_hybrid_adds_visual_omega_on_top_of_moment():
    class _Position:
        last_path_twist = np.zeros(6)
        last_feedback_twist = np.zeros(6)
        last_vel_ff = np.zeros(6)
        last_pose_d = np.zeros(6)
        last_err_mm = 0.0
        cfg = SimpleNamespace(euler_order='xyz', track_axes=np.ones(6))

        def sample(self, t_s, current_pose, f_ext):
            del t_s, current_pose, f_ext
            return np.zeros(6)

    class _Law:
        controller = SimpleNamespace(contact_present=True)
        tilt = SimpleNamespace(cfg=SimpleNamespace(vmax_rad_s=0.28))

        def reset(self, **_kwargs):
            return None

        def update(self, **kwargs):
            assert kwargs.get('visual_velocity_target_rad_s') is None
            force = np.zeros(6)
            force[4] = 0.04
            return ForceOutput(v_force=force, v_force_z=0.0, contact_active=True, f_des_z=4.0)

    class _Guide:
        last_alpha = 1.0
        last_visual_omega = 0.05

        def preview(self, **_kwargs):
            return None

        def commit(self, *_args):
            return None

    outer = HybridTffOuter(
        _Position(), _Law(), desired_force=np.array([0., 0., 4., 0., 0., 0.]),
        selection=np.array([1., 1., 0., 1., 0., 1.]),
    )
    outer.pad_visual = _Guide()
    command = outer.sample(0., np.zeros(6), np.array([0., 0., 4., 0., 0.2, 0.]),
                           contact=True, dt_actual=.005)
    assert command[4] == pytest.approx(0.09)


def test_explicit_legacy_pad_phase_retains_progress_guide():
    from peirastic.core.modes import Mode, ModeRequest
    from peirastic.realman8dof.session import compile_request
    from peirastic.tests.test_api_facade import _SEED, _ctx

    raw, ctx = _ctx()
    payload = pad_hybrid_payload('ultrapoc', desired_z=4.0)
    payload['v_cmd'] = [0.02, 0.0, 0.0, 0.0, 0.0, 0.0]
    phase = compile_request(
        ctx, ModeRequest(Mode.TRACK_HYBRID, payload), raw=raw,
    )
    try:
        guide = phase.outer.pad_visual
        assert guide is not None and guide.progress is True
        pose = ctx.kin.fk_pose(_SEED)
        phase.outer.set_origin(pose)
        wrench = np.array([0., 0., 4., 0., 0.2, 0.])
        command = phase.outer.sample(0., pose, wrench, contact=True, dt_actual=.005)
        moment = command[4]
        assert abs(moment) > 1e-4
        n = FeatureConfig(**load_study_config(CONFIG)['feature']['config']).region_count
        hole = np.ones(n)
        hole[0] = 0.5
        import time
        now = time.monotonic()
        receiver = SimpleNamespace(observation=_region_frame(hole, now=now))
        guide._receiver = receiver
        first = phase.outer.sample(0.01, pose, wrench, contact=True, dt_actual=.005)
        assert first[0] < 0.02
        assert first[0] > 0.015
        assert abs(first[4]) > 1e-4
        for step in range(1, 120):
            receiver.observation = _region_frame(hole, now=now + 0.005 * step)
            slowed = phase.outer.sample(0.01 + 0.005 * step, pose, wrench,
                                        contact=True, dt_actual=.005)
        assert slowed[0] == pytest.approx(0.005, abs=0.002)
        assert phase.outer.last_pad_alpha == pytest.approx(0.25, abs=0.02)
        assert abs(slowed[4]) > 1e-4
    finally:
        if phase.on_exit is not None:
            phase.on_exit()


def test_ac2d_pad_phase_keeps_moment_and_does_not_scale():
    from peirastic.core.modes import Mode, ModeRequest
    from peirastic.realman8dof.force.comparison_laws import ComparisonAdmittance2D
    from peirastic.realman8dof.session import compile_request
    from peirastic.tests.test_api_facade import _SEED, _ctx

    raw, ctx = _ctx()
    payload = pad_hybrid_payload('ac2d', desired_z=4.0)
    payload['v_cmd'] = [0.02, 0.0, 0.0, 0.0, 0.0, 0.0]
    phase = compile_request(
        ctx, ModeRequest(Mode.TRACK_HYBRID, payload), raw=raw,
        twist_read=lambda: np.array([0.02, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )
    assert getattr(phase.outer, 'pad_visual', None) is None
    assert isinstance(phase.outer.force_law, ComparisonAdmittance2D)
    pose = ctx.kin.fk_pose(_SEED)
    phase.outer.set_origin(pose)
    wrench = np.array([0., 0., 4., 0., 0.2, 0.])
    command = phase.outer.sample(0., pose, wrench, contact=True, dt_actual=.005)
    assert command[0] == pytest.approx(0.02)
    assert abs(command[4]) > 1e-4
    assert phase.outer.last_pad_alpha == pytest.approx(1.0)


def test_pad_hybrid_payload_log_path():
    payload = pad_hybrid_payload('ultrapoc', desired_z=4.0, log_path='/tmp/pad.jsonl')
    assert payload['pad_log'] == '/tmp/pad.jsonl'
    assert pad_hybrid_payload('ac2d').get('pad_log') is None


def test_allocate_pad_log_path_uniquifies(tmp_path):
    first = tmp_path / 'pad.jsonl'
    first.write_text('{}\n')
    allocated = allocate_pad_log_path(first)
    assert allocated == tmp_path / 'pad_001.jsonl'
    directory = allocate_pad_log_path(tmp_path)
    assert directory.parent == tmp_path
    assert directory.suffix == '.jsonl'
    assert not directory.exists()


def test_pad_hybrid_log_records_stick_alpha_and_visual(tmp_path):
    from rm75_control.control.joint_admittance_8dof.loop import Phase

    class _Position:
        def __init__(self):
            self.last_path_twist = np.array([0.02, 0.0, 0.0, 0.0, 0.0, 0.0])
            self.last_feedback_twist = np.zeros(6)
            self.last_vel_ff = np.zeros(6)
            self.last_pose_d = np.zeros(6)
            self.last_err_mm = 0.0
            self.cfg = SimpleNamespace(euler_order='xyz', track_axes=np.ones(6))

        def sample(self, t_s, current_pose, f_ext):
            del t_s, current_pose, f_ext
            return self.last_path_twist.copy()

    class _Law:
        controller = SimpleNamespace(contact_present=True)
        tilt = SimpleNamespace(cfg=SimpleNamespace(vmax_rad_s=0.28))

        def reset(self, **_kwargs):
            return None

        def update(self, **_kwargs):
            force = np.zeros(6)
            force[4] = 0.04
            return ForceOutput(
                v_force=force, v_force_z=0.0, contact_active=True, f_des_z=4.0,
                telemetry={"omega_y": 0.04, "tau_y": 0.2, "tau_error_y": 0.05},
            )

    class _Guide:
        last_alpha = 0.25
        last_visual_omega = 0.05
        last_facts = {
            'visual_reason': 'regional_deficit_moment',
            'visual_policy': 'pad_ultrapoc_additive',
            'alpha_target': 0.25,
            'visual_request_rad_s': 0.08,
            'visual_omega_rad_s': 0.05,
            'image_age_s': 0.12,
            'max_deficit': 0.2,
            'bad_region_indices': [0],
            'region_quality': [0.5] + [1.0] * 9,
        }

        def preview(self, **_kwargs):
            return None

        def commit(self, *_args):
            return None

    path = tmp_path / 'pad.jsonl'
    outer = HybridTffOuter(
        _Position(), _Law(), desired_force=np.array([0., 0., 4., 0., 0., 0.]),
        selection=np.array([1., 1., 0., 1., 0., 1.]),
    )
    outer.pad_visual = _Guide()
    phase = Phase(outer=outer, label='track_hybrid_pad_ultrapoc')
    wrap_pad_hybrid_log(phase, {'pad_log': {'path': str(path), 'hybrid_law': 'ultrapoc'}})
    command = phase.outer.sample(
        0., np.zeros(6), np.array([0., 0., 4., 0., 0.2, 0.]),
        contact=True, dt_actual=.005,
    )
    assert command[0] == pytest.approx(0.005)
    assert command[4] == pytest.approx(0.09)
    if phase.on_exit is not None:
        phase.on_exit()
    drain_study_records()
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    samples = [row for row in rows if row.get('event') == 'control_sample']
    assert samples
    row = samples[0]
    assert row['record_kind'] == 'pad_hybrid'
    assert row['alpha'] == pytest.approx(0.25)
    assert row['alpha_target'] == pytest.approx(0.25)
    assert row['v_stick'][0] == pytest.approx(0.02)
    assert row['v_scaled'][0] == pytest.approx(0.005)
    assert row['omega_moment'] == pytest.approx(0.04)
    assert row['omega_visual'] == pytest.approx(0.05)
    assert row['omega_cmd'] == pytest.approx(0.09)
    assert row['visual_reason'] == 'regional_deficit_moment'
    assert row['max_deficit'] == pytest.approx(0.2)
    assert row['my'] == pytest.approx(0.2)


def test_ultrapoc_pad_phase_writes_hybrid_log(tmp_path):
    from peirastic.core.modes import Mode, ModeRequest
    from peirastic.realman8dof.session import compile_request
    from peirastic.tests.test_api_facade import _SEED, _ctx

    raw, ctx = _ctx()
    path = tmp_path / 'compile.jsonl'
    payload = pad_hybrid_payload('ultrapoc', desired_z=4.0, log_path=str(path))
    payload['v_cmd'] = [0.02, 0.0, 0.0, 0.0, 0.0, 0.0]
    phase = compile_request(ctx, ModeRequest(Mode.TRACK_HYBRID, payload), raw=raw)
    try:
        import time
        clock = [time.monotonic()]
        phase.outer.clock = lambda: clock[0]
        pose = ctx.kin.fk_pose(_SEED)
        phase.outer.set_origin(pose)
        wrench = np.array([0., 0., 4., 0., 0.2, 0.])
        command = phase.outer.sample(0., pose, wrench, contact=True, dt_actual=.005,
                                     sensor_age_s=.001, feedback_age_s=.001)
        assert command.shape == (6,)
        assert np.isfinite(command).all()
    finally:
        if phase.on_exit is not None:
            phase.on_exit()
        drain_study_records()
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    assert any(row.get('event') == 'study_start' and row.get('record_kind') == 'pad_hybrid' for row in rows)
    samples = [row for row in rows if row.get('event') == 'control_sample']
    assert samples
    assert samples[0]['v_stick'][0] == pytest.approx(0.02)
    assert 'alpha' in samples[0]
    assert samples[0]['command_semantics'] == 'delay_kf_cop_v1'
    assert samples[0].get('cop_reason') is not None


def test_allocate_pad_analysis_dir_uniquifies(tmp_path, monkeypatch):
    from peirastic.realman8dof.modes.pad_analysis import allocate_pad_analysis_dir

    monkeypatch.setattr('peirastic.realman8dof.modes.pad_analysis.time.strftime', lambda *_a, **_k: '20260921_150000')
    first = allocate_pad_analysis_dir(tmp_path)
    second = allocate_pad_analysis_dir(tmp_path)
    assert first == tmp_path / 'apps/logs/pad_hybrid/20260921_150000'
    assert second == tmp_path / 'apps/logs/pad_hybrid/20260921_150000_001'
    assert first.is_dir() and second.is_dir()


def test_pad_analysis_recorder_writes_session_and_stops(tmp_path, monkeypatch):
    from peirastic.realman8dof.modes.pad_analysis import PadAnalysisRecorder

    created = {}

    class _Proc:
        def __init__(self):
            self._code = None

        def poll(self):
            path = created['raw']
            if not path.exists():
                path.write_bytes(b'h5')
            return self._code

        def send_signal(self, sig):
            created['sig'] = sig
            self._code = 0

        def wait(self, timeout=None):
            return 0

        def terminate(self):
            self._code = 0

        def kill(self):
            self._code = 0

    def fake_popen(command, **_kwargs):
        created['command'] = command
        created['raw'] = Path(command[command.index('--output-file') + 1])
        return _Proc()

    monkeypatch.setattr('peirastic.realman8dof.modes.pad_analysis.subprocess.Popen', fake_popen)
    directory = tmp_path / 'session'
    recorder = PadAnalysisRecorder(
        repo=tmp_path, record_script=tmp_path / 'record.py', python='/bin/true',
    )
    raw = recorder.start(directory, extra={'hybrid_law': 'ultrapoc'})
    assert raw.exists()
    session = json.loads((directory / 'session.json').read_text())
    assert session['schema'] == 'pad_hybrid_analysis_v1'
    assert session['hybrid_law'] == 'ultrapoc'
    assert session['raw_h5'] == str(raw)
    assert '--raw-only' in created['command']
    assert recorder.close_if_inactive(True) is None
    assert recorder.proc is not None
    assert recorder.close_if_inactive(False) == raw
    import signal
    assert created['sig'] == signal.SIGINT
    assert recorder.path is None and recorder.proc is None
    assert recorder.close_if_inactive(False) is None
    second = recorder.start(tmp_path / 'second_session')
    assert second != raw and second.exists()
    recorder.stop()
