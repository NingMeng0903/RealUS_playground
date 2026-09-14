"""Pad visual ωy injection: image-ready opens the visual branch, dropout returns to moment."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from peirastic.contact_qp.confidence_fusion import FEATURE_VERSION
from peirastic.contact_qp.runtime_config import load_study_config
from peirastic.realman8dof.force.torque_tilt import TorqueTilt, TorqueTiltConfig
from peirastic.realman8dof.modes.pad_visual import (
    PadVisualGuide,
    attach_pad_visual,
    resolve_pad_visual_config,
)
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
