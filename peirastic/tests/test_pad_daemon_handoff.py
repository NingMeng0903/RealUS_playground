"""Pad hot entry inherits the same Cartesian model that final review accepts."""

from types import SimpleNamespace

import numpy as np
import pytest

from peirastic.core.ipc import Cmd
from peirastic.core.modes import Mode, ModeRequest
from peirastic.realman8dof import daemon
from peirastic.tests.test_daemon_boundary import _Outer, _hot_swap_service
from rm75_control.control.joint_admittance_8dof.loop import LoopResult, Phase


class SeedOuter(_Outer):
    def __init__(self, *, pad):
        if pad:
            self.command_lifecycle = self
        self.seeds = []

    def begin_hybrid_episode(self, applied_twist, current_pose):
        self.seeds.append((applied_twist.copy(), current_pose.copy()))


def run_handoff(monkeypatch, *, pad, estimate):
    svc = _hot_swap_service()
    # The rail command is 50 mm/s, while execution is only 10 mm/s.
    # The arm contribution is -20 mm/s: total execution-model speed is
    # -10 mm/s, unlike the +30 mm/s all-commanded model.
    jacobian = np.zeros((6, 8))
    jacobian[0, 0] = jacobian[0, 1] = 1.
    svc.inner.kin.jacobian = lambda _q: jacobian
    svc.inner.core.qdot_prev = np.array([.05, -.02, 0., 0., 0., 0., 0., 0.])
    joint_seeds = []
    svc.inner.begin_hybrid_episode = lambda q, qdot: joint_seeds.append((q.copy(), qdot.copy()))
    outer = SeedOuter(pad=pad)
    exited = []

    def compile_mode(ctx, req, **kwargs):
        if req.mode == Mode.TRACK_HYBRID:
            return Phase(outer=outer, label="pad_hybrid" if pad else "legacy_hybrid")
        return Phase(outer=_Outer(), label="servo", on_exit=lambda: exited.append(True))

    def runner(sess, phases, inner, **kwargs):
        phase = phases[0]
        original = phase.outer.child
        svc.hub.polls = [(Cmd.SET_MODE, 72, ModeRequest(Mode.TRACK_HYBRID, {}))]
        step = SimpleNamespace(q_send=inner.q_cmd.copy(), slack_norm=0.,
                               v_tcp_estimated=estimate)
        kwargs['on_step']('servo', 1., step, np.zeros(6), np.zeros(6), 1.)
        installed = phase.outer.child is outer
        if not installed:
            assert phase.outer.child is original
        svc._stop = True
        return LoopResult(1, .005, 0., False)

    monkeypatch.setattr(daemon, 'compile_request', compile_mode)
    monkeypatch.setattr(daemon, 'run_joint_admittance_phases', runner)
    svc.run(SimpleNamespace(robot=None), None, None)
    return svc, outer, joint_seeds, exited


@pytest.mark.parametrize('pad', [True, False])
def test_pad_uses_execution_rail_seed_and_legacy_keeps_commanded_seed(monkeypatch, pad):
    estimate = np.array([-.01, 0., 0., 0., 0., 0.])
    svc, outer, joint_seeds, exited = run_handoff(monkeypatch, pad=pad, estimate=estimate)
    assert len(outer.seeds) == len(joint_seeds) == len(exited) == 1
    expected = estimate.copy() if pad else np.array([.03, 0., 0., 0., 0., 0.])
    np.testing.assert_allclose(outer.seeds[0][0], expected)
    np.testing.assert_array_equal(joint_seeds[0][1], svc.inner.core.qdot_prev)
    estimate[:] = 99.
    np.testing.assert_allclose(outer.seeds[0][0], expected)


@pytest.mark.parametrize('estimate', [None, np.zeros(5), np.zeros((2, 3)), np.full(6, np.nan)])
def test_bad_pad_handoff_estimate_leaves_existing_outer_installed(monkeypatch, estimate):
    svc, outer, joint_seeds, exited = run_handoff(monkeypatch, pad=True, estimate=estimate)
    assert outer.seeds == joint_seeds == exited == []
    assert svc.mode == Mode.SERVO_TWIST
    assert svc._compile_fault is not None
    assert 'finite applied base twist' in svc._compile_fault['msg']


def test_legacy_handoff_does_not_require_execution_estimate(monkeypatch):
    svc, outer, joint_seeds, exited = run_handoff(monkeypatch, pad=False, estimate=None)
    assert len(outer.seeds) == len(joint_seeds) == len(exited) == 1
    np.testing.assert_allclose(outer.seeds[0][0], [.03, 0., 0., 0., 0., 0.])
