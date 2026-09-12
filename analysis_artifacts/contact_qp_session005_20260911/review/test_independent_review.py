"""ULTRA audit checks; no hardware imports with side effects or device startup."""
import ast
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from peirastic.tests.test_contact_qp_active import make_active
from peirastic.tests.test_contact_qp_transient_feedback import image


@pytest.mark.parametrize('invalid_window', [0, 2])
def test_initial_registered_frame_requires_valid_lateral_windows(monkeypatch, invalid_window):
    active, _, clock, _ = make_active(monkeypatch)
    try:
        active.config['feature']['required'] = True
        active._image_dropout_policy = 'pause_visual'
        flags = np.ones(3, dtype=bool)
        flags[invalid_window] = False
        active.features.observation = replace(image(active, clock[0]), valid=flags)
        assert not active.features.observation.fresh(clock[0], .3)
        with pytest.raises(RuntimeError, match='confidence feedback'):
            active._image_observation(clock[0], contact_enabled=True)
        assert not active._image_seen_in_contact
    finally:
        active.close()


def test_invalid_lateral_frame_cannot_announce_recovery(monkeypatch):
    active, _, clock, sink = make_active(monkeypatch)
    try:
        active.config['feature']['required'] = True
        active._image_dropout_policy = 'pause_visual'
        active.features.observation = image(active, clock[0])
        active._image_observation(clock[0], contact_enabled=True)
        clock[0] += .301
        assert active._image_observation(clock[0], contact_enabled=True)[0] is None
        unavailable_since = active._image_unavailable_since_s
        active.features.observation = replace(image(active, clock[0]), valid=[False, True, False])
        try:
            observed, reason = active._image_observation(clock[0], contact_enabled=True)
        except RuntimeError:
            pass
        else:
            assert observed is None and reason != 'ok'
        assert active._image_unavailable_since_s == unavailable_since
        assert not any(row['event'] == 'image_feedback_recovered' for row in sink.records)
    finally:
        active.close()


@pytest.mark.parametrize('stop_requested,watchdog_fired,fault_epoch', [
    (True, False, 0), (False, True, 0), (False, False, 1),
])
def test_real_runner_loop_guards_interrupt_retry_before_source_wait(stop_requested, watchdog_fired, fault_epoch):
    """Execute the real top-of-loop guards enclosing the new retry branch."""
    from rm75_control.control.joint_admittance_8dof import loop
    tree = ast.parse(Path(loop.__file__).read_text())
    repeated = next(node for node in ast.walk(tree) if isinstance(node, ast.While)
        and len(node.body) > 1 and isinstance(node.body[0], ast.If)
        and any(isinstance(x, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'study_termination_reason'
            for t in x.targets) for x in node.body[0].body))
    events = []
    marker = ast.Expr(value=ast.Call(func=ast.Name(id='unexpected_wait', ctx=ast.Load()), args=[], keywords=[]))
    body = repeated.body[:2] + [marker]
    module = ast.Module(body=[ast.For(target=ast.Name(id='_once', ctx=ast.Store()),
        iter=ast.Tuple(elts=[ast.Constant(1)], ctx=ast.Load()), body=body, orelse=[])], type_ignores=[])
    env = dict(stop_check=lambda: stop_requested, wd=NS(fired=watchdog_fired),
        fault_epoch=[fault_epoch], _fault_stop=events.append,
        unexpected_wait=lambda: events.append('unexpected_wait'), phase_stopped=False)
    exec(compile(ast.fix_missing_locations(module), '<real runner retry guard>', 'exec'), env)
    assert env['phase_stopped']
    assert 'unexpected_wait' not in events
    if stop_requested:
        assert env['study_termination_reason'] == 'stop_requested'
    else:
        assert events == ['watchdog_latched']


def test_overdue_retry_schedule_resynchronizes_before_burst():
    from rm75_control.control.joint_admittance_8dof.loop import _resync_late_tick
    # A 15.2 ms solve can miss two nominal 5 ms deadlines. The shared top
    # of loop resets that old schedule, so the next same-source wait is paced.
    next_tick, late_ms = _resync_late_tick(10.005, 10.0152, .005)
    assert next_tick == 10.0152
    assert late_ms == pytest.approx(10.2)
    assert next_tick + .005 > 10.0152
