"""Image-only, short-horizon feedback wrapper for a planned probe motion.

The policy receives one public packet and the measured motor state after each
original-frame block. It never receives the scene, belief, or controller.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from view_geometry import FRAME_STEPS, Motion, prior_run


def _blocks(motion):
    """Yield fixed-clock blocks without changing the source motion samples."""
    for end in range(FRAME_STEPS, len(motion.x), FRAME_STEPS):
        yield Motion(motion.x[end-FRAME_STEPS:end+1],
                     motion.v[end-FRAME_STEPS:end+1]), end


def _frame_count(plan):
    if (len(plan.x) - 1) % FRAME_STEPS:
        raise ValueError('motion length is not an integer number of frames')
    return (len(plan.x) - 1) // FRAME_STEPS


def _quintic_from_state(q, v, target, frames):
    """Quintic interpolation with measured initial velocity and zero end speed."""
    dt = float(prior_run.legacy_scene.FRAME_DT)
    t = np.linspace(0., 1., frames * FRAME_STEPS + 1)[:, None]
    duration = frames * dt
    delta = np.asarray(target) - q
    v0t = np.asarray(v) * duration
    c3 = 10. * delta - 6. * v0t
    c4 = -15. * delta + 8. * v0t
    c5 = 6. * delta - 3. * v0t
    x = q + v0t*t + c3*t**3 + c4*t**4 + c5*t**5
    dx = v0t + 3.*c3*t**2 + 4.*c4*t**3 + 5.*c5*t**4
    return Motion(x, dx / duration)


def feasible_redirect(scene, state, target, frames_left):
    """Plan to an absolute target from measured q,v within a frame budget.

    Native motor replanning is preferred. If that fails or misses the exact
    stationary endpoint, try bounded-duration quintics and admit only those
    accepted by the motor's full feasibility checker.
    """
    target = np.clip(np.asarray(target, dtype=float), *prior_run.legacy_control.CONTACT)
    q, v = np.asarray(state.q, dtype=float), np.asarray(state.v, dtype=float)
    if target.shape != q.shape or not np.isfinite(target).all() or not np.isfinite(v).all():
        return None
    budget = max(0, int(frames_left))
    native = scene.motor.replan(q, v, target)
    if native is not None and _frame_count(native) <= budget:
        if (np.max(np.abs(native.x[0]-q)) <= 1.e-7 and
                np.max(np.abs(native.v[0]-v)) <= 1.e-7 and
                np.max(np.abs(native.x[-1]-target)) <= 1.e-7 and
                np.max(np.abs(native.v[-1])) <= 1.e-7 and scene.motor.feasible(native)):
            return native
    for frames in (4, 5, 7, 10, 15, 22, 33, 48, 72, 105):
        if frames > budget:
            break
        candidate = _quintic_from_state(q, v, target, frames)
        if (np.max(np.abs(candidate.x[0]-q)) <= 1.e-7 and
                np.max(np.abs(candidate.v[0]-v)) <= 1.e-7 and
                np.max(np.abs(candidate.x[-1]-target)) <= 1.e-7 and
                np.max(np.abs(candidate.v[-1])) <= 1.e-7 and scene.motor.feasible(candidate)):
            return candidate
    return None


def advance_block(ctrl, tape, state, motion, frame, trace, history, phase):
    """Execute one 1-indexed original-clock frame, enforcing exact q,v join."""
    frame = int(frame)
    if frame < 1 or frame > _frame_count(motion):
        raise IndexError('frame is outside motion')
    end = frame * FRAME_STEPS
    block = Motion(motion.x[end-FRAME_STEPS:end+1],
                   motion.v[end-FRAME_STEPS:end+1])
    if (np.max(np.abs(block.x[0]-state.q)) > 1.e-7 or
            np.max(np.abs(block.v[0]-state.v)) > 1.e-7):
        raise RuntimeError('motion block does not join current q,v')
    state = ctrl.advance(state, block, tape, trace=trace, phase=phase)
    if history is not None:
        history.append(state.recent[-1])
    return state


def assert_admissible(scene, state, plan, deadline):
    """Raise unless plan joins current state, fits the tick deadline, and rests."""
    count = _frame_count(plan)
    if state.tick + count > int(deadline):
        raise ValueError('plan exceeds deadline')
    if (np.max(np.abs(plan.x[0]-state.q)) > 1.e-7 or
            np.max(np.abs(plan.v[0]-state.v)) > 1.e-7):
        raise ValueError('plan does not join current q,v')
    if np.max(np.abs(plan.v[-1])) > 1.e-7:
        raise ValueError('plan does not end at rest')
    if not scene.motor.feasible(plan):
        raise ValueError('motor rejects plan as infeasible')
    return plan


def _advance_motion(scene, ctrl, tape, state, motion, trace, history, phase,
                    deadline, *, stop_on_deadline=True):
    for block, _ in _blocks(motion):
        if stop_on_deadline and state.tick + 1 > deadline:
            break
        state = ctrl.advance(state, block, tape, trace=trace, phase=phase)
        if history is not None:
            history.append(state.recent[-1])
    return state


def _continuous_brake(scene, state):
    """Try checked brakes from actual q,v; return None when none is feasible."""
    accel = np.asarray(prior_run.ACCEL, dtype=float)
    v = np.asarray(state.v, dtype=float)
    stopping = v * np.abs(v) / (2. * accel)
    for factor in (1., 1.25, 1.5, 2.):
        target = np.clip(state.q + factor * stopping, *prior_run.legacy_control.CONTACT)
        plan = scene.motor.replan(state.q, v, target)
        if plan is None:
            continue
        if (np.max(np.abs(plan.x[0] - state.q)) <= 1.e-7 and
                np.max(np.abs(plan.v[0] - v)) <= 1.e-7 and
                np.max(np.abs(plan.v[-1])) <= 1.e-7):
            return plan
    return None


def follow(scene, ctrl, tape, state, motion, trace, history=None, phase='feedback_probe',
           *, deadline, policy: Callable | None = None, max_interventions=1,
           max_dwell=12):
    """Follow a motion while scoring truthful images and allowing local control.

    ``policy(public, q, v, tick)`` returns ``{'action': 'continue'|'brake'|
    'micro', 'delta': optional four-vector, 'score': optional scalar,
    'dwell': optional frame count}``. ``public`` contains only packet image
    fields. ``micro`` first brakes, then replans a bounded local recenter/tilt
    from the actual stopped pose. Failed brakes leave the original suffix
    untouched. The return is ``(state, events)``.
    """
    events, interventions = [], 0
    initial_tick = int(state.tick)
    source = motion
    for block, end_index in _blocks(source):
        if state.tick >= deadline:
            events.append({'event': 'deadline', 'tick': int(state.tick)})
            break
        # Every planned block begins at the currently executed q,v exactly.
        pe = float(np.max(np.abs(block.x[0] - state.q)))
        ve = float(np.max(np.abs(block.v[0] - state.v)))
        if pe > 1.e-7 or ve > 1.e-7:
            raise RuntimeError('feedback block does not join current q,v')
        state = ctrl.advance(state, block, tape, trace=trace, phase=phase)
        packet = state.recent[-1]
        if history is not None:
            history.append(packet)
        directive = {} if policy is None else policy(
            {'tick': int(packet.tick), 'q': np.asarray(packet.q).copy(),
             'reading': np.asarray(packet.reading).copy(),
             'marks': np.asarray(packet.marks).copy()},
            np.asarray(state.q).copy(), np.asarray(state.v).copy(), int(state.tick)) or {}
        action = directive.get('action', 'continue')
        score = directive.get('score')
        if score is not None:
            events.append({'event': 'score', 'tick': int(state.tick), 'score': float(score)})
        if action == 'continue' or interventions >= max_interventions:
            continue
        if action not in ('brake', 'micro'):
            raise ValueError(f"unknown feedback action: {action}")
        brake = _continuous_brake(scene, state)
        if brake is None:
            events.append({'event': 'brake_unavailable_resume_suffix', 'tick': int(state.tick),
                           'action': action})
            continue
        remain = int(deadline - state.tick)
        brake_frames = (len(brake.x) - 1) // FRAME_STEPS
        if brake_frames > remain:
            events.append({'event': 'brake_over_deadline_resume_suffix', 'tick': int(state.tick),
                           'brake_frames': brake_frames})
            continue
        interventions += 1
        state = _advance_motion(scene, ctrl, tape, state, brake, trace, history,
                                phase + '_brake', deadline)
        events.append({'event': 'braked', 'tick': int(state.tick), 'from_tick': int(packet.tick),
                       'action': action})
        if np.max(np.abs(state.v)) > 1.e-7:
            raise RuntimeError('checked brake did not end at rest')

        if action == 'micro':
            delta = np.asarray(directive.get('delta', np.zeros(4)), dtype=float)
            if delta.shape != (4,) or not np.isfinite(delta).all():
                raise ValueError('micro delta must be a finite four-vector')
            # Tight pose cap; tilt remains small and uses current, measured pose.
            delta = np.clip(delta, (-6., -3., -5., -4.), (6., 3., 5., 4.))
            target = np.clip(state.q + delta, *prior_run.legacy_control.CONTACT)
            local = scene.motor.replan(state.q, state.v, target)
            if local is not None and (len(local.x)-1)//FRAME_STEPS <= deadline-state.tick:
                if (np.max(np.abs(local.x[0]-state.q)) > 1.e-7 or
                        np.max(np.abs(local.v[0]-state.v)) > 1.e-7):
                    raise RuntimeError('micro-replan does not join measured q,v')
                state = _advance_motion(scene, ctrl, tape, state, local, trace, history,
                                        phase + '_micro', deadline)
                events.append({'event': 'micro_recenter', 'tick': int(state.tick),
                               'delta': delta.tolist()})
            else:
                events.append({'event': 'micro_unavailable', 'tick': int(state.tick)})

        dwell = int(np.clip(directive.get('dwell', 0), 0, max_dwell))
        held = 0
        for _ in range(dwell):
            if state.tick >= deadline:
                break
            x = np.broadcast_to(state.q, (FRAME_STEPS + 1, 4)).copy()
            hold = Motion(x, np.zeros_like(x))
            state = ctrl.advance(state, hold, tape, trace=trace, phase=phase + '_dwell')
            if history is not None:
                history.append(state.recent[-1])
            held += 1
        if dwell:
            events.append({'event': 'dwell', 'frames': held, 'tick': int(state.tick)})
        # Stop after intervention; a stationary feedback endpoint is deliberate.
        break

    if state.tick > deadline:
        raise RuntimeError('feedback execution exceeded deadline')
    if state.tick > initial_tick and np.max(np.abs(state.v)) > 1.e-7:
        # Preserve the planned suffix when a final stop is not already present.
        # If the deadline cannot fit a brake, report it rather than falsifying rest.
        brake = _continuous_brake(scene, state)
        if brake is not None and (len(brake.x)-1)//FRAME_STEPS <= deadline-state.tick:
            state = _advance_motion(scene, ctrl, tape, state, brake, trace, history,
                                    phase + '_final_brake', deadline)
            events.append({'event': 'final_brake', 'tick': int(state.tick)})
        else:
            events.append({'event': 'moving_at_deadline_or_suffix_end', 'tick': int(state.tick),
                           'speed_max': float(np.max(np.abs(state.v)))})
    return state, events
