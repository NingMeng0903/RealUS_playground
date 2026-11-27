"""Closed-loop execution from the shared costal sweep or a saved moving exit."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from policy import (KINDS, QUERY_EVERY, QUERY_START, Choice, Query, action_targets, choice_features,
                    dist, frame_count, motion_and_cine, options_from_targets, select, visible_frame)
from acquire import (MAX_STATIONS, MIN_HOLD_FRAMES, NOMINAL_RIB, fit_anatomy, prefix_scan,
                     propose_station, stationary_confirmation)
from view_geometry import AtlasViews, FRAME_STEPS, Motion, measure_view, prior_run
from motion_feedback import advance_block, assert_admissible


@dataclass(frozen=True)
class SensorExit:
    state: object
    history: tuple
    accepted: object = None
    station: object = None


def station_cine(scene, ctrl, tape, point, anatomy, deadline, trace=None, prediction=None):
    state, history = point.state, list(point.history)
    arrived = state.tick
    if np.max(np.abs(state.v)) >= 1.e-7 or arrived > deadline-MIN_HOLD_FRAMES:
        raise RuntimeError('stationary cine did not have a zero-velocity, in-budget start')
    if prediction is None:
        prediction = measure_view(scene.body, state.q, anatomy, NOMINAL_RIB)
    state, accepted, repeats, strength = stationary_confirmation(
        scene, ctrl, state, tape, trace, history, deadline, prediction)
    if accepted is not None and (accepted.tick <= arrived or dist(accepted.q, state.q) > 1.e-7):
        raise RuntimeError('selected image was not acquired at rest in the current clip')
    stats = {'q': state.q.tolist(), 'clip_strength': float(strength), 'arrival': int(arrived),
             'depart_after_cine': int(state.tick), 'repeats': repeats.tolist(), 'kind': 'continue'}
    return SensorExit(state, tuple(history), accepted, stats)


def branch_for_choice(scene, ctrl, tape, query, choice, anatomy, trace=None, prediction=None):
    result = motion_and_cine(scene, ctrl, tape, query, choice, anatomy, trace, prediction)
    return SensorExit(result[0], tuple(result[1]), result[2], result[3])


def run_station(scene, ctrl, tape, point, target, plan, anatomy, prediction, deadline,
                *, controller='off', model=None, margin=0., collect=None, trace=None):
    """Visit scheduled image checkpoints until a public selector switches the tail."""
    initial = state = point.state
    history = list(point.history)
    assert_admissible(scene, state, plan, deadline-MIN_HOLD_FRAMES)
    decisions, queries = [], 0
    frames = frame_count(plan)
    for step in range(1, frames+1):
        state = advance_block(ctrl, tape, state, plan, step, trace, history, 'seek_centered_long_axis')
        if ((controller == 'off' and collect is None) or step < QUERY_START or step % QUERY_EVERY
                or frames-step < 2 or deadline-state.tick < MIN_HOLD_FRAMES+4):
            continue
        if not any(visible_frame(p) for p in history[-6:]):
            continue
        query = Query(state, tuple(history), Motion(plan.x[step*FRAME_STEPS:], plan.v[step*FRAME_STEPS:]),
                      initial.tick, deadline, initial)
        targets = action_targets(scene, query)
        if len(targets) < 2:
            continue
        options = options_from_targets(scene, query, targets)
        if len(options) < 2:
            continue
        queries += 1
        if collect is not None:
            collect(query, options, anatomy)
        if controller == 'off':
            continue
        choice = select(scene, query, options, model, stale=controller == 'stale', margin=margin)
        if choice.kind == 'continue':
            continue
        decisions.append({'query_tick': int(state.tick), 'kind': choice.kind,
                          'from_q': state.q.tolist(), 'from_v': state.v.tolist(),
                          'prior_goal': target.tolist(), 'target': choice.target.tolist(),
                          'planned_frames_left': frame_count(query.suffix),
                          'new_motion_frames': frame_count(choice.plan),
                          'allowed': [o.kind for o in options]})
        return branch_for_choice(scene, ctrl, tape, query, choice, anatomy, trace), decisions, queries
    return station_cine(scene, ctrl, tape, SensorExit(state, tuple(history)), anatomy,
                        deadline, trace, prediction), decisions, queries


def strict_frame(scene, packet, tape):
    """Offline truth evaluation of individual frames, independent of the public clip gate."""
    section, _, _ = scene.truth_packet(packet.q)
    if not section.qualified or abs(section.gall[1]) > 12.:
        return False
    frame = scene.body.view(packet.q, scene.case.rib)
    if abs(frame.normal @ scene.truth.axis) > np.sin(np.deg2rad(15.)):
        return False
    return bool(scene._offline_observe(packet.q, packet.tick, tape).genuine)


def endpoint(scene, point, tape, start, deadline):
    state = point.state
    stopped = np.max(np.abs(state.v)) < 1.e-7
    if state.tick > deadline or not stopped:
        raise RuntimeError('terminal scan breached deadline or had nonzero speed')
    correct = point.accepted is not None and strict_frame(scene, point.accepted, tape)
    return {'correct': bool(correct), 'wrong': point.accepted is not None and not correct,
            'seconds': float((state.tick-start.tick) / 10.), 'travel': float(state.travel-start.travel),
            'accepted_tick': None if point.accepted is None else int(point.accepted.tick),
            'end_tick': int(state.tick), 'end_q': state.q.tolist(), 'end_speed': float(np.max(np.abs(state.v)))}


def diagnostic(scene, point, tape, start, deadline, stations):
    out = endpoint(scene, point, tape, start, deadline)
    strict = [p.tick for p in point.history if strict_frame(scene, p, tape)]
    stopped = any(s['arrival'] < tick <= s['depart_after_cine'] for s in stations for tick in strict)
    out.update({'ever': bool(strict), 'ever_after_prefix': any(tick > start.tick for tick in strict),
                'strict_at_cine': bool(stopped), 'strict_ticks': strict})
    if out['correct'] and not stopped:
        raise RuntimeError('strict delivery lacks a genuine certified stationary frame')
    return out


def rewards(o):
    return np.asarray([o[k] for k in ('correct', 'wrong', 'seconds', 'travel')], dtype=float)


class Scanner:
    """One simulated body, one time-indexed sensor tape, one identical costal prefix."""

    def __init__(self, seed, profile, stream, *, trace_prefix=False):
        self.seed = seed
        self.scene, self.ctrl, self.tape = prior_run.configure(seed, profile, stream)
        self.search = AtlasViews(self.scene.body)
        history = []
        self.trace = [] if trace_prefix else None
        self.prefix = prefix_scan(self.scene, self.ctrl, self.tape, self.trace, history, seed % 3)
        self.initial = SensorExit(self.prefix, tuple(history))
        self.deadline = self.prefix.tick + prior_run.MAX_TICKS

    def propose(self, point, modes, previous, stations):
        fitted, discrete, n_mark = fit_anatomy(self.scene, point.state, point.history, previous)
        anatomy = self.scene.body.anatomy(fitted)
        visited = [np.asarray(s['q']) for s in stations]
        left = self.deadline-point.state.tick-MIN_HOLD_FRAMES
        diag = {'landmarks': n_mark}
        if modes:
            from belief import propose_multimodal

            multi = propose_multimodal(self.scene, point.state, self.search, fitted, point.history,
                                       visited, stations, left)
            if multi is not None:
                return multi[:3], fitted, anatomy, dict(diag, **multi[3])
        alternative = self.scene.body.anatomy(discrete)
        chosen = propose_station(self.scene, point.state, self.search, anatomy, alternative,
                                 point.history, visited, stations, left)
        return chosen, fitted, anatomy, diag

    def scan(self, modes, controller, model=None, *, margin=0., gate=None, collect=None,
             keep_trace=False, max_stations=MAX_STATIONS, start_point=None,
             previous_stations=(), previous_fit=None, with_diagnostic=True):
        start = perf_counter()
        point = self.initial if start_point is None else start_point
        stations, interventions, gates = list(previous_stations), [], []
        trace = list(self.trace) if keep_trace and self.trace is not None else ([] if keep_trace else None)
        fitted = previous_fit
        query_count = 0
        self.ctrl._motion_trace.clear()
        for index in range(len(stations), max_stations):
            if self.deadline - point.state.tick < MIN_HOLD_FRAMES+4 or point.accepted is not None:
                break
            previous = point
            target, fitted, anatomy, beliefs = self.propose(point, modes, fitted, stations)
            if target is None:
                break
            q, plan, prediction = target
            use = controller if gate is None or gate(point, q, plan, beliefs, index) else 'off'
            gates.append(use != 'off')
            point, events, queries = run_station(self.scene, self.ctrl, self.tape, point, q, plan,
                                                 anatomy, prediction, self.deadline, controller=use, model=model,
                                                 margin=margin, collect=collect, trace=trace)
            interventions.extend(events)
            query_count += queries
            stations.append(dict(point.station, departed=int(previous.state.tick), expected=q.tolist(),
                                 used_mode_count=beliefs.get('mode_count', 0)))
        if with_diagnostic:
            out = diagnostic(self.scene, point, self.tape, self.prefix, self.deadline, stations)
        else:
            out = endpoint(self.scene, point, self.tape, self.prefix, self.deadline)
        out.update({'interventions': interventions, 'queries': query_count, 'station_log': stations,
                    'gate_exposures': gates, 'wall_seconds': perf_counter()-start})
        if keep_trace:
            out['trace'] = trace
        return out
