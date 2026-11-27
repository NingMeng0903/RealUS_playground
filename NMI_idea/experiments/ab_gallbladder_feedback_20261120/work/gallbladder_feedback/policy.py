"""Public short feedback options for a costal scan with a checked stopping tail."""

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'gallbladder_recovery'))

from acquire import HOLD_FRAMES, MIN_HOLD_FRAMES, NOMINAL_RIB, stationary_confirmation
from view_geometry import CONTACT, FRAME_STEPS, Motion, POSE_DISTANCE_SCALE, frame_count, measure_view, prior_run


QUERY_EVERY = 3
QUERY_START = 3
KINDS = ('continue', 'keep_view', 'center', 'fan_left', 'fan_right', 'recall')
UTILITY = np.array([1., -.8, -.006, -.00025])


@dataclass(frozen=True)
class Query:
    state: object
    history: tuple
    suffix: Motion
    origin_tick: int
    deadline: int
    origin_state: object = None


@dataclass(frozen=True)
class Choice:
    kind: str
    target: np.ndarray
    plan: Motion


def dist(a, b):
    return np.linalg.norm((a-b) * POSE_DISTANCE_SCALE, axis=-1)


def visible_frame(packet):
    reading = packet.reading[0]
    return (np.isfinite(reading[:4]).all() and reading[0] >= 45.
            and np.isfinite(packet.marks[:, 0]).any())


def view_features(packet, length):
    read = packet.reading[0]
    visible = np.isfinite(packet.marks[:, 0])
    area, x, depth, span = np.nan_to_num(read[:4], nan=0.)
    lateral = np.tanh(x / 25.)
    marks = packet.marks
    if visible.all():
        axis = marks[0]-marks[2]
        normalized_length = np.linalg.norm(axis)
        fraction = np.clip((marks[1]-marks[2]) @ axis / max(normalized_length**2, 1.), -1., 2.)
        residual = np.linalg.norm(marks[1]-marks[2] - fraction*axis)
    else:
        normalized_length = fraction = residual = 0.
    return np.r_[visible_frame(packet), visible, np.clip(area/400., 0., 3.), lateral, depth/100.,
                 np.clip(span/length, 0., 2.), normalized_length/length, fraction, residual/20.]


def image_rank(packet, length):
    f = view_features(packet, length)
    if not f[0]:
        return 0.
    marks = f[1:4]
    ordered_triplet = marks.all() and -.1 <= f[9] <= 1.1 and f[10] <= .75 and f[8] >= .50
    return (marks.sum() + .65 * ordered_triplet + .8 * min(f[7], 1.15)
            + .20 * min(f[4], 1.) - .75 * abs(f[5]))


def anchors(state, history, length):
    nearby = [p for p in history[-24:] if visible_frame(p) and dist(p.q, state.q) <= 25.]
    current = [p for p in nearby if state.tick-p.tick <= 5 and dist(p.q, state.q) <= 15.]

    def rating(p):
        return image_rank(p, length) - .032 * (state.tick-p.tick) - .034 * dist(p.q, state.q)

    if current:
        primary = max(current, key=rating)
        others = [p for p in nearby if dist(p.q, primary.q) >= 4.5]
        return primary, max(others, key=rating) if others else None
    return None, max(nearby, key=rating) if nearby else None


def center_preserving_target(scene, packet, delta_angles=(0., 0.)):
    """Lift the measured in-plane centroid and locally adjust its projection."""
    start = packet.q
    view = scene.body.view(start)
    _, x, depth, _, _ = packet.reading[0]
    target_point = view.origin + x*view.right + depth*view.forward
    target = np.clip(start + np.r_[0., 0., delta_angles], *CONTACT)

    def lateral(q):
        frame = scene.body.view(q)
        return float((target_point-frame.origin) @ frame.right)

    prediction = lateral(target)
    perturb = np.array([[1., 0., 0., 0.], [0., .65, 0., 0.]])
    jac = np.asarray([(lateral(np.clip(target + step, *CONTACT)) -
                       lateral(np.clip(target - step, *CONTACT))) / 2. for step in perturb])
    # Damping and a trust region prevent a singular image Jacobian from proposing a chart jump.
    change = np.clip(-jac * prediction / (jac @ jac + .23), [-7., -4.5], [7., 4.5])
    target[:2] += change * [1., .65]
    return np.clip(target, *CONTACT)


def action_targets(scene, query):
    """Both image ablations use these same controls, including local visual servos."""
    state, history = query.state, query.history
    _, _, length = scene.estimate(state)
    primary, previous = anchors(state, history, length)
    targets = {'continue': query.suffix.x[-1]}
    if primary is not None:
        targets['keep_view'] = primary.q
        targets['center'] = center_preserving_target(scene, primary)
        targets['fan_left'] = center_preserving_target(scene, primary, (-5.5, -3.))
        targets['fan_right'] = center_preserving_target(scene, primary, (5.5, 3.))
    if previous is not None:
        targets['recall'] = previous.q
    return targets


def options_from_targets(scene, query, targets):
    from motion_feedback import feasible_redirect

    state = query.state
    options = [Choice('continue', query.suffix.x[-1], query.suffix)]
    frames_left = min(30, query.deadline - state.tick - HOLD_FRAMES)
    for kind, target in targets.items():
        if kind == 'continue' or frames_left < 4:
            continue
        if any(dist(target, c.target) < 1.3 for c in options):
            continue
        plan = feasible_redirect(scene, state, target, frames_left)
        if plan is not None:
            options.append(Choice(kind, target, plan))
    return options


def choice_features(scene, query, options, *, stale=False):
    state, hist = query.state, query.history
    if stale:
        hist = tuple(p for p in hist if p.tick <= query.origin_tick)
    evidence_state = query.origin_state if stale and query.origin_state is not None else state
    _, _, length = scene.estimate(evidence_state)
    window = hist[-8:]
    block = np.asarray([view_features(p, length) for p in window]).reshape(-1, 11)
    last = block[-1] if len(block) else np.zeros(11)
    average = block.mean(axis=0) if len(block) else np.zeros(11)
    recent_peak = max((image_rank(p, length) for p in window), default=0.)
    current_age = state.tick-window[-1].tick if len(window) else 100.
    weights = scene.gall_weights(evidence_state)
    prefix = np.r_[state.q, state.v, length, (query.deadline-state.tick) / 10.,
                   frame_count(query.suffix), recent_peak, current_age / 10., last, average,
                   np.linalg.norm(state.v * POSE_DISTANCE_SCALE), scene.condition == 'stress']
    rows = []
    for option in options:
        target = option.target
        _, support, quality = scene.project(target)
        nearby = [p for p in window if dist(target, p.q) <= 9.5]
        evidence = max((image_rank(p, length) - .05 * dist(target, p.q) for p in nearby), default=0.)
        rows.append(np.r_[prefix, target-state.q, frame_count(option.plan), evidence,
                          weights @ support, weights @ quality,
                          [option.kind == name for name in KINDS]])
    return np.stack(rows)


def heuristic_kind(scene, query, options):
    """Greedy public servo used as a fixed feedback comparator."""
    _, _, length = scene.estimate(query.state)
    primary, _ = anchors(query.state, query.history, length)
    if primary is not None and image_rank(primary, length) >= 3.4:
        available = {o.kind for o in options}
        if abs(primary.reading[0, 1]) >= 9. and 'center' in available:
            return 'center'
        if 'keep_view' in available:
            return 'keep_view'
    return 'continue'


def select(scene, query, options, model, *, stale=False, margin=0.):
    if len(options) < 2:
        return options[0]
    if model is None:
        kind = heuristic_kind(scene, query, options)
        return next(option for option in options if option.kind == kind)
    values = model.predict(choice_features(scene, query, options, stale=stale)) @ UTILITY
    values[0] = margin
    return options[int(np.argmax(values))]


def motion_and_cine(scene, ctrl, tape, query, choice, anatomy, trace=None, prediction=None):
    from acquire import execute

    history = list(query.history)
    state = execute(scene, ctrl, tape, query.state, choice.plan, trace, history, 'short_' + choice.kind)
    arrival = state.tick
    if arrival > query.deadline or np.max(np.abs(state.v)) > 1.e-7:
        raise RuntimeError('short option did not reach a safe terminal state before the deadline')
    if query.deadline - arrival < MIN_HOLD_FRAMES:
        raise RuntimeError('short option used the reserve for confirmation')
    predicted = measure_view(scene.body, state.q, anatomy, NOMINAL_RIB) if prediction is None else prediction
    state, accepted, repeats, strength = stationary_confirmation(
        scene, ctrl, state, tape, trace, history, query.deadline, predicted)
    if accepted is not None and (accepted.tick <= arrival or dist(accepted.q, state.q) > 1.e-7):
        raise RuntimeError('selected image was not acquired at rest in its current clip')
    record = {'q': state.q.tolist(), 'clip_strength': float(strength), 'arrival': int(arrival),
              'depart_after_cine': int(state.tick), 'repeats': repeats.tolist(), 'kind': choice.kind}
    return state, history, accepted, record
