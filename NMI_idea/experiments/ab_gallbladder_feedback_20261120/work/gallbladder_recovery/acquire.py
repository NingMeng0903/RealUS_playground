"""Use landmark history to propose and verify reachable, stationary scan views."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import gzip
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from view_geometry import (
    AtlasViews, CONTACT, FRAME_STEPS, Motion, POSE_DISTANCE_SCALE, REGIONS,
    frame_count, measure_view, offsets_around, prior_run, settled_plan,
)
from gallbladder_belief import ORGAN_LIMIT, SENSOR


CONDITIONS = ('wide', 'stress')
HOLD_FRAMES = 18
MIN_HOLD_FRAMES = 12
MAX_STATIONS = 12
NOMINAL_RIB = np.zeros(2)


def prefix_scan(scene, controller, tape, trace, history, index):
    """Execute the same costal sweep and keep each usable public observation."""
    state = scene.at_start()
    final_angle, offset = prior_run.legacy_scene.SWEEPS[index]
    plan = prior_run.sweep_motion(scene.motor, scene.envelope, scene.start, final_angle, offset)
    return execute(scene, controller, tape, state, plan, trace, history, 'liver_sweep')


def execute(scene, controller, tape, state, plan, trace, history, label):
    for i in range(FRAME_STEPS, len(plan.x), FRAME_STEPS):
        motion = Motion(plan.x[i-FRAME_STEPS:i+1], plan.v[i-FRAME_STEPS:i+1])
        state = controller.advance(state, motion, tape, trace=trace, phase=label)
        history.append(state.recent[-1])
    return state


def fit_anatomy(scene, state, history, previous):
    """Fit continuous template parameters to noisy labelled points from multiple views."""
    mean_liver, spread = scene.liver_prior.moment(state.liver)
    weights = scene.gall_weights(state)
    discrete = weights @ scene.parameters
    liver_seed = np.r_[mean_liver, discrete[3:]]
    marked = [p for p in history if np.isfinite(p.reading[0, 0]) and p.reading[0, 0] >= 45.
              and np.isfinite(p.marks[:, 0]).any()]
    if not marked:
        return liver_seed, discrete, 0

    retained = []
    for packet in marked:
        if (not retained or packet.tick - retained[-1].tick >= 3
                or np.isfinite(packet.marks[:, 0]).sum() > np.isfinite(retained[-1].marks[:, 0]).sum()):
            retained.append(packet)
    # Downsample the history by time so a single final cine cannot erase earlier
    # oblique views. All calibration uses the known registered chart.
    if len(retained) > 58:
        retained = [retained[i] for i in np.linspace(0, len(retained)-1, 58, dtype=int)]
    frames = [scene.body.view(p.q) for p in retained]
    marks = np.stack([p.marks for p in retained])
    time_id, region_id = np.where(np.isfinite(marks[:, :, 0]))
    measured = marks[time_id, region_id]
    origin = np.stack([f.origin for f in frames])[time_id]
    axes = np.stack([np.stack((f.right, f.forward, f.normal)) for f in frames])[time_id]
    nominal_points = np.stack([scene.nominal.landmarks[name] for name in REGIONS])
    neck = nominal_points[0]
    relative = nominal_points - neck
    axial = (relative @ scene.nominal.axis)[:, None] * scene.nominal.axis
    transverse = relative - axial
    mark_sigma = 3.3 if scene.condition == 'wide' else 5.1
    prior_sd = np.r_[np.full(3, np.clip(spread * .75, 4.2, 8.)), 15., 11., .17, .19]

    def residual(p):
        length, width = np.exp(p[5:])
        rotation = Rotation.from_euler('yx', p[3:5], degrees=True).as_matrix()
        landmarks = neck + p[:3] + (length * axial + width * transverse) @ rotation.T
        relative_to_probe = landmarks[region_id] - origin
        reprojected = np.einsum('nki,ni->nk', axes, relative_to_probe)
        delta = (reprojected[:, :2] - measured) / mark_sigma
        thickness = np.maximum(np.abs(reprojected[:, 2]) - 2.5, 0.) / 7.5
        return np.r_[delta.ravel(), thickness, (p-liver_seed)/prior_sd]

    initial = previous if previous is not None else liver_seed
    fit = least_squares(residual, np.clip(initial, -ORGAN_LIMIT, ORGAN_LIMIT),
                        bounds=(-ORGAN_LIMIT, ORGAN_LIMIT), x_scale=ORGAN_LIMIT,
                        max_nfev=25, loss='soft_l1', f_scale=1.5)
    return fit.x, discrete, len(measured)


def revisit_candidates(history):
    """Retain useful observations from the entire initial sweep and later probes."""
    options = []
    for p in history:
        area, center, _, span, _ = p.reading[0]
        if not np.isfinite(area) or area < 70. or not np.isfinite(span):
            continue
        marks = np.isfinite(p.marks[:, 0]).sum()
        if marks == 0:
            continue
        quality = .66 * marks + .023 * min(span, 82.) - .035 * abs(center)
        options.append((quality, p.q))
    selected = []
    for _, q in sorted(options, key=lambda p: -p[0]):
        if all(np.linalg.norm((q-old) * POSE_DISTANCE_SCALE) >= 4.5 for old in selected):
            selected.append(q)
        if len(selected) >= 8:
            break
    return selected


def propose_station(scene, state, search, anatomy, alternative, history, visited, stations, frames_left):
    """Shortlist diverse views cheaply and grade their full visible mesh sections."""
    body = scene.body
    regional = search.proposals(anatomy, current=state.q, limit=53)
    alternatives = search.proposals(alternative, current=state.q, limit=19)
    past = revisit_candidates(history)
    promising = sorted(stations, key=lambda s: -s['clip_strength'])[:2]
    revisits = [offsets_around(np.asarray(s['q'])) for s in promising if s['clip_strength'] > .4]
    poses = np.unique(np.round(np.vstack((regional, alternatives, offsets_around(state.q),
                                        np.asarray(past).reshape(-1, 4), *revisits)), 6), axis=0)
    start_position = body.surface(state.q)
    distances = np.linalg.norm(body.surface(poses) - start_position, axis=1)
    motion_distance = np.linalg.norm((poses-state.q) * POSE_DISTANCE_SCALE, axis=1)
    if visited:
        previous = np.asarray(visited)
        delta = (poses[:, None, :] - previous[None, :, :]) * POSE_DISTANCE_SCALE
        repetition = np.exp(-.5 * np.sum(delta**2, axis=-1) / 9.**2).sum(axis=1)
    else:
        repetition = np.zeros(len(poses))
    nearby_data = np.zeros(len(poses))
    for station in promising:
        delta = (poses - station['q']) * POSE_DISTANCE_SCALE
        neighborhood = np.exp(-.5 * np.sum(delta**2, axis=1) / 24.**2)
        nearby_data = np.maximum(nearby_data, station['clip_strength'] * neighborhood)
    cutoff = (distances / 49. * 1.9 * 10. < frames_left) & (motion_distance >= 1.0)
    metrics = {}

    def score(q):
        key = tuple(q)
        if key not in metrics:
            metrics[key] = measure_view(body, q, anatomy, NOMINAL_RIB)
        return metrics[key]

    def ranking(i):
        metric = score(poses[i])
        return metric.score - .012 * distances[i] - .005 * motion_distance[i] - .92 * repetition[i] + .65 * nearby_data[i]

    order = sorted(np.flatnonzero(cutoff), key=ranking, reverse=True)
    if not order:
        return None
    # Grading a second public anatomical hypothesis reduces substitution of a
    # perfect view for one posterior mode for a view that survives uncertainty.
    def robust_rank(i):
        alt = measure_view(body, poses[i], alternative, NOMINAL_RIB)
        return ranking(i) + .32 * alt.score

    order = sorted(order[:20], key=robust_rank, reverse=True)
    for i in order:
        if repetition[i] > 1.85 or any(np.linalg.norm((poses[i]-v)*POSE_DISTANCE_SCALE) < 2. for v in visited):
            continue
        q = poses[i]
        plan = settled_plan(scene.motor, state.q, q, frames_left)
        if plan is not None:
            return q, plan, score(q)
    return None


def clip_evidence(packets):
    seen = [p for p in packets if np.isfinite(p.reading[0, 0]) and p.reading[0, 0] >= 45.]
    if not seen:
        return 0, np.zeros(3), None, 0.
    marks = np.stack([p.marks for p in seen])
    repeats = np.zeros(3)
    for region in range(3):
        points = marks[:, region]
        points = points[np.isfinite(points[:, 0])]
        if len(points):
            center = np.median(points, axis=0)
            repeats[region] = np.sum(np.linalg.norm(points-center, axis=1) <= 13.) / len(seen)
    centroid = float(np.median([p.reading[0, 1] for p in seen]))
    strength = .45 * repeats.mean() + .55 * repeats.min()
    strength *= np.exp(-.5 * (centroid / 22.)**2) * min(len(seen) / 7., 1.)
    return len(seen), repeats, centroid, float(strength)


def stationary_confirmation(scene, ctrl, state, tape, trace, history, deadline, predicted):
    hold = np.broadcast_to(state.q, (FRAME_STEPS+1, 4)).copy()
    motion = Motion(hold, np.zeros_like(hold))
    arrival = state.tick
    proposed = {}
    clip = []
    for i in range(min(HOLD_FRAMES, deadline-state.tick)):
        state = execute(scene, ctrl, tape, state, motion, trace, history, 'verify_station')
        clip.append(history[-1])
        accepted = prior_run.public_confirm(state, scene)
        if accepted is not None and accepted.tick > arrival:
            proposed[accepted.tick] = accepted
        if i + 1 < MIN_HOLD_FRAMES:
            continue
        count, repeats, centroid, strength = clip_evidence(clip)
        min_repeat = .54 if scene.condition == 'wide' else .50
        supported = count >= 8 and repeats.min() >= min_repeat and abs(centroid) <= 11.
        if supported and (len(proposed) >= 2 or (len(proposed) and predicted.qualifies)):
            best = min(proposed.values(), key=lambda p: abs(p.reading[0, 1]))
            return state, best, repeats, strength
    _, repeats, _, strength = clip_evidence(clip)
    return state, None, repeats, strength


def run_recovery(scene, ctrl, tape, seed):
    started = perf_counter()
    trace, history, visited, stations = [], [], [], []
    state = prefix_scan(scene, ctrl, tape, trace, history, seed % 3)
    prefix = state
    deadline = prefix.tick + prior_run.MAX_TICKS
    search = AtlasViews(scene.body)
    fitted = None
    result = None
    for index in range(MAX_STATIONS):
        frames_left = deadline - state.tick - MIN_HOLD_FRAMES
        if frames_left < 4:
            break
        fitted, discrete, nmarks = fit_anatomy(scene, state, history, fitted)
        anatomy, alternative = scene.body.anatomy(fitted), scene.body.anatomy(discrete)
        target = propose_station(scene, state, search, anatomy, alternative, history, visited, stations, frames_left)
        if target is None:
            break
        q, plan, predicted = target
        depart = state.tick
        state = execute(scene, ctrl, tape, state, plan, trace, history, 'seek_centered_long_axis')
        arrival = state.tick
        visited.append(q)
        state, result, repeats, strength = stationary_confirmation(scene, ctrl, state, tape, trace, history, deadline, predicted)
        stations.append({'index': index, 'depart': depart, 'arrival': arrival, 'depart_after_cine': state.tick,
                         'q': q.tolist(), 'observed_mark_count': nmarks, 'fitted_parameters': fitted.tolist(),
                         'predicted_strict': predicted.qualifies,
                         'predicted_image_centroid_x_mm': predicted.center,
                         'observed_region_repeat_fractions': repeats.tolist(),
                         'clip_strength': strength,
                         'submitted_tick': None if result is None else result.tick})
        if result is not None:
            break
    elapsed = perf_counter() - started
    assert state.tick <= deadline and np.max(np.abs(state.v)) < 1.e-7
    ctrl._accepted_frame = result
    ctrl._stop_reason = 'confirmed_at_station' if result is not None else 'stopped_without_confirmation'
    scoring_state = state if result is not None else replace(state, captures=())
    outcome = prior_run.enforce_execution_success(prior_run.score_offline(scene, scoring_state, trace, result), state, ctrl)
    outcome['trace'] = trace
    outcome['stations'] = stations
    outcome['simulated_scan_seconds'] = state.tick / 10.
    outcome['wall_compute_seconds'] = elapsed
    outcome['travel_mm'] = state.travel
    outcome['deadline_tick'] = deadline
    outcome['prefix_tick'] = prefix.tick
    outcome['uses_only_public_observations_for_decisions'] = True
    return outcome


def run_case(task):
    seed, condition_index, include_legacy = task
    condition = CONDITIONS[condition_index]
    scene, ctrl, tape = prior_run.configure(seed, condition, condition_index)
    new = run_recovery(scene, ctrl, tape, seed)
    row = {'seed': seed, 'condition': condition, 'stationary_search': new}
    if include_legacy:
        old = prior_run.run_case(seed, condition, condition_index)
        row['frozen_ordered'] = old['ordered_skill_shared_confirmation']
        row['frozen_mpc'] = old['baseline_old_mpc_shared_confirmation']
        old_prefix = row['frozen_ordered']['trace'][:old['shared_prefix_end_tick']]
        new_prefix = new['trace'][:new['prefix_tick']]
        public_observations_match = len(old_prefix) == len(new_prefix) and all(
            np.max(np.abs(np.asarray(a[:9], float) - np.asarray(b[:9], float))) < 1.e-7
            and a[17:28] == b[17:28]
            for a, b in zip(old_prefix, new_prefix)
        )
        if not public_observations_match:
            raise RuntimeError(f'initial sweep or observations differ for {seed}/{condition}')
        if old['absolute_deadline_tick'] != new['deadline_tick']:
            raise RuntimeError(f'budgets differ for {seed}/{condition}')
        row['shared_prefix_geometry_readouts_and_budget_verified'] = True
    return row


def summarize(rows):
    names = [key for key in ('stationary_search', 'frozen_ordered', 'frozen_mpc') if rows and key in rows[0]]
    metrics = {}
    for key in names:
        arms = {}
        for noise in ('both', *CONDITIONS):
            cases = [r[key] for r in rows if noise == 'both' or r['condition'] == noise]
            arms[noise] = {
                'n': len(cases), 'correct': sum(a['strict_success'] for a in cases),
                'wrong': sum(a['wrong_delivery'] for a in cases),
                'strict_acquired': sum(a['ever_strict_genuine'] for a in cases),
                'execution_failures': sum(a['execution_failure'] is not None for a in cases),
                'final_speed_nonzero': sum(a['final_velocity_max_abs'] >= 1.e-6 for a in cases),
                'travel_mm_mean': float(np.mean([a['travel_mm'] for a in cases])),
                'simulated_duration_mean_s': float(np.mean([a.get('simulated_scan_seconds', a.get('time_s')) for a in cases])),
            }
        metrics[key] = arms
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', type=int, required=True)
    parser.add_argument('--count', type=int, default=8)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--legacy', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    tasks = [(seed, i, args.legacy) for seed in range(args.start, args.start+args.count) for i in range(2)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=prior_run.initialize) as pool, gzip.open(args.output, 'wt') as f:
        for i, row in enumerate(pool.map(run_case, tasks), start=1):
            rows.append(row)
            f.write(json.dumps(row, allow_nan=False, separators=(',', ':'))+'\n')
            arm = row['stationary_search']
            print(i, row['seed'], row['condition'], 'correct', arm['strict_success'],
                  'wrong', arm['wrong_delivery'], 'seen', arm['ever_strict_genuine'],
                  'stations', len(arm['stations']), flush=True)
    summary = summarize(rows)
    args.output.with_name(args.output.name.removesuffix('.jsonl.gz')+'_summary.json').write_text(
        json.dumps(summary, indent=2, allow_nan=False)+'\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
