"""Offline search for a feasible path, strict view, and stationary dwell per case."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from view_geometry import AtlasViews, FRAME_STEPS, Motion, frame_count, measure_view, offsets_around, prior_run, settled_plan


def find_witness(scene, state):
    body = scene.body
    search = AtlasViews(body)
    anatomy, rib = scene.truth, scene.case.rib
    poses = search.proposals(anatomy, current=state.q, limit=150)
    tested = {}

    def inspect(q):
        key = tuple(q)
        if key not in tested:
            tested[key] = measure_view(body, q, anatomy, rib)
        return tested[key]

    for q in poses:
        inspect(q)
    best = sorted(tested, key=lambda q: tested[q].score, reverse=True)
    # Refine the best sections, including small coupled translations/tilts.
    for q in best[:12]:
        for target in offsets_around(np.asarray(q)):
            inspect(target)
    if not any(m.qualifies for m in tested.values()):
        best = sorted(tested, key=lambda q: tested[q].score, reverse=True)
        for q in best[:12]:
            for target in offsets_around(np.asarray(q)):
                inspect(target)
    strict = [q for q, data in tested.items() if data.qualifies]
    strict.sort(key=lambda q: -tested[q].score)
    for key in strict:
        target = np.asarray(key)
        plan = settled_plan(scene.motor, state.q, target, prior_run.MAX_TICKS - 20)
        if plan is not None:
            return plan, target, tested[key], len(tested), len(strict)
    return None, None, None, len(tested), len(strict)


def run_case(seed):
    scene, _, tape = prior_run.configure(seed, 'wide', 0)
    trace = []
    state = scene.start_prefix(seed % 3, tape, trace=trace)
    plan, q, quality, tested, strict = find_witness(scene, state)
    result = {
        'seed': seed, 'from_tick': state.tick, 'from_q': state.q.tolist(),
        'tested_pose_count': tested, 'strict_pose_count': strict, 'path_found': plan is not None,
        'privileged_anatomy_and_ribs_used_for_offline_search': True,
    }
    if plan is None:
        return result
    result['target_q'] = q.tolist()
    result['truth_geometry'] = asdict(quality)
    result['motion_ticks'] = frame_count(plan)
    result['path_length_mm'] = float(np.linalg.norm(np.diff(scene.body.surface(plan.x), axis=0), axis=1).sum())
    result['verified_motor'] = bool(scene.motor.feasible(plan))
    result['final_speed_max'] = float(np.max(np.abs(plan.v[-1])))
    result['native_replan_exists'] = scene.motor.replan(state.q, state.v, q) is not None
    result['dwell'] = {}
    for stream, condition in enumerate(('wide', 'stress')):
        other, ctrl, other_tape = prior_run.configure(seed, condition, stream)
        records = []
        s = other.start_prefix(seed % 3, other_tape, trace=records)
        assert s.tick == state.tick and np.max(np.abs(s.q - plan.x[0])) < 1.e-7
        for i in range(FRAME_STEPS, len(plan.x), FRAME_STEPS):
            s = ctrl.advance(s, Motion(plan.x[i-FRAME_STEPS:i+1], plan.v[i-FRAME_STEPS:i+1]),
                             other_tape, phase='witness_transit')
        arrival_tick = s.tick
        idle = np.broadcast_to(q, (FRAME_STEPS+1, 4)).copy()
        hold = Motion(idle, np.zeros_like(idle))
        correct = []
        submissions = []
        for _ in range(20):
            s = ctrl.advance(s, hold, other_tape, phase='witness_dwell')
            packet = other._offline_observe(q, s.tick, other_tape)
            if packet.genuine:
                correct.append(s.tick)
            accepted = prior_run.public_confirm(s, other)
            if accepted is not None and accepted.tick > arrival_tick and accepted.tick not in submissions:
                submissions.append(accepted.tick)
        result['dwell'][condition] = {
            'arrival_tick': arrival_tick, 'end_tick': s.tick,
            'deadline_tick': state.tick + prior_run.MAX_TICKS,
            'genuine_detection_ticks': correct, 'old_public_confirmation_ticks': submissions,
            'first_old_confirmation_strict': bool(submissions and submissions[0] in correct),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', type=int, default=58650000)
    parser.add_argument('--count', type=int, default=32)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    seeds = range(args.start, args.start + args.count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers, initializer=prior_run.initialize) as pool:
        cases = []
        for row in pool.map(run_case, seeds):
            cases.append(row)
            print(f"{row['seed']} path={row['path_found']} strict={row['strict_pose_count']} tested={row['tested_pose_count']}", flush=True)
    summary = {
        'cases': len(cases), 'strict_pose_found': sum(row['strict_pose_count'] > 0 for row in cases),
        'feasible_trajectory_found': sum(row['path_found'] for row in cases),
        'max_motion_ticks': max((row.get('motion_ticks', 0) for row in cases), default=0),
        'max_final_speed': max((row.get('final_speed_max', 0.) for row in cases), default=0.),
        'noise_runs_with_genuine_detection_at_witness': sum(bool(d['genuine_detection_ticks']) for row in cases
                                                         for d in row.get('dwell', {}).values()),
        'noise_runs_first_old_confirmation_correct': sum(d['first_old_confirmation_strict'] for row in cases
                                                       for d in row.get('dwell', {}).values()),
    }
    args.output.write_text(json.dumps({'summary': summary, 'cases': cases}, indent=2, allow_nan=False)+'\n')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
