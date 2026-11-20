"""Freeze a first-option PSDP gate and evaluate it on independent anatomy roots."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from policy import UTILITY, select, view_features
from acquire import CONDITIONS
from engine import (Scanner, branch_for_choice, frame_count, prior_run, rewards, run_station, strict_frame,
                    endpoint)
from train import HERE, OUT, save_json, save_rows


GATE_START = 58658000
GATE_COUNT = 28
EVAL_START = 58659000
EVAL_COUNT = 40
LOCAL_EFFECT_MARGIN = 0.


def gate_features(scanner, point, target, plan, beliefs):
    state, scene = point.state, scanner.scene
    center, axis, length = scene.estimate(state)
    _, width = scene.liver_prior.moment(state.liver)
    weights = scene.gall_weights(state)
    observations = np.stack([view_features(p, length) for p in point.history[-20:]])
    pose_ids = np.linspace(0, len(plan.x)-1, 5, dtype=int)
    forecast = np.asarray([scene.project(plan.x[i])[-1] @ weights for i in pose_ids])
    return np.r_[state.q, target-state.q, frame_count(plan), center, axis, length, width,
                 (scanner.deadline-state.tick)/10., observations.mean(axis=0), observations.max(axis=0),
                 forecast, (weights**2).sum(), scene.condition == 'stress', beliefs['landmarks']]


def gate_case(seed):
    local = joblib.load(OUT/'frozen_local.joblib')
    examples = []
    for i, profile in enumerate(CONDITIONS):
        scanner = Scanner(seed, profile, 100+i)
        target, fitted, anatomy, diag = scanner.propose(scanner.initial, False, None, [])
        if target is None:
            continue
        pose, motion, prediction = target
        y, counts = [], []
        for use in ('off', 'fresh'):
            scanner.ctrl._motion_trace.clear()
            exit, interventions, _ = run_station(
                scanner.scene, scanner.ctrl, scanner.tape, scanner.initial, pose, motion,
                anatomy, prediction, scanner.deadline, controller=use, model=local['models']['fresh'],
                margin=local['margin'])
            # Both first options are judged after the same frozen public tail succeeds or times out.
            outcome = scanner.scan(False, 'fresh', local['models']['fresh'], margin=local['margin'],
                                   start_point=exit, previous_stations=[exit.station], previous_fit=fitted,
                                   with_diagnostic=False)
            y.append(outcome)
            counts.append({'cut_tick': int(exit.state.tick), 'first_interventions': len(interventions)})
        examples.append({'seed': seed, 'profile': profile, 'feature': gate_features(
            scanner, scanner.initial, pose, motion, diag).tolist(), 'outcomes': y,
            'fit_target': (rewards(y[1])-rewards(y[0])).tolist(), 'first_exits': counts})
    return examples


def hash_path(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze_config():
    relevant = ('belief.py', 'motion_feedback.py', 'curriculum.py', 'policy.py', 'engine.py', 'train.py', 'evaluate.py',
                '../gallbladder_recovery/acquire.py', '../gallbladder_recovery/view_geometry.py',
                '../gallbladder_intent/pilot.py', '../gallbladder_intent/landmark_repair.py')
    modelnames = ('frozen_local.joblib', 'reverse_policy.joblib', 'frozen_gate.joblib')
    return {'test_anatomies': [EVAL_START, EVAL_START+EVAL_COUNT], 'test_noise_streams': [120, 121],
            'gate_training_anatomies': [GATE_START, GATE_START+GATE_COUNT],
            'source_sha256': {f: hash_path(HERE/f) for f in relevant},
            'model_sha256': {f: hash_path(OUT/f) for f in modelnames}, 'utility': UTILITY.tolist(),
            'station_confirmation': 'original stationary gate, actual stopped view and exact accepted frame',
            'comparison': 'same probe-frame cadence and geometry; read vs masked ranker evaluated after one cloned exit'}


def train_gate(pool):
    rows = []
    for i, data in enumerate(pool.map(gate_case, range(GATE_START, GATE_START+GATE_COUNT)), 1):
        rows.extend(data)
        if i % 4 == 0 or i == GATE_COUNT:
            print('PSDP gate anatomies', i, '/', GATE_COUNT, 'cases', len(rows), flush=True)
    save_rows(OUT/'gate_supervision.jsonl.gz', rows)
    model = ExtraTreesRegressor(n_estimators=128, min_samples_leaf=5, max_features=.82,
                                n_jobs=6, random_state=904)
    x = np.asarray([r['feature'] for r in rows])
    y = np.asarray([r['fit_target'] for r in rows])
    model.fit(x, y).set_params(n_jobs=1)
    joblib.dump(model, OUT/'frozen_gate.joblib', compress=3)
    save_json(OUT/'freeze.json', freeze_config())
    print('frozen forward decision', len(rows), 'paired first continuations', flush=True)


def stats_outcome(p):
    return {k: p[k] for k in ('correct', 'wrong', 'seconds', 'travel', 'accepted_tick', 'end_tick')}


def replay_one(seed):
    artifacts = joblib.load(OUT/'frozen_local.joblib')
    reverse = joblib.load(OUT/'reverse_policy.joblib')['extended']
    model, old_model = artifacts['models']['fresh'], artifacts['models']['stale']
    margin = artifacts['margin']
    gate_model = joblib.load(OUT/'frozen_gate.joblib')
    rows = []
    for i, profile in enumerate(CONDITIONS):
        scanner = Scanner(seed, profile, 120+i)
        arms = {}
        saved = []

        def first_exit(query, options, anatomy):
            if not saved:
                saved.append((query, options, anatomy))

        arms['stationary'] = scanner.scan(False, 'off', collect=first_exit)
        arms['multimodal'] = scanner.scan(True, 'off')
        arms['hand_servo'] = scanner.scan(False, 'heuristic')
        arms['reverse_only'] = scanner.scan(False, 'fresh', reverse, margin=artifacts['reverse_margin'])
        arms['student_feedback'] = scanner.scan(False, 'fresh', model, margin=margin)
        arms['joint'] = scanner.scan(True, 'fresh', model, margin=margin)
        gate_predictions = []

        def gate(point, target, plan, diag, index):
            if index != 0:
                return True
            values = gate_model.predict(gate_features(scanner, point, target, plan, diag)[None])[0]
            gate_predictions.append(values.tolist())
            return bool(values @ UTILITY > LOCAL_EFFECT_MARGIN)

        arms['psdp'] = scanner.scan(False, 'fresh', model, margin=margin, gate=gate)
        effect = None
        if saved:
            query, options, anatomy = saved[0]
            # Same physical snapshot, same targets and motion authority. Only the selector's
            # own evidence snapshot differs; the low-level visual servo menu is common.
            fresh = select(scanner.scene, query, options, model, margin=margin)
            stale = select(scanner.scene, query, options, old_model, margin=margin, stale=True)
            table = {}
            for option in options:
                scanner.ctrl._motion_trace.clear()
                result = branch_for_choice(scanner.scene, scanner.ctrl, scanner.tape, query, option, anatomy)
                table[option.kind] = stats_outcome(endpoint(scanner.scene, result, scanner.tape,
                                                             query.state, query.deadline))
            effect = {'tick': int(query.state.tick), 'history_origin_tick': int(query.origin_tick),
                      'q': query.state.q.tolist(), 'v': query.state.v.tolist(), 'deadline': query.deadline,
                      'fresh_choice': fresh.kind, 'stale_choice': stale.kind, 'alternatives': table}
        rows.append({'seed': seed, 'profile': profile, 'prefix_tick': int(scanner.prefix.tick),
                     'prefix_q': scanner.prefix.q.tolist(), 'deadline': scanner.deadline,
                     'pre_gall_visible': any(np.isfinite(p.marks[:, 0]).any() for p in scanner.initial.history),
                     'gate_prediction': gate_predictions, 'same_exit': effect, 'arms': arms})
    return rows


def binary_interval(keys, diff):
    key = np.asarray(keys)
    diffs = np.asarray(diff, dtype=float)
    # Equal-sized anatomy groups for complete end-to-end pairing. For any selected
    # exit subset, resample clusters and retain their observed episode counts.
    sums = np.array([diffs[key == k].sum() for k in np.unique(key)])
    counts = np.array([int((key == k).sum()) for k in np.unique(key)])
    rng = np.random.default_rng(6920)
    picks = rng.integers(len(sums), size=(12000, len(sums)))
    estimates = sums[picks].sum(1)/counts[picks].sum(1)
    return {'percentage_points': float(diffs.mean()*100.), 'paired_95ci_pp': (100.*np.percentile(estimates, [2.5, 97.5])).tolist(),
            'win': int((diffs > 0).sum()), 'loss': int((diffs < 0).sum()), 'anatomies': len(sums)}


def summarize(rows):
    arms = list(rows[0]['arms'])
    report = {}
    for k in arms:
        y = [r['arms'][k] for r in rows]
        report[k] = {
            'n': len(rows), 'correct': sum(o['correct'] for o in y), 'wrong': sum(o['wrong'] for o in y),
            'ever': sum(o['ever'] for o in y), 'ever_after_prefix': sum(o['ever_after_prefix'] for o in y),
            'strict_at_cine': sum(o['strict_at_cine'] for o in y),
            'final_speed_nonzero': sum(o['end_speed'] >= 1.e-7 for o in y),
            'repairs': sum(len(o['interventions']) for o in y),
            'cases_with_redirect': sum(bool(o['interventions']) for o in y),
            'fraction_first_gate_off': sum(bool(o['gate_exposures']) and not o['gate_exposures'][0] for o in y)/len(y),
            'seconds_mean': float(np.mean([o['seconds'] for o in y])),
            'travel_mean': float(np.mean([o['travel'] for o in y])),
        }
    differences = {}
    for a, b in (('multimodal', 'stationary'), ('hand_servo', 'stationary'), ('reverse_only', 'stationary'),
                 ('student_feedback', 'stationary'), ('student_feedback', 'reverse_only'),
                 ('student_feedback', 'hand_servo'), ('joint', 'multimodal'), ('joint', 'stationary'),
                 ('psdp', 'stationary'), ('psdp', 'student_feedback')):
        diff = [int(r['arms'][a]['correct'])-int(r['arms'][b]['correct']) for r in rows]
        differences[f'{a}_vs_{b}'] = binary_interval([r['seed'] for r in rows], diff)
    pairs = [r for r in rows if r['same_exit'] is not None]
    same = [{'seed': r['seed'], **r['same_exit']} for r in pairs]
    selected = lambda r, arm: r['alternatives'][r[arm+'_choice']]
    paired_report = None
    if same:
        diff = [int(selected(r, 'fresh')['correct'])-int(selected(r, 'stale')['correct']) for r in same]
        paired_report = {'n': len(same), 'fresh': sum(selected(r, 'fresh')['correct'] for r in same),
                         'stale': sum(selected(r, 'stale')['correct'] for r in same),
                         'continue': sum(r['alternatives']['continue']['correct'] for r in same),
                         'any_correct': sum(any(o['correct'] for o in r['alternatives'].values()) for r in same),
                         'changed': sum(r['fresh_choice'] != r['stale_choice'] for r in same),
                         'paired': binary_interval([r['seed'] for r in same], diff)}
    return {'arms': report, 'paired_correct': differences, 'same_exit_first_eligible': paired_report}


def test(pool):
    frozen = json.loads((OUT/'freeze.json').read_text())
    if frozen != freeze_config():
        raise RuntimeError('frozen controller, libraries or model files have changed')
    rows = []
    for i, data in enumerate(pool.map(replay_one, range(*frozen['test_anatomies'])), 1):
        rows.extend(data)
        if i % 4 == 0 or i == EVAL_COUNT:
            print('heldout anatomy', i, '/', EVAL_COUNT, flush=True)
    save_rows(OUT/'heldout.jsonl.gz', rows)
    report = summarize(rows)
    save_json(OUT/'heldout_summary.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('gate', 'test'))
    parser.add_argument('--workers', type=int, default=18)
    args = parser.parse_args()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=prior_run.initialize) as pool:
        (train_gate if args.phase == 'gate' else test)(pool)
