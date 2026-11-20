"""Fit one guarded correction to the frozen public hand servo using full scan returns."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from policy import UTILITY, choice_features, heuristic_kind
from acquire import CONDITIONS, MAX_STATIONS, MIN_HOLD_FRAMES
from engine import Scanner, branch_for_choice, endpoint, prior_run, rewards, run_station
from evaluate import binary_interval, freeze_config, hash_path
from train import HERE, OUT, save_json, save_rows


FIT_START, FIT_COUNT = 58661000, 36
CAL_START, CAL_COUNT = 58662000, 16
TEST_START, TEST_COUNT = 58663000, 32
NOISE_STREAMS = {'fit': 130, 'calibrate': 140, 'test': 150}


def first_hand_decision(scanner):
    point, stations, fitted = scanner.initial, [], None
    for _ in range(MAX_STATIONS):
        if point.accepted is not None or scanner.deadline-point.state.tick < MIN_HOLD_FRAMES+4:
            break
        target, fitted, anatomy, _ = scanner.propose(point, False, fitted, stations)
        if target is None:
            break
        pose, plan, prediction = target
        found = []

        def recall(query, options, _):
            kind = heuristic_kind(scanner.scene, query, options)
            if not found and kind != 'continue':
                found.append((query, options, kind))

        point, events, _ = run_station(scanner.scene, scanner.ctrl, scanner.tape, point, pose, plan,
                                       anatomy, prediction, scanner.deadline, controller='heuristic', collect=recall)
        if found:
            query, choices, kind = found[0]
            if len(events) != 1 or events[0]['kind'] != kind or events[0]['query_tick'] != query.state.tick:
                raise RuntimeError('saved exit was not reached by the original feedback controller')
            return query, choices, kind, anatomy, fitted, list(stations), point
        stations.append(point.station)
    return endpoint(scanner.scene, point, scanner.tape, scanner.prefix, scanner.deadline)


def predict_index(model, row, margin):
    baseline = row['baseline_index']
    if margin is None or baseline is None:
        return baseline
    estimate = model.predict(row['feature'])
    value = estimate @ UTILITY
    # Suppress candidate transitions that predict increased erroneous delivery.
    value[estimate[:, 1] > .04] = -np.inf
    value[baseline] = margin
    return int(np.argmax(value))


def case_rollouts(job):
    seed, phase = job
    rows = []
    fit = joblib.load(OUT/'residual_policy.joblib') if phase == 'test' else None
    for i, profile in enumerate(CONDITIONS):
        scanner = Scanner(seed, profile, NOISE_STREAMS[phase]+i)
        scout = first_hand_decision(scanner)
        row = {'seed': seed, 'profile': profile, 'baseline_index': None}
        if phase == 'test':
            row['stationary'] = scanner.scan(False, 'off', with_diagnostic=False)
        if isinstance(scout, dict):
            row.update({'hand': scout, 'residual': scout, 'zero_margin': scout})
            rows.append(row)
            continue
        query, choices, baseline, anatomy, fitted, stations, baseline_point = scout
        actions = [a.kind for a in choices]
        b = actions.index(baseline)
        row.update({'baseline_index': b, 'kinds': actions, 'tick': int(query.state.tick),
                    'q': query.state.q.tolist(), 'v': query.state.v.tolist(), 'deadline': query.deadline,
                    'feature': choice_features(scanner.scene, query, choices).tolist()})
        if phase == 'test':
            chosen = predict_index(fit['model'], row, fit['margin'])
            eager = predict_index(fit['model'], row, 0.)
        outputs = []
        for choice in choices:
            point = baseline_point if choice.kind == baseline else branch_for_choice(
                scanner.scene, scanner.ctrl, scanner.tape, query, choice, anatomy)
            output = scanner.scan(False, 'heuristic', start_point=point, previous_fit=fitted,
                                  previous_stations=[*stations, point.station], with_diagnostic=False)
            outputs.append({k: output[k] for k in ('correct', 'wrong', 'seconds', 'travel', 'accepted_tick', 'end_tick')})
        row.update({'outcomes': outputs, 'hand': outputs[b]})
        if phase == 'test':
            row.update({'residual': outputs[chosen], 'zero_margin': outputs[eager], 'chosen_index': chosen,
                        'eager_index': eager})
        rows.append(row)
    return rows


def run_set(pool, name, start, count):
    rows = []
    for j, result in enumerate(pool.map(case_rollouts, [(seed, name) for seed in range(start, start+count)]), start=1):
        rows.extend(result)
        if j % 4 == 0 or j == count:
            print('residual', name, j, '/', count, flush=True)
    return rows


def learn(pool):
    rows = run_set(pool, 'fit', FIT_START, FIT_COUNT)
    valid = run_set(pool, 'calibrate', CAL_START, CAL_COUNT)
    save_rows(OUT/'residual_train.jsonl.gz', rows)
    save_rows(OUT/'residual_calibration.jsonl.gz', valid)
    fit = [r for r in rows if r['baseline_index'] is not None]
    x = np.vstack([r['feature'] for r in fit])
    y = np.vstack([np.stack([rewards(o)-rewards(r['hand']) for o in r['outcomes']]) for r in fit])
    model = ExtraTreesRegressor(n_estimators=128, min_samples_leaf=8, max_features=.82,
                                n_jobs=6, random_state=2023).fit(x, y).set_params(n_jobs=1)
    choices = []
    # Validation favors returning to the real public default in any utility tie.
    for margin in (None, .25, .15, .10, .06, .025, 0.):
        decisions = [predict_index(model, r, margin) for r in valid]
        evaluated = [r['outcomes'][k] if k is not None else r['hand'] for r, k in zip(valid, decisions)]
        differences = [rewards(o)-rewards(r['hand']) for o, r in zip(evaluated, valid)]
        gains = np.stack(differences)
        wrong_delta = int(gains[:, 1].sum())
        choices.append({'margin': margin, 'utility_delta': float((gains @ UTILITY).mean()),
                        'wrong_delta': wrong_delta, 'correct_delta': int(gains[:, 0].sum()),
                        'changed': sum(k != r['baseline_index'] for r, k in zip(valid, decisions))})
    top = max([c for c in choices if c['wrong_delta'] <= 0], key=lambda r: r['utility_delta'])
    joblib.dump({'model': model, 'margin': top['margin']}, OUT/'residual_policy.joblib', compress=3)
    lock = {'secondary_study': 'one correction to the first rule-based decision using whole-scan returns',
            'fit_range': [FIT_START, FIT_START+FIT_COUNT], 'calibration_range': [CAL_START, CAL_START+CAL_COUNT],
            'test_range': [TEST_START, TEST_START+TEST_COUNT], 'streams': NOISE_STREAMS,
            'labels': len(y), 'paired_training_exits': len(fit), 'calibration': choices, 'chosen': top,
            'software_sha256': hash_path(HERE/'residual.py'), 'policy_sha256': hash_path(OUT/'residual_policy.joblib')}
    save_json(OUT/'residual_freeze.json', lock)
    print('residual frozen', lock, flush=True)


def test(pool):
    if json.loads((OUT/'freeze.json').read_text()) != freeze_config():
        raise RuntimeError('shared frozen policy and scripts changed')
    lock = json.loads((OUT/'residual_freeze.json').read_text())
    if lock['software_sha256'] != hash_path(HERE/'residual.py') or lock['policy_sha256'] != hash_path(OUT/'residual_policy.joblib'):
        raise RuntimeError('secondary study changed after freezing')
    rows = run_set(pool, 'test', *[lock['test_range'][0], lock['test_range'][1]-lock['test_range'][0]])
    arms, comparisons = {}, {}
    for arm in ('stationary', 'hand', 'residual', 'zero_margin'):
        y = [r[arm] for r in rows]
        arms[arm] = {'n': len(y), 'correct': sum(o['correct'] for o in y), 'wrong': sum(o['wrong'] for o in y),
                     'seconds': float(np.mean([o['seconds'] for o in y])),
                     'travel_mm': float(np.mean([o['travel'] for o in y]))}
    for a, b in (('hand', 'stationary'), ('residual', 'hand'), ('zero_margin', 'hand')):
        diff = [int(r[a]['correct'])-int(r[b]['correct']) for r in rows]
        comparisons[f'{a}_vs_{b}'] = binary_interval([r['seed'] for r in rows], diff)
    choices = [r for r in rows if r['baseline_index'] is not None]
    summary = {'arms': arms, 'comparisons': comparisons, 'eligible_hand_exits': len(choices),
               'changed': sum(r['chosen_index'] != r['baseline_index'] for r in choices),
               'changed_no_margin': sum(r['eager_index'] != r['baseline_index'] for r in choices),
               'cases_the_same_menu_can_rescue': sum(not r['hand']['correct'] and any(o['correct'] for o in r['outcomes']) for r in choices)}
    save_rows(OUT/'residual_holdout.jsonl.gz', rows)
    save_json(OUT/'residual_holdout_summary.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('fit', 'test'))
    p.add_argument('--workers', type=int, default=20)
    args = p.parse_args()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=prior_run.initialize) as pool:
        (learn if args.phase == 'fit' else test)(pool)
