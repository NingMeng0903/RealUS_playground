"""Build paired near-success and student-visited labels, then freeze the local policy."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from policy import (QUERY_EVERY, Query, UTILITY, action_targets, choice_features, dist, options_from_targets)
from acquire import CONDITIONS, fit_anatomy
from curriculum import reverse_checkpoints
from engine import Scanner, branch_for_choice, endpoint, rewards
from view_geometry import prior_run


HERE = Path(__file__).resolve().parent
OUT = HERE / 'results'
STUDENT_START = 58657000
STUDENT_COUNT = 32
STUDENT_VALIDATE = 8
REVERSE_LIMIT = 32
WITNESSES = HERE.parent / 'gallbladder_psdp_v2/results/holdout_witnesses.json'


def save_json(file, value):
    file.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def save_rows(file, rows):
    with gzip.open(file, 'wt', compresslevel=4) as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False, separators=(',', ':')) + '\n')


def read_rows(file):
    with gzip.open(file, 'rt') as handle:
        return [json.loads(line) for line in handle]


def label_options(scanner, stage, query, anatomy, options, *, witness=None):
    scene = scanner.scene
    if len(options) < 2:
        return None
    values, records = [], []
    # Each branch begins at precisely the same recorded position, velocity and clock.
    for choice in options:
        scanner.ctrl._motion_trace.clear()
        result = branch_for_choice(scene, scanner.ctrl, scanner.tape, query, choice, anatomy)
        output = endpoint(scene, result, scanner.tape, query.state, query.deadline)
        records.append(output)
        values.append(rewards(output))
    scaled = np.stack(values) - values[0]
    evidence = [{'tick': p.tick, 'q': p.q.tolist(), 'reading': prior_run.finite(p.reading),
                 'marks': prior_run.finite(p.marks)} for p in query.history[-14:]]
    return {'seed': scanner.seed, 'profile': scene.condition, 'stage': stage, 'tick': query.state.tick,
            'q': query.state.q.tolist(), 'v': query.state.v.tolist(), 'absolute_deadline': query.deadline,
            'choice_kinds': [c.kind for c in options], 'target': [c.target.tolist() for c in options],
            'from_witness_chart_distance': None if witness is None else float(dist(query.state.q, witness)),
            'fresh': choice_features(scene, query, options).tolist(),
            'stale': choice_features(scene, query, options, stale=True).tolist(),
            'delta': scaled.tolist(), 'outcome': records, 'last_packets': evidence}


def curriculum_case(item):
    seed, target = item
    q_witness = np.asarray(target)
    data = []
    for i, profile in enumerate(CONDITIONS):
        scanner = Scanner(seed, profile, 80+i)
        scene = scanner.scene
        for stage, state, history, suffix in reverse_checkpoints(
                scene, scanner.ctrl, scanner.tape, scanner.prefix, scanner.initial.history, q_witness,
                variant=seed+i):
            query = Query(state, tuple(history), suffix, scanner.prefix.tick, scanner.deadline, scanner.prefix)
            candidates = action_targets(scene, query)
            options = options_from_targets(scene, query, candidates)
            if len(options) < 2:
                continue
            fitted, _, _ = fit_anatomy(scene, state, history, None)
            example = label_options(scanner, stage, query, scene.body.anatomy(fitted), options, witness=q_witness)
            if example:
                data.append(example)
    return data


def fit_regression(rows, mode):
    x = np.vstack([r[mode] for r in rows])
    y = np.vstack([r['delta'] for r in rows])
    model = ExtraTreesRegressor(n_estimators=144, min_samples_leaf=9, max_features=.8,
                                n_jobs=6, random_state=8327)
    model.fit(x, y).set_params(n_jobs=1)
    return model


def collect_reverse(pool):
    data = json.loads(WITNESSES.read_text())['anatomies'][:REVERSE_LIMIT]
    tasks = [(r['seed'], r['target']) for r in data if r['stop_feasible']]
    rows = gather(pool, curriculum_case, tasks, 'reverse')
    rows.sort(key=lambda r: (r['seed'], r['profile'], r['tick'], r['stage']))
    save_rows(OUT/'reverse_queries.jsonl.gz', rows)
    easy = [r for r in rows if r['stage'] != 'wide_near_miss']
    model_early = fit_regression(easy, 'fresh')
    model_extended = fit_regression(rows, 'fresh')
    joblib.dump({'near': model_early, 'extended': model_extended}, OUT/'reverse_policy.joblib', compress=3)
    print('reverse', data_summary(rows), flush=True)


def student_case(seed):
    data = []
    stage_model = joblib.load(OUT/'reverse_policy.joblib')['extended']
    for i, profile in enumerate(CONDITIONS):
        scanner = Scanner(seed, profile, 90+i)
        snapshots, stages = [], []

        def remember(query, options, anatomy):
            if not snapshots or query.state.tick-snapshots[-1][0].state.tick >= QUERY_EVERY*4:
                snapshots.append((query, options, anatomy))

        # Actual student rollouts after learning the reverse neighborhood.
        run = scanner.scan(False, 'fresh', stage_model, margin=.09, collect=remember)
        if len(snapshots) > 5:
            snapshots = [snapshots[k] for k in np.linspace(0, len(snapshots)-1, 5, dtype=int)]
        for query, options, anatomy in snapshots:
            example = label_options(scanner, 'student', query, anatomy, options)
            if example:
                data.append(example)
        stages.append({'condition': profile, 'success': run['correct'], 'queries_used': len(snapshots),
                       'interventions': len(run['interventions'])})
    return data


def gather(pool, function, tasks, label):
    rows = []
    for k, result in enumerate(pool.map(function, tasks), start=1):
        rows.extend(result)
        if k % 4 == 0 or k == len(tasks):
            print(label, 'anatomies', k, '/', len(tasks), 'exits', len(rows), flush=True)
    return rows


def data_summary(rows):
    out = {'rows': len(rows), 'anatomies': len({r['seed'] for r in rows}),
           'actions': sum(len(r['outcome']) for r in rows), 'stages': dict(Counter(r['stage'] for r in rows)),
           'any_correct': sum(any(o['correct'] for o in r['outcome']) for r in rows),
           'continue_correct': sum(r['outcome'][0]['correct'] for r in rows),
           'changed_can_rescue': sum(not r['outcome'][0]['correct'] and any(o['correct'] for o in r['outcome']) for r in rows)}
    return out


def choose_margin(rows, model):
    deltas = [r['outcome'] for r in rows]
    predictions = [model.predict(r['fresh']) @ UTILITY for r in rows]
    selections = []
    for margin in (0., .025, .05, .075, .10, .15, .20, .30, .45):
        def selected(values):
            copy = values.copy()
            copy[0] = margin
            return int(np.argmax(copy))
        outcomes = [r[selected(p)] for r, p in zip(deltas, predictions)]
        utility = np.mean([rewards(r) @ UTILITY for r in outcomes])
        selections.append({'margin': margin, 'utility': float(utility),
                           'correct': sum(r['correct'] for r in outcomes), 'wrong': sum(r['wrong'] for r in outcomes)})
    return sorted(selections, key=lambda r: (r['utility'], r['margin']), reverse=True)


def collect_student(pool):
    rows = gather(pool, student_case, range(STUDENT_START, STUDENT_START+STUDENT_COUNT), 'student')
    save_rows(OUT/'student_queries.jsonl.gz', rows)
    reverse = read_rows(OUT/'reverse_queries.jsonl.gz')
    border = STUDENT_START+STUDENT_COUNT-STUDENT_VALIDATE
    student, valid = [r for r in rows if r['seed'] < border], [r for r in rows if r['seed'] >= border]
    joint = reverse+student
    models = {mode: fit_regression(joint, mode) for mode in ('fresh', 'stale')}
    margins = choose_margin(valid, models['fresh'])
    # Compare the pre-student model on the same new exits, with the same action library.
    reverse_model = joblib.load(OUT/'reverse_policy.joblib')['extended']
    pre_scores = choose_margin(valid, reverse_model)
    save_json(OUT/'local_training.json', {'reverse': data_summary(reverse), 'student_fit': data_summary(student),
                                          'student_validation': data_summary(valid), 'selection': margins,
                                          'reverse_selection': pre_scores,
                                          'joint_train_uses_all_feasible_success_and_failure_labels': True})
    joblib.dump({'models': models, 'margin': margins[0]['margin'], 'reverse_margin': pre_scores[0]['margin']},
                OUT/'frozen_local.joblib', compress=3)
    print('frozen local', data_summary(student), 'validation', data_summary(valid), 'choice', margins[0], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('reverse', 'student'))
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers, initializer=prior_run.initialize) as pool:
        (collect_reverse if args.phase == 'reverse' else collect_student)(pool)
