"""Test local retention on witness cases held out from this training round."""

from concurrent.futures import ProcessPoolExecutor
import json

import joblib
import numpy as np

from policy import UTILITY
from train import OUT, REVERSE_LIMIT, WITNESSES, curriculum_case, prior_run, save_json, save_rows


def take(row, model, margin):
    values = model.predict(row['fresh']) @ UTILITY
    values[0] = margin
    return int(np.argmax(values))


if __name__ == '__main__':
    witnesses = json.loads(WITNESSES.read_text())['anatomies'][REVERSE_LIMIT:]
    tasks = [(r['seed'], r['target']) for r in witnesses if r['stop_feasible']]
    with ProcessPoolExecutor(max_workers=8, initializer=prior_run.initialize) as pool:
        rows = [r for sample in pool.map(curriculum_case, tasks) for r in sample]
    reverse = joblib.load(OUT/'reverse_policy.joblib')
    fit = joblib.load(OUT/'frozen_local.joblib')
    conditions = {'near_initial': (reverse['near'], fit['reverse_margin']),
                  'reverse_extended': (reverse['extended'], fit['reverse_margin']),
                  'student_feedback': (fit['models']['fresh'], fit['margin'])}
    for row in rows:
        row['evaluated'] = {name: take(row, model, margin) for name, (model, margin) in conditions.items()}
    summary = {}
    for stage in ['all', *sorted({r['stage'] for r in rows})]:
        subset = [r for r in rows if stage == 'all' or r['stage'] == stage]
        angles = [r['from_witness_chart_distance'] for r in subset]
        summary[stage] = {'n': len(subset), 'median_chart_distance_from_witness': float(np.median(angles)),
                          'any_correct': sum(any(o['correct'] for o in r['outcome']) for r in subset),
                          'continue': sum(r['outcome'][0]['correct'] for r in subset)}
        for name in conditions:
            out = [r['outcome'][r['evaluated'][name]] for r in subset]
            summary[stage][name] = {'correct': sum(o['correct'] for o in out), 'wrong': sum(o['wrong'] for o in out)}
    save_rows(OUT/'near_success_holdout.jsonl.gz', rows)
    save_json(OUT/'near_success_holdout_summary.json', summary)
    print(json.dumps(summary), flush=True)
