"""Train an anonymous candidate ranker and re-read invariant test_main trajectories."""
from argparse import ArgumentParser
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import joblib

from identify import CandidateModel, HERE, public_reader
from core import HORIZON, advance, band, fixed_slot, get_world, legal_slots, req, scan, terminal, visit
from experiments import SEEDS

TRAIN_SEED = 382014981
DEVELOPMENT_SEED = 702816979
TEST = HERE.parent / 'test_main'
ARMS = ('equal', 'teacher')
GOAL = 'Posterior_tibial_vein.r'


def archived(case):
    document = json.loads((TEST / f'{case.world_seed}.json').read_text())
    if document['world'] != case.world_seed or document['shape'] != case.shape:
        raise RuntimeError('Test world provenance mismatch')
    return document


def exercise(env, spec, trial, limits, item, mode, random, saved):
    goal, entry, original, history, kind, first, clock, plan = item
    node = original.fork()
    h = history.fork()
    if saved is not None:
        trace = bool(saved['trace'])
    else:
        trace = mode == 'teacher' and plan is not None and random.random() < .65
    visit(env, spec, node, trial, plan, trace)
    h.append(node.reader.frames)
    cut = len(node.reader.frames)
    spans, bands, slots = [], [], []
    for i in range(HORIZON):
        legal = legal_slots(env, spec, node, limits, clock, HORIZON - i)
        if saved is not None:
            slot = int(saved['slots'][i])
        elif mode == 'equal' or random.random() < .30:
            slot = int(fixed_slot(env, spec, node, limits, legal))
        else:
            slot = int(random.choice(legal))
        if slot not in legal:
            raise RuntimeError('Invalid archived action')
        start = len(node.reader.frames)
        advance(env, spec, node, trial, limits, slot, clock)
        h.append(node.reader.frames)
        bands.append(band(node.reader.frames, h, start))
        spans.append((start, len(node.reader.frames)))
        slots.append(slot)
    base = terminal(node, h, entry, cut, bands)
    if saved is not None:
        actual = base['bands'] + [base['entry'], base['fresh']]
        expected = saved['bands'] + [saved['entry'], saved['fresh']]
        if not all(x['role'] == y['role'] and x['captured'] == y['captured'] and
                   np.isclose(x['conf'], y['conf'], rtol=1e-8, atol=1e-8)
                   for x,y in zip(actual, expected, strict=True)):
            raise RuntimeError('Archived candidate scores did not reproduce')
        if node.tick - original.tick != saved['ticks'] or not np.isclose(node.distance - original.distance, saved['mm']):
            raise RuntimeError('Archived movement costs did not reproduce')
    return node.reader.frames, cut, spans, base


def identify_track(env, item, frames, cut, spans, *, training):
    entry = item[1]
    reader = public_reader(env)
    # All feature sessions use only the frames already acquired when the decision is made.
    safe = reader.public(frames)
    sets = {f'band{j+1}': list(range(lo,hi)) for j,(lo,hi) in enumerate(spans)}
    sets['entry'] = [j for j in range(cut) if entry - req.LOOKBACK - 1.e-8 <= frames[j][0] <= entry + 2. + 1.e-8]
    sets['fresh'] = list(range(cut, len(frames)))
    out = {}
    for j, (_, hi) in enumerate(spans):
        names = [f'band{j+1}'] + (['entry', 'fresh'] if j == len(spans) - 1 else [])
        matrices = reader.feature_sets(safe[:hi], [sets[name] for name in names])
        for name, (x, ix) in zip(names, matrices, strict=True):
            labels = np.asarray([int(frames[k][3][r]) for k,r in ix], dtype=np.int8)
            out[name] = (x, labels)
    if training:
        return [pair for key,pair in out.items() if len(pair[0]) and key != 'fresh']
    return out


def collect(job):
    case, phase, requests = job
    cache = archived(case) if phase == 'test' else None
    env, spec, trial, limits, views, items = get_world(case)
    layout = hashlib.sha256(np.nan_to_num(trial.layout.centers).tobytes()+bytes([case.shape])).hexdigest()
    if cache is not None and cache['layout'] != layout:
        raise RuntimeError('Archived layout did not reproduce')
    random = np.random.default_rng(scan.rng_seed(case.action_seed, 195127))
    datasets, rows = [], []
    for item in items:
        goal, entry = item[:2]
        if requests != 'all' and goal != GOAL:
            continue
        for arm in ARMS:
            saved = None if cache is None else cache['requests'][goal]['arms'][arm]
            frames, cut, spans, base = exercise(env, spec, trial, limits, item, arm, random, saved)
            features = identify_track(env, item, frames, cut, spans, training=phase=='train')
            if phase == 'train':
                datasets.extend(features)
            else:
                rows.append(dict(world=case.world_seed, layout=layout, vessel=goal, mode=arm,
                                 profile=case.profile, shape=case.shape, station=entry, base=base,
                                 features=features))
    return (layout, datasets) if phase == 'train' else rows


def classifier(x, y, args):
    from sklearn.ensemble import HistGradientBoostingClassifier
    # Candidate-level proper binary loss, shared by all named target portions.
    model = HistGradientBoostingClassifier(max_iter=args.trees, learning_rate=.10, max_leaf_nodes=23,
                                            min_samples_leaf=40, l2_regularization=2.,
                                            early_stopping=False, random_state=118736)
    model.fit(x, y == 0)
    return model


def read_scores(model, rows):
    blocks = [x for row in rows for x,y in row['features'].values() if len(y)]
    probs = model.predict(np.concatenate(blocks))
    raw_index = 0
    out = []
    for row in rows:
        res = {}
        for name, (x,y) in row['features'].items():
            if len(y):
                q = probs[raw_index:raw_index + len(y)]
                raw_index += len(y)
                i = int(q.argmax())
                # Keep Brier on every acquired candidate, separately from selection and rejection.
                res[name] = dict(conf=float(q[i]), role=int(y[i]), captured=int(np.any(y == 0)),
                                 brier=float(np.mean((q - (y == 0))**2)),
                                 registered_brier=float(np.mean((x[:,0] - (y == 0))**2)),
                                 native_brier=float(np.mean((x[:,8] - (y == 0))**2)))
            else:
                res[name] = dict(conf=0., role=None, captured=0, brier=None,
                                 registered_brier=None, native_brier=None)
        out.append({key:value for key,value in row.items() if key != 'features'} | {'learned':res})
    if raw_index != len(probs):
        raise RuntimeError('Lost a candidate prediction')
    return out


def scores(row, source, cutoff):
    if source == 'native':
        r = row['base']
        raw = {name:r[name] for name in ('entry','fresh')} | {f'band{i+1}':v for i,v in enumerate(r['bands'])}
    else:
        raw = row['learned']
    values = {}
    for name,value in raw.items():
        submitted = value['role'] is not None and value['conf'] >= cutoff
        correct = int(submitted and value['role'] == 0)
        wrong = int(submitted and value['role'] != 0)
        values[name] = dict(correct=correct, wrong=wrong, reject=int(not submitted),
                            forced=int(value['role'] == 0), captured=int(value['captured']))
    entry, fresh = values['entry'], values['fresh']
    bands = [values['band'+str(i)] for i in (1,2,3)]
    return {
        'entry':entry['correct'], 'fresh':fresh['correct'], 'both':entry['correct']*fresh['correct'],
        'wrong':entry['wrong']+fresh['wrong'], 'reject':entry['reject']+fresh['reject'],
        'utility': .5*(entry['correct']+fresh['correct']-entry['wrong']-fresh['wrong']),
        'forced_entry':entry['forced'], 'forced_fresh':fresh['forced'], 'forced_both':entry['forced']*fresh['forced'],
        'capture_both':entry['captured']*fresh['captured'],
        'capture_entry':entry['captured'], 'capture_fresh':fresh['captured'],
        'chain':int(all(b['correct'] for b in bands)),
        'force_chain':int(all(b['forced'] for b in bands)),
        'capture_chain':int(all(b['captured'] for b in bands)),
        'prefix_mm':5*next((i for i,b in enumerate(bands) if not b['correct']),3),
        'wrong_bands':sum(b['wrong'] for b in bands), 'reject_bands':sum(b['reject'] for b in bands),
    }


def summarize(rows, cutoff):
    groups = {'all': rows}
    for name in sorted({r['vessel'] for r in rows}):
        groups[name] = [r for r in rows if r['vessel'] == name]
    output = {}
    for group, data in groups.items():
        output[group] = {}
        for arm in ARMS:
            chosen = [r for r in data if r['mode'] == arm]
            outcomes = {}
            for name, threshold in (('native',.5),('learned',cutoff),('learned_at_half',.5)):
                source = 'native' if name == 'native' else 'learned'
                values = [scores(row, source, threshold) for row in chosen]
                statistics = {k:(float(np.mean([v[k] for v in values])) if k in ('utility','prefix_mm')
                                 else sum(v[k] for v in values)) for k in values[0]}
                outcomes[name] = dict(n=len(chosen), threshold=threshold, **statistics)
            output[group][arm] = outcomes
    return output


def paired(rows, cutoff, group=GOAL):
    output = {}
    for arm in ARMS:
        sub = [r for r in rows if r['mode'] == arm and (group=='all' or r['vessel'] == group)]
        clusters = defaultdict(list)
        for r in sub:
            learned = scores(r, 'learned', cutoff)
            base = scores(r, 'native', .5)
            clusters[r['layout']].append([learned[k]-base[k] for k in ('entry','fresh','both','chain','wrong','utility','prefix_mm')])
        n = len(clusters)
        weights = np.asarray([len(x) for x in clusters.values()])
        totals = np.asarray([np.sum(x,axis=0) for x in clusters.values()])
        index = np.random.default_rng(1989179).integers(n, size=(4000,n))
        draws = totals[index].sum(axis=1) / weights[index].sum(axis=1)[:,None]
        output[arm] = {k:dict(mean=float(totals[:,j].sum()/weights.sum()), bounds=np.quantile(draws[:,j],[.025,.975]).tolist())
                       for j,k in enumerate(('entry','fresh','both','chain','wrong','utility','prefix_mm'))}
    return output


def choose_threshold(rows):
    # One threshold, frozen before test. Correct and wrong images have equal cost.
    candidates = []
    target = [r for r in rows if r['vessel']==GOAL]
    for i in range(16,31):
        t = i / 40.
        values = [scores(row,'learned',float(t)) for row in rows]
        goals = [scores(row,'learned',float(t)) for row in target]
        def metric(values):
            return np.mean([v['utility'] + .20*v['chain'] - .04*v['wrong_bands'] for v in values])
        candidates.append((.25*metric(values) + .75*metric(goals), -abs(t-.5), float(t)))
    return max(candidates)


def run(args):
    tests = list(scan.cases('test', 96, SEEDS['test']))[:args.test]
    train = list(scan.cases('train', args.train, TRAIN_SEED))
    development = list(scan.cases('train', args.dev, DEVELOPMENT_SEED))
    phases = dict(train=train, development=development, test=tests)
    def phase(pool,name):
        start=time.monotonic()
        data=[]
        for i, result in enumerate(pool.map(collect,[(c,name,args.requests) for c in phases[name]],chunksize=1)):
            if name == 'train':
                data.append(result)
            else:
                data.extend(result)
            if (i+1)%24==0 or i+1==len(phases[name]):
                print(name,i+1,len(phases[name]),round(time.monotonic()-start,2),flush=True)
        return data

    with ProcessPoolExecutor(args.workers) as pool:
        samples = phase(pool,'train')
        training_layouts = {layout for layout,_ in samples}
        pairs = [pair for _,group in samples for pair in group]
        x, y = zip(*pairs)
        fit_x, fit_y = np.concatenate(x), np.concatenate(y)
        classifier_fit = classifier(fit_x, fit_y, args)
        print('training candidates',len(fit_y),'feature cols',fit_x.shape[1],flush=True)
        del samples, pairs, x, y, fit_x, fit_y
        dev_raw = [r for r in phase(pool,'development') if r['layout'] not in training_layouts]
        ensemblers = [CandidateModel(classifier_fit, w) for w in (0.,.25,.50)]
        selection = []
        for m in ensemblers:
            scored = read_scores(m, dev_raw)
            value, close, threshold = choose_threshold(scored)
            selection.append((value,close,-m.registered_weight,threshold))
        best = max(range(len(selection)),key=selection.__getitem__)
        model = ensemblers[best]
        cut = selection[best][-1]
        dev = read_scores(model, dev_raw)
        stats = summarize(dev, cut)
        del dev_raw
        save = HERE / args.tag
        save.mkdir(exist_ok=True)
        joblib.dump(dict(model=model, threshold=cut, train_seed=TRAIN_SEED, development_seed=DEVELOPMENT_SEED,
                         fit_shapes=(0,1,2,5)), save/'model.joblib',compress=3)
        p = args.requests if args.requests!='all' else 'all'
        print('frozen threshold',cut,'registered weight',model.registered_weight, 'development',stats.get(GOAL,stats['all']),flush=True)
        test = read_scores(model,phase(pool,'test'))
        out = dict(train_worlds=len(train), development_worlds=len(development), test_worlds=len(tests),
                   train_seed=TRAIN_SEED, development_seed=DEVELOPMENT_SEED, threshold=cut, requests=p,
                   train_layouts=len(training_layouts), development_independent_layouts=len({r['layout'] for r in dev}),
                   registered_weight=model.registered_weight, selection=selection,
                   test_summary=summarize(test,cut), paired=paired(test,cut,GOAL), development_summary=stats,
                   matched_native_replayed=True, test_rows=test, development_rows=dev)
        (save/'study.json').write_text(json.dumps(out,separators=(',',':')))
        print('test',out['test_summary'].get(GOAL,out['test_summary']['all']),flush=True)


if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('--train', type=int, default=160)
    p.add_argument('--dev',type=int,default=64)
    p.add_argument('--test',type=int,default=96)
    p.add_argument('--trees',type=int,default=100)
    p.add_argument('--workers',type=int,choices=range(1,13),default=8)
    p.add_argument('--requests',choices=('all','posterior'),default='all')
    p.add_argument('--tag',default='freeze1')
    run(p.parse_args())
