"""Freeze the v15 reader, label actual continuations, train and test the student."""

from argparse import ArgumentParser
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
import gzip
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path
import sys

import numpy as np
from scipy.special import softmax

import data_v15 as data
import identity_v15 as identity
import teacher_v15 as teacher
from control_v15 import Controller

from fit_v12 import hydrate
from run_v11 import outcome


ROOT = data.ROOT
REF_COMMIT = '24b17882d6e2e92e46fc3f3da465db6db2a71ebd'
READER_SEED, STUDENT_SEED, BOOTSTRAP_SEED = 8614173, 8615183, 8616359
RULES = tuple({'margin': m, 'r6_bias': 0.} for m in (0., .025, .055, .095))
BANDS = dict(zip(('actual', 'proof_0', 'proof_1'), data.BLIND_TAPES, strict=True))
MODEL_FILES = ('reader.json', 'student.json', 'student.pt')
BASELINE = 'v13'
LEARNER = 'v15_student'
ARMS = (BASELINE, 'v14_terminal', 'v15_terminal', 'v15_online_reference', LEARNER)
_CONTROLLER = _VERSION = _STUDENT = _RULE = _V14 = None


def _hash_files(root, files):
    return {name: data.digest(root / name) for name in files}


def protocol(args):
    out = args.out
    gpu = data.read_json(out/'gpu_preflight.json')
    if not gpu['cuda_optimizer_step_verified']:
        raise RuntimeError('CUDA optimizer preflight failed')
    v14 = data.V14/'run'
    old = data.read_json(data.OLD/'test_freeze.json')
    cfg = {'experiment': 'leg_scan_shared_belief_20261126_v15', 'parent_commit': REF_COMMIT,
           'source_sha256': _hash_files(ROOT, sorted(p.name for p in ROOT.glob('*.py'))),
           'atlas_sha256': data.digest(args.atlas), 'gpu_preflight_sha256': data.digest(out/'gpu_preflight.json'),
           'gpu_driver_sha256': data.digest(out/'gpu_driver.txt'), 'data_freeze_sha256': data.digest(out/'data_freeze.json'),
           'v13_freeze_sha256': data.digest(data.OLD/'test_freeze.json'),
           'v14_inputs_sha256': _hash_files(v14, ('freeze.json', 'graph.json')),
           'frozen_continuation_student_sha256': old['actor_sha256'], 'frozen_continuation_rule': old['selection']['candidate_rule'],
           'counts': data.COUNTS, 'world_seeds': data.SEED, 'search_offsets': list(teacher.SEARCH),
           'teacher_proof_offset': teacher.PROOF, 'blind_offsets': BANDS,
           'training_seeds': {'reader': READER_SEED, 'student': STUDENT_SEED}, 'bootstrap_seed': BOOTSTRAP_SEED,
           'threshold_candidates': list(RULES), 'blind_arms': list(ARMS),
           'layout': 'Open3DModel right, full groin to distal calf, ordinary and stress; synthetic layouts',
           'readout': 'R4-R6 full v13 public embedding, shared track/scene identity, unknown per region',
           'motion': 'Existing contact, legal route and observation clock; no new time or distance budget or penalty'}
    p = out/'protocol.json'
    if p.exists() and data.read_json(p) != cfg:
        raise RuntimeError('A frozen v15 protocol input changed')
    if not p.exists():
        data.atomic_json(p, cfg)
    return cfg


def _keys(snapshot):
    return list(zip(snapshot['frame'].astype(int).tolist(), snapshot['slot'].astype(int).tolist(), strict=True))


def pair_audit(reader, examples):
    """Give the re-read exactly the old candidate pool and newly observed context."""
    audit = []
    for ex in examples:
        for pair in ex['pairs']:
            first, final, region = pair['before'], pair['after'], pair['region']
            old_keys, all_keys = _keys(first), _keys(final)
            lookup = {k: i for i,k in enumerate(all_keys)}
            old_set = set(old_keys)
            fresh = np.asarray([k not in old_set for k in all_keys], dtype=bool)
            old = np.flatnonzero(first['region'] == region)
            if not len(old):
                continue
            after = np.asarray([lookup[old_keys[i]] for i in old])
            if not np.array_equal(first['label'][old], final['label'][after]):
                raise RuntimeError('A fixed old candidate changed its reference identity')
            b, a = identity.score_snapshot(reader, first), identity.score_snapshot(reader, final)
            rows = []
            for name, pre, post in (('v13', first['prior'], final['prior']),
                                    ('v15', b['identity_score'], a['identity_score'])):
                i = int(old[np.argmax(pre[old])])
                j = int(after[np.argmax(post[after])])
                same_i = lookup[old_keys[i]]
                labels = [int(first['label'][i]), int(final['label'][j])]
                same = (first['instance'][i] >= 0) and np.any(
                    fresh & (final['instance'] == first['instance'][i]))
                rows.append((name, {'before': labels[0], 'after': labels[1], 'before_key': old_keys[i],
                                    'after_key': all_keys[j], 'fixed_score_before': float(pre[i]),
                                    'fixed_score_after': float(post[same_i]), 'new_echo_same_instance': bool(same)}))
            i = int(old[np.argmax(first['prior'][old])])
            same_i = lookup[old_keys[i]]
            fixed = {'key': old_keys[i], 'label': int(first['label'][i]),
                     'v15_target_role_before': float(b['role_posterior'][i,0]),
                     'v15_target_role_after': float(a['role_posterior'][same_i,0]),
                     'new_echo_same_instance': rows[0][1]['new_echo_same_instance']}
            audit.append({'case': ex['case']['identifier'], 'region': region+1, 'program': pair['program'],
                          'start_tick': pair['start_tick'], 'end_tick': pair['end_tick'],
                          'old_candidate_count': len(old), 'new_echo_count_all_regions': int(fresh.sum()),
                          'fixed_v13_old_winner': fixed, 'arms': dict(rows)})
    by = {}
    for region in range(4,7):
        selected = [r for r in audit if r['region'] == region]
        by[str(region)] = {'pairs': len(selected)}
        for name in ('v13','v15'):
            pairs = [r['arms'][name] for r in selected]
            by[str(region)][name] = {'before_correct': sum(p['before']==0 for p in pairs),
                                     'after_correct': sum(p['after']==0 for p in pairs),
                                     'rescued': sum(p['before']!=0 and p['after']==0 for p in pairs),
                                     'lost': sum(p['before']==0 and p['after']!=0 for p in pairs)}
        fixed = [r['fixed_v13_old_winner'] for r in selected]
        def correct(p, field):
            return (p[f'v15_target_role_{field}'] >= .5) == (p['label'] == 0)
        by[str(region)]['fixed_v13_winner_role'] = {
            'reclassified_correct': sum(not correct(p,'before') and correct(p,'after') for p in fixed),
            'reclassified_wrong': sum(correct(p,'before') and not correct(p,'after') for p in fixed),
            'with_new_echo_same_instance': sum(p['new_echo_same_instance'] for p in fixed)}
    return {'by_region': by, 'pairs': audit}


def fit_reader(args):
    # Reader training may run concurrently with implementation of the CPU
    # controller. Its own source/data inputs are committed before using it.
    from reader_fit_v15 import evaluate_reader, fit_reader as train

    out = args.out
    stage = out/'reader_freeze.json'
    cfg = {'source_sha256': _hash_files(ROOT, ('data_v15.py', 'identity_v15.py','reader_fit_v15.py')),
           'data_sha256': data.digest(out/'data_freeze.json'), 'seed': READER_SEED,
           'gpu_preflight_sha256': data.digest(out/'gpu_preflight.json')}
    if stage.exists():
        want = cfg | {'reader_sha256': data.digest(out/'models/reader.json')}
        if data.read_json(stage) != want:
            raise RuntimeError('The fitted reader or its inputs changed after freeze')
        return
    if not data.read_json(out/'gpu_preflight.json')['cuda_optimizer_step_verified']:
        raise RuntimeError('CUDA optimizer preflight failed')
    splits = data.load_splits(out)
    path = out/'models/reader_training.json'
    trained = data.read_json(path) if path.exists() else train(splits, out/'models', READER_SEED)
    asset = out/'models/reader.json'
    if (trained['reader_sha256'] != data.digest(asset) or trained['seed'] != READER_SEED or
            trained['unique_fit']['complete_histories'] != len(splits['fit'])):
        raise RuntimeError('Reader model, seed or fitted input does not match its training record')
    reader = identity.load_reader(asset)
    measured = evaluate_reader(reader,splits['dev'])
    if measured != trained['same_snapshot_development']:
        raise RuntimeError('Exact-candidate reread failed to reproduce')
    # A truth-removal check on the inference fields, across large real histories.
    for ex in splits['dev'][:8]:
        terminal = ex['terminal']
        direct = identity.score_snapshot(reader,terminal)
        public = identity.score_snapshot(reader,{k:terminal[k] for k in identity.PUBLIC_KEYS})
        if any(not np.array_equal(direct[k],public[k]) for k in direct):
            raise RuntimeError('A reference label influenced the public reader')
    audit = pair_audit(reader, splits['dev'])
    data.atomic_json(out/'reader_evidence_pairs.json',audit)
    data.atomic_json(stage,cfg | {'reader_sha256':data.digest(asset)})
    print('same candidate strict and wrong', {a:(measured[a]['strict'],measured[a]['wrong']) for a in ('v13','v15')}, flush=True)
    print('evidence pairs',audit['by_region'],flush=True)


def _write_gzip(path, row):
    temp = path.with_suffix('.tmp')
    with temp.open('wb') as file, gzip.GzipFile(fileobj=file, mode='wb', filename='', mtime=0) as zipped:
        zipped.write(json.dumps(row, separators=(',', ':'), allow_nan=False).encode())
    temp.replace(path)


def _read_gzip(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        return json.load(stream)


def initialize(atlas, out, version, stage):
    global _CONTROLLER, _VERSION, _STUDENT, _RULE, _V14
    data.init_base(atlas)
    _VERSION = version
    models = out/'models'
    _CONTROLLER = Controller(identity.load_reader(models/'reader.json'), data.policy_v13.READER, data.policy_v13.MODE)
    if stage in ('student_dev', 'blind'):
        _STUDENT = hydrate(data.read_json(models/'student.json'))
    if stage == 'blind':
        _RULE = data.read_json(out/'test_freeze.json')['development']['rule']
        sys.path.insert(1,str(data.V14))
        import experiment_v14 as prior
        prior.init_cpu(atlas, data.OLD)
        _V14 = prior, data.read_json(data.V14/'run/graph.json')


def _teacher_job(case):
    x, y, meta = teacher.teacher_job(case, _CONTROLLER, data.old_actor(), data.OLD_RULE, _VERSION)
    return meta | {'features':x.tolist(), 'labels':y.tolist()}


def _counters(result):
    actions = Counter((a['region'],a['program']) for a in result.actions)
    forward = [v for t in result.live.trace for v in t.get('belief_windows',())]
    bridges = [v for a in result.actions for v in a.get('bridge_windows',())]
    return {'review_count':len(result.actions),'review_before_last':sum(a['before_last'] for a in result.actions),
            'reviews':[{'region':r,'program':name,'count':n} for (r,name),n in sorted(actions.items())],
            'forward_by_region':dict(Counter(str(v['region']) for v in forward)),
            'forward_with_echo_key':sum(v['key'] is not None for v in forward),
            'forward_different_from_local_mpc':sum(v['sector']!=v['frozen_mpc_sector'] for v in forward),
            'bridge_windows':len(bridges), 'bridge_with_echo_key':sum(v['source_key'] is not None for v in bridges),
            'clock_preserved':result.live.scan.node.tick==len(result.live.scan.node.reader.frames)}


def _signature(result):
    frames = [(round(float(f[0]),5),round(float(f[1]),5),int(f[-1])) for f in result.live.scan.node.reader.frames]
    return hashlib.sha256(np.asarray(frames,dtype=np.float64).tobytes()).hexdigest()


def _summary(result, reader=None):
    if reader is None:
        grade = _CONTROLLER.grade(result)
    else:
        grade = data.tr.score(result, reader)
    compact = outcome(grade)
    if (len(compact['selected_ids'])!=6 or compact['strict']!=all(i==0 for i in compact['selected_ids']) or
            compact['wrong_branch']!=any(i>0 for i in compact['selected_ids'])):
        raise RuntimeError('The six-region delivery standard changed')
    return compact | {'acquisition_sha256':_signature(result), 'echo_candidates':len(result.read.public['frames']),
                      'execution':_counters(result)}


def _development_job(case):
    start = data.tr.live_case(case)
    ref = _CONTROLLER.chooser(data.old_actor(), data.OLD_RULE)
    arms = {'reference': _summary(_CONTROLLER.roll(start.fork(),ref))}
    for i, rule in enumerate(RULES):
        chosen = _CONTROLLER.chooser(_STUDENT,rule)
        arms[str(i)] = _summary(_CONTROLLER.roll(start.fork(),chosen))
    return {'case':asdict(case),'layout_id':start.scan.tape.layout.identifier,'version':_VERSION,'arms':arms}


def _v14_terminal(result, case, band, base):
    prior, model = _V14
    snapshot = data.tr.save_terminal(result,'student')
    row = prior.record(snapshot,asdict(case),base['selected_ids'][:3],band,result.live.scan.tape.layout.identifier)
    if row['baseline_ids'] != base['selected_ids']:
        raise RuntimeError('Frozen v14 uses a different v13 same-trajectory input')
    prediction = [row['baseline_row']]
    if row['state']:
        features = prior.materialize([row])
        prob = softmax(prior.numpy_logits(model,features)/model['temperature'],axis=-1)
        prediction = prior.decide(prob,[row],features,prior.Rule(**model['rule']))
    fixed = prior.outcome([row],prediction)[0]
    # Only the v14 selected distal image may change. Regrade using its exact
    # candidate row so all reported lengths and class outcomes are consistent.
    candidates = np.flatnonzero(result.read.public['regions'] == 5)
    late = np.flatnonzero(result.read.public['regions'] >= 3)
    pred = data.deployment(result.read,data.policy_v13.READER,data.policy_v13.MODE)
    score = pred['score'].copy()
    if prediction[0] != row['baseline_row']:
        score[candidates] = -1
        if prediction[0] >= 0:
            if result.read.public['regions'][late[prediction[0]]] != 5:
                raise RuntimeError('v14 selected a candidate outside R6')
            score[late[prediction[0]]] = 1
    grade = _summary(result,pred | {'score':score})
    if grade['selected_ids'] != fixed['labels']:
        raise RuntimeError('Frozen v14 candidate address did not reproduce')
    return grade


def _blind_job(case):
    ref = _CONTROLLER.chooser(data.old_actor(),data.OLD_RULE)
    student = _CONTROLLER.chooser(_STUDENT,_RULE)
    tapes = {}
    for band,offset in BANDS.items():
        start = data.tr.live_case(case,offset=offset)
        original = data.tr.roll(start.fork(),data.policy_v13.learned(data.OLD_RULE))
        base = _summary(original,data.deployment(original.read,data.policy_v13.READER,data.policy_v13.MODE))
        v14 = _v14_terminal(original,case,band,base)
        same = identity.apply(original.read,_CONTROLLER.reader,data.policy_v13.READER,data.policy_v13.MODE)
        terminal = _summary(replace(original,read=same))
        new_ref = _summary(_CONTROLLER.roll(start.fork(),ref))
        new_student = _summary(_CONTROLLER.roll(start.fork(),student))
        if any(x['acquisition_sha256']!=base['acquisition_sha256'] or
               x['echo_candidates']!=base['echo_candidates'] or x['capture']!=base['capture'] for x in (v14,terminal)):
            raise RuntimeError('A terminal reader changed the fixed observation pool')
        tapes[band] = dict(zip(ARMS,(base,v14,terminal,new_ref,new_student),strict=True))
    return {'case':asdict(case),'layout_id':start.scan.tape.layout.identifier,'version':_VERSION,'tapes':tapes}


def _parallel(args, stages, jobs, version):
    out = args.out
    rows = {s:[] for s in stages}
    marker = stages[0]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=get_context('spawn'),
                             initializer=initialize,initargs=(args.atlas,out,version,marker)) as pool:
        pending = {}
        for stage in stages:
            store = out/'shards'/stage
            store.mkdir(parents=True,exist_ok=True)
            for case in data.roster(stage):
                path = store/f'{case.identifier}.json.gz'
                if path.exists():
                    row = _read_gzip(path)
                    if row['case']!=asdict(case) or row['version']!=version:
                        raise RuntimeError('A scheduled layout belongs to a different frozen configuration')
                    rows[stage].append(row)
                else:
                    pending[pool.submit(jobs,case)] = stage,path
        for future in as_completed(pending):
            stage,path = pending[future]
            row = future.result()
            _write_gzip(path,row)
            rows[stage].append(row)
            n = len(rows[stage])
            if n==1 or n%4==0 or n==data.COUNTS[stage]:
                print(stage,n,'/',data.COUNTS[stage],flush=True)
    layout_ids = set()
    for stage, group in rows.items():
        group.sort(key=lambda r:r['case']['identifier'])
        layout = {r['layout_id'] for r in group}
        if len(layout)!=data.COUNTS[stage] or layout & layout_ids:
            raise RuntimeError('The execution roster contains missing or repeated layouts')
        layout_ids |= layout
    return rows


def _freeze(path, inputs):
    if path.exists() and data.read_json(path) != inputs:
        raise RuntimeError(f'Mismatched frozen inputs for {path.name}')
    if not path.exists():
        data.atomic_json(path,inputs)
    return data.digest(path)


def teach(args):
    out = args.out
    protocol(args)
    lock = data.read_json(out/'reader_freeze.json')
    if lock['reader_sha256'] != data.digest(out/'models/reader.json'):
        raise RuntimeError('Reader changed before teacher generation')
    file = out/'teacher_freeze.json'
    version = _freeze(file,{'protocol':data.digest(out/'protocol.json'), 'reader':lock['reader_sha256'],
                            'continuation_student':data.digest(data.OLD/'models/student.json'),
                            'reference_action':'normally advance or submit, then same continuation',
                            'search':list(teacher.SEARCH), 'independent_proof':teacher.PROOF})
    rows = _parallel(args,('teach','teach_cal'),_teacher_job,version)
    summary = {stage:teacher.summarize_teacher(rec) for stage,rec in rows.items()}
    data.atomic_json(out/'teacher_summary.json',summary)
    print('paired teacher',summary,flush=True)


def _load_teacher(out, stage, version):
    rows = []
    for c in data.roster(stage):
        row = _read_gzip(out/'shards'/stage/f'{c.identifier}.json.gz')
        if row['case']!=asdict(c) or row['version']!=version:
            raise RuntimeError('Teacher labels use a different student or reader')
        rows.append((np.asarray(row['features'],dtype=np.float32),np.asarray(row['labels'],dtype=np.float32),row))
    return rows


def fit_student(args):
    out = args.out
    protocol(args)
    version = data.digest(out/'teacher_freeze.json')
    if (out/'student_freeze.json').exists():
        if data.read_json(out/'student_freeze.json') != {'teacher':version,'models':_hash_files(out/'models',MODEL_FILES)}:
            raise RuntimeError('The trained student changed')
        return
    fit, cal = (_load_teacher(out,s,version) for s in ('teach','teach_cal'))
    teacher.fit_student(fit,cal,out/'models',STUDENT_SEED,data.old_actor())
    data.atomic_json(out/'student_freeze.json',{'teacher':version,'models':_hash_files(out/'models',MODEL_FILES)})


def measures(steps):
    labels = np.asarray([x['selected_ids'] for x in steps],dtype=int)
    captures = np.asarray([x['capture'] for x in steps],dtype=bool)
    return {'layouts':len(steps),'strict':sum(x['strict'] for x in steps),
            'wrong':sum(x['wrong_branch'] for x in steps), 'wrong_regions':int((labels>0).sum()),
            'correct_regions':int((labels==0).sum()),'all_captured':int(np.all(captures,axis=1).sum()),
            'r_correct':(labels==0).sum(axis=0).tolist(),'r_wrong':(labels>0).sum(axis=0).tolist(),
            'r_artifact':np.isin(labels,(-1,-2)).sum(axis=0).tolist(),
            'r_rejected':(labels==-99).sum(axis=0).tolist(), 'r_capture':captures.sum(axis=0).tolist(),
            'seconds':float(np.mean([x['time_s'] for x in steps])),
            'travel_mm':float(np.mean([x['travel_mm'] for x in steps])),
            'correct_coverage_mm':float(np.mean([x['mm'] for x in steps])),
            'reviews':sum(x['execution']['review_count'] for x in steps),
            'reviews_before_last':sum(x['execution']['review_before_last'] for x in steps),
            'bridge_windows':sum(x['execution']['bridge_windows'] for x in steps),
            'bridge_with_echo_key':sum(x['execution']['bridge_with_echo_key'] for x in steps),
            'forward_windows':sum(sum(x['execution']['forward_by_region'].values()) for x in steps),
            'forward_with_echo_key':sum(x['execution']['forward_with_echo_key'] for x in steps),
            'forward_different_from_local_mpc':sum(x['execution']['forward_different_from_local_mpc'] for x in steps)}


def develop(args):
    out = args.out
    protocol(args)
    student = data.read_json(out/'student_freeze.json')
    if student['models'] != _hash_files(out/'models',MODEL_FILES):
        raise RuntimeError('A model changed before closed loop development')
    version = data.digest(out/'student_freeze.json')
    rows = _parallel(args,('student_dev',),_development_job,version)['student_dev']
    scores = {name:measures([r['arms'][name] for r in rows]) for name in rows[0]['arms']}
    selected = max(range(len(RULES)),key=lambda i:(scores[str(i)]['strict'],-scores[str(i)]['wrong'],
                                                 scores[str(i)]['correct_regions']))
    choice = {'rule':RULES[selected],'rule_index':selected,'scores':scores,
              'criterion':'most strict, then fewer wrong layouts, then more correct regions on student development'}
    _write_gzip(out/'student_development.json.gz',rows)
    _freeze(out/'test_freeze.json',{'protocol':data.digest(out/'protocol.json'),
            'student':version,'models':student['models'],'blind_offsets':BANDS,'development':choice})
    print('development selected',selected,{k:(v['strict'],v['wrong']) for k,v in scores.items()},flush=True)


def _pair_result(rows, bands, left, right):
    score = {m: np.asarray([[int(r['tapes'][b][right][m])-int(r['tapes'][b][left][m]) for b in bands]
                           for r in rows]) for m in ('strict','wrong_branch')}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    strata = [np.flatnonzero([r['case']['profile']==p for r in rows]) for p in data.PROFILE]
    sampled = np.column_stack([rng.choice(s,size=(8000,len(s)),replace=True) for s in strata if len(s)])
    stats = {}
    for name,d in score.items():
        layout = d.mean(axis=1) * 100.
        stats[name] = {'increase':int((d>0).sum()),'decrease':int((d<0).sum()),'net':int(d.sum()),
                       'point_pp':float(layout.mean()),'ci95_pp':np.quantile(layout[sampled].mean(axis=1),[.025,.975]).tolist()}
    stats['same_acquisition'] = sum(r['tapes'][b][left]['acquisition_sha256']==r['tapes'][b][right]['acquisition_sha256']
                                    for r in rows for b in bands)
    return stats


def summarize_blind(rows):
    comparisons = ((BASELINE,'v14_terminal'),(BASELINE,'v15_terminal'),('v14_terminal','v15_terminal'),
                   ('v15_terminal','v15_online_reference'),('v15_online_reference',LEARNER),
                   ('v14_terminal',LEARNER), (BASELINE,LEARNER))
    result = {'layouts':len(rows),'bands':{},'pairs':{},'profiles_actual':{}}
    for band in BANDS:
        result['bands'][band] = {arm:measures([r['tapes'][band][arm] for r in rows]) for arm in ARMS}
    for k,bands in tuple((key,(key,)) for key in BANDS) + (('proof_only',('proof_0','proof_1')),('all',tuple(BANDS))):
        result['pairs'][k] = {f'{right}-vs-{left}':_pair_result(rows,bands,left,right) for left,right in comparisons}
    for profile in data.PROFILE:
        chosen = [r for r in rows if r['case']['profile']==profile]
        result['profiles_actual'][profile] = {arm:measures([r['tapes']['actual'][arm] for r in chosen]) for arm in ARMS}
    return result


def check_separation(out, rows):
    groups = {'blind':{r['layout_id'] for r in rows}}
    for name in ('reader_fit','reader_cal','reader_dev'):
        groups[name] = {data.read_json(f)['layout_id'] for f in (out/'reader_data'/name).glob('*/commit.json')}
    for name in ('teach','teach_cal','student_dev'):
        groups[name] = {_read_gzip(f)['layout_id'] for f in (out/'shards'/name).glob('*.json.gz')}
    seen = set()
    for name,group in groups.items():
        if len(group)!=data.COUNTS[name] or seen & group:
            raise RuntimeError(f'Reused or missing layout in {name}')
        seen |= group
    return {k:len(v) for k,v in groups.items()}


def blind(args):
    out = args.out
    protocol(args)
    lock = out/'test_freeze.json'
    test = data.read_json(lock)
    if test['models'] != _hash_files(out/'models',MODEL_FILES) or test['protocol']!=data.digest(out/'protocol.json'):
        raise RuntimeError('A frozen test dependency changed')
    version = data.digest(lock)
    rows = _parallel(args,('blind',),_blind_job,version)['blind']
    result = summarize_blind(rows)
    result['verified_layout_split_counts'] = check_separation(out,rows)
    result['test_freeze_sha256'] = version
    data.atomic_json(out/'results.json',result)
    _write_gzip(out/'blind_paired.json.gz',rows)
    print('blind strict and wrong', {b:{a:(x['strict'],x['wrong']) for a,x in t.items()} for b,t in result['bands'].items()},flush=True)


def main():
    stages = {'fit-reader':fit_reader,'teach':teach,'fit-student':fit_student,'develop':develop,'blind':blind}
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=stages)
    parser.add_argument('--atlas',type=Path,default=data.DEFAULT_PACK)
    parser.add_argument('--out',type=Path,default=ROOT/'run')
    parser.add_argument('--workers',type=int,default=18)
    args = parser.parse_args()
    stages[args.stage](args)


if __name__ == '__main__':
    main()
