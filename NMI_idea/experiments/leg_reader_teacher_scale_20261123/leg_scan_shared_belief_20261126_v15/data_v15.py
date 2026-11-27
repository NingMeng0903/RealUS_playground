"""Causal v15 training records from actual frozen v13 student scans."""

from argparse import ArgumentParser
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from multiprocessing import get_context
from pathlib import Path
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
V13 = ROOT.parent / 'leg_scan_student_history_20261124_v13'
V14 = ROOT.parent / 'leg_scan_topology_reader_20261126_v14'
sys.path.insert(1, str(V13))

import policy_v13
import trajectory_v13 as tr
from environment import Case
from fit_v12 import hydrate
from reader_v13 import _adjust, deployment
from recovery_task import DEFAULT_PACK
from study_v9 import atomic_json, digest, read_json
from study_worlds import truth_at


OLD = V13 / 'run'
PROFILE = ('ordinary', 'stress')
COUNTS = {'reader_fit':256, 'reader_cal':48, 'reader_dev':72,
          'teach':108, 'teach_cal':36, 'student_dev':36, 'blind':108}
SEED = {k:8_614_150_000_000_000+i*1_000_000 for i,k in enumerate(COUNTS)}
BLIND_TAPES = (None, 227_194_021, 227_216_577)
OLD_RULE = None
ATLAS = None


def roster(name):
    if name not in COUNTS:
        raise ValueError(name)
    shapes = (0,1,2,5) if name in ('reader_fit','teach') else (3,4,6)
    for i in range(COUNTS[name]):
        # Cycle the assigned target region fastest. Each zone sees both noise
        # profiles and each shape, including when the teacher roster is small.
        q, profile = divmod(i//3, len(PROFILE))
        shape = shapes[q%len(shapes)]
        seed = SEED[name] + 43*i
        yield Case(f'v15_{name}_{i:05d}', f'v15_{name}', 2, shape,
                   'groin_to_distal_calf', PROFILE[profile], seed, seed+1, seed+2, 0, False)


def ordinal(case):
    return int(case.identifier.rsplit('_',1)[-1])


def init_base(atlas=DEFAULT_PACK):
    global OLD_RULE, ATLAS
    anchor = read_json(OLD/'test_freeze.json')
    protocol = read_json(OLD/'protocol.json')
    if digest(atlas)!=protocol['atlas_sha256'] or any(digest(V13/n)!=h for n,h in protocol['sources'].items()):
        raise RuntimeError('The original source or atlas differs from frozen v13')
    actor = OLD/'models/student.json'
    if digest(actor)!=anchor['actor_sha256'] or anchor['selection']['selected']!='student_0':
        raise RuntimeError('Expected frozen v13 student_0')
    policy_v13.init(atlas,OLD,anchor['teacher_sha256'],actor)
    OLD_RULE = anchor['selection']['candidate_rule']
    ATLAS = atlas


def old_actor():
    return policy_v13.NEW_ACTOR if OLD_RULE is not None else hydrate(read_json(OLD/'models/student.json'))


def old_roll(case, offset=None, **callbacks):
    live = tr.live_case(case,offset=offset,augment=case.split=='v15_reader_fit')
    live = tr.ENGINE.to_frontier(live,3)
    return tr.roll(live,policy_v13.learned(OLD_RULE),**callbacks)


def snapshot(live, read):
    """Full 177-channel causal v13 representation plus offline truth arrays."""
    p = read.public
    base = deployment(read,policy_v13.READER,policy_v13.MODE)
    x = tr.observation(read)
    score = _adjust(policy_v13.READER,x,tr.old_logits(read))
    truth = truth_at(live.scan.node,p,live.scan.tape)
    data = {'x':x.astype(np.float16,copy=False), 'prior':score.astype(np.float32),
            'roles':np.asarray(read.decision['role_posterior'],dtype=np.float32),
            'track':np.asarray(read.decision['track_id'],dtype=np.int32),
            'frame':np.asarray(p['frames'],dtype=np.int32), 'slot':np.asarray(p['slots'],dtype=np.int8),
            'region':np.asarray(p['regions'],dtype=np.int8), 'station':np.asarray(p['stations'],dtype=np.float32),
            'sector':np.asarray(p['sectors'],dtype=np.float32), 'xy':np.asarray(p['points'],dtype=np.float32),
            'quality':np.asarray(p['quality'],dtype=np.float32),
            'label':truth['labels'].astype(np.int8), 'instance':truth['instance'].astype(np.int32),
            'baseline_ids':np.asarray(live.scan.grade(p,base)[0]['selected_ids'], dtype=np.int8)}
    # The grader does not communicate its labels to any subsequent rollout.
    if x.ndim!=2 or x.shape[1]!=177 or len(x)!=len(p['frames']):
        raise RuntimeError('Incomplete public embedding')
    return data


def _pair(live, before, after, action, captured, detail):
    proposal = before.hypotheses.get(action.region)
    keys = [] if proposal is None else [list(k) for k in proposal.keys if k is not None]
    post = snapshot(live,after)
    first = captured
    if first is None:
        return None
    addresses = {(int(i),int(j)):int(y) for i,j,y in zip(post['frame'],post['slot'],post['label'],strict=True)}
    if any(addresses.get((int(f),int(s)))!=int(ty)
           for f,s,ty in zip(first['frame'],first['slot'],first['label'],strict=True)):
        raise RuntimeError('A pre-action echo changed its address or truth post-action')
    return {'before':first,'after':post,'region':int(action.region),'program':str(action.name),
            'keys':keys,'start_tick':int(detail['start_tick']),'end_tick':int(detail['end_tick'])}


def collect(case):
    pairs = {}
    wanted = 3 + ordinal(case)%3

    def before(live, read, action):
        # One natural pair in the assigned region, with one backup sample
        # from another region if the student never chooses the assigned one.
        if action.region in pairs or (pairs and action.region!=wanted):
            return None
        return snapshot(live,read)

    def after(live, previous, current, action, captured, detail):
        if captured is not None:
            p = _pair(live,previous,current,action,captured,detail)
            pairs[action.region] = p

    result = old_roll(case,before_action=before,after_action=after)
    terminal = snapshot(result.live,result.read)
    chosen = [pairs[wanted]] if wanted in pairs else list(pairs.values())
    grade = policy_v13.grade(result)
    assert grade['selected_ids']==terminal['baseline_ids'].tolist() and grade['completed_legal']
    return {'case':asdict(case),'terminal':terminal,'pairs':chosen,
            'layout_id':result.live.scan.tape.layout.identifier,
            'relative':bool(result.live.scan.tape.layout.metadata.get('v13_relative',False)),
            'acoustic':result.live.scan.tape.window is not None}


def save_case(root, example):
    case = example['case']['identifier']
    out = root/case
    out.mkdir(parents=True,exist_ok=True)
    arrays = {'term_'+k:v for k,v in example['terminal'].items()}
    pairs = []
    for i,pair in enumerate(example['pairs']):
        for stage in ('before','after'):
            arrays |= {f'p{i}_{stage}_{k}':v for k,v in pair[stage].items()}
        pairs.append({k:v for k,v in pair.items() if k not in ('before','after')})
    file = out/'observations.npz'
    temp = file.with_suffix('.new.npz')
    np.savez_compressed(temp,**arrays)
    temp.replace(file)
    meta = {k:v for k,v in example.items() if k not in ('terminal','pairs')}
    meta |= {'pairs':pairs,'sha256':digest(file)}
    atomic_json(out/'commit.json',meta)
    return meta


def collect_job(case, root):
    return save_case(root,collect(case))


def _group(a, prefix):
    return {k[len(prefix):]:a[k] for k in a.files if k.startswith(prefix)}


def load_case(root, marker):
    m = read_json(marker)
    data = marker.parent/'observations.npz'
    if digest(data)!=m['sha256'] or marker.parent.parent!=root:
        raise RuntimeError('Reader data checksum or roster path changed')
    with np.load(data,allow_pickle=False) as a:
        term = _group(a,'term_')
        pairs = [row | {k:_group(a,f'p{i}_{k}_') for k in ('before','after')}
                 for i,row in enumerate(m['pairs'])]
    return m | {'terminal':term,'pairs':pairs}


def load_splits(out):
    stages = {'fit':'reader_fit','cal':'reader_cal','dev':'reader_dev'}
    data = {}
    ids = set()
    for key,stage in stages.items():
        root = out/'reader_data'/stage
        works = sorted(root.glob('*/commit.json'))
        data[key] = [load_case(root,m) for m in works]
        if [d['case'] for d in data[key]] != [asdict(c) for c in roster(stage)]:
            raise RuntimeError(f'Incomplete or different {stage} roster')
        new = {d['layout_id'] for d in data[key]}
        if len(new)!=len(works) or ids & new:
            raise RuntimeError('Repeated or overlapping reader layouts')
        ids |= new
    return data


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument('--atlas',type=Path,default=DEFAULT_PACK)
    p.add_argument('--out',type=Path,default=ROOT/'run')
    p.add_argument('--workers',type=int,default=16)
    a = p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    lock = a.out/'data_freeze.json'
    cfg = {'counts':COUNTS,'stages_seeds':SEED,'blind_tapes':BLIND_TAPES,'atlas_sha256':digest(a.atlas),
           'source_sha256':digest(Path(__file__)),'v13_freeze':digest(OLD/'test_freeze.json'),
           'assignment':'R4/R5/R6 by layout ordinal; capture first naturally selected action in assigned region',
           'input_schema':'177 original embedding channels including identity + public history and exact frame, slot'}
    if lock.exists() and json.dumps(read_json(lock),sort_keys=True)!=json.dumps(cfg,sort_keys=True):
        raise RuntimeError('Conflicting reader data protocol')
    if not lock.exists():
        atomic_json(lock,cfg)
    hist = {}
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=get_context('spawn'),
                             initializer=init_base,initargs=(a.atlas,)) as pool:
        jobs = {}
        for stage in ('reader_fit','reader_cal','reader_dev'):
            hist[stage] = []
            root = a.out/'reader_data'/stage
            for case in roster(stage):
                marker = root/case.identifier/'commit.json'
                if marker.exists():
                    m = read_json(marker)
                    if m['case']!=asdict(case) or m['sha256']!=digest(marker.parent/'observations.npz'):
                        raise RuntimeError('Existing reader record failed verification')
                    hist[stage].append(m)
                else:
                    jobs[pool.submit(collect_job,case,root)] = stage
        for job in as_completed(jobs):
            stage = jobs[job]
            hist[stage].append(job.result())
            if len(hist[stage])%12==0 or len(hist[stage])==COUNTS[stage]:
                print(stage,len(hist[stage]),'/',COUNTS[stage],flush=True)
    summary = {s:{'layouts':len(v),'pairs':sum(len(x['pairs']) for x in v),
                  'pairs_by_region':dict(Counter(p['region']+1 for x in v for p in x['pairs'])),
                  'relative_layouts':sum(x['relative'] for x in v),'acoustic_layouts':sum(x['acoustic'] for x in v)}
               for s,v in hist.items()}
    atomic_json(a.out/'data_summary.json',summary)
    print(summary,flush=True)


if __name__ == '__main__':
    main()
