"""Train and test a distal relational readout with the frozen v13 trajectories."""

from argparse import ArgumentParser
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import copy
from dataclasses import asdict, replace
import gzip
import json
import math
from multiprocessing import get_context
from pathlib import Path
import sys

import numpy as np
from scipy.special import softmax

ROOT = Path(__file__).resolve().parent
V13 = ROOT.parent/'leg_scan_student_history_20261124_v13'
sys.path.insert(1, str(V13))

import policy_v13
import trajectory_v13 as tr
from collect_v13 import SNAPSHOT_COLUMNS
from reader_v13 import _adjust, logistic, winners
from recovery_task import DEFAULT_PACK
from run_v11 import outcome as v13_outcome
from study_v9 import verify_shard

from fit_v14 import (Rule, decide, digest, materialize, numpy_logits, outcome, save_json,
                     totals, train)
from relations_v14 import (Junction, POSITION_LIMIT, _pair_features, addresses,
                           junction_compatibility, load_prior, panel, target_set)


BANDS = (('actual',None),('proof_0',226_208_173),('proof_1',226_249_871))
SPLITS = {'fit':'fit','cal':'cal','dev':'reader_dev'}
NAMES = ('relative','graph')
FIT_SEED = 26_112_614
RESAMPLE_SEED = 282_717_611
_PRIOR = _READER = _RUN = None


def read_json(path):
    return json.loads(path.read_text())


def write_gzip(path, data):
    temp = path.with_suffix('.tmp')
    with gzip.open(temp,'wt',encoding='utf-8') as stream:
        json.dump(data,stream,ensure_ascii=False,separators=(',',':'))
    temp.replace(path)


def read_gzip(path):
    with gzip.open(path,'rt',encoding='utf-8') as stream:
        return json.load(stream)


def init_cpu(atlas, old_run, execute=False, junctions=None):
    global _PRIOR, _READER, _RUN
    _RUN = old_run
    _READER, mode = policy_v13.model_reader(old_run)
    if mode != 'new':
        raise ValueError('v14 uses the frozen v13 residual terminal reader')
    events = [Junction(**r) for r in read_json(junctions)] if junctions is not None else ()
    _PRIOR = load_prior(atlas, events)
    if execute:
        frozen = read_json(old_run/'test_freeze.json')
        policy_v13.init(atlas, old_run, frozen['teacher_sha256'], old_run/'models/student.json')


def record(snapshot, case, initial_ids, band, layout_id):
    y = snapshot['label']
    scores = _adjust(_READER,snapshot['x'],snapshot['logit'])
    selected = [-99] * 3
    old_row = -1
    for r,i,f in winners(snapshot,scores):
        confidence = logistic(_READER['calibration'], f)
        if confidence >= _READER['calibration']['threshold']:
            selected[r-3] = int(y[i])
            if r==5:
                old_row = i
    public_graph = panel(snapshot,scores,case['shape'],_PRIOR)
    out = {'case':case['identifier'],'band':band,'shape':case['shape'],'profile':case['profile'],
           'layout_id':layout_id, 'baseline_ids':list(initial_ids)+selected,'baseline_row':old_row,
           'captured':bool(np.any(y[snapshot['region']==5] == 0))}
    if public_graph is None:
        return out | {k:[] for k in ('state','position','motif','valid','truth','rows','label','keys')} | {
            'candidate_has_target':False}
    valid, rows = public_graph['valid'], public_graph['pairs']
    truth = target_set(public_graph, y)
    label = np.where(valid,y[rows.clip(min=0)],-99)
    keys = np.stack((snapshot['frame'],snapshot['slot']),axis=-1)[rows.clip(min=0)]
    keys = np.where(valid[...,None],keys,-1)
    return out | {k:public_graph[k].tolist() for k in ('state','position','motif','valid')} | {
        'rows':rows.tolist(),'label':label.tolist(),'truth':truth.tolist(),'keys':keys.tolist(),
        'candidate_has_target':bool(truth[:,:2].any())}


def archive(path, first=0, limit=None):
    tensors, metadata = verify_shard(path)
    out = []
    i = 0
    with np.load(tensors,allow_pickle=False) as data:
        offsets = data['offsets']
        cols = {k:data[k] for k in SNAPSHOT_COLUMNS}
        for num, rec in enumerate(metadata):
            for view in rec['snapshots']:
                lo, hi = offsets[i:i+2]
                i += 1
                if view['scope']!='student' or num < first or (limit is not None and len(out)>=limit):
                    continue
                s = {k:v[lo:hi] for k,v in cols.items()}
                out.append(record(s,rec['case'],rec['v10']['selected_ids'][:3],'actual',rec['layout_id']))
    if i+1 != len(offsets):
        raise RuntimeError(f'Broken archive offsets for {path.name}')
    return out


def collect_archives(args):
    shards = {name:sorted((args.old/'history'/source).glob('*/commit.json')) for name,source in SPLITS.items()}
    if any(not files for files in shards.values()):
        raise RuntimeError('The frozen v13 history shards must be available')
    dest = args.out/'features'
    dest.mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=get_context('spawn'),
                             initializer=init_cpu,initargs=(args.atlas,args.old,False,args.junctions)) as pool:
        futures = {}
        for split, commits in shards.items():
            for commit in commits:
                path = dest/(split+'_'+commit.parent.name+'.json.gz')
                if path.exists():
                    existing = read_gzip(path)
                    if existing['archive_sha256']!=digest(commit):
                        raise RuntimeError('Archived input changed after feature extraction')
                    continue
                futures[pool.submit(archive,commit.parent)] = (path,digest(commit))
        done = 0
        for future in as_completed(futures):
            file, lock = futures[future]
            write_gzip(file,{'archive_sha256':lock,'rows':future.result()})
            done += 1
            if done%10==0 or done==len(futures):
                print('archived trajectory batches',done,'/',len(futures),flush=True)
    results = load_features(args.out)
    layouts = [r['layout_id'] for rows in results.values() for r in rows]
    if len(layouts) != len(set(layouts)) or {k:len(v) for k,v in results.items()}!={'fit':1024,'cal':64,'dev':72}:
        raise RuntimeError('The frozen fit, calibration and developer rosters changed')
    summary = {}
    for name, rows in results.items():
        n = Counter()
        for r in rows:
            outcomes = np.array(r['truth'])
            if len(outcomes):
                n.update(A_and_B=int(np.any(outcomes[:,0]&outcomes[:,1])),
                         alternative_only=int(not outcomes[:,0].any() and outcomes[:,1].any()),
                         candidate_correct=int(np.any(outcomes[:,:2])),
                         no_target_proposal=int(not np.any(outcomes[:,:2])))
        summary[name] = {'layouts':len(rows),'baseline':totals(outcome(rows,[r['baseline_row'] for r in rows])),
                         'proposal_coverage':dict(n)}
    save_json(args.out/'archive_summary.json',summary)
    print(summary,flush=True)


def load_features(out):
    return {name:[row for p in sorted((out/'features').glob(name+'_*.json.gz'))
                  for row in read_gzip(p)['rows']] for name in SPLITS}


def _transformed_layout(result, kind, ordinal):
    """One static smooth warp of all vessels, preserving their source links."""
    from motion import _angular_mix
    scan = result.live.scan
    surface = scan.env.scanner.surface
    layout = scan.tape.layout
    s = np.arange(scan.spec[1],scan.spec[2]+1)
    old = layout.centers[s].copy()
    observed = np.isfinite(old).all(-1)
    support = np.flatnonzero(observed.any(-1))
    orig, rad = surface.shape[:2]
    local = old-orig[s,None,:]
    safe = np.where(observed[...,None],old,orig[s,None,:])
    occ0 = np.stack([surface.occupancy(int(k),u) for k,u in zip(s,safe,strict=True)])

    def cross_distance(delta):
        finite = np.where(observed[...,None],delta,0.)
        t = np.arctan2(finite[...,1],finite[...,0])*12./np.pi
        r = np.stack([_angular_mix(rad[k],theta) for k,theta in zip(s,t,strict=True)])
        return np.linalg.norm(finite,axis=-1)/r

    bound = np.maximum(.965,cross_distance(local)+.005)
    occ_bound = np.maximum(.35,occ0+.045)
    offset = 2.*np.pi*(ordinal*0.38196601125 % 1.)
    final_s = np.flatnonzero(scan.spec[4] == 5)
    # A common displacement across R6, with a smooth entrance in R5. This
    # keeps proximal bifurcations registered and avoids moving unrelated
    # superficial vessels outside the available skin in other regions.
    u = np.clip((np.arange(len(s))-final_s[0]+8.)/8.,0.,1.)
    taper = u*u*(3.-2.*u)
    phase = (np.arange(len(s))-final_s[0])*.16 + offset
    directions = offset+np.arange(12)*np.pi/6.
    refused = Counter()
    for fraction in (1.,.80,.60,.40,.25,.15):
        for angle in directions:
            shift = np.array([np.cos(angle),np.sin(angle)])
            if kind == 'translation':
                delta = local + 9.*fraction*taper[:,None,None]*shift
                turn = np.zeros(len(s))
            else:
                turn = .57*fraction*np.sin(phase)*taper
                c,sine = np.cos(turn),np.sin(turn)
                stretch = fraction*np.cos(phase[:,None])*taper[:,None]
                x0 = local[:,:,0]*(1.+.25*stretch)
                z0 = local[:,:,1]*(1.-.21*stretch)
                delta = np.stack((x0*c[:,None]-z0*sine[:,None],x0*sine[:,None]+z0*c[:,None]),-1)
                delta += 4.0*stretch[...,None]*shift
            if np.any((cross_distance(delta) > bound) & observed):
                refused['skin'] += 1
                continue
            new = orig[s,None,:]+delta
            points = np.where(observed[...,None],new,orig[s,None,:])
            occup = np.stack([surface.occupancy(int(k),p) for k,p in zip(s,points,strict=True)])
            if np.any((occup > occ_bound) & observed):
                refused['bone'] += 1
                continue
            moved = layout.centers.copy()
            moved[s] = new
            d = np.linalg.norm(new-old,axis=-1)
            val = d[final_s][observed[final_s]]
            motion = {'fraction':fraction,'r6_median_mm':float(np.median(val)),
                      'r6_max_mm':float(val.max()),'max_rotation_degrees':float(np.max(abs(turn))*180./np.pi),
                      'retained_station_links':True,'supported_station_count':len(support)}
            return replace(layout,centers=moved), motion
    return None, {'fraction':0.,'reason':'no checked R6 transform','rejected':dict(refused)}


def fixed_pose_replay(result, layout):
    # Re-render with the same contact poses, pulse clock and future innovation
    # tape. The baseline and v14 read exactly these same acquired frames.
    reference = result.live.scan
    arc = result.live.fork()
    new = arc.scan
    new.tape = copy(reference.tape)
    new.tape.layout = layout
    old_reader = reference.node.reader
    reader = type(old_reader)(old_reader.model,old_reader.roles,old_reader.history)
    new.node.reader = reader
    new.prefix = type(reference.prefix)(new.env,new.spec)
    for station,sector,_,_,_,_,_,_,tick in old_reader.frames:
        new.env.scanner._observe(new.node,new.spec,new.tape,station,sector,tick)
    return tr.Result(arc,tr.ENGINE.read(arc),result.actions,result.seen)


def replay_row(result, case, band):
    snapshot = tr.save_terminal(result,'student')
    base = v13_outcome(policy_v13.grade(result))
    row = record(snapshot,case,base['selected_ids'][:3],band,result.live.scan.tape.layout.identifier)
    if row['baseline_ids'] != base['selected_ids'] or not base['completed_legal']:
        raise RuntimeError('Archived baseline ranker or trajectory failed parity')
    row['time_s'] = base['time_s']
    row['travel_mm'] = base['travel_mm']
    return row


def run_case(case_row, *, smoke=False):
    from environment import Case
    raw = case_row['case']
    case = Case(**raw)
    chosen = read_json(_RUN/'test_freeze.json')['selection']
    if chosen['selected']!='student_0':
        raise RuntimeError('Expected the recorded v13 student')
    output = []
    for band,offset in BANDS[:1] if smoke else BANDS:
        live = tr.ENGINE.to_frontier(tr.live_case(case,offset=offset),3)
        result = tr.roll(live,policy_v13.learned(chosen['candidate_rule']))
        row = replay_row(result,raw,band)
        if not smoke:
            known = case_row['tapes'][band]['arms']['frozen_best']
            if row['baseline_ids']!=known['selected_ids'] or not math.isclose(row['time_s'],known['time_s']):
                raise RuntimeError(f'Frozen trajectory did not reproduce {case.identifier}/{band}')
        output.append(row)
        if band=='actual':
            if smoke:
                zero = replay_row(fixed_pose_replay(result,result.live.scan.tape.layout),raw,'zero')
                if row['baseline_ids']!=zero['baseline_ids'] or not np.array_equal(row['state'],zero['state']):
                    raise RuntimeError('Fixed-pose zero warp did not reproduce the actual observations')
            ordinal = int(case.identifier.rsplit('_',1)[-1])
            for kind in ('translation','deformation'):
                layout, deformation = _transformed_layout(result,kind,ordinal)
                if layout is None:
                    output.append({'case':case.identifier,'band':kind,'invalid_warp':deformation})
                    continue
                changed = replay_row(fixed_pose_replay(result,layout),raw,kind)
                changed['deformation'] = deformation
                output.append(changed)
    return output


def smoke(args):
    init_cpu(args.atlas,args.old,True,args.junctions)
    # The observed parent persists across the branch. The second candidate
    # separates over consecutive co-observed stations. A named event is a
    # fixed fixture here; the real distal atlas does not supply that edge.
    group = np.array([0]*5+[1]*4)
    samples = [np.arange(5),np.arange(5,9)]
    station = np.array([10,11,12,13,14,11,12,13,14])
    point = np.array([[0.,0.]]*5+[[2.,0.],[3.,0.],[8.,0.],[11.,0.]])
    frame = station.copy()
    role = np.zeros((9,8))
    role[:5,0],role[5:,1] = 1.,1.
    join = Junction(11,0,1,'bifurcation',1.)
    split, yes = _pair_features(2,6,group,samples,station,point,frame,role,(join,))
    _, oppose = _pair_features(2,6,group,samples,station,point,frame,role,(replace(join,kind='confluence'),))
    _, absent = _pair_features(2,6,group,samples,station,point,frame,role,())
    yaw = 1.4
    rigid = np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
    rotated, stable = _pair_features(2,6,group,samples,station,point@rigid+[34.,-42.],frame,role,(join,))
    np.testing.assert_allclose(split,rotated,atol=1.e-6)
    np.testing.assert_allclose(yes,stable,atol=1.e-6)
    assert yes[0]>.20 and oppose[0]==-yes[0] and not absent.any() and not yes[1]
    # Mirrored along the material axis, sort the per-track samples back into
    # scan order. A confluence supplies evidence on the opposite side.
    _, merge = _pair_features(2,6,group,[v[::-1] for v in samples],24-station,point,frame,role,
                              (Junction(13,0,1,'confluence',1.),))
    assert merge[0]>.20 and not merge[1]
    assert junction_compatibility(None,'bifurcation',1.) == 0.
    bent = (point @ np.diag([1.65,.76]))@rigid + np.column_stack((station**1.4*.5,2.*station))
    _, bent_value = _pair_features(2,6,group,samples,station,bent,frame,role,(join,))
    assert bent_value[0]>.20 and not bent_value[1]
    rows = []
    for case in list(tr.cases('reader_dev',4)):
        rows.extend(run_case({'case':asdict(case)},smoke=True))
    regions = {b:[r for r in rows if r['band']==b and 'baseline_ids' in r] for b in
               ('actual','translation','deformation')}
    summary = {'named_bifurcation_bonus':float(yes[0]),'named_confluence_bonus':float(merge[0]),
               'contradiction_bonus':float(oppose[0]),'missing_bonus':float(absent[0]),
               'rigid_relation_invariance':True,'checked_bent_branch_bonus':float(bent_value[0]),
               'zero_warp_baseline_parity':True,
               'pretraining_developer_replays':{k:{'layouts':len(v),'capture':sum(r['captured'] for r in v),
                  'candidate_target_present':sum(r['candidate_has_target'] for r in v)} for k,v in regions.items()},
               'invalid_warps':sum('invalid_warp' in r for r in rows)}
    write_gzip(args.out/'smoke_rows.json.gz',rows)
    save_json(args.out/'smoke.json',summary)
    print(summary,flush=True)
    if len(regions['translation'])<2 or len(regions['deformation'])<2:
        raise RuntimeError('No suitable checked geometry stress development replay')


def _paired(base, plus):
    assert [(r['case'],r['band']) for r in base]==[(r['case'],r['band']) for r in plus]
    deltas = {}
    for metric in ('strict','wrong','r6_correct','r6_wrong_branch','r6_artery'):
        d = np.array([r[metric]-s[metric] for s,r in zip(base,plus,strict=True)])
        deltas[metric] = {'gained':int((d>0).sum()),'lost':int((d<0).sum()),'net':int(d.sum())}
    return deltas


def _block_intervals(base, alt):
    order = sorted(set(r['case'] for r in base))
    if not order:
        return {}
    ranges = {}
    rng = np.random.default_rng(RESAMPLE_SEED)
    indexes = rng.integers(0,len(order),size=(6000,len(order)))
    for metric in ('strict','wrong','r6_correct','r6_wrong_branch','r6_artery'):
        groups = {c:[] for c in order}
        for x,y in zip(base,alt,strict=True):
            groups[x['case']].append(y[metric]-x[metric])
        v = np.asarray([np.mean(groups[c]) for c in order])*100.
        ranges[metric] = {'point_pp':float(v.mean()),
                          'ci95_pp':np.quantile(v[indexes].mean(axis=1),[.025,.975]).tolist()}
    return ranges


def evaluate(args):
    freeze = read_json(args.out/'freeze.json')
    for name in NAMES:
        if digest(args.out/f'{name}.json')!=freeze['models'][name]['model_sha256']:
            raise RuntimeError('A readout changed after freeze')
    test = read_json(args.old/'test_paired.json')
    if len(test)!=108:
        raise RuntimeError('Expected the 108 frozen v13 full trajectories')
    directory = args.out/'replay'
    directory.mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=get_context('spawn'),
                             initializer=init_cpu,initargs=(args.atlas,args.old,True,args.junctions)) as pool:
        jobs = {}
        for case in test:
            file = directory/(case['case']['identifier']+'.json.gz')
            if not file.exists():
                jobs[pool.submit(run_case,case)] = file
        for i, future in enumerate(as_completed(jobs),1):
            write_gzip(jobs[future],future.result())
            if i%12==0 or i==len(jobs):
                print('paired actual + two proof tapes with fixed-pose stress',i,'/',len(jobs),flush=True)
    data = [r for f in sorted(directory.glob('*.json.gz')) for r in read_gzip(f)]
    actual = [r for r in data if r['band'] == 'actual']
    model = {k:read_json(args.out/f'{k}.json') for k in NAMES}
    res = {'test_layouts':len(actual),'invalid_warps':[r for r in data if 'invalid_warp' in r],
           'branches':{},'band_intervals':{},'mean_time_s_actual':float(np.mean([r['time_s'] for r in actual]))}
    chosen = {}
    for tag in [name for name,_ in BANDS]+['translation','deformation']:
        rows = [r for r in data if r['band']==tag and 'invalid_warp' not in r]
        if not rows:
            continue
        x = materialize(rows)
        base = outcome(rows,[r['baseline_row'] for r in rows])
        arms = {'v13':base}
        detail = {'layouts':len(rows),'v13':totals(base)}
        if tag in ('translation','deformation'):
            detail['realized_geometry'] = {k:float(np.median([r['deformation'][k] for r in rows]))
                                           for k in ('r6_median_mm','r6_max_mm','max_rotation_degrees')}
        for name,head in model.items():
            p = softmax(numpy_logits(head,x)/head['temperature'],axis=-1)
            out = outcome(rows,decide(p,rows,x,Rule(**head['rule'])))
            arms[name] = out
            detail[name] = totals(out) | {'paired':_paired(base,out), 'layout_ci':_block_intervals(base,out)}
        res['branches'][tag] = detail
        chosen[tag] = arms
    for group,tags in [('three_tapes',['actual','proof_0','proof_1']),('proof_only',['proof_0','proof_1'])]:
        baseline = [row for tag in tags for row in chosen[tag]['v13']]
        res['band_intervals'][group] = {name:_block_intervals(baseline,[row for tag in tags for row in chosen[tag][name]])
                                        for name in model}
    write_gzip(args.out/'paired.json.gz',chosen)
    save_json(args.out/'results.json',res)
    print({b:{a:v['strict'] for a,v in s.items() if a in ('v13',*NAMES)} for b,s in res['branches'].items()},flush=True)


def fit(args):
    if not read_json(args.out/'gpu_preflight.json')['cuda_optimizer_step_verified']:
        raise RuntimeError('A real CUDA self-check is required')
    smoke_info = read_json(args.out/'smoke.json')
    if not smoke_info['zero_warp_baseline_parity'] or any(
            entry['layouts']<2 for entry in smoke_info['pretraining_developer_replays'].values()):
        raise RuntimeError('Readout validation must precede training')
    summary = train(load_features(args.out),args.out,seed=FIT_SEED)
    probes = load_features(args.out)['cal'][:5]
    x = materialize(probes)
    swapped = x | {key:x[key][:,:,::-1].copy() for key in ('state','position','motif','valid')}
    for name in NAMES:
        model = read_json(args.out/f'{name}.json')
        v, w = numpy_logits(model,x),numpy_logits(model,swapped)
        np.testing.assert_allclose(v[..., [1,0,2]],w,atol=2.e-6)
    selected = {k:{f:summary[k][f] for f in ('model_sha256','checkpoint_sha256','selected','parity_max')}
                for k in NAMES}
    save_json(args.out/'freeze.json',{'models':selected,'swap_invariance_verified':True,
              'protocol_sha256':digest(args.out/'protocol.json'),
              'selection_rule':'Developer strict deliveries, then fewer wrong branches and R6 success; no test selection',
              'readout_scope':'R6 only on the subsequential frozen acquisition. Full six-region strict and wrong branches scored.',
              'comparison':'v13 / v14 relative layout only / v14 graph', 'r6_joint_outcomes':['A','B','unknown'],
              'bootstrap_seed':RESAMPLE_SEED})


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument('step',choices=('smoke','prepare','fit','evaluate'))
    p.add_argument('--old',type=Path,default=V13/'run')
    p.add_argument('--atlas',type=Path,default=DEFAULT_PACK)
    p.add_argument('--junctions',type=Path,help='Optional public atlas Junction records; omit for the archived distal arm')
    p.add_argument('--out',type=Path,default=ROOT/'run')
    p.add_argument('--workers',type=int,default=10)
    args = p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    args.old, args.atlas = args.old.resolve(),args.atlas.resolve()
    old = read_json(args.old/'protocol.json')
    if digest(args.atlas)!=old['atlas_sha256'] or any(digest(V13/k)!=v for k,v in old['sources'].items()):
        raise RuntimeError('Frozen v13 source or atlas changed')
    cfg = {'source':{f.name:digest(f) for f in sorted(ROOT.glob('*.py'))},'atlas_sha256':digest(args.atlas),
           'v13_test_freeze_sha256':digest(args.old/'test_freeze.json'),
           'v13_reader_freeze_sha256':digest(args.old/'reader_freeze.json'),
           'v13_data_freeze_sha256':digest(args.old/'data_freeze.json'),
           'gpu_preflight_sha256':digest(args.out/'gpu_preflight.json'),
           'seed':FIT_SEED,'archive_splits':{'fit':1024,'cal':64,'dev':72},
           'relative_location_basis':'Observed displacements between different echoes at a common station and frame',
           'position_limit':POSITION_LIMIT,'tested_current_scans':108,'test_bands':[list(x) for x in BANDS],
           'declared_named_junctions_sha256':digest(args.junctions) if args.junctions else None,
           'declared_named_junction_count':len(read_json(args.junctions)) if args.junctions else 0}
    protocol = args.out/'protocol.json'
    if protocol.exists() and read_json(protocol)!=cfg:
        raise RuntimeError('Run protocol changed. Start a new run directory for changed code.')
    if not protocol.exists():
        save_json(protocol,cfg)
    {'smoke':smoke,'prepare':collect_archives,'fit':fit,'evaluate':evaluate}[args.step](args)


if __name__=='__main__':
    main()
