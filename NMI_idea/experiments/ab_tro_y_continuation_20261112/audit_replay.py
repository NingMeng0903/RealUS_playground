"""Portable version of the independent Astra replay of the frozen moving-Y study.

Uses the exact previously transported branch and logged mechanics. It replays
the observation likelihood, not atlas rebaking or controller execution.
"""
from pathlib import Path
from collections import Counter
import hashlib, json, lzma, sys
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
ARCHIVED_REPO = Path('/media/camp/EXT_DRIVE/RealUS_playground')
OUT=ROOT/'holdout'
assert (OUT/'COMPLETE.json').exists()
manifest=json.loads((ROOT/'FROZEN_MANIFEST.json').read_text())
complete=json.loads((OUT/'COMPLETE.json').read_text())
sha=lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
portable = json.loads((ROOT/'review_inputs'/'MANIFEST.json').read_text())
branch_path = ROOT/'review_inputs'/portable['asset']
assert sha(branch_path) == portable['sha256']
excluded = {Path('NMI_idea/experiments') / p for p in portable['excluded_upstream']}
skipped, checked = set(), 0
for origin, checksum in manifest['sources'].items():
    relative = Path(origin).relative_to(ARCHIVED_REPO)
    local = REPO / relative
    if not local.is_file() and relative in excluded:
        skipped.add(relative)
        continue
    assert sha(local) == checksum, relative
    checked += 1
def archived_jsonl(name, checksum):
    with lzma.open(OUT / (name + '.xz'), 'rb') as stream:
        data = stream.read()
    assert hashlib.sha256(data).hexdigest() == checksum, name
    return [json.loads(line) for line in data.splitlines()]

common = archived_jsonl('COMMON.jsonl', complete['common_sha256'])
runs = archived_jsonl('RUNS.jsonl', complete['rollouts_sha256'])
run_manifest=json.loads((OUT/'MANIFEST.json').read_text())
assert run_manifest['frozen_manifest_sha256']==sha(ROOT/'FROZEN_MANIFEST.json')
assert run_manifest['executed_seed_list']==manifest['holdout_seed_list']
arms=('online_B','online_MPC','frozen_B')
reserves=('narrow','wide')
assert len(common)==128 and len(runs)==768
lookup={(r['index'],r['reserve'],r['arm']):r for r in runs}
assert len(lookup)==768 and set(lookup)=={(i,z,a) for i in range(128) for z in reserves for a in arms}
assert [c['seed'] for c in common]==manifest['holdout_seed_list']
assert len(set(manifest['holdout_seed_list']))==128
assert manifest['holdout_seed_list']==[int(np.random.SeedSequence([2026111217,i]).generate_state(1)[0]) for i in range(128)]
json_digest=lambda x: hashlib.sha256(json.dumps(x,sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()

metrics=('travel_mm','turn_deg','duration_s')
max_cost_error=0.; motion_count=0; max_speed=0.; max_handoff=0.
def check_trace(m):
    global max_cost_error,motion_count,max_speed
    s=m['samples']; t=np.asarray(s['t_s']); x=np.asarray(s['atlas_xyz_mm']); q=np.asarray(s['q'])
    rotations=Rotation.from_matrix(s['robot_tcp_rotation'])
    assert len(t)==len(x)==len(q)==len(rotations) and np.all(np.diff(t)>0)
    speed=float(np.max(abs(np.diff(q,axis=0))/np.diff(t)[:,None]/np.asarray(manifest['mechanic_preflight']['model_joint_velocity_limits'])))
    max_speed=max(max_speed,speed)
    assert speed<=1.+1e-8
    assert np.allclose(q[:,0],manifest['mechanic_preflight']['fixed_rail_m'],atol=1e-10,rtol=0)
    cost=np.array([np.linalg.norm(np.diff(x,axis=0),axis=1).sum(),
                   np.rad2deg((rotations[:-1].inv()*rotations[1:]).magnitude()).sum(),t[-1]-t[0]])
    err=float(abs(cost-np.array([m[k] for k in metrics])).max()); max_cost_error=max(err,max_cost_error)
    assert err<1e-8
    assert abs(t[0]-m['t_start'])<1e-9 and abs(t[-1]-m['t_end'])<1e-9
    if m['legal_completed']:
        assert m['min_self_collision_m']>=.01-1e-10
        assert m['max_reference_fk_error_mm']<=.25+1e-8 and m['max_reference_fk_error_deg']<=.15+1e-8
        assert m['image_pose_fk_difference_mm']<=.25+1e-8 and m['image_pose_fk_difference_deg']<=.15+1e-8
        assert abs(m['stroke_mm']-m['commanded_stroke_mm'])<1e-10
        assert abs(cost[2]-m['commanded_duration_s'])<1e-9
    motion_count+=1
    return cost

for c in common:
    assert len(c['prefix_frames'])==28 and c['index'] in range(128)
    p=[[f['t'],f['q'],f['velocity'],f['site'],f['view'],f['centroids_mm']] for f in c['prefix_frames']]
    assert hashlib.sha256(json.dumps(p,separators=(',',':')).encode()).hexdigest()==c['prefix_sha']
    assert json_digest(c['common_probe_frame'])==c['common_probe_sha']
    labels=c['probe_labels_for_evaluation']
    assert c['common_probe_visible_for_evaluation']==bool(labels is not None and 1 in labels)
    check_trace(c['probe_motion'])

for r in runs:
    c=common[r['index']]
    assert r['seed']==c['seed'] and r['prefix_sha']==c['prefix_sha'] and r['common_probe_sha']==c['common_probe_sha']
    assert r['common_probe_visible_for_evaluation']==c['common_probe_visible_for_evaluation']
    previous=c['probe_motion']; cost=np.zeros(3); stroke=0.
    for stage in r['stages']:
        m=stage['motion']; f=stage['frame']; samples=m['samples']
        error=float(abs(np.asarray(previous['samples']['q'][-1])-samples['q'][0]).max())
        max_handoff=max(max_handoff,error); assert error<1e-10
        assert abs(previous['t_end']-m['t_start'])<1e-9
        assert previous['goal']==m['start']
        cost+=check_trace(m); stroke+=m['stroke_mm']; previous=m
        assert (f is not None)==m['legal_completed']
        if f is not None:
            assert [f['site'],f['view']]==m['goal'] and abs(f['t']-m['t_end'])<1e-9
            assert len(f['centroids_mm'])==len(stage['labels_for_evaluation'])
    assert np.allclose(cost,[r[k] for k in metrics],atol=1e-8,rtol=0)
    legal=(len(r['stages'])==2 and all(s['motion']['legal_completed'] for s in r['stages'])
           and cost[0]<=60.+1e-9 and cost[1]<=60.+1e-9 and cost[2]<=1.5+1e-9
           and stroke<=manifest['reserve_mm'][r['reserve']]+1e-9)
    assert r['legal_completed']==legal
    if len(r['stages'])==2:
        assert r['stages'][-1]['motion']['path_id'] in manifest['mechanic_preflight']['legal_graph'][r['reserve']][str(r['first_view'])]
    claim=r['final_claim']; assert claim['event']==30 and claim['site']==4
    declared=claim['confidence']>=.87; assert declared==claim['confirmed']
    label=None
    if claim['ordinal'] is not None:
        last=r['stages'][-1]; ordinal=claim['ordinal']; label=last['labels_for_evaluation'][ordinal]
        assert last['frame']['site']==4 and last['frame']['t']>c['common_probe_frame']['t']
        assert np.allclose(last['frame']['centroids_mm'][ordinal],claim['centroid_mm'])
    assert r['strict']==bool(legal and declared and label==1)
    assert r['correct_fresh']==bool(declared and label==1)
    assert r['false_confirm']==bool(declared and label!=1)
    assert r['false_confirm_trunk']==bool(declared and label==0)
    assert r['false_confirm_clutter']==bool(declared and label==-1)
    if 'planned' in r:
        choices=r['planning_candidates']; selected=max(choices,key=lambda k:k['score'])
        assert selected==r['planned'] and selected['first_view']==r['first_view']
        for candidate in choices:
            hist=candidate['adaptive_tail_histogram']
            assert sum(hist.values())==32 and candidate['recourse_gap']>=-1e-12
            assert np.isclose(candidate['conditional_tail_value']-candidate['open_loop_tail_value'],candidate['recourse_gap'])
            assert candidate['open_loop_tail_view']==candidate['legal_tail_views'][int(np.argmax(candidate['open_loop_tail_scores']))]
        tail=r['real_tail_scores']
        assert r['actual_tail_view']==tail['views'][int(np.argmax(tail['values']))]
        assert r['tail_switched']==(r['actual_tail_view']!=selected['open_loop_tail_view'])

for i in range(128):
    for z in reserves:
        a,b=lookup[i,z,'online_B'],lookup[i,z,'online_MPC']
        for ca,cb in zip(a['planning_candidates'],b['planning_candidates']):
            assert {k:v for k,v in ca.items() if k!='score'}=={k:v for k,v in cb.items() if k!='score'}
        if a['first_view']==b['first_view']:
            assert a['stages']==b['stages'] and a['final_claim']==b['final_claim'] and a['strict']==b['strict']

# Reconstruct the filter from public masks only, then directly integrate the new
# terminal mask's daughter responsibility. Do not invoke run_study.target_claim.
sys.path.insert(0,str(ROOT.parent/'ab_map_recheck_20261112'))
from map_model import MapFilter,MapWorld
from belief import Reading
# Use the recorded public rule route and already transported branch. The audit
# integrates image likelihoods and checks logged motion independently below;
# it does not call or replace the executor used to produce the frozen traces.
from unittest.mock import patch
import dynamic_scan
import knee_world
import learned_prodmp

with np.load(branch_path, allow_pickle=False) as data:
    posed = (data['dense'], data['s'], data['daughter'], tuple(map(int, data['sites'])), int(data['fork']))
rule_route = tuple((f['site'], f['view']) for f in common[0]['prefix_frames'])
assert all(tuple((f['site'], f['view']) for f in row['prefix_frames']) == rule_route for row in common)
with (patch.object(dynamic_scan, 'posed_branch', return_value=posed),
      patch.object(knee_world, 'Edges', return_value=None),
      patch.object(knee_world.KneeWorld, '_route', return_value=rule_route),
      patch.object(learned_prodmp, 'REFERENCE_MOTION', ROOT.parent/'ab_observation_stress_20261111'/'motion.py')):
    world = MapWorld(mode='stress')
def assimilate(b,f,event):
    b.observe(world,Reading(f['t'],f['q'],f['velocity']),f['site'],f['view'],
              np.asarray(f['centroids_mm']).reshape(-1,2),event=event,retain=False,track=False)
max_conf_error=0.; max_mass_error=0.; replayed=0
for c in common:
    p=MapFilter()
    for event,f in enumerate(c['prefix_frames']): assimilate(p,f,event)
    original=p.weights.sum(axis=1).copy()
    starts={False:p.copy(),True:p.copy().freeze_map()}
    for b in starts.values(): assimilate(b,c['common_probe_frame'],28)
    for z in reserves:
        for a in arms:
            r=lookup[c['index'],z,a]; b=starts[a=='frozen_B'].copy()
            for event,stage in enumerate(r['stages'],29):
                f=stage['frame']
                if f is None: continue
                if event==30:
                    reading=Reading(f['t'],f['q'],f['velocity'])
                    prior,_=b.prior_at(reading)
                    density,responsibility=b.emission(world,reading,f['site'],f['view'],np.asarray(f['centroids_mm']).reshape(-1,2))
                    if len(f['centroids_mm']):
                        if b.frozen_marginal is None:
                            probabilities=(responsibility[:,1]*prior).sum(axis=(1,2))/(density*prior).sum()
                        else:
                            evidence=(density*prior).sum(axis=1)
                            conditional=(responsibility[:,1]*prior).sum(axis=2)/evidence[None,:]
                            probabilities=(conditional*b.frozen_marginal[None,:]).sum(axis=1)
                        ordinal=int(np.argmax(probabilities)); conf=float(probabilities[ordinal])
                    else: ordinal,conf=None,0.
                    assert ordinal==r['final_claim']['ordinal']
                    max_conf_error=max(max_conf_error,abs(conf-r['final_claim']['confidence']))
                    assert abs(conf-r['final_claim']['confidence'])<1e-10
                    replayed+=1
                assimilate(b,f,event)
            change=float(abs(b.weights.sum(axis=1)-original).sum())
            assert abs(change-r['map_marginal_change_l1'])<1e-10
            if a=='frozen_B': max_mass_error=max(max_mass_error,change)

def exact_empirical_bootstrap(d,scale=2):
    """Exact convolution of the empirical subject distribution; no Monte Carlo."""
    d=np.asarray(d); values=np.rint(d*scale).astype(int); assert np.max(abs(values/scale-d))<1e-10
    lo,hi=int(values.min()),int(values.max()); pmf=np.bincount(values-lo,minlength=hi-lo+1)/len(values)
    distribution=np.array([1.])
    for _ in values: distribution=np.convolve(distribution,pmf)
    grid=(np.arange(len(distribution))+len(values)*lo)/(len(values)*scale)
    cdf=np.cumsum(distribution); cdf[-1]=1.
    quant=lambda p:float(grid[np.searchsorted(cdf,p)])
    return dict(mean=float(d.mean()),ci95=[quant(.025),quant(.975)],lower_one_sided_975=quant(.025),upper_one_sided_975=quant(.975))

def values(a,k,ids,z=None):
    return np.asarray([lookup[i,z,a][k] if z else np.mean([lookup[i,x,a][k] for x in reserves]) for i in ids],dtype=float)
def group(ids):
    result={}
    for z in (*reserves,'pooled'):
        zi=None if z=='pooled' else z
        table={}
        for a in arms:
            rows=[lookup[i,zz,a] for i in ids for zz in (reserves if zi is None else (zi,))]
            table[a]={k:sum(bool(r[k]) for r in rows) for k in ('strict','false_confirm','false_confirm_trunk','false_confirm_clutter','legal_completed','target_present')}
            table[a]['n_rows']=len(rows)
            table[a]['cost_mean']={k:float(values(a,k,ids,zi).mean()) for k in metrics}
            table[a]['tail_switches']=sum(bool(r['tail_switched']) for r in rows)
            table[a]['planned_contingent_selected']=sum(sum(v>0 for v in r['planned']['adaptive_tail_histogram'].values())>1 for r in rows)
            table[a]['first_views']=dict(Counter(r['first_view'] for r in rows))
            table[a]['tail_views']=dict(Counter(r['actual_tail_view'] for r in rows))
        comparisons={}
        for b in ('online_MPC','frozen_B'):
            d=values('online_B','strict',ids,zi)-values(b,'strict',ids,zi)
            risk=values('online_B','false_confirm',ids,zi)-values(b,'false_confirm',ids,zi)
            v=exact_empirical_bootstrap(d); v['wrong_confirm']=exact_empirical_bootstrap(risk)
            v['subject_positive_negative_zero']=[int((d>0).sum()),int((d<0).sum()),int((d==0).sum())]
            if zi:
                wins,losses=int((d>0).sum()),int((d<0).sum())
                v['exact_mcnemar_p']=float(binomtest(wins,wins+losses,.5).pvalue) if wins+losses else 1.
            comparisons[b]=v
        result[z]={'table':table,'contrasts':comparisons}
    return result

ids=list(range(128)); visible=[i for i,c in enumerate(common) if c['common_probe_visible_for_evaluation']]
all_case=group(ids); sensitivity=group(visible)
cost_d=np.column_stack([values('online_B',k,ids)-values(b,k,ids) for b in ('online_MPC','frozen_B') for k in metrics])
rng=np.random.default_rng(20261113017); boot=[]
for _ in range(20):
    index=rng.integers(0,128,size=(10000,128)); boot.append(cost_d[index].mean(axis=1))
boot=np.concatenate(boot)
quant=np.quantile(boot,[.05/12,1-.05/12],axis=0)
costs={}; margins=(2.,2.,.1)
for bi,b in enumerate(('online_MPC','frozen_B')):
    costs[b]={}
    for ki,k in enumerate(metrics):
        j=bi*3+ki; ci=quant[:,j].tolist()
        costs[b][k]=dict(mean=float(cost_d[:,j].mean()),simultaneous95_bonferroni_ci=ci,margin=margins[ki],equivalent=bool(ci[0]>=-margins[ki] and ci[1]<=margins[ki]))

mechanism={}
for a in arms:
    rows=[lookup[i,'wide',a] for i in ids]
    mechanism[a]=dict(reserve_first_or_tail_difference=sum(lookup[i,'wide',a]['first_view']!=lookup[i,'narrow',a]['first_view'] or lookup[i,'wide',a]['actual_tail_view']!=lookup[i,'narrow',a]['actual_tail_view'] for i in ids),
                      switched_success=sum(r['strict'] and r['tail_switched'] for r in rows),
                      switched_wrong=sum(r['false_confirm'] and r['tail_switched'] for r in rows),
                      selected_recourse_gap_mean=float(np.mean([r['planned']['recourse_gap'] for r in rows])),
                      selected_recourse_gap_max=float(max(r['planned']['recourse_gap'] for r in rows)))
output=dict(independent_checks=dict(source_hashes=34,cohort_subjects=128,run_rows=768,trace_records_checked=motion_count,
                                   fresh_confidences_independently_replayed=replayed,max_confidence_error=max_conf_error,
                                   max_frozen_mass_error=max_mass_error,max_cost_recalculation_error=max_cost_error,
                                   max_q_handoff_error=max_handoff,max_observed_joint_rate_fraction=max_speed),
            common_probe=dict(target_present=len(visible),empty=sum(c['common_probe_empty'] for c in common),
                              wrong_only=sum(not c['common_probe_visible_for_evaluation'] and not c['common_probe_empty'] for c in common)),
            all_cases=all_case,target_present_sensitivity=sensitivity,cost_comparisons=costs,
            cost_bootstrap=dict(seed=20261113017,repetitions=200000,subject_cluster=True,interval='Bonferroni 6 comparisons, two-sided simultaneous 95%'),
            primary_bootstrap='Exact convolution of empirical paired subject distribution, averaging the two reserves first',
            mechanism=mechanism)
output['independent_checks']['source_hashes'] = checked
output['independent_checks']['upstream_hashes_not_checked'] = sorted(str(p) for p in skipped)
output['independent_checks']['portable_branch_sha256'] = portable['sha256']
print(json.dumps(output,indent=2))
