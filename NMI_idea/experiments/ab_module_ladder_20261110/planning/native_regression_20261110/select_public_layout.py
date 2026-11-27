"""Place the local scan patch using public commands/preferences, never anatomy GT."""
from pathlib import Path
import json
import sys
import argparse
import numpy as np
import yaml

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from run_controlled_comparison import ROOT,CapabilityEvaluator,MOTION,body_transform,clean_json,sha
from system import env
from observation.servo import template_candidates


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--basis',choices=['commands','prior'],default='commands')
    args=parser.parse_args()
    output=HERE/('public_layout' if args.basis=='commands' else 'public_prior_layout');output.mkdir(exist_ok=False)
    cfg=Path('/media/camp/EXT_DRIVE/RealUS_playground/peirastic/configs/controller.yaml')
    raw=yaml.safe_load(cfg.read_text())
    nominal=np.deg2rad(np.array(raw['inner']['nullspace']['q_nominal_deg']))
    nominal[0]=.4
    cases=[('nominal',nominal.copy())]
    for axis,offset in [(1,np.deg2rad(10)),(2,np.deg2rad(5)),(4,np.deg2rad(10)),(0,.1)]:
        for sign in [-1,1]:
            q=nominal.copy();q[axis]+=sign*offset
            cases.append((f'q{axis}_{sign:+d}',q))
    cohort_path=ROOT/'planning/test_frozen/results.json'
    paths=[];targets=[]
    if args.basis=='commands':
        cohort=json.loads(cohort_path.read_text())
        for row in cohort['rows']:
            if row['method']!='B_two_step' or row['object_index_for_audit_only'] not in (0,1):continue
            previous=np.zeros(3)
            for event in row['events']:
                if 'motion_s' not in event:continue
                target=np.array(event['target']);paths.extend(MOTION.path(previous,target));targets.append(target)
                previous=target
    else:
        # This branch does not open heldout records or observations. Use only
        # the public coarse prior's four mode summaries at three scan stations,
        # plus the same small yaw probes available to every online policy.
        particles=env.sample_prior(720502,32);weights=np.ones(32)/32
        for x in [-.025,0.,.025]:
            for base in template_candidates(particles,weights,np.array([x,0.,0.])):
                for yaw_delta in [-.06,0.,.06]:
                    target=base.copy();target[2]+=yaw_delta;target=env.clip_pose(target)
                    paths.extend(MOTION.path(np.zeros(3),target));targets.append(target)
    targets=np.unique(np.round(targets,7),axis=0)
    evaluator=CapabilityEvaluator()
    result=dict(scope='New local scan-patch/robot setup, public geometric commands only; no GT, no full-leg validation',
                basis=args.basis,cohort_sha256=sha(cohort_path) if args.basis=='commands' else None,
                prior_seed=720502 if args.basis=='prior' else None,config_sha256=sha(cfg),nominal_q=nominal.tolist(),
                selection='max minimum path dIRD calibrated margin - .05*normalized squared nominal-q displacement, subject to all checked endpoints found',candidates=[])
    for name,q in cases:
        check=evaluator.check_q(q)
        if not check['valid']:
            result['candidates'].append(dict(id=name,q_seed=q.tolist(),valid=False,check=check));continue
        anchor=evaluator.fk(q)
        matrices=np.array([body_transform(p,anchor) for p in paths])
        margins=np.array(evaluator.score(matrices,q[0])['raw_calibrated_margin'])
        checks=[evaluator._static_pose(body_transform(p,anchor),float(q[0]),q) for p in targets]
        found=all(c['found'] for c in checks)
        delta=(q-nominal)/np.array([.1]+[.2]*7)
        score=float(margins.min()-.05*np.dot(delta,delta))
        row=dict(id=name,q_seed=q.tolist(),T_world_body=anchor.tolist(),valid=bool(found),
                 checked_endpoint_count=len(checks),found_count=sum(c['found'] for c in checks),
                 min_dird_margin=float(margins.min()),score=score,initial_check=check)
        result['candidates'].append(row)
        print(name,'min_dird',margins.min(),'endpoints',row['found_count'],'/',len(checks),'score',score,flush=True)
    legal=[r for r in result['candidates'] if r['valid']]
    if not legal:raise ValueError('No proposed setup passed public endpoint checks')
    selected=max(legal,key=lambda x:x['score'])
    result['selected_id']=selected['id']
    (output/'CANDIDATES.json').write_text(json.dumps(clean_json(result),indent=2,allow_nan=False))
    (output/'SELECTED.json').write_text(json.dumps(clean_json(selected),indent=2,allow_nan=False))
    print('SELECTED',selected['id'],flush=True)
