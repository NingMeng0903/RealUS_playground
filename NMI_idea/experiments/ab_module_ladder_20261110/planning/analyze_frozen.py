"""Object-clustered paired analysis, retaining all registered failures."""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def analyze(path):
    x=json.loads(path.read_text()); rows=x['rows']
    methods=sorted({r['method'] for r in rows})
    ids=sorted({r['object_index_for_audit_only'] for r in rows})
    contexts=sorted({r['context'] for r in rows})
    assert len(rows)==len(methods)*len(ids)*len(contexts)
    by={(r['object_index_for_audit_only'],r['context'],r['method']):r for r in rows}
    assert len(by)==len(rows)
    rng=np.random.default_rng(720504)
    boot=rng.integers(0,len(ids),size=(2000,len(ids)))
    metrics={'success':'correct_public_stop','wrong_stop':'false_public_stop','RMT_s':'rmt_s'}
    means={}; arrays={}
    for method in methods:
        m={}
        for label,key in metrics.items():
            a=np.array([[float(by[h,c,method][key]) for c in contexts] for h in ids])
            arrays[method,label]=a.mean(1)
            v=arrays[method,label][boot].mean(1)
            m[label]=dict(mean=float(a.mean()),object_clustered_95pct=list(np.quantile(v,[.025,.975])))
        selected=[r for r in rows if r['method']==method]
        logged_plan=[sum(e.get('planning_s',0) for e in r['events']) for r in selected]
        logged_bayes=[sum(e.get('belief_update_s',0) for e in r['events']) for r in selected]
        logged_motion=[sum(e.get('motion_s',0) for e in r['events']) for r in selected]
        logged_hold=[.5*sum('motion_s' in e for e in r['events']) for r in selected]
        logged_endpoint=[sum(e.get('endpoint_s',0) for e in r['events']) for r in selected]
        unlogged=[r['elapsed_s']-sum(s[i] for s in [logged_plan,logged_bayes,logged_motion,
                  logged_hold,logged_endpoint]) for i,r in enumerate(selected)]
        m['accounting']=dict(mean_logged_planning_s=float(np.mean(logged_plan)),
            mean_logged_bayes_s=float(np.mean(logged_bayes)),
            mean_logged_motion_s=float(np.mean(logged_motion)),
            mean_logged_hold_s=float(np.mean(logged_hold)),
            mean_logged_endpoint_s=float(np.mean(logged_endpoint)),
            mean_unlogged_termination_or_budget_s=float(np.mean(unlogged)),
            note='Unlogged work includes terminal planning with no acquisition and budget exhaustion; failures receive RMT60 independently of elapsed time.')
        m['endpoint_queries']=sum(r['endpoint_queries'] for r in selected)
        m['invalid_endpoint_attempts']=sum(r['invalid_endpoint_attempts'] for r in selected)
        m['correct_count']=sum(r['correct_public_stop'] for r in selected)
        m['wrong_count']=sum(r['false_public_stop'] for r in selected)
        m['n_episodes']=len(selected)
        m['success_by_s']={str(t):sum(r['correct_public_stop'] and r['rmt_s']<=t for r in selected)
                           for t in [15,30,45,60]}
        m['mean_final_true_hypothesis_weight']=float(np.mean([
            r['posterior'][r['object_index_for_audit_only']] for r in selected]))
        m['mean_final_raw_ESS']=float(np.mean([1/np.square(r['posterior']).sum() for r in selected]))
        m['branch_decisions']=sum(e.get('decision',{}).get('kind')=='branch'
                                  for r in selected for e in r['events'])
        means[method]=m
    paired={}
    for base in methods:
        if base=='B_two_step':continue
        paired[base]={}
        for label in metrics:
            difference=arrays['B_two_step',label]-arrays[base,label]
            boot_diff=difference[boot].mean(1)
            paired[base][label]=dict(B_minus_baseline=float(difference.mean()),
                                    object_clustered_95pct=list(np.quantile(boot_diff,[.025,.975])))
    return dict(scope=x['scope'],source=digest(path),protocol_sha256=x['protocol_sha256'],
                objects=len(ids),contexts=contexts,methods=means,paired=paired,
                uncertainty='Paired bootstrap over16 anatomy objects, both contexts kept within a cluster; small synthetic cohort, descriptive interval only.')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('results',type=Path)
    args=parser.parse_args()
    output=analyze(args.results)
    (args.results.parent/'ANALYSIS.json').write_text(json.dumps(output,indent=2,allow_nan=False))
    for method,row in output['methods'].items():
        print(method,row['correct_count'],'/',row['n_episodes'],'wrong',row['wrong_count'],
              'RMT',row['RMT_s']['mean'],'plan',row['accounting']['mean_logged_planning_s'])
    print(json.dumps(output['paired'],indent=2))
