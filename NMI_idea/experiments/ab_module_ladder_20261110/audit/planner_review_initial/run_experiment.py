"""Frozen geometric A/B runs; test release requires the approved protocol hash."""
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np
from system import ROOT,CFG,env,Robot,CapabilityEvaluator,run_episode


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def contexts(evaluator):
    rows=json.loads((ROOT/'capability/frozen_public/candidates.json').read_text())
    checks=json.loads((ROOT/'capability/endpoint_checks.json').read_text())
    result=[]
    for key in CFG['contexts']:
        j=next(j for j,row in enumerate(rows) if row['id']==key)
        assert checks[j]['found']
        q=np.asarray(checks[j]['q_solution'])
        result.append(dict(id=key,q_seed=q,T_world_body=evaluator.fk(q)))
    return result


def summarize(rows):
    output={}
    for method in sorted(set(x['method'] for x in rows)):
        selected=[r for r in rows if r['method']==method]
        output[method]=dict(n=len(selected),correct_public_stops=sum(r['correct_public_stop'] for r in selected),
            false_public_stops=sum(r['false_public_stop'] for r in selected),
            ever_audit_quality=sum(r['ever_audit_quality'] for r in selected),
            mean_RMT_s=float(np.mean([r['rmt_s'] for r in selected])),
            mean_actions=float(np.mean([r['actions'] for r in selected])),
            total_endpoint_queries=sum(r['endpoint_queries'] for r in selected))
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort',choices=['dev','test'],default='dev')
    parser.add_argument('--approved-protocol-sha')
    parser.add_argument('--limit',type=int,default=2)
    parser.add_argument('--methods',nargs='+')
    parser.add_argument('--max-actions',type=int,default=6)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    protocol=Path(__file__).parent/'PROTOCOL.json';fingerprint=sha(protocol)
    if args.cohort=='test' and args.approved_protocol_sha!=fingerprint:
        raise SystemExit('test cohort needs independently reviewed protocol SHA')
    args.output.mkdir(parents=True,exist_ok=False)
    dev=args.cohort=='dev'
    particles=env.sample_prior(CFG['catalogue']['dev_seed' if dev else 'test_seed'],
        CFG['catalogue']['dev_n' if dev else 'test_n'])
    indices=list(range(min(args.limit,len(particles)))) if dev else CFG['catalogue']['test_hidden_indices']
    evaluator=CapabilityEvaluator()
    scenes=contexts(evaluator)
    if dev:scenes=scenes[:1]
    methods=args.methods or CFG['methods']
    out=dict(schema='AB_geometric_finite_v1',cohort=args.cohort,protocol_sha256=fingerprint,
        system_sha256=sha(Path(__file__).parent/'system.py'),environment_sha256=sha(ROOT/'observation/environment.py'),
        robot=evaluator.manifest(),contexts=[dict(id=s['id'],q_seed=s['q_seed'].tolist(),T_world_body=s['T_world_body'].tolist()) for s in scenes],
        scope='exact finite support geometric model + real static robot endpoint realizations; no whole-episode native execution',rows=[],summary={})
    started=time.perf_counter()
    for h in indices:
        for scene in scenes:
            # Rotate method order so host cache/load does not favor one fixed method.
            order=np.random.default_rng(720505+h).permutation(methods)
            for method in order:
                robot=Robot(evaluator,scene['T_world_body'],scene['q_seed'])
                result=run_episode(particles[h],particles,str(method),robot,
                    (args.cohort,h,scene['id']),max_actions=args.max_actions)
                result.update(object_index_for_audit_only=h,context=scene['id'])
                out['rows'].append(result);out['summary']=summarize(out['rows'])
                out['wall_s']=time.perf_counter()-started
                (args.output/'results.json').write_text(json.dumps(out,indent=2,allow_nan=False))
                print(args.cohort,h,scene['id'],method,'correct',result['correct_public_stop'],
                    'false',result['false_public_stop'],'actions',result['actions'],'RMT',result['rmt_s'],
                    'wall',out['wall_s'],flush=True)
