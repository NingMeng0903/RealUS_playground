"""Offline numerical replay; never starts a controller or accesses hardware."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from peirastic.contact_qp.qp import ContactQp, QpConfig, QpInput
from peirastic.contact_qp.types import ProbeGeometry, TwistConstraints, ContactObservation
from peirastic.contact_qp.port_constraint import PortEnergyConstraint
from peirastic.contact_qp.numeric_record import SCHEMA, decode
from peirastic.realman8dof.modes.contact_recording import json_value


def replay_record(path):
    results=[]
    for line in Path(path).open():
        row=json.loads(line)
        if row.get('event')!='qp_prepare_rejected': continue
        wire=row.get('diagnostics',{}).get('replay_input')
        if wire is None or wire.get('schema')!=SCHEMA:
            raise ValueError('This legacy rejection has no exact replay payload; use the historical reconstruction mode.')
        raw=decode(wire['payload']); solver=ContactQp(QpConfig(**raw['qp_config']))
        matrices=raw['numeric_problem']
        if matrices is not None:
            x,status,success,iterations=solver._solve_numeric(*(matrices[k] for k in ('H','g','C','l','u')),
                energy_active=raw['qp_input']['energy'] is not None)
            result=dict(control_id=row['control_id'],success=bool(success),status=status,iterations=iterations,
                attempts=solver._numeric_attempts,solution=x,mode='exact_numeric_problem')
        else:
            data=raw['qp_input'];data['geometry']=ProbeGeometry(**data['geometry'])
            data['mechanical']=TwistConstraints(**data['mechanical'])
            if data['observation'] is not None:data['observation']=ContactObservation(**data['observation'])
            if data['energy'] is not None:data['energy']=PortEnergyConstraint(**data['energy'])
            solved=solver.solve(QpInput(**data),online=False)
            result=dict(control_id=row['control_id'],success=solved.success,status=solved.status.value,
                        mode='exact_input_rebuild',diagnostics=dict(solved.diagnostics))
        result['original_reason']=row['diagnostics']['reason']
        result['live_deadline_replayed']=False
        results.append(result)
    return dict(mode='offline_numeric_only',rejections=len(results),results=results)


def replay_t7():
    # Legacy logs did not record rejected wrench or measured angle. Preserve
    # the original reconstruction assumptions; this is not exact raw replay.
    sys.path.insert(0,str(ROOT/'analysis_artifacts/contact_qp_t7_20260912/solver'))
    from replay import failure_files, settings, data_for
    cases=[]
    for index,(path,start,last,failed,samples) in enumerate(failure_files()):
        config,geometry=settings(start)
        solver=ContactQp(replace(config,solver_policy='bounded_retry_v1'))
        refusals=[];retries=0;count=0
        for row,prior,is_failed in [(r,r,False) for r in samples]+[(failed,last,True)]:
            solved=solver.solve(data_for(row,prior,geometry,is_failed))
            count+=1
            retries+=int(len(solved.diagnostics.get('numeric_attempts',()))>1)
            if not solved.success:refusals.append(dict(control_id=row['control_id'],reason=solved.diagnostics.get('reason')))
        cases.append(dict(case=index,path=str(path),inputs=count,retries=retries,refusals=refusals))
        print(f'case {index}: {count} inputs, {retries} retries, {len(refusals)} refusals',flush=True)
    if not cases:raise ValueError('No original T7 failure histories were found')
    return dict(mode='legacy_open_loop_reconstruction',cases=cases,inputs=sum(x['inputs'] for x in cases),
                refusals=sum(len(x['refusals']) for x in cases),
                caveat='Unlogged failed wrench uses previous row; unlogged success angle is zero. No physical/closed-loop claim.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--record',type=Path)
    mode.add_argument('--t7-history',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=replay_t7() if args.t7_history else replay_record(args.record)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(json_value(result),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(args.output)
    if args.t7_history and result['refusals']:raise SystemExit(1)


if __name__=='__main__':main()
