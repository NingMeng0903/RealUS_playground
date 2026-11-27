"""Post-run contract checks; never changes predictions, models or outcomes."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path

def run(folder):
    folder=Path(folder)
    protocol=json.loads((folder/'protocol.json').read_text())
    freeze=json.loads((folder/'prediction_freeze.json').read_text())
    cases=json.loads((folder/'test_manifest.json').read_text())
    preps=json.loads((folder/'preparations.json').read_text())
    train=[json.loads(s) for s in (folder/'training_manifest.jsonl').read_text().splitlines()]
    predictions=[json.loads(s) for s in (folder/'predictions.jsonl').read_text().splitlines()]
    executions=[json.loads(s) for s in (folder/'executions.jsonl').read_text().splitlines()]
    status=json.loads((folder/'run_status.json').read_text())
    case_map={c['case_id']:c for c in cases}
    train_ids={x['spec']['object_id'] for x in train};test_ids={c['spec']['object_id'] for c in cases}
    train_seeds={x['spec']['geometry_seed'] for x in train};test_seeds={c['spec']['geometry_seed'] for c in cases}
    errors=[]
    for name,digest in freeze['sha256'].items():
        if hashlib.sha256((folder/name).read_bytes()).hexdigest()!=digest:errors.append(f'changed frozen artifact: {name}')
    if train_ids&test_ids or train_seeds&test_seeds:errors.append('train-test anatomy overlap')
    hashes={};hidden_reads=early_reads=old_fresh=wrong_anatomy=0
    failed=0;inconsistent_success=0
    for row in executions:
        case_id=row['case_id'];hashes.setdefault(case_id,set()).add(row['packet_hash'])
        if row['world_hash']!=preps[case_id]['old_world_hash']:wrong_anatomy+=1
        failed+=not row['success']
        for d in row['decisions']:
            hidden_reads+=bool(row['hidden'] and d['near_visible'])
            early_reads+=bool(d['near_visible'] and d['time']<row['release']-1e-7)
        for c in row['calls']:
            if c['success'] and (not c['public_success'] or not c['independent_success']):
                inconsistent_success+=1
            if c['program']=='f' and c['capture_min'] is not None:
                old_fresh+=bool(c['capture_min']<c['start_time']-1e-7)
    for k,v in hashes.items():
        if len(v)!=1:errors.append(f'packet content changed across interventions: {k}')
    for name,value in [('hidden_reads',hidden_reads),('premature_reads',early_reads),('old_frames_fresh',old_fresh),('changed_anatomy',wrong_anatomy)]:
        if value:errors.append(f'{name}: {value}')
    if inconsistent_success:errors.append(f'inconsistent acceptance fields: {inconsistent_success}')
    for p in predictions:
        if (p['arm']=='hide_old' or p['arm'].startswith('release_')) and p['visible_evidence']:
            errors.append('forecast sees unreleased test evidence')
    expected=len(cases)*len(protocol['arms'])*len(protocol['trees'])
    if len(executions)!=expected:errors.append(f'missing failed or completed runs: {len(executions)} != {expected}')
    result={'schema':'v852_contract_audit_v1','pass':not errors,'errors':errors,
            'n_independent_test_configurations':len(cases),'n_interventions':len(executions),
            'failed_interventions_included':failed,'training_test_overlap':False,
            'same_historical_content_across_gates':all(len(v)==1 for v in hashes.values()),
            'hidden_reads':hidden_reads,'premature_reads':early_reads,'old_frames_counted_fresh':old_fresh,
            'same_material_anatomy_across_pose':wrong_anatomy==0,
            'prediction_sha_unchanged':status['predictions_unchanged'],
            'inconsistent_public_and_audit_success_fields':inconsistent_success,
            'scope':'contract audit only; not scientific effect, coverage, contact or hardware validation'}
    (folder/'contract_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);args=p.parse_args()
    result=run(args.input);print(json.dumps(result,ensure_ascii=False));raise SystemExit(0 if result['pass'] else 1)
if __name__=='__main__':main()
