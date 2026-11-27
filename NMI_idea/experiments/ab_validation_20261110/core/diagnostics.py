from pathlib import Path
import json,numpy as np,hashlib
R=Path(__file__).resolve().parent
G=json.loads((R/'graph.json').read_text());B=json.loads((R/'b_results.json').read_text());P=json.loads((R/'b_policies.json').read_text())
checks={'child_pose_exit_max_error':0.,'child_q_exit_max_error':0.,'valid_edge_q_seed_max_error':0.,'common_graph_queries':G['total_queries']};contexts=[];traces=[]
for ci,g in G['contexts'].items():
 edges=list(g['edges'].values());valid=[e for e in edges if e['exact_validated']];bad=[e for e in edges if not e['exact_validated']]
 contexts.append({'context':int(ci),'translation_m':g['placement_translation_m'],'queries':g['queries'],'exact_valid':len(valid),'exact_invalid':len(bad),'proxy_pass':sum(e['capability']['proxy_pass'] for e in edges),'proxy_pass_but_exact_invalid':sum(e['capability']['proxy_pass'] for e in bad),'raw_score_range':[min(e['capability']['min_raw_score'] for e in edges),max(e['capability']['min_raw_score'] for e in edges)],'reachable_nodes':sum(n['reachable_under_exact'] for n in g['nodes'].values()),'nodes':len(g['nodes'])})
 for e in edges:
  source=g['nodes'][e['id'].split('/')[0]]
  if source['reachable_under_exact']:checks['valid_edge_q_seed_max_error']=max(checks['valid_edge_q_seed_max_error'],float(np.max(np.abs(np.array(e['q_seed'])-source['q_exit']))))
  if e['next']:
   child=g['nodes'][e['next']];checks['child_pose_exit_max_error']=max(checks['child_pose_exit_max_error'],float(np.max(np.abs(np.array(e['pose_exit'])-child['pose_exit']))));checks['child_q_exit_max_error']=max(checks['child_q_exit_max_error'],float(np.max(np.abs(np.array(e['q_exit'])-child['q_exit']))))
 for row in [r for r in B['rows'] if str(r['context'])==ci]:
  bi=row['belief_id'];b=P[f'{ci}:{bi}']['B_observation_contingent'];n=P[f'{ci}:{bi}']['B_without_capability']
  if b.get('edge')!=n.get('edge') and len(traces)<4:
   def desc(p):
    if 'edge' not in p:return p
    e=g['edges'][p['edge']];return {'edge':p['edge'],'exact_validated':e['exact_validated'],'exact_reason':e['capability']['exact'].get('reason'),'proxy_score':e['capability']['min_raw_score'],'path_length_m':e['path_length_m'],'q_seed':e['q_seed'],'q_exit':e['q_exit']}
   traces.append({'context':int(ci),'belief_id':bi,'belief':row['belief'],'B':desc(b),'without_capability':desc(n),'B_loss':row['methods']['B_observation_contingent']['expected_taskloss_plus_cost'],'without_capability_loss':row['methods']['B_without_capability']['expected_taskloss_plus_cost']})
work={m:{k:float(np.mean([r['methods'][m]['planning_cached_model_work'].get(k,0) for r in B['rows']])) for k in ['likelihood_branch_expansions','cached_edge_cost_reads','cached_feasibility_reads']} for m in B['rows'][0]['methods']}
for name in ['b_results.json','extension_results.json','prior_factorial_results.json']:
 data=json.loads((R/name).read_text());err=0.;wrongerr=0.
 for row in data['rows']:
  for d in row['methods'].values():
   err=max(err,abs(sum(d[k] for k in ['correct_success_probability','wrong_identity_probability','explicit_failure_probability','invalid_attempt_probability'])-1))
   wrongerr=max(wrongerr,abs(d['wrong_identity_probability']-d['false_acceptance_probability']))
 checks[name+'_outcome_sum_max_error']=err;checks[name+'_wrong_replay_max_error']=wrongerr
 assert err<1e-10 and wrongerr<1e-12
checks['graph_recorded_source_hashes_match_current']={p:hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in G['source_hashes'].items()}
D={'checks':checks,'contexts':contexts,'causal_capability_traces':traces,'mean_cached_planning_work_per_scenario':work,'note':'Cached model work counts are operations, not new dIRD queries; graph setup cost is reported separately.'};(R/'diagnostics.json').write_text(json.dumps(D,indent=2,allow_nan=False));print(json.dumps(D,indent=2))
assert max(checks[k] for k in ['child_pose_exit_max_error','child_q_exit_max_error','valid_edge_q_seed_max_error'])<1e-12
files=[p for p in R.iterdir() if p.is_file() and p.suffix in ['.py','.txt','.json','.npz','.sh'] and p.name!='source_manifest.json']
(R/'source_manifest.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2))
