from pathlib import Path
import hashlib,json,time
import numpy as np
import b_planner as bp
from ab_integrated_extension import kernels
R=Path(__file__).resolve().parent

def main():
 graph=json.loads((R/'graph.json').read_text());a=json.loads((R/'a_results.json').read_text());W=np.array(a['conditions']['context_dependent_response_shift']['coefficient']);hidden=json.loads((R/'extension_hidden_manifest.json').read_text())
 source=np.array([.4,.3,.2,.1]);counts=np.random.default_rng(41108).multinomial(256,source);fit=(counts+1)/260.;priors={'broad_uniform':np.ones(4)/4,'source_fitted':fit};populations={'source':source,'shifted':np.roll(source,1)};rows=[]
 for item in hidden:
  ci=str(item['context']);bi=item['belief_id'];g=graph['contexts'][ci];T=np.array(g['nodes']['root']['pose_exit']);targets=T[:3,3]+np.array(g['placement_translation_m'])+np.array(graph['config']['target_offsets_m']);c=np.array(item['public_context']);E=np.array(item['persistent_true_E_by_hypothesis']);KL,KA,KT,_,_=kernels(g,T,targets,c,E,W)
  for control in ['greedy','B']:
   policies={}
   for pn,p in priors.items():
    for rn,K in [('analytic',KA),('learned',KL)]:
     t=time.perf_counter();v,pol=bp.greedy(g,'root',p,2,K) if control=='greedy' else bp.solve(g,'root',p,2,K);wall=time.perf_counter()-t;policies[pn+':'+rn]=(v,pol,wall)
   for population,truth in populations.items():
    row={'context':int(ci),'scenario_id':bi,'controller':control,'test_population':population,'methods':{}}
    for method,(v,pol,wall) in policies.items():
     metric=bp.evaluate(g,pol,truth,KT);row['methods'][method]={**bp.outcomes(g,pol,truth,KT),'planning_objective':v,'true_expected_taskloss_plus_cost':float(metric[0]),'false_acceptance_probability':float(metric[1]),'infeasible_attempt_probability':float(metric[4]),'movement_distance_m':float(metric[2]),'observations':float(metric[3]),'planner_wall_seconds':wall}
    rows.append(row)
 summary={};comparisons={}
 for pop in populations:
  for control in ['greedy','B']:
   subset=[r for r in rows if r['controller']==control and r['test_population']==pop];key=pop+':'+control
   summary[key]={m:{k:float(np.mean([r['methods'][m][k] for r in subset])) for k in ['correct_success_probability','wrong_identity_probability','explicit_failure_probability','invalid_attempt_probability','true_expected_taskloss_plus_cost','false_acceptance_probability']} for m in subset[0]['methods']}
   for response in ['analytic','learned']:
    comparisons[key+':source_prior_minus_broad:'+response]=bp.bootstrap([r['methods']['source_fitted:'+response]['true_expected_taskloss_plus_cost']-r['methods']['broad_uniform:'+response]['true_expected_taskloss_plus_cost'] for r in subset])
   for prior in priors:
    comparisons[key+':learned_minus_analytic:'+prior]=bp.bootstrap([r['methods'][prior+':learned']['true_expected_taskloss_plus_cost']-r['methods'][prior+':analytic']['true_expected_taskloss_plus_cost'] for r in subset])
 out={'evidence':'SYNTHETIC POPULATION prior x response factorial, requested post-main; no clinical prior claim','design_sha256':hashlib.sha256((R/'PRIOR_FACTORIAL_DESIGN.txt').read_bytes()).hexdigest(),'seed_source_prior':41108,'source_probability':source.tolist(),'source_counts':counts.tolist(),'fitted_prior':fit.tolist(),'broad_prior':priors['broad_uniform'].tolist(),'shifted_test_probability':populations['shifted'].tolist(),'public_response_contexts':72,'new_actual_capability_queries':0,'summary':summary,'comparisons':comparisons,'rows':rows}
 (R/'prior_factorial_results.json').write_text(json.dumps(out,indent=2,allow_nan=False));print(json.dumps({k:out[k] for k in ['source_counts','fitted_prior','summary','comparisons']},indent=2))
if __name__=='__main__':main()
