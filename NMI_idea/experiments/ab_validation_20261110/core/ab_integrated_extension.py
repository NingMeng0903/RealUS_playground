"""Prespecified post-main synthetic learned-A -> B dataflow extension."""
from pathlib import Path
import hashlib,json,time
import numpy as np
import b_planner as bp
ROOT=Path(__file__).resolve().parent.parent

def feature(c,u):return np.concatenate([u]+[c[j]*u for j in range(3)])
def sensor_probability(z):
 p=.1+.85*np.exp(-float(z@z)/(2*.8**2));logits=np.array([z[0],-z[0],z[1],-z[1]])/.4
 logits-=max(logits);m=np.exp(logits);m/=sum(m)
 return np.r_[1-p,p*m]
def kernels(g,rootT,targets,c,E,W):
 learned={};analytic={};true={};calls=0;feature_errors=[]
 G=.45*np.array([[c[0],.5*c[1]],[-.35*c[1],c[2]]])
 for node,n in g['nodes'].items():
  if not n['inspect']:continue
  KL=np.zeros((3,4,5));KA=KL.copy();KT=KL.copy()
  for a,eid in n['inspect'].items():
   edge=g['edges'][eid];u=(np.array(edge['pose_exit'])[:2,3]-rootT[:2,3])/.04
   for h,target in enumerate(targets):
    e=(rootT[:2,3]-target[:2])/.04;zt=e+(np.eye(2)+G+E[h])@u;za=e+u;zl=za+feature(c,u)@W
    KT[int(a),h]=sensor_probability(zt);KA[int(a),h]=sensor_probability(za);KL[int(a),h]=sensor_probability(zl);calls+=1;feature_errors.append([float(np.sum((za-zt)**2)),float(np.sum((zl-zt)**2))])
  learned[node]=KL;analytic[node]=KA;true[node]=KT
 return learned,analytic,true,calls,np.mean(feature_errors,axis=0)

def main():
 graph=json.loads((ROOT/'core/graph.json').read_text());a=json.loads((ROOT/'core/a_results.json').read_text());W=np.array(a['conditions']['context_dependent_response_shift']['coefficient']);rng=np.random.default_rng(41107);rows=[];saved_policies={};hidden_manifest=[];max_ref=0.;maxnorm=0.;allpositive=True;noninterference=True
 methods=['analytic_greedy','learned_greedy','analytic_open_loop','learned_open_loop','analytic_B','learned_B','learned_reference','learned_no_capability','learned_no_dird_proxy','learned_collapsed_belief']
 for ci,g in graph['contexts'].items():
  rootT=np.array(g['nodes']['root']['pose_exit']);shift=np.array(g['placement_translation_m']);targets=rootT[:3,3]+shift+np.array(graph['config']['target_offsets_m'])
  for bi,b in enumerate(bp.beliefs()):
   c=rng.uniform(-1,1,3);E=rng.normal(0,.04,(4,2,2));hidden_manifest.append({'context':int(ci),'belief_id':bi,'public_context':c.tolist(),'persistent_true_E_by_hypothesis':E.tolist()})
   t=time.perf_counter();KL,KA,KT,calls,fe=kernels(g,rootT,targets,c,E,W);kernel_s=time.perf_counter()-t
   maxnorm=max(maxnorm,max(float(np.max(np.abs(K.sum(2)-1))) for Ks in [KL,KA,KT] for K in Ks.values()));allpositive &= all(np.min(K)>0 for Ks in [KL,KA,KT] for K in Ks.values())
   kmse={label:float(np.mean([(K[node]-KT[node])**2 for node in KT])) for label,K in [('analytic',KA),('learned',KL)]}
   row={'context':int(ci),'belief_id':bi,'belief':b.tolist(),'public_context':c.tolist(),'kernel_build_seconds':kernel_s,'local_response_predictions_per_model':calls,'feature_prediction_squared_error':{'analytic':float(fe[0]),'learned':float(fe[1])},'kernel_probability_mse':kmse,'methods':{}};saved_policies[f'{ci}:{bi}']={}
   for method in methods:
    K=KA if method.startswith('analytic') else KL;bp.COUNTERS.clear();t=time.perf_counter()
    if method.endswith('greedy'):v,p=bp.greedy(g,'root',b,2,K)
    elif method.endswith('open_loop'):v,p=bp.open_loop(g,b,K)
    elif method=='learned_reference':v,p=bp.reference_joint(g,'root',b,2,K)
    elif method=='learned_no_capability':v,p=bp.solve(g,'root',b,2,K,'no_capability')
    elif method=='learned_no_dird_proxy':v,p=bp.solve(g,'root',b,2,K,'no_proxy')
    elif method=='learned_collapsed_belief':v,p=bp.solve(g,'root',np.eye(4)[np.argmax(b)],2,K)
    else:v,p=bp.solve(g,'root',b,2,K)
    wall=time.perf_counter()-t;work=dict(bp.COUNTERS);metric=bp.evaluate(g,p,b,KT);model_metric=bp.evaluate(g,p,b,K)
    row['methods'][method]={**bp.outcomes(g,p,b,KT),'planning_objective':v,'assumed_model_evaluated_objective':float(model_metric[0]),'true_expected_taskloss_plus_cost':float(metric[0]),'false_acceptance_probability':float(metric[1]),'movement_distance_m':float(metric[2]),'acquired_observations_including_terminal':float(metric[3]),'infeasible_attempt_probability':float(metric[4]),'planner_wall_seconds':wall,'root_action':p.get('edge','fail'),'cached_model_work':work,'new_actual_capability_queries':0};saved_policies[f'{ci}:{bi}'][method]=p
   max_ref=max(max_ref,abs(row['methods']['learned_B']['planning_objective']-row['methods']['learned_reference']['planning_objective']))
   # Hidden truth is changed outside the policy API, public Khat/c/b are fixed.
   expected=json.dumps(bp.signature(saved_policies[f'{ci}:{bi}']['learned_B']),sort_keys=True)
   for hidden in range(4):
    _,p=bp.solve(g,'root',b,2,KL);noninterference &= json.dumps(bp.signature(p),sort_keys=True)==expected
   rows.append(row)
 summary={m:{k:float(np.mean([r['methods'][m][k] for r in rows])) for k in ['correct_success_probability','wrong_identity_probability','explicit_failure_probability','invalid_attempt_probability','true_expected_taskloss_plus_cost','false_acceptance_probability','movement_distance_m','acquired_observations_including_terminal','infeasible_attempt_probability','planner_wall_seconds']} for m in methods}
 comparisons={}
 for m1,m0 in [('learned_B','learned_greedy'),('learned_B','learned_open_loop'),('learned_B','analytic_B'),('learned_greedy','analytic_greedy'),('learned_B','learned_no_capability'),('learned_B','learned_no_dird_proxy'),('learned_B','learned_reference')]:
  comparisons[m1+'_minus_'+m0]=bp.bootstrap([r['methods'][m1]['true_expected_taskloss_plus_cost']-r['methods'][m0]['true_expected_taskloss_plus_cost'] for r in rows])
 result={'evidence':'POST-MAIN PREDECLARED SYNTHETIC integrated A-response -> B-likelihood -> actual primitive/capability graph; no physical ultrasound','seed':41107,'design_sha256':hashlib.sha256((ROOT/'core/EXTENSION_DESIGN.txt').read_bytes()).hexdigest(),'a_coefficients_sha256':hashlib.sha256(W.tobytes()).hexdigest(),'sensor_outcomes':['no_detection','positive_x','negative_x','positive_y','negative_y'],'shared_graph_setup_actual_capability_queries':graph['total_queries'],'new_actual_capability_queries':0,'checks':{'kernel_normalization_max_error':maxnorm,'all_kernel_entries_positive':allpositive,'learned_B_reference_assumed_model_max_error':max_ref,'hidden_state_noninterference':noninterference,'persistent_perturbations':'one E[scenario,h], reused all edges/branches'},'summary':summary,'comparisons':comparisons,'kernel_model_error_means':{k:float(np.mean([r['kernel_probability_mse'][k] for r in rows])) for k in ['analytic','learned']},'response_feature_error_means':{k:float(np.mean([r['feature_prediction_squared_error'][k] for r in rows])) for k in ['analytic','learned']},'rows':rows,'limits':['Sensor map and response generator are synthetic, independent held-out contexts but not clinical physics.','Planning reference minimizes the learned assumed model; true-model expected performance can be worse.','Same finite action graph and all mechanical validation limits as main experiment.','No retraining or parameter tuning after extension outcome inspection.']}
 (ROOT/'core/extension_hidden_manifest.json').write_text(json.dumps(hidden_manifest,indent=2,allow_nan=False));(ROOT/'core/extension_policies.json').write_text(json.dumps(saved_policies,indent=2,allow_nan=False));(ROOT/'core/extension_results.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps({k:result[k] for k in ['summary','comparisons','checks','kernel_model_error_means','response_feature_error_means']},indent=2));assert max_ref<1e-9 and maxnorm<1e-12 and allpositive and noninterference
if __name__=='__main__':main()
