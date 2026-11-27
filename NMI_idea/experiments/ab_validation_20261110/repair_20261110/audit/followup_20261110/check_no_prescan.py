"""Read-only no-prescan pilot audit; recompute rows and public contracts."""
from pathlib import Path
import json,hashlib,math,sys
import numpy as np
from scipy.special import betaln,logsumexp
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]/'no_prescan_ab';sys.path.insert(0,str(ROOT))
import benchmark as b
data=json.loads((ROOT/'results.json').read_text());bank=b.PathBank();cfg=b.CFG
out={'source_sha256':hashlib.sha256((ROOT/'benchmark.py').read_bytes()).hexdigest(),'results_sha256':hashlib.sha256((ROOT/'results.json').read_bytes()).hexdigest(),'scope':'row/likelihood/seed/DMP recomputation; no full experiment or native execution','cohorts':{},'violations':[]}

def independent_loglike(h,pose,obs):
 lane,sgn=b.HYP[h];s=pose.s;target=(lane*.022+.2*math.tan(sgn*.17)*(s-.5),sgn*.17,.04,True);decoy=(-lane*.022-.2*math.tan(sgn*.17)*(s-.5),-sgn*.17,.052,False);v=min([target,decoy],key=lambda z:(abs(z[0]-pose.y),z[0]));inside=abs(v[0]-pose.y)<=.032;p=.88 if inside else .01
 if not obs['visible']:return math.log1p(-p)
 u,ang,score,dep=[obs[k] for k in ['image_u_m','tangent_rad','target_identity_score','depth_m']]
 aa,bb=(8,2) if inside and v[3] else (2,8);beta=(aa-1)*math.log(score)+(bb-1)*math.log1p(-score)-betaln(aa,bb)
 if not inside:
  if abs(u)>.032 or abs(ang)>.35 or not .02<=dep<=.07:return -math.inf
  return math.log(p)-math.log(.064*.7*.05)+beta
 ll=math.log(p)+beta
 for value,mean,sd in [(u,v[0]-pose.y,.0025),(ang,v[1],.03),(dep,v[2],.002)]:ll+=-.5*((value-mean)/sd)**2-math.log(sd)-.5*math.log(2*math.pi)
 return ll

max_posterior=max_likelihood=max_sensor=0.;observation_count=0
for cohort,n,seed in [('test',256,20261110),('shift',128,20261111)]:
 rows=[r for r in data['rows'] if r['cohort']==cohort];worlds=b.draw_worlds(n,seed,shifted=cohort=='shift');methods=sorted(set(r['method'] for r in rows));recs={}
 for method in methods:
  arr=[r for r in rows if r['method']==method];correct_stops=public_stops=wrong_stops=ever_quality=over_budget=0;recs[method]={}
  for r in arr:
   belief=np.full(4,.25);previous=-1;clock=0.;quality_times=[];stop_events=[]
   for step,t in enumerate(r['trace']):
    observation_count+=1;obs=t['observation'];a=t['action_id'];pose=b.pose_for_action(a);max_posterior=max(max_posterior,float(abs(belief-np.array(t['prior_belief'])).max()))
    event=b.sensor(worlds[r['world_id']],pose,r['world_id'],step,a,seed)
    if event.public.public_dict()!=obs:out['violations'].append([cohort,method,r['world_id'],step,'seeded_sensor_disagrees'])
    stop=bool(obs['visible'] and abs(obs['image_u_m'])<=.007 and abs(obs['tangent_rad']-pose.yaw)<=.08 and obs['target_identity_score']>=.8 and .02<=obs['depth_m']<=.07)
    if stop!=t['public_stop']:out['violations'].append([cohort,method,r['world_id'],step,'public_stop_disagrees'])
    if not obs['visible'] and any(obs[k] is not None for k in obs if k!='visible'):out['violations'].append([cohort,method,r['world_id'],step,'missing_exposes_numeric'])
    good=b.independent_quality(event,worlds[r['world_id']],pose)
    if good!=t['evaluator_quality']:out['violations'].append([cohort,method,r['world_id'],step,'truth_evaluation_disagrees'])
    clock+=t['planning_s']+bank.duration(previous,a)+cfg['planning']['observation_time_s']
    if abs(clock-t['clock_s'])>1e-10:out['violations'].append([cohort,method,r['world_id'],step,'clock_disagrees'])
    if clock>cfg['planning']['deadline_s']:over_budget+=1
    if good:quality_times.append(clock)
    if stop:stop_events.append(bool(good))
    ll=np.array([independent_loglike(h,pose,obs) for h in range(4)]);o=b.Observation(obs['visible'],*[obs[k] if obs[k] is not None else math.nan for k in ['image_u_m','tangent_rad','target_identity_score','depth_m']]);actual=np.array([b.log_likelihood(h,pose,o) for h in range(4)]);finite=np.isfinite(ll)&np.isfinite(actual)
    max_likelihood=max(max_likelihood,float(abs(ll[finite]-actual[finite]).max()) if finite.any() else 0.)
    lp=np.log(np.maximum(belief,1e-300))+ll
    if np.isfinite(lp).any():belief=np.exp(lp-logsumexp(lp))
    previous=a
   public_stops+=bool(stop_events);correct_stops+=bool(stop_events and stop_events[-1]);wrong_stops+=bool(stop_events and not stop_events[-1]);ever_quality+=bool(quality_times)
   if bool(stop_events)!=r['public_stopped'] or bool(stop_events and not stop_events[-1])!=r['false_accept']:out['violations'].append([cohort,method,r['world_id'],'stop_summary_disagrees'])
  recs[method]={'n':len(arr),'ever_single_frame_quality_count':ever_quality,'public_stop_count':public_stops,'correct_public_stop_count':correct_stops,'false_stop_count':wrong_stops,'outside_deadline_observations':over_budget,'mean_compute_s':float(np.mean([r['compute_s'] for r in arr])),'mean_censored_time_s':float(np.mean([r['censored_time_s'] for r in arr]))}
 aa={r['world_id']:r for r in rows if r['method']=='A_one_step_greedy'};bb={r['world_id']:r for r in rows if r['method']=='B_two_step_contingent'};mm={r['world_id']:r for r in rows if r['method']=='two_step_belief_MPC_reference'}
 recs['paired']={'A_B_equal_actions':sum([t['action_id'] for t in aa[i]['trace']]==[t['action_id'] for t in bb[i]['trace']] for i in aa),'A_B_equal_success':sum(aa[i]['success']==bb[i]['success'] for i in aa),'B_MPC_equal_actions':sum([t['action_id'] for t in mm[i]['trace']]==[t['action_id'] for t in bb[i]['trace']] for i in bb),'B_minus_A_mean_compute_s':float(np.mean([bb[i]['compute_s']-aa[i]['compute_s'] for i in aa])),'B_minus_A_mean_censored_time_s':float(np.mean([bb[i]['censored_time_s']-aa[i]['censored_time_s'] for i in aa]))}
 out['cohorts'][cohort]=recs
out['observations_recomputed']=observation_count;out['max_log_likelihood_difference']=max_likelihood;out['max_public_posterior_difference']=max_posterior
out['DMP']={'position_max_endpoint_error_m':max(float(np.linalg.norm(p['path'][-1,:2]-b.pose_for_action(dst).vector()[:2])) for (src,dst),p in bank.paths.items()),'yaw_max_endpoint_error_rad':max(float(abs(p['path'][-1,2]-b.pose_for_action(dst).yaw)) for (src,dst),p in bank.paths.items()),'reported_mixed_metre_radian_norm':bank.max_endpoint_error,'max_discrete_translation_speed_m_s':bank.max_nominal_speed,'max_discrete_yaw_rate_rad_s':max(float(abs(np.diff(p['path'][:,2])).max()/(p['duration']/20)) for p in bank.paths.values()),'sensor_pose_uses_nominal_endpoint_not_actual_DMP_last_sample':True}
out['root_policy_recomputed']={'B':b.choose_b(bank,b.PRIOR),'MPC':b.exhaustive_reference(bank,b.PRIOR)}
out['quadrature_mass_max_error']=max(abs(sum(w for w,_,_ in nodes)-1) for nodes in b.MODEL_NODES.values())
(HERE/'no_prescan_independent.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
