import json
from pathlib import Path
import numpy as np
ROOT=Path('/media/camp/yameng/icra 2027/uncalibrated/005');OUT=Path(__file__).parent
s=json.loads((ROOT/'session.json').read_text());epoch=s['clock']['anchor_time_ns']-s['clock']['anchor_monotonic_ns'];out=[]
for trial in s['trials']:
 for a in trial['attempts']:
  if a.get('status')!='completed':continue
  rows=[];pubs={};reported_work=0
  for line in (ROOT/a['directory']/'contact_qp.jsonl').open():
   r=json.loads(line)
   if r['event']=='control_sample':rows.append(r)
   if r['event']=='publication' and r['success']:pubs[r['control_id']]=r
   if r['event']=='logical_command_energy':
    reported_work+=sum(e.get('work_j',0) for e in r['events'] if e['event']=='logical_epoch_work')
  t=np.array([r['record_monotonic_s'] for r in rows]);tend=next((a['stages'][k]-epoch)*1e-9 for k in ['path_done']);tstart=(a['stages']['tracking_observed']-epoch)*1e-9
  w=-np.array([r['physical_wrench_candidate_tool'] for r in rows]);nom=np.array([r['nominal_twist_tool'] for r in rows]);can=np.array([r['candidate_twist_tool'] for r in rows]);final=np.array([pubs[r['control_id']]['final_command_model_tool'] if r['control_id'] in pubs else [0.]*6 for r in rows]);rout=dict(attempt=a['directory'],logged_logical_epoch_net_work_j=reported_work,segments={})
  for label,start in [('whole_control',t[0]),('scan',tstart)]:
   dt=np.maximum(0,np.minimum(np.r_[t[1:],tend],tend)-np.maximum(t,start));ok=np.array([r['control_id'] in pubs for r in rows]);dt[~ok]=0
   result={}
   def stats(p):return dict(net_j=float(np.sum(p*dt)),positive_j=float(np.sum(np.maximum(p,0)*dt)),negative_magnitude_j=float(np.sum(np.maximum(-p,0)*dt)),max_abs_w=float(np.max(abs(p[dt>0]))))
   for n,v in [('nominal',nom),('candidate',can),('final',final),('candidate_minus_nominal',can-nom),('final_minus_nominal',final-nom),('final_minus_candidate',final-can)]:
    result[n]={axis:stats(np.sum(w[:,ix]*v[:,ix],axis=1)) for axis,ix in [('total',slice(None)),('translation',slice(0,3)),('angular',slice(3,6)),('wy',slice(4,5))]}
   pn=np.sum(w*nom,axis=1);pf=np.sum(w*final,axis=1);savail=np.maximum(-pn,0);excess=np.maximum(-pf-savail,0);supply=np.minimum(savail,np.maximum(-pf,0));dE=np.maximum(pf,0)-excess
   cum=np.cumsum(dE*dt);peak=np.maximum.accumulate(np.r_[0,cum]);draw=peak[1:]-cum
   result['proposed_nominal_supply']=dict(supply_used_j=float(np.sum(supply*dt)),excess_outgoing_j=float(np.sum(excess*dt)),recovery_j=float(np.sum(np.maximum(pf,0)*dt)),net_change_j=float(np.sum(dE*dt)),largest_cumulative_drawdown_j=float(np.max(draw)),peak_excess_power_w=float(np.max(excess)),max_50ms_excess_liability_j=float(np.max(excess)*.05))
   result['abs_raw_wrench_p50_p95_max']=np.quantile(abs(w[dt>0]),[.5,.95,1],axis=0).tolist();result['energy_available_min_j']=min(r['energy']['available_j'] for r,d in zip(rows,dt) if d>0)
   rout['segments'][label]=result
  out.append(rout);print(a['directory'],rout['segments']['whole_control']['proposed_nominal_supply'],flush=True)
(OUT/'work.json').write_text(json.dumps(dict(method='Approximate same-control W=-physical_wrench_candidate_tool, successful final publication joined by control_id, dt to next control capped at path_done; skips unsuccessful controls. Positive W.V is recovery, negative is outgoing. Baseline allowance counterfactual arithmetic only, not replay; no physical passivity claim.',attempts=out),indent=2)+'\n')
