from pathlib import Path
import json,collections
import numpy as np
ROOT=Path('/media/camp/yameng/icra 2027/uncalibrated/006');OUT=Path(__file__).parent
s=json.loads((ROOT/'session.json').read_text());epoch=s['clock']['anchor_time_ns']-s['clock']['anchor_monotonic_ns'];out=[]
metrics=json.loads((OUT/'metrics.json').read_text());mf={r['attempt']:r for r in metrics['attempts']}
for trial in s['trials']:
 for a in trial['attempts']:
  if 'tracking_observed' not in a['stages']:continue
  rows=[];pubs={};events=[]
  for l in (ROOT/a['directory']/'contact_qp.jsonl').open():
   r=json.loads(l)
   if r['event']=='control_sample':rows.append(r)
   elif r['event']=='publication' and r['success']:pubs[r['control_id']]=r
   elif r['event'] not in ['publication_review','publication_transport_result','nonspendable_measured_port','logical_command_energy']:events.append(r)
  start=(a['stages']['tracking_observed']-epoch)*1e-9;end=(a['stages']['path_done']-epoch)*1e-9 if 'path_done' in a['stages'] else rows[-1]['record_monotonic_s'];t=np.array([r['record_monotonic_s'] for r in rows]);dt=np.maximum(0,np.minimum(np.r_[t[1:],end],end)-np.maximum(t,start));idx=np.flatnonzero((t>=start)&(t<end));scan=[rows[i] for i in idx];nom=np.array([r['nominal_twist_tool'][4] for r in rows]);can=np.array([r['candidate_twist_tool'][4] if r['candidate_twist_tool'] else np.nan for r in rows]);req=np.array([r['allocation_diagnostics'].get('differential_request_m_s',0) for r in rows]);sgn=np.array([r['allocation_diagnostics'].get('differential_sign',0) for r in rows]);active=(dt>0)&(req>0)
  q=np.array([r['feature']['quality'] if r['feature'] else [np.nan]*3 for r in rows]);gate=np.array([r['allocation_diagnostics']['repair_force_gate'] for r in rows]);allow=np.array([r['repair_episode']['repair_allowed'] for r in rows]);n=-sgn*.031*nom;c=-sgn*.031*can
  byframe={}
  for i in idx:
   r=rows[i]
   if r['feature']:byframe.setdefault((r['feature']['source_id'],r['feature']['frame_seq']),i)
  unique=list(byframe.values());uq=q[unique];difference=abs(uq[:,2]-uq[:,0]);
  report=dict(attempt=a['directory'],status=a['status'],scan_s=end-start,feature_status_cycles=dict(collections.Counter(r['image_feedback_status'] for r in scan)),episode_reason_cycles=dict(collections.Counter(r['repair_episode']['reason'] for r in scan)),feature_status_seconds={key:float(sum(dt[i] for i in idx if rows[i]['image_feedback_status']==key)) for key in set(r['image_feedback_status'] for r in scan)},unique_image_frames=len(unique),unique_quality_lcr_min_p05_p50_p95=np.quantile(uq,[0,.05,.5,.95],axis=0).tolist(),unique_side_min_below08_fraction=float(np.mean(uq[:,[0,2]].min(axis=1)<.8)),unique_imbalance_over03_fraction=float(np.mean(difference>.03)),unique_imbalance_between03and10_fraction=float(np.mean((difference>.03)&(difference<=.1))),unique_imbalance_over10_fraction=float(np.mean(difference>.1)),request_cycles=int(active.sum()),request_duration_s=float(dt[active].sum()),request_in_positive_wy_s=float(dt[active&(sgn<0)].sum()),request_in_negative_wy_s=float(dt[active&(sgn>0)].sum()),nominal_already_covers_request_fraction=None if not active.any() else float(np.mean(n[active]>=req[active])),candidate_covers_request_fraction=None if not active.any() else float(np.mean(c[active]>=req[active]-1e-9)),request_mm_s_p50_p95_max=None if not active.any() else (1000*np.quantile(req[active],[.5,.95,1])).tolist(),positive_qp_increment_duration_s=float(dt[active&(-sgn*(can-nom)>1e-6)].sum()),abs_qp_increment_integral_deg=float(np.degrees(np.nansum(abs(can-nom)*dt))),pauses=[],request_episodes=[],selected_frame_controls=[])
  for key in ['image_feedback_status','permission']:
   vals=np.array([r['image_feedback_status']!='ok' for r in rows]) if key=='image_feedback_status' else ~allow
   marked=vals&(dt>0);edges=np.diff(np.r_[False,marked,False].astype(int))
   for lo,hi in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):
    endidx=min(hi,len(rows)-1);report['pauses'].append(dict(type=key,start_scan_s=float(t[lo]-start),duration_s=float(dt[lo:hi].sum()),reason=rows[lo]['repair_episode']['reason'],image_status=rows[lo]['image_feedback_status'],resumed=bool(hi<len(rows) and t[hi]<end and not vals[hi]),next_reason=rows[endidx]['repair_episode']['reason']))
  # Group visual request spans separated by <0.25s into broader observations,
  # preserving actual requested duration separately.
  ari=np.flatnonzero(active);groups=[]
  for i in ari:
   if not groups or t[i]-t[groups[-1][-1]]>.25:groups.append([i])
   else:groups[-1].append(i)
  for group in groups:
   lo,hi=group[0],group[-1];tail=next((i for i in unique if t[i]>=t[hi]+.5),None);report['request_episodes'].append(dict(start_scan_s=float(t[lo]-start),end_scan_s=float(t[hi]-start),request_s=float(dt[group].sum()),min_quality_lcr=q[group].min(axis=0).tolist(),first_quality=q[lo].tolist(),last_quality=q[hi].tolist(),later_05s_quality=None if tail is None else q[tail].tolist(),qp_increment_deg=float(np.degrees(np.sum((can[group]-nom[group])*dt[group])))))
  for f in mf[a['directory']]['selected_frames']:
   matching=[r for r in scan if r['feature'] and r['feature']['frame_seq']==f['frame_id']];r=matching[0];p=pubs.get(r['control_id']);d=r['allocation_diagnostics'];report['selected_frame_controls'].append(dict(label=f['label'],frame=f['frame_id'],scan_time_s=f['scan_time_s'],quality=r['feature']['quality'],request_mm_s=1000*d.get('differential_request_m_s',0),gate=d['repair_force_gate'],nominal_wy_deg_s=float(np.degrees(r['nominal_twist_tool'][4])),candidate_wy_deg_s=float(np.degrees(r['candidate_twist_tool'][4])),published_wy_deg_s=None if p is None else float(np.degrees(p['final_command_model_tool'][4])),diagnostics=d))
  out.append(report);print(a['directory'],report['feature_status_cycles'],report['episode_reason_cycles'],'nomcovers',report['nominal_already_covers_request_fraction'],'q<.8',report['unique_side_min_below08_fraction'],flush=True)
(OUT/'runtime.json').write_text(json.dumps(out,indent=2)+'\n')
