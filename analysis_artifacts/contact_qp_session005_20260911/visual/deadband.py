from pathlib import Path
import json,numpy as np
ROOT=Path('/media/camp/yameng/icra 2027/uncalibrated/005');OUT=Path(__file__).parent;s=json.loads((ROOT/'session.json').read_text());epoch=s['clock']['anchor_time_ns']-s['clock']['anchor_monotonic_ns'];out=[];baseline=[]
for trial in s['trials']:
 for a in trial['attempts']:
  if a.get('status')!='completed':continue
  start=(a['stages']['tracking_observed']-epoch)*1e-9;end=(a['stages']['path_done']-epoch)*1e-9;byframe={}
  for l in (ROOT/a['directory']/'contact_qp.jsonl').open():
   r=json.loads(l)
   if r['event']=='control_sample' and r['feature'] and start<r['record_monotonic_s']<end:byframe.setdefault((r['feature']['source_id'],r['feature']['frame_seq']),r)
  rows=sorted(byframe.values(),key=lambda r:r['feature']['effective_time_s']);t=np.array([r['feature']['effective_time_s'] for r in rows]);q=np.array([r['feature']['quality'] for r in rows]);dif=q[:,2]-q[:,0];parts={}
  for label,m in [('early2to8',(t>=start+2)&(t<=start+8)),('latehalf',(t>=(start+end)/2)&(t<=end))]:
   x=dif[m];changes=np.diff(x);z=dict(unique_frames=int(m.sum()),abs_imbalance_p50_p95_p99_max=np.quantile(abs(x),[.5,.95,.99,1]).tolist(),signed_median=float(np.median(x)),consecutive_abs_change_p50_p95_p99_max=np.quantile(abs(changes),[.5,.95,.99,1]).tolist(),robust_change_sigma=float(1.4826*np.median(abs(changes-np.median(changes)))/np.sqrt(2)),trigger_fractions={str(d):float(np.mean(abs(x)>d)) for d in [.01,.02,.03,.05,.1]})
   parts[label]=z
   if label=='latehalf' and any(k in a['directory'] for k in ['L_DtP','C_DtP']):baseline.extend(x.tolist())
  out.append(dict(attempt=a['directory'],segments=parts))
b=np.array(baseline);summary=dict(unique_frames=len(b),abs_imbalance_p50_p95_p99_max=np.quantile(abs(b),[.5,.95,.99,1]).tolist(),trigger_fractions={str(d):float(np.mean(abs(b)>d)) for d in [.01,.02,.03,.05,.1]})
result=dict(method='Deduplicated feature source_id/frame_seq, one statistic per image, not repeated 200Hz controls. Consecutive frames may remain temporally correlated; not an IID noise estimate or physical image calibration. Baseline chosen as late halves of completed L_DtP/C_DtP with stable high confidence and no side band in sampled panels; other early segments shown for comparison.',baseline=summary,attempts=out,recommendation='0.03 initial engineering balance deadband for controlled replay/test: larger than baseline maximum .0166 and about 2.4x baseline p99, while exposing S observed .0975 imbalance. Keep ROI and registration unchanged. This setting is empirical, not calibrated ground truth.')
(OUT/'deadband.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
