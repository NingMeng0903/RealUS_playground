import json,numpy as np
from pathlib import Path
out=[]
for session in ['005','006']:
 root=Path('/media/camp/yameng/icra 2027/uncalibrated')/session;s=json.loads((root/'session.json').read_text());epoch=s['clock']['anchor_time_ns']-s['clock']['anchor_monotonic_ns']
 for trial in s['trials']:
  for a in trial['attempts']:
   if a['status']!='completed':continue
   start=(a['stages']['tracking_observed']-epoch)*1e-9;end=(a['stages']['path_done']-epoch)*1e-9;uniq={}
   for l in (root/a['directory']/'contact_qp.jsonl').open():
    r=json.loads(l)
    if r['event']=='control_sample' and r['feature'] and start<r['record_monotonic_s']<end:uniq.setdefault((r['feature']['source_id'],r['feature']['frame_seq']),r)
   rows=list(uniq.values());parts={}
   for part,startp in [('all',start),('latehalf',(start+end)/2)]:
    q=np.array([r['feature']['quality'] for r in rows if r['record_monotonic_s']>=startp]);bad=q[:,[0,2]].min(axis=1);difference=abs(q[:,2]-q[:,0]);parts[part]=dict(n=len(q),quality_lcr_min_p05_p50=np.quantile(q,[0,.05,.5],axis=0).tolist(),side_min_under08_fraction=float(np.mean(bad<.8)),side_min_under09_fraction=float(np.mean(bad<.9)),imbalance_over03_fraction=float(np.mean(difference>.03)),imbalance_over10_fraction=float(np.mean(difference>.1)))
   out.append(dict(session=session,attempt=a['directory'],scan_s=end-start,segments=parts))
Path('analysis_artifacts/contact_qp_session006_20260911/visual/comparison005.json').write_text(json.dumps(dict(method='One retained control per unique image source/frame, successful scans only. Unmatched trajectory/contact/scan duration; descriptive observations, not causal validation.',attempts=out),indent=2)+'\n')
for r in out:
 z=r['segments']['latehalf'];print(r['session'],r['attempt'],'latebad<08',round(z['side_min_under08_fraction']*100,2),'q05',np.round(z['quality_lcr_min_p05_p50'][1],4).tolist(),flush=True)
