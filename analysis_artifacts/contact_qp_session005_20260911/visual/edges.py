from pathlib import Path
import json,sys
import numpy as np,h5py,cv2
sys.path.insert(0,str(Path.cwd()))
from peirastic.contact_qp.features import FeatureConfig,random_walk_confidence,window_quality,confidence_features
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path('/media/camp/yameng/icra 2027/uncalibrated/005');OUT=Path(__file__).parent
s=json.loads((ROOT/'session.json').read_text());epoch=s['clock']['anchor_time_ns']-s['clock']['anchor_monotonic_ns'];results=[]
for trial in s['trials']:
 for a in trial['attempts']:
  if a.get('status')!='completed' or not any(x in a['directory'] for x in ['S_DtP','L_PtD']):continue
  name=a['directory'].split('/')[1];side=2 if 'S_DtP' in name else 0
  controls=[];byframe={}
  for l in (ROOT/a['directory']/'contact_qp.jsonl').open():
   r=json.loads(l)
   if r['event']=='control_sample' and r['feature']:controls.append(r);byframe.setdefault(r['feature']['frame_seq'],r)
  start=(a['stages']['tracking_observed']-epoch)*1e-9;end=(a['stages']['path_done']-epoch)*1e-9;cfg=FeatureConfig(**a['contact_qp']['config']['feature']['config'])
  with h5py.File(ROOT/a['directory']/'raw.h5') as h:
   ids=h['ultrasound/frame_index'][:];ts=h['ultrasound/timestamp_mono_ns'][:]*1e-9;eligible=[i for i in range(len(ids)) if ts[i]>(start+end)/2 and ts[i]<end and int(ids[i]) in byframe]
   stats=[]
   for i in eligible:
    im=cv2.imdecode(np.asarray(h['ultrasound/jpeg'][i],np.uint8),0);hh,ww=im.shape;xl,xr=(.90,1) if side==2 else (0,.1);cut=im[int(.22*hh):,int(xl*ww):int(xr*ww)];r=byframe[int(ids[i])]
    stats.append(dict(i=i,frame=int(ids[i]),scan_t=float(ts[i]-start),outer10_below22_dark=float(np.mean(cut<20)),outer10_below22_mean=float(cut.mean()),q=r['feature']['quality'],request=r['allocation_diagnostics'].get('differential_request_m_s',0)))
   worstq=min(stats,key=lambda r:r['q'][side]);darkest=max(stats,key=lambda r:r['outer10_below22_dark']);blind=max([r for r in stats if r['request']==0],key=lambda r:r['outer10_below22_dark'])
   selection=[('Lowest side confidence',worstq),('Darkest outer 10%',darkest),('Darkest outer 10%, zero request',blind)]
   fig,axes=plt.subplots(3,3,figsize=(15,16));fig.suptitle(name+' late-half edge audit (raw gray <20 is descriptive, not contact truth)')
   chosen=[]
   for j,(label,row) in enumerate(selection):
    im=cv2.imdecode(np.asarray(h['ultrasound/jpeg'][row['i']],np.uint8),0);conf=random_walk_confidence(im,cfg);q=window_quality(conf,cfg);details=confidence_features(im,conf,cfg);r=byframe[row['frame']];hh,ww=im.shape
    row=dict(row,label=label,recomputed_q=q.tolist(),max_q_error=float(np.max(abs(q-r['feature']['quality']))),online_force_gate=r['allocation_diagnostics']['repair_force_gate'],nominal_wy_deg_s=float(np.degrees(r['nominal_twist_tool'][4])),candidate_wy_deg_s=float(np.degrees(r['candidate_twist_tool'][4])))
    chosen.append(row);axes[j,0].imshow(im,cmap='gray',vmin=0,vmax=255);axes[j,0].axhline(.22*hh,color='cyan');axes[j,0].set_title(f'{label}\nt={row["scan_t"]:.2f}s, frame={row["frame"]}\nouter10 below22 dark={row["outer10_below22_dark"]:.1%}')
    axes[j,1].imshow(conf,cmap='gray',vmin=0,vmax=1);axes[j,1].axhline(22,color='cyan');axes[j,1].set_title(f'qL/C/R={q[0]:.3f}/{q[1]:.3f}/{q[2]:.3f}\nrequest={row["request"]*1000:.3f}mm/s')
    for lo,hi in cfg.lateral_windows:
     for x in [lo,hi]:axes[j,0].axvline(x*ww,color='lime',lw=.6);axes[j,1].axvline(x*cfg.width,color='lime',lw=.6)
    axes[j,2].plot(np.arange(cfg.width)/cfg.width,details['column_confidence']);axes[j,2].axhline(.8,color='red',ls='--');axes[j,2].set_ylim(0,1.03);axes[j,2].set_title(f'Shallow side imbalance={q[2]-q[0]:.3f}\nnom/candidate wy={row["nominal_wy_deg_s"]:.3f}/{row["candidate_wy_deg_s"]:.3f}deg/s')
    for k,(lo,hi) in enumerate(cfg.lateral_windows):axes[j,2].axvspan(lo,hi,alpha=.15,color=['green','yellow','orange'][k])
   fig.tight_layout();fig.savefig(OUT/(name+'_edges.png'),dpi=120);plt.close(fig)
   results.append(dict(attempt=a['directory'],eligible_exact_frames=len(stats),selected_frames=chosen,all_frame_descriptive_stats=stats))
   print(name,chosen,flush=True)
(OUT/'edge_metrics.json').write_text(json.dumps(results,indent=2)+'\n')
