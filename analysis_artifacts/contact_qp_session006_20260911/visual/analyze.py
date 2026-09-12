from pathlib import Path
import json,sys
import numpy as np,h5py,cv2
from scipy.spatial.transform import Rotation,Slerp
sys.path.insert(0,str(Path.cwd()))
from peirastic.contact_qp.features import FeatureConfig,random_walk_confidence,window_quality,confidence_features
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path('/media/camp/yameng/icra 2027/uncalibrated/006');OUT=Path(__file__).parent
s=json.loads((ROOT/'session.json').read_text());epoch=s['clock']['anchor_time_ns']-s['clock']['anchor_monotonic_ns']
results=[]
for trial in s['trials']:
 for a in trial['attempts']:
  if 'tracking_observed' not in a['stages']:continue
  p=ROOT/a['directory']; controls=[]; pubs={}
  for l in (p/'contact_qp.jsonl').open():
   r=json.loads(l)
   if r['event']=='control_sample':controls.append(r)
   elif r['event']=='publication' and r['success']:pubs[r['control_id']]=r
  t=np.array([r['record_monotonic_s'] for r in controls]);start=(a['stages']['tracking_observed']-epoch)*1e-9;end=(a['stages']['path_done']-epoch)*1e-9 if 'path_done' in a['stages'] else t[-1]
  cfg=FeatureConfig(**a['contact_qp']['config']['feature']['config'])
  q=np.array([r['feature']['quality'] if r['feature'] else [np.nan]*3 for r in controls]); nom=np.array([r['nominal_twist_tool'] for r in controls]);can=np.array([r['candidate_twist_tool'] if r['candidate_twist_tool'] else [np.nan]*6 for r in controls]); fin=np.array([pubs[r['control_id']]['final_command_model_tool'] if r['control_id'] in pubs else [np.nan]*6 for r in controls])
  request=np.array([r['allocation_diagnostics'].get('differential_request_m_s',0) for r in controls]);sign=np.array([r['allocation_diagnostics'].get('differential_sign',0) for r in controls]);gate=np.array([r['allocation_diagnostics']['repair_force_gate'] for r in controls]);avail=np.array([r['energy']['available_j'] for r in controls]);permission=np.array([r['repair_episode']['repair_allowed'] for r in controls]);exhaust=np.array([r['repair_episode']['exhausted'] for r in controls]);valid=np.array([r['image_feedback_status']=='ok' for r in controls]);
  # Flat 50 mm aperture, centered TCP, endpoint delta=delta_vz +/- .025*delta_wy.
  d=can-nom; aperture=np.abs(d[:,2])+0.025*np.abs(d[:,4]);power=np.array([-np.dot(r['physical_wrench_candidate_tool'],c) for r,c in zip(controls,can)]); liability=np.maximum(-power,0)*.05
  byframe={}
  for i,r in enumerate(controls):
   if r['feature']:byframe.setdefault(r['feature']['frame_seq'],[]).append(i)
  result=dict(attempt=a['directory'],status=a['status'],scan_s=end-start,segments={},selected_frames=[])
  with h5py.File(p/'raw.h5') as h:
   pt=h['tcp/timestamp_mono_ns'][:]*1e-9;rot=Slerp(pt,Rotation.from_euler('xyz',h['tcp/pose_m_rad'][:,3:]));ft=h['force/timestamp_mono_ns'][:]*1e-9;force=h['force/contact_force_n'][:]
   for name,lo,hi in [('all',0,1),('first_half',0,.5),('late_half',.5,1),('last_quarter',.75,1)]:
    t0=start+lo*(end-start);t1=start+hi*(end-start); mask=(t>=t0)&(t<t1);idx=np.flatnonzero(mask); dt=np.maximum(0,np.minimum(np.r_[t[1:],end],t1)-np.maximum(t,t0));m=mask&np.isfinite(fin[:,4]);req=m&(request>0);rpair=rot([t0,t1]); net=(rpair[0].inv()*rpair[1]).as_rotvec();
    sampled=rot(np.linspace(t0,t1,int((t1-t0)*30)+1));incs=(sampled[:-1].inv()*sampled[1:]).as_rotvec();
    def quant(x):return np.quantile(x,[0,.5,.95,1]).tolist()
    z=dict(duration_s=t1-t0,cycles=int(mask.sum()),request_fraction=float(np.mean(request[mask]>0)),request_s=float(dt[request>0].sum()),permission_fraction=float(np.mean(permission[mask])),exhausted_cycles=int(exhaust[mask].sum()),image_ok_fraction=float(np.mean(valid[mask])),quality_lcr_p05_p50_p95=np.quantile(q[mask],[.05,.5,.95],axis=0).tolist(),imbalance_abs_p50_p95_max=np.quantile(abs(q[mask,2]-q[mask,0]),[.5,.95,1]).tolist(),both_sides_below08_fraction=float(np.mean(np.all(q[mask][:,[0,2]]<.8,axis=1))),force_gate_zero_fraction=float(np.mean(gate[mask]==0)),force_gate_partial_fraction=float(np.mean((gate[mask]>0)&(gate[mask]<1))),force_n_min_p50_p95_max=quant(force[(ft>=t0)&(ft<t1)]),request_mm_s_min_p50_p95_max=quant(1000*request[mask]),nominal_wy_deg_s_min_p50_p95_max=quant(np.degrees(nom[mask,4])),candidate_wy_deg_s_min_p50_p95_max=quant(np.degrees(can[mask,4])),candidate_minus_nominal_abs_wy_deg_s_min_p50_p95_max=quant(np.degrees(abs(d[mask,4]))),final_abs_wy_deg_s_min_p50_p95_max=quant(np.degrees(abs(fin[m,4]))),final_candidate_max_diff_deg_s=float(np.degrees(np.max(abs(fin[m,4]-can[m,4])))),qp_adjustment_requested_direction_fraction=None if not req.any() else float(np.mean(-sign[req]*d[req,4]>1e-9)),final_requested_direction_fraction=None if not req.any() else float(np.mean(-sign[req]*fin[req,4]>1e-9)),nominal_integral_deg=float(np.degrees(np.sum(nom[:,4]*dt))),qp_minus_nominal_integral_deg=float(np.degrees(np.nansum(d[:,4]*dt))),final_integral_deg=float(np.degrees(np.nansum(fin[:,4]*dt))),measured_relative_tool_xyz_deg=np.degrees(net).tolist(),measured_30hz_tool_y_abs_travel_deg=float(np.degrees(abs(incs[:,1]).sum())),energy_available_j_min_p50_p95_max=quant(avail[mask]),energy_50ms_liability_max_j=float(np.max(liability[mask])),aperture_used_mm_s_min_p50_p95_max=quant(aperture[mask]*1000),aperture_at_limit_fraction=float(np.mean(aperture[mask]>.00075-1e-8)),aperture_at_limit_requested_fraction=None if not req.any() else float(np.mean(aperture[req]>.00075-1e-8)))
    result['segments'][name]=z
   ids=h['ultrasound/frame_index'][:];ut=h['ultrasound/timestamp_mono_ns'][:]*1e-9;lookup={int(f):i for i,f in enumerate(ids)}
   selected=[('early',int(np.argmin(abs(ut-(start+2))))),('middle',int(np.argmin(abs(ut-(start+.5*(end-start)))))),('late',int(np.argmin(abs(ut-(end-2)))))]
   eligible=[i for i in range(len(ids)) if start<ut[i]<end and int(ids[i]) in byframe]
   selected=[(label,min(eligible,key=lambda i:abs(ut[i]-ut[index]))) for label,index in selected]
   selected.append(('lowest side q',min(eligible,key=lambda i:min(q[byframe[int(ids[i])][0]][[0,2]]))))
   fig,axes=plt.subplots(4,3,figsize=(16,22));fig.suptitle(p.parent.name+' / '+p.name+' — saved grayscale, confidence, shallow column confidence')
   for j,(label,index) in enumerate(selected):
    im=cv2.imdecode(np.asarray(h['ultrasound/jpeg'][index],np.uint8),cv2.IMREAD_GRAYSCALE);conf=random_walk_confidence(im,cfg);fq=window_quality(conf,cfg);details=confidence_features(im,conf,cfg);fid=int(ids[index]); online=byframe.get(fid,[]); online=[i for i in online if t[i]>=start and t[i]<=end];hh,ww=im.shape
    dark={}
    for band,(xl,xr) in {'left_outer4':(0,.04),'left_outer10':(0,.1),'left_window':(.04,.34),'center':(.34,.66),'right_window':(.66,.96),'right_outer10':(.9,1),'right_outer4':(.96,1)}.items():
     dark[band]={}
     for dep,(yl,yr) in {'all':(0,1),'top22':(0,.22),'below22':(.22,1)}.items():
      cut=im[int(hh*yl):int(hh*yr),int(ww*xl):int(ww*xr)]; dark[band][dep]=dict(mean_gray=float(cut.mean()),fraction_gray_lt20=float(np.mean(cut<20)))
    item=dict(label=label,frame_id=fid,scan_time_s=float(ut[index]-start),shape=im.shape,quality_lcr=fq.tolist(),online_quality_lcr=None if not online else q[online[0]].tolist(),sameframe_quality_max_abs_error=None if not online else float(np.max(abs(q[online]-fq))),online_request_mm_s=None if not online else float(1000*request[online[0]]),online_force_gate=None if not online else float(gate[online[0]]),dark_statistics=dark,top_column_confidence=details['column_confidence'],unknown_columns=details['unknown_columns'])
    result['selected_frames'].append(item)
    axes[j,0].imshow(im,cmap='gray',vmin=0,vmax=255);axes[j,0].set_title(f'{label} t={ut[index]-start:.2f}s frame={fid}');axes[j,0].axhline(.22*hh,color='cyan')
    for lo,hi in cfg.lateral_windows:
     for x in [lo,hi]:axes[j,0].axvline(x*ww,color='lime',lw=.6)
    axes[j,1].imshow(conf,cmap='gray',vmin=0,vmax=1);axes[j,1].set_title(f'qL/C/R={fq[0]:.3f}/{fq[1]:.3f}/{fq[2]:.3f}');axes[j,1].axhline(22,color='cyan')
    for lo,hi in cfg.lateral_windows:
     for x in [lo,hi]:axes[j,1].axvline(x*cfg.width,color='lime',lw=.6)
    axes[j,2].plot(np.arange(cfg.width)/cfg.width,details['column_confidence']);axes[j,2].axhline(.8,color='red',ls='--');axes[j,2].set_ylim(0,1.03)
    for k,(lo,hi) in enumerate(cfg.lateral_windows):axes[j,2].axvspan(lo,hi,alpha=.15,color=['green','yellow','orange'][k])
    axes[j,2].set_title(f'Shallow means delta R-L={fq[2]-fq[0]:.3f}, deadband=.03')
   fig.tight_layout();fig.savefig(OUT/(p.parent.name+'_'+p.name+'.png'),dpi=120);plt.close(fig)
  results.append(result); print(a['directory'],json.dumps(result['segments']['all']),flush=True)
(OUT/'metrics.json').write_text(json.dumps(dict(method='Read-only raw recorded data; scan intervals from shared session clock. Statistics time-sample weighted except explicit integrated durations. Measured rotation via SO(3), includes all control/path contributions. Dark gray<20 is descriptive, not contact ground truth. Aperture uses configured identity TCP-face and default .75mm/s budget; energy liability uses worst configured .05s command interval.',attempts=results),indent=2)+'\n')
