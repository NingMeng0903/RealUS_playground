"""Offline description of executed QP requests; never connects to hardware."""
from pathlib import Path
import json, collections
import h5py, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE=Path('/media/camp/EXT_DRIVE/ICRA_2027/icra 2027_contact/real_characterization/active_probe50/003')
OUT=Path(__file__).resolve().parent
NAMES=['RH_Per_L_DtP','RH_Per_C_DtP','RH_Per_S_DtP','RH_Per_L_PtD','RH_Per_C_PtD']
report={'scope':'Descriptive executed-command analysis of five successful, separately recorded scans. No acoustic contact ground truth. Command integrals are not measured displacement.',
 'thresholds':{'c_min':.8,'force_sign_band_n':.1,'left_window_x_m':.0155,'aperture_half_length_m':.025,
 'visible_added_action_normal_mm_s':.01,'visible_added_action_rocking_deg_s':.01},'scans':{}}
fig,axs=plt.subplots(5,4,figsize=(18,14),constrained_layout=True)
for row,name in enumerate(NAMES):
 with h5py.File(BASE/(name+'.h5')) as f:
  raw=Path(f.attrs['postprocess_source']);force=f['force/contact_force_n'][:]
  frame_count=len(f['ultrasound/jpeg']);complete=bool(f.attrs['complete'])
 rr=[json.loads(l) for l in (raw.parent/'contact_qp.jsonl').read_text().splitlines()]
 cs=[r for r in rr if r['event']=='control_sample' and r['h_ref_s']>0]
 t=np.array([r['source']['source_t_s'] for r in cs]);elapsed=t-t[0]
 n=np.array([r['nominal_twist_tool'] for r in cs]);c=np.array([r['candidate_twist_tool'] for r in cs]);d=c-n
 q=np.array([r['feature']['quality'] if r.get('feature') else [np.nan]*3 for r in cs]);F=np.array([r['control_wrench_tool'][2] for r in cs]);a=np.array([r['alpha'] for r in cs])
 hold=np.array([r['command_hold_model_s'] for r in cs])
 low=q[:,0]<.8;req=np.maximum(.8-q[:,0],0)/.8*.002-np.maximum(q[:,0]-.8,0)/.2*.004
 nominal_local=n[:,2]-.0155*n[:,4];actual_request_local=c[:,2]-.0155*c[:,4]
 meets=nominal_local>=req;high=F>4.1
 visible=(abs(d[:,2])>1e-5)|(abs(d[:,4])>np.deg2rad(.01))
 item={'complete':complete,'frames':frame_count,'events':dict(collections.Counter(r['event'] for r in rr)),
  'stop_reasons':[r['reason'] for r in rr if r['event']=='stop'],
  'saved_force':{'mean_n':float(force.mean()),'rmse_n':float(np.sqrt(np.mean((force-4)**2))),
                 'min_n':float(force.min()),'max_n':float(force.max())},
  'scan_samples':len(cs),'scan_elapsed_s':float(elapsed[-1]),'low_left_count':int(low.sum()),
  'low_left_and_nominal_meets_local_request_count':int(np.sum(low&meets)),
  'low_left_nominal_below_request_force_high_count':int(np.sum(low&~meets&high)),
  'low_left_with_visible_added_action_count':int(np.sum(low&visible)),
  'max_added_vn_mm_s':float(np.max(abs(d[:,2]))*1000),'max_added_omega_deg_s':float(np.rad2deg(np.max(abs(d[:,4])))),
  'signed_added_rocking_command_hold_integral_deg':float(np.rad2deg(np.sum(d[:,4]*hold))),
  'absolute_added_rocking_command_hold_integral_deg':float(np.rad2deg(np.sum(abs(d[:,4])*hold))),
  'qp_alpha_mean':float(a.mean()),'qp_alpha_min':float(a.min()),'segments':{},'examples':[]}
 for label,mask in [('first5s',elapsed<5),('last5s',elapsed>elapsed[-1]-5)]:
  item['segments'][label]={'samples':int(mask.sum()),'q_min_lcr':np.nanmin(q[mask],axis=0).tolist(),
   'low_left_fraction':float(low[mask].mean()),'low_left_force_high_fraction':float(np.sum(mask&low&high)/max(np.sum(mask&low),1)),
   'max_added_omega_deg_s':float(np.rad2deg(np.max(abs(d[mask,4])))),
   'max_added_vn_mm_s':float(np.max(abs(d[mask,2]))*1000)}
 examples={'minimum_left_quality':int(np.nanargmin(q[:,0])), 'maximum_added_rocking':int(np.argmax(abs(d[:,4])))}
 noadded=np.flatnonzero(low&~visible)
 if len(noadded):examples['low_quality_no_material_added_action']=int(noadded[np.argmin(q[noadded,0])])
 for reason,i in examples.items():
  r=cs[i];item['examples'].append({'selection':reason,'control_id':r['control_id'],'elapsed_s':float(elapsed[i]),'frame_seq':r['feature']['frame_seq'],
   'quality_lcr':q[i].tolist(),'force_n':float(F[i]),'nominal_vn_mm_s':float(n[i,2]*1000),'command_vn_mm_s':float(c[i,2]*1000),
   'nominal_omega_deg_s':float(np.rad2deg(n[i,4])),'command_omega_deg_s':float(np.rad2deg(c[i,4])),
   'left_required_mm_s':float(req[i]*1000),'left_nominal_mm_s':float(nominal_local[i]*1000),
   'left_command_mm_s':float(actual_request_local[i]*1000),'alpha':float(a[i])})
 report['scans'][name]=item
 ax=axs[row];ax[0].plot(elapsed,q[:,0],label='Left');ax[0].plot(elapsed,q[:,2],label='Right');ax[0].axhline(.8,c='k',ls='--');ax[0].set_ylim(.3,1.02)
 ax[0].set_ylabel(name.replace('RH_Per_','')+'\nConfidence');ax[0].legend(fontsize=7)
 ax[1].plot(elapsed,F,lw=.5);ax[1].axhline(4.1,c='r',ls='--');ax[1].axhline(4,c='k',ls=':');ax[1].set_ylim(3.4,4.6);ax[1].set_ylabel('Tool-Z force (N)')
 ax[2].plot(elapsed,np.rad2deg(n[:,4]),label='Nominal');ax[2].plot(elapsed,np.rad2deg(c[:,4]),alpha=.7,label='QP command');ax[2].plot(elapsed,np.rad2deg(d[:,4]),lw=.6,label='Added');ax[2].set_ylabel('Rocking (deg/s)');ax[2].legend(fontsize=7)
 ax[3].plot(elapsed,a);ax[3].set_ylim(0,1.02);ax[3].set_ylabel('QP alpha')
 for aa in ax:aa.grid(alpha=.2);aa.set_xlabel('Elapsed scan time (s)')
 print(name,'low',int(low.sum()),'nominal_meets',int(np.sum(low&meets)),'blocked_force',int(np.sum(low&~meets&high)),flush=True)
fig.suptitle('Active 003: recorded requested motion, not measured motion | first/last 5 s use scan clock')
fig.savefig(OUT/'five_scan_diagnostics.png',dpi=130);plt.close(fig)
(OUT/'inspection.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
