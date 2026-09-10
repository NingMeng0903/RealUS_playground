"""Read-only descriptive comparison of three independently taught real scans."""
from pathlib import Path
import json, collections, hashlib
import cv2, h5py, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation, Slerp
from peirastic.contact_qp.features import FeatureExtractor, load_feature_config

OUT=Path(__file__).resolve().parent
BASE=Path('/media/camp/EXT_DRIVE/ICRA_2027/icra 2027_contact/real_characterization')
SOURCES={'baseline':'baseline_probe50/002','shadow':'shadow_probe50/001','active':'active_probe50/002'}
CFG=load_feature_config('peirastic/config/contact_qp/active_probe50.yaml')
report={'scope':'One independently taught scan per condition. Descriptive, not paired causal validation. Confidence is a policy proxy, not contact ground truth.',
        'feature_version':CFG.window_version,'threshold':.8,'image_pose_delay_s':CFG.effective_delay_s,
        'alignment':'Contact-segment H5 retains capture timestamps; subtract image delay once only for pose projection. Force uses latest force timestamp; videos use capture elapsed time.',
        'scans':{}}
fig,axs=plt.subplots(3,3,figsize=(15,10),constrained_layout=True)
def stats(f):
    e=f-4.
    return dict(mean_n=float(f.mean()),rmse_n=float(np.sqrt(np.mean(e*e))),abs_bias_n=float(abs(e.mean())),
                peak_abs_error_n=float(np.max(abs(e))),min_n=float(f.min()),max_n=float(f.max()))

for col,(label,rel) in enumerate(SOURCES.items()):
    path=BASE/rel/'RH_Per_L_DtP.h5'
    with h5py.File(path) as f:
        meta=json.loads(f.attrs['scan_metadata_json']);raw=Path(f.attrs['postprocess_source'])
        tf=f['force/source_time_s'][:];force=f['force/contact_force_n'][:];wrench=f['force/wrench_tcp'][:]
        ti=f['ultrasound/source_time_s'][:];seq=f['ultrasound/frame_index'][:]
        md=json.loads(f['ultrasound/metadata_json'][0]);n=len(ti)
        cache=OUT/(label+'_features.npz')
        if cache.exists():
            data=np.load(cache);q=data['quality'];valid=data['valid'];assert np.array_equal(data['frame_index'],seq)
            assert data['window_version'].item()==CFG.window_version
        else:
            ex=FeatureExtractor(CFG);q=[];valid=[]
            for i in range(n):
                image=cv2.imdecode(np.asarray(f['ultrasound/jpeg'][i],dtype=np.uint8),cv2.IMREAD_GRAYSCALE)
                obs,_=ex.extract(image,frame_seq=int(seq[i]),source_id='offline',capture_time_s=float(ti[i]),received_time_s=float(ti[i]))
                q.append(obs.quality);valid.append(obs.valid)
                if i%300==0: print(label,'features',i,'/',n,flush=True)
            q=np.array(q);valid=np.array(valid)
            np.savez_compressed(cache,quality=q,valid=valid,frame_index=seq,capture_time_s=ti,window_version=CFG.window_version)
        assert bool(f.attrs['complete'])
    with h5py.File(raw) as f:
        tp=f['tcp/source_time_s'][:];poses=f['tcp/pose_m_rad'][:]
    keep=np.r_[True,np.diff(tp)>0];tp=tp[keep];poses=poses[keep]
    start=np.array(meta['taught']['distal']['pose_m_rad'][:3]);end=np.array(meta['taught']['proximal']['pose_m_rad'][:3])
    length=np.linalg.norm(end-start);axis=(end-start)/length
    effective=ti-CFG.effective_delay_s
    assert effective.min()>=tp.min() and effective.max()<=tp.max()
    xyz=np.array([np.interp(effective,tp,poses[:,k]) for k in range(3)]).T
    pos=(xyz-start)@axis;delta=np.maximum(np.diff(pos),0.)
    known=valid[:,[0,2]].all(axis=1);bad=known & (q[:,[0,2]]<.8).any(axis=1);good=known & ~bad
    quality=lambda a:float(np.mean(a))
    gap=float(np.sum(delta*(bad[:-1].astype(float)+bad[1:])/2)*1000)
    # Body orientation relative to segment-start, sampled at 10 Hz to limit
    # counting high-rate SDK quantization as physical rocking.
    times=np.arange(max(tf[0],tp[0]),min(tf[-1],tp[-1]),.1)
    rr=Slerp(tp,Rotation.from_euler('xyz',poses[:,3:]))(times)
    relrot=(rr[0].inv()*rr).as_rotvec()*180/np.pi
    bodyinc=(rr[:-1].inv()*rr[1:]).as_rotvec()*180/np.pi
    log=raw.parent/'contact_qp.jsonl'
    rows=[json.loads(line) for line in log.read_text().splitlines()]
    counts=dict(collections.Counter(r['event'] for r in rows))
    controls=[r for r in rows if r['event']=='control_sample']
    timestamp=lambda r:r.get('source',{}).get('source_t_s',r.get('wrench_source_time_s',r['record_monotonic_s']))
    selected=[r for r in controls if tf[0]<=timestamp(r)<=tf[-1]]
    entry={'source':str(path),'frames':n,'frame_duration_s':float(ti[-1]-ti[0]),'force_samples':len(tf),'force':stats(force),
           'taught_length_mm':float(length*1000),'taught_start_m':start.tolist(),'taught_end_m':end.tolist(),
           'image_metadata':{k:md.get(k) for k in ('crop_box','hflip','width','height')},'stage_timestamps':meta['stages'],
           'both_windows_good_frame_fraction':quality(good),'low_window_frame_fraction':quality(bad),'unknown_frame_fraction':quality(~known),
           'quality_mean_lcr':q.mean(axis=0).tolist(),'low_left_right_fraction':(q[:,[0,2]]<.8).mean(axis=0).tolist(),
           'projected_image_span_mm':float((pos[-1]-pos[0])*1000),'forward_exposure_mm':float(delta.sum()*1000),
           'low_quality_forward_exposure_mm':gap,'gap_metric':'Trapezoid low-confidence indicator times positive measured axial displacement; revisits count. Not unique anatomical coverage.',
           'measured_relative_rotation_y_range_deg':[float(relrot[:,1].min()),float(relrot[:,1].max())],
           'measured_body_y_total_variation_10hz_deg':float(np.abs(bodyinc[:,1]).sum()),
           'lateral_force_norm_mean_max_n':[float(np.linalg.norm(wrench[:,:2],axis=1).mean()),float(np.linalg.norm(wrench[:,:2],axis=1).max())],
           'events':counts,'stop_reasons':[r['reason'] for r in rows if r['event']=='stop'],
           'dropped_records_max':max(r.get('dropped_records',0) for r in rows),'selected_control_samples':len(selected)}
    tt=np.array([timestamp(r) for r in selected]);nom=np.array([r['nominal_twist_tool'] for r in selected])
    if label=='active':
        cmd=np.array([r['candidate_twist_tool'] for r in selected]);d=cmd-nom;a=np.array([r['alpha'] for r in selected])
        dt=np.array([r['control_actual_dt_s'] for r in selected]);repair=(abs(d[:,2])>1e-5)|(abs(d[:,4])>np.deg2rad(.01))
        entry['active_commands']=dict(alpha_min=float(a.min()),alpha_mean=float(a.mean()),alpha_timeweighted=float(np.average(a,weights=dt)),
             slowed_fraction=float(np.mean(a<.99)),added_normal_or_rocking_fraction=float(repair.mean()),
             added_normal_mm_s_min_max=[float(d[:,2].min()*1000),float(d[:,2].max()*1000)],
             added_rocking_deg_s_min_max=[float(np.rad2deg(d[:,4].min())),float(np.rad2deg(d[:,4].max()))],
             added_rocking_absolute_command_integral_deg=float(np.sum(abs(d[:,4])*dt)*180/np.pi),
             valid_measured_twist_fraction=float(np.mean([r['measured_twist_valid'] for r in selected])),
             compatible_image_fraction=float(np.mean([r['image_compatible'] for r in selected])))
        entry['active_commands']['display_activation_threshold']='0.01 mm/s normal or 0.01 deg/s rocking; descriptive display threshold, not preregistered acceptance. Exact maxima also reported.'
        pubs=[r for r in rows if r['event']=='publication'];entry['reference_final_s']=pubs[-1]['reference_s']
        entry['accepted_alpha_mean']=float(np.mean([r['accepted_alpha'] for r in pubs]))
        af,aa=plt.subplots(2,1,figsize=(11,6),constrained_layout=True)
        aa[0].plot(tt-tt[0],np.rad2deg(d[:,4]),label='Added rocking (deg/s)')
        aa[0].plot(tt-tt[0],d[:,2]*1000,label='Added normal (mm/s)')
        aa[0].set_ylim(-.01,.01);aa[0].set_title('Added QP actions are negligible on this scan; nominal motion is retained')
        aa[1].plot(tt-tt[0],a,label='QP alpha')
        chosen=[r for r in pubs if selected[0]['control_id']<=r['control_id']<=selected[-1]['control_id']]
        aa[1].plot([r['record_monotonic_s']-tt[0] for r in chosen],[r['accepted_alpha'] for r in chosen],alpha=.6,lw=.6,label='Reference credit factor (final model; not measured speed)')
        for ax in aa:ax.legend(fontsize=8);ax.grid(alpha=.2);ax.set_xlabel('Elapsed seconds')
        af.savefig(OUT/'active_commands.png',dpi=150);plt.close(af)
    axs[2,col].plot(times-times[0],relrot[:,1],label='Measured relative Y angle (deg)');axs[2,col].legend(fontsize=8)
    axs[2,col].set_ylim(-1,12)
    axs[0,col].plot(tf-tf[0],force,lw=.6);axs[0,col].axhline(4,color='k',ls='--');axs[0,col].set_ylim(3.5,4.8)
    axs[0,col].set_title(label+f' | RMSE {entry["force"]["rmse_n"]:.3f} N');axs[0,col].set_ylabel('Force (N)')
    axs[1,col].plot(ti-ti[0],q[:,0],label='Left');axs[1,col].plot(ti-ti[0],q[:,2],label='Right');axs[1,col].axhline(.8,color='k',ls='--')
    axs[1,col].set_ylim(0,1.02);axs[1,col].set_ylabel('Confidence');axs[1,col].legend(fontsize=8)
    for row in axs:row[col].set_xlabel('Elapsed seconds in saved contact segment');row[col].grid(alpha=.2)
    report['scans'][label]=entry
    print(label,json.dumps({k:entry[k] for k in ('frames','force','both_windows_good_frame_fraction','low_quality_forward_exposure_mm')}),flush=True)
fig.suptitle('Real scans, independently taught paths | confidence proxy, no physical contact labels',fontsize=13)
fig.savefig(OUT/'comparison_metrics.png',dpi=150);plt.close(fig)
(OUT/'inspection.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
print('Saved',OUT/'inspection.json',flush=True)
