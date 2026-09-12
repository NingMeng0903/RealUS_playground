from pathlib import Path
import json,sys,collections
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py,cv2
sys.path.insert(0,str(Path.cwd()))
from peirastic.contact_qp.features import FeatureConfig,random_walk_confidence,window_quality

BASE=Path('/media/camp/PEI_T7/icra 2027_contact/uncalibrated');OUT=Path(__file__).parent
metrics=json.loads((OUT/'metrics.json').read_text());out=[]
selected=[]
for subject in ['yangming','yuhan_L','yuhan_R','zhongyao2']:
    eligible=[a for a in metrics if a['attempt'].startswith(subject+'/') and a.get('visual') and a['status']=='completed']
    selected.append(max(eligible,key=lambda a:a['visual']['qp_increment_abs_integral_deg'])['attempt'])
for a in metrics:
    p=BASE/a['attempt'];rows=[];pubs={}
    with (p/'contact_qp.jsonl').open('rb') as f:
        for l in f:
            if b'"event":"control_sample"' in l or b'"event": "control_sample"' in l:
                rows.append(json.loads(l))
            elif b'"event":"publication"' in l or b'"event": "publication"' in l:
                r=json.loads(l)
                if r.get('success'):pubs[r['control_id']]=r
    if not rows:continue
    t=np.array([r['record_monotonic_s'] for r in rows]);end=t[-1];dt=np.diff(np.r_[t,end]);
    v=a.get('visual');start=v['scan_start_s'] if v else None
    result=dict(attempt=a['attempt'],all_control_image_pauses=[],segments={},selected_frames=[])
    marked=np.array([r['image_feedback_status']!='ok' for r in rows]);edges=np.diff(np.r_[False,marked,False].astype(int))
    for lo,hi in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):
        result['all_control_image_pauses'].append(dict(start_monotonic_s=float(t[lo]),start_scan_s=float(t[lo]-start) if start else None,duration_s=float(sum(dt[lo:hi])),resumed=bool(hi<len(rows) and not marked[hi]),image_status=rows[lo]['image_feedback_status'],force_fresh_cycles=sum(r['source']['fresh'] for r in rows[lo:hi]),cycles=int(hi-lo),request_cycles=sum(r['allocation_diagnostics'].get('differential_request_m_s',0)>0 for r in rows[lo:hi]),publication_cycles=sum(r['control_id'] in pubs for r in rows[lo:hi])))
    if not v:out.append(result);continue
    end=v['scan_end_s'];byframe={}
    for i,r in enumerate(rows):
        f=r.get('feature')
        if f and start<=t[i]<end:byframe.setdefault((f['source_id'],f['frame_seq']),i)
    for label,lo,hi in [('all',0,1),('first_quarter',0,.25),('last_quarter',.75,1),('first_half',0,.5),('last_half',.5,1)]:
        ids=[i for i in byframe.values() if start+(end-start)*lo<=t[i]<start+(end-start)*hi]
        if not ids:continue
        q=np.array([rows[i]['feature']['quality'] for i in ids]);d=abs(q[:,2]-q[:,0])
        result['segments'][label]=dict(unique_frames=len(ids),lcr_p05_p50_p95=np.quantile(q,[.05,.5,.95],axis=0).tolist(),side_min_below08_fraction=float(np.mean(q[:,[0,2]].min(axis=1)<.8)),both_sides_below08_fraction=float(np.mean(np.all(q[:,[0,2]]<.8,axis=1))),side_difference_gt003_fraction=float(np.mean(d>.03)))
    if a['attempt'] in selected:
        times=t-start;q=np.array([r['feature']['quality'] if r['feature'] else [np.nan]*3 for r in rows]);energy=np.array([r['energy']['balance_j'] for r in rows]);nom=np.degrees([r['nominal_twist_tool'][4] for r in rows]);can=np.degrees([r['candidate_twist_tool'][4] if r['candidate_twist_tool'] else np.nan for r in rows]);final=np.degrees([pubs[r['control_id']]['final_command_model_tool'][4] if r['control_id'] in pubs else np.nan for r in rows]);request=np.array([r['allocation_diagnostics'].get('differential_request_m_s',0)*1000 for r in rows]);
        fig,ax=plt.subplots(4,1,figsize=(12,9),sharex=True);fig.suptitle(a['attempt'])
        ax[0].plot(times,energy,label='Tank balance');ax[0].axhline(.05,c='r',ls='--',label='Reserve');ax[0].axhline(.15,c='gray',ls='--');ax[0].set_ylabel('Energy (J)');ax[0].legend(loc='lower right')
        for k,label in enumerate(['Left','Center','Right']):ax[1].plot(times,q[:,k],label=label,lw=1)
        ax[1].axhline(.8,c='gray',ls='--');ax[1].set_ylabel('Shallow confidence');ax[1].legend(loc='lower right')
        ax[2].plot(times,request,c='purple');ax[2].set_ylabel('Visual request (mm/s)')
        ax[3].plot(times,nom,label='Nominal',lw=.9);ax[3].plot(times,can,label='QP candidate',lw=.9);ax[3].plot(times,final,label='Published',lw=.7);ax[3].set_ylabel('Tool wy (deg/s)');ax[3].set_xlabel('Seconds from scan start');ax[3].legend()
        for pause in v['pauses']:
            if pause['type']=='image':
                for axis in ax:axis.axvspan(pause['start_scan_s'],pause['start_scan_s']+pause['duration_s'],color='red',alpha=.3)
        for axis in ax:axis.grid(alpha=.2);axis.set_xlim(0,end-start)
        fig.tight_layout();fig.savefig(OUT/(a['attempt'].replace('/','_')+'_timeline.png'),dpi=150);plt.close(fig)
        cfg=FeatureConfig(**a['config']['feature']['config'])
        with h5py.File(p/'raw.h5') as h:
            lookup={int(fid):i for i,fid in enumerate(h['ultrasound/frame_index'][:])}
            picked=[r for r in v['representative'] if r['label'] in ['maximum_aligned_increment','worst_side_quality','last_image'] and r['frame'] in lookup]
            fig,axes=plt.subplots(len(picked),2,figsize=(10,5*len(picked)),squeeze=False)
            for j,c in enumerate(picked):
                index=lookup[c['frame']];im=cv2.imdecode(np.asarray(h['ultrasound/jpeg'][index],np.uint8),cv2.IMREAD_GRAYSCALE);conf=random_walk_confidence(im,cfg);fq=window_quality(conf,cfg)
                result['selected_frames'].append(dict(label=c['label'],frame=c['frame'],scan_time_s=c['scan_time_s'],quality=fq.tolist(),same_frame_max_error=float(np.max(abs(fq-np.array(c['q'])))),nominal_wy_deg_s=float(np.degrees(c['nominal'])),candidate_wy_deg_s=float(np.degrees(c['candidate'])),published_wy_deg_s=float(np.degrees(c['published_wy'])) if c['published_wy'] is not None else None,request_mm_s=c['request']*1000))
                axes[j,0].imshow(im,cmap='gray',vmin=0,vmax=255);axes[j,0].axhline(.22*im.shape[0],c='cyan');axes[j,0].set_title(f"{c['label']} / frame {c['frame']} / t={c['scan_time_s']:.2f}s")
                axes[j,1].imshow(conf,cmap='gray',vmin=0,vmax=1);axes[j,1].axhline(.22*conf.shape[0],c='cyan');axes[j,1].set_title('qL/C/R = '+ '/'.join(f'{x:.3f}' for x in fq))
                for axis,imx in [(axes[j,0],im),(axes[j,1],conf)]:
                    for lo,hi in cfg.lateral_windows:
                        for x in [lo,hi]:axis.axvline(x*imx.shape[1],c='lime',lw=.5)
            fig.tight_layout();fig.savefig(OUT/(a['attempt'].replace('/','_')+'_frames.png'),dpi=130);plt.close(fig)
    out.append(result);print(a['attempt'],'pauses',len(result['all_control_image_pauses']),'frames',len(result['selected_frames']),flush=True)
(OUT/'visual_detail.json').write_text(json.dumps(dict(selected=selected,attempts=out),indent=2,allow_nan=False))
