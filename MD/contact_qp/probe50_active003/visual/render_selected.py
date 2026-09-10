"""Selected real saved JPEGs and unchanged runtime feature configuration."""
from pathlib import Path
import json
import h5py,cv2,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from peirastic.contact_qp.features import FeatureConfig,random_walk_confidence,window_quality,confidence_features
ROOT=Path('/media/camp/EXT_DRIVE/ICRA_2027/icra 2027_contact/real_characterization/active_probe50/003')
OUT=Path(__file__).resolve().parent
report={'scope':'Selected original JPEG frames only. No flip/crop/intensity enhancement. Confidence is unchanged logged config. Gray <20 occupancy is descriptive, not contact truth.', 'selection':'Saved contact segment +0.5s, minimum logged left quality with h_ref>0 and frame in saved segment, segment end-0.5s. Same-frame online matches only; no temporal substitution.', 'scans':{}}
for path in sorted(ROOT.glob('*.h5')):
    logpath=ROOT/'attempts'/path.stem/'001/contact_qp.jsonl'
    rows=[json.loads(s) for s in logpath.read_text().splitlines()]
    config=next(r['config'] for r in rows if r['event']=='study_start')
    cfg=FeatureConfig(**config['feature']['config'])
    byframe={}
    for r in rows:
        if r.get('event')=='control_sample' and r.get('feature'):
            byframe.setdefault(r['feature']['frame_seq'],[]).append(r)
    with h5py.File(path,'r') as f:
        ids=f['ultrasound/frame_index'][:].astype(int);ts=f['ultrasound/timestamp_ns'][:].astype(np.int64)
        times=(ts-ts[0])/1e9;lookup={int(x):i for i,x in enumerate(ids)}
        eligible=[r for r in rows if r.get('event')=='control_sample' and r.get('feature') and r.get('h_ref_s',0)>0 and r['feature']['frame_seq'] in lookup]
        worst=min(eligible,key=lambda r:r['feature']['quality'][0])
        selected=[('start +0.5s',int(np.argmin(abs(times-.5)))),('worst online LEFT during scan',lookup[worst['feature']['frame_seq']]),('end -0.5s',int(np.argmin(abs(times-(times[-1]-.5)))))]
        fig,axes=plt.subplots(3,3,figsize=(21,24),dpi=130)
        fig.suptitle(f'{path.stem}: original grayscale | v3 confidence | top-ROI column confidence\nSame logged feature config; no spatial registration; black fraction is NOT ground truth',fontsize=17,y=.994)
        metrics=[]
        for row,(label,index) in enumerate(selected):
            im=cv2.imdecode(np.asarray(f['ultrasound/jpeg'][index],np.uint8),cv2.IMREAD_GRAYSCALE)
            confidence=random_walk_confidence(im,cfg);q=window_quality(confidence,cfg);detail=confidence_features(im,confidence,cfg)
            frame_id=int(ids[index]);online=byframe.get(frame_id,[])
            onlineq=None if not online else np.asarray(online[0]['feature']['quality'])
            errors=[] if not online else [float(np.max(abs(np.asarray(r['feature']['quality'])-q))) for r in online]
            h,w=im.shape;l0,l1=cfg.lateral_windows[0];dark=float(np.mean(im[:,int(w*l0):int(w*l1)]<20))
            regions=[]
            for lo,hi in cfg.lateral_windows:
                columns=np.asarray(detail['column_confidence'])[int(cfg.width*lo):int(cfg.width*hi)]
                regions.append({'minimum_top_column_confidence':float(columns.min()),'fraction_columns_below_08':float(np.mean(columns<.8))})
            item=dict(selection=label,frame_index=frame_id,saved_index=index,relative_segment_time_s=float(times[index]),timestamp_ns=int(ts[index]),quality_lcr=q.tolist(),full_depth_left_window_dark_fraction_gray_lt20=dark,
                      outermost_left_4percent_dark_fraction=float(np.mean(im[:,:int(w*.04)]<20)),
                      online_quality_lcr=None if onlineq is None else onlineq.tolist(),online_matching_control_samples=len(online),max_sameframe_quality_abs_error=None if not errors else max(errors),
                      windows=regions,feature_window_version=cfg.window_version,feature_algorithm=cfg.algorithm_version,
                      online_control_ids=[r['control_id'] for r in online])
            metrics.append(item)
            a=axes[row,0];a.imshow(im,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
            a.axhline(cfg.near_depth[1]*h,color='cyan',lw=1)
            for k,(lo,hi) in enumerate(cfg.lateral_windows):
                for x in [lo,hi]:a.axvline(x*w,color=['lime','yellow','orange'][k],lw=.8)
                a.text((lo+hi)*w/2,35,'LCR'[k],color=['lime','yellow','orange'][k],ha='center',fontsize=12)
            a.set_title(f'{label}\nt={times[index]:.3f}s, frame_index={frame_id}\nFull-depth left-window gray<20: {dark:.1%}',fontsize=12)
            a.set_xlim(0,w-1);a.set_ylim(h-1,0);a.axis('off')
            a=axes[row,1];a.imshow(confidence,cmap='gray',vmin=0,vmax=1,interpolation='nearest',extent=(0,w,h,0))
            a.axhline(cfg.near_depth[1]*h,color='cyan',lw=1)
            for lo,hi in cfg.lateral_windows:
                a.axvline(lo*w,color='lime',lw=.8);a.axvline(hi*w,color='lime',lw=.8)
            a.set_title(f'v3 confidence fixed display [0,1]\nqL/C/R={q[0]:.4f} / {q[1]:.4f} / {q[2]:.4f}\nTop ROI: rows 0:{detail["top_roi_rows"][1]} / {cfg.height}',fontsize=12)
            a.axis('off')
            a=axes[row,2];x=np.arange(cfg.width)/cfg.width;a.plot(x,detail['column_confidence'],color='navy',lw=1.3);a.axhline(.8,color='red',ls='--',label='0.8 threshold')
            for k,(lo,hi) in enumerate(cfg.lateral_windows):
                a.axvspan(lo,hi,alpha=.13,color=['green','gold','orange'][k]);a.text((lo+hi)/2,1.02,'LCR'[k],ha='center')
            a.set_ylim(0,1.08);a.set_xlim(0,1);a.set_xlabel('Image column / width (no flip)');a.set_ylabel('Mean confidence in top 22% depth');a.grid(alpha=.2);a.legend(loc='center right')
            match='No exact-frame online sample (not substituted)' if onlineq is None else f'Online q={onlineq[0]:.4f}, {onlineq[1]:.4f}, {onlineq[2]:.4f}\nMax same-frame |online-recomputed|={max(errors):.3g}; n={len(online)}'
            a.set_title('Top-ROI columns; shaded L/C/R windows\nWindow means can mask narrower defects',fontsize=12)
            a.text(.02,.04,match,transform=a.transAxes,fontsize=10,bbox=dict(facecolor='white',alpha=.9))
        fig.tight_layout(rect=[0,0,1,.967],h_pad=3,w_pad=2)
        png=OUT/(path.stem+'.png');fig.savefig(png);plt.close(fig)
        report['scans'][path.stem]={'input_h5':str(path),'online_log':str(logpath),'config':config['feature'],'duration_s':float(times[-1]),'selected_frames':metrics,'image':png.name}
        print(path.stem,[(m['selection'],np.round(m['quality_lcr'],4).tolist(),m['max_sameframe_quality_abs_error']) for m in metrics],flush=True)
(OUT/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
