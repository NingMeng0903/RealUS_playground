"""Candidate episodes and synchronized plots for ultrasound_contact_audit.py."""
from __future__ import annotations
import csv
import html
import json
from pathlib import Path
import cv2
import h5py
import numpy as np
from scipy.ndimage import median_filter
from ultrasound_contact_audit import decode, nearest, write_csv


def runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(edges==1), np.flatnonzero(edges==-1)))


def spatial_width(occ, occupancy=.75):
    mask = (occ[:,0] >= occupancy) & (occ[:,1] >= occupancy)
    width = np.zeros(len(mask)); side=[]; spans=[]
    for i,m in enumerate(mask):
        rr=runs(m)
        lo,hi=max(rr,key=lambda r:r[1]-r[0]) if rr else (0,0)
        width[i]=(hi-lo)/m.size
        has_left=any(a<=3 and b-a>=4 for a,b in rr)
        has_right=any(b>=29 and b-a>=4 for a,b in rr)
        side.append("both" if has_left and has_right else "left" if (hi+lo)/2<16 and hi>lo else "right" if hi>lo else "none")
        spans.append((lo,hi))
    return width, np.array(side), np.array(spans)


def episodes(t,mask,min_duration=.2,max_gap=.1):
    mask=mask.copy()
    for a,b in runs(~mask):
        if a>0 and b<len(t) and t[b]-t[a]<=max_gap+1e-6:
            mask[a:b]=True
    step=float(np.median(np.diff(t)))
    keep=np.zeros(len(t),bool); rr=[]
    for a,b in runs(mask):
        if t[b-1]-t[a]+step >= min_duration-1e-6:
            keep[a:b]=True; rr.append((a,b))
    return keep,rr


def stat_range(v):
    return [float(np.percentile(v,q)) for q in (5,50,95)] if len(v) else [float('nan')]*3


def delay_sensitivity(root, events):
    rows=[]
    for e in events:
        d=np.load(Path(root)/e['person']/e['scan']/'signals.npz')
        ft=d['force_time_s']; my=d['wrench'][:,4]
        a=float(e['start_time_s']); b=float(e['end_exclusive_time_s']); vv=[]
        for shift in (-.02,0,.02):
            m=(ft+shift>=a)&(ft+shift<b)
            vv.append(float(np.mean(abs(my[m])<=.025)) if m.any() else np.nan)
        rows.append(dict(person=e['person'],scan=e['scan'],event=e['event'],
            fraction_delay_minus20ms=vv[0],fraction_nominal=vv[1],fraction_delay_plus20ms=vv[2],
            max_fraction_change=max(abs(vv[0]-vv[1]),abs(vv[2]-vv[1]))))
    write_csv(Path(root)/'delay_sensitivity.csv',rows)


def plot_cases(root):
    import matplotlib.pyplot as plt
    root=Path(root)
    cases=[('zhongyao','RH_Per_C_DtP',1.5,'Broad near-field echoes'),
           ('chenwei','RH_Per_L_PtD',3.5,'Dark edge with little rotation'),
           ('chenwei','RH_Per_C_DtP',3.0,'Dark edge during rotation')]
    fig,axs=plt.subplots(4,3,figsize=(14,12),gridspec_kw={'height_ratios':[3,1,1,1]})
    for j,(person,scan,stamp,title) in enumerate(cases):
        dest=root/person/scan; d=np.load(dest/'signals.npz'); meta=json.loads((dest/'metadata.json').read_text())
        t=d['us_time_s'];ft=d['force_time_s'];vt=d['velocity_time_s'];i=int(np.argmin(abs(t-stamp)))
        with h5py.File(meta['source'],'r') as h: im=decode(h['ultrasound/jpeg'][i])
        axs[0,j].imshow(im,cmap='gray',vmin=0,vmax=255);axs[0,j].axis('off')
        axs[0,j].set_title(f'{person}  {scan}\n{title}\n{t[i]:.2f} s',fontsize=10)
        width,_,_=spatial_width(d['dark_occupancy']);_,rr=episodes(t,width>=.125)
        axs[1,j].plot(ft,d['force'],color='#1678a0');axs[1,j].axhline(4,ls='--',color='gray',lw=.8)
        axs[1,j].set_ylabel('Force N');axs[1,j].set_ylim(2.8,5.3)
        axs[2,j].plot(ft,d['wrench'][:,4],color='#963dad');axs[2,j].axhspan(-.025,.025,color='gray',alpha=.2)
        axs[2,j].set_ylabel('My Nm');axs[2,j].set_ylim(-.12,.12)
        axs[3,j].plot(vt,median_filter(d['body_omega'][:,1],size=5)*180/np.pi,color='#269365')
        axs[3,j].set_ylabel('Body y deg/s');axs[3,j].set_ylim(-18,18);axs[3,j].set_xlabel('Time s')
        for ax in axs[1:,j]:
            for a,b in rr: ax.axvspan(t[a],t[b] if b<len(t) else t[-1],color='tomato',alpha=.12)
            ax.axvline(t[i],color='black',ls=':',lw=.8);ax.set_xlim(0,t[-1]);ax.grid(alpha=.2)
    fig.suptitle('Ultrasound, force and actual probe rotation',fontsize=15)
    fig.tight_layout();fig.savefig(root/'case_comparison.png',dpi=140);plt.close(fig)


def report(root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root=Path(root)
    events=[]; scans=[]
    for p in sorted(root.glob("*/*/signals.npz")):
        dest=p.parent; person=dest.parent.name; scan=dest.name
        d=np.load(p); meta=json.loads((dest/"metadata.json").read_text())
        t=d['us_time_s']; ft=d['force_time_s']; vt=d['velocity_time_s']
        w=d['wrench']; fz=d['force']; om=d['body_omega']; vel=d['body_velocity']
        width,side,spans=spatial_width(d['dark_occupancy'])
        mask,rr=episodes(t,width>=.125)
        # Smoothing is only for plots and robust actual-speed summaries, not
        # detection timing. 5 samples ~27 ms at the observed TCP rate.
        wy=median_filter(om[:,1],size=5,mode="nearest")
        angle=np.r_[0,np.cumsum(om[:,1]*np.diff(d['tcp_time_s']))]*180/np.pi
        # This is integrated body-y motion, NOT an absolute Euler angle.
        nvalid=int(d['valid'].sum())
        for number,(a,b) in enumerate(rr,1):
            peak=a+int(np.argmax(width[a:b])); end=float(t[b-1])
            end_exclusive=float(t[b]) if b<len(t) else end+float(np.median(np.diff(t)))
            fm=(ft>=t[a])&(ft<end_exclusive); vm=(vt>=t[a])&(vt<end_exclusive)
            wp=w[fm,4]; sp=wy[vm]
            interp_wy=np.interp(ft[fm],vt,wy)
            signvalid=(np.abs(wp)>.025)&(np.abs(interp_wy)>.01)
            signok=np.mean((wp*interp_wy)[signvalid]<0) if signvalid.any() else np.nan
            e=dict(person=person,scan=scan,event=number,start_frame=a,end_frame=b-1,
                start_time_s=float(t[a]),end_time_s=end,end_exclusive_time_s=end_exclusive,duration_s=end_exclusive-float(t[a]),
                start_timestamp_ns=int(d['us_timestamp_ns'][a]),
                end_timestamp_ns=int(d['us_timestamp_ns'][b-1]),side=side[peak],
                max_dark_width_fraction=float(width[peak]),peak_frame=peak,
                left_censored=a==0,right_censored=b==len(t),
                aligned_frame_fraction=float(d['valid'][a:b].mean()),
                force_samples=int(fm.sum()),velocity_samples=int(vm.sum()),
                force_p05=stat_range(fz[fm])[0],force_median=stat_range(fz[fm])[1],force_p95=stat_range(fz[fm])[2],
                my_p05=stat_range(wp)[0],my_median=stat_range(wp)[1],my_p95=stat_range(wp)[2],
                mx_median=float(np.median(w[fm,3])) if fm.any() else np.nan,
                fy_median=float(np.median(w[fm,1])) if fm.any() else np.nan,
                my_deadzone_fraction=float(np.mean(np.abs(wp)<=.025)) if len(wp) else np.nan,
                omega_y_median_deg_s=float(np.median(sp)*180/np.pi) if len(sp) else np.nan,
                omega_y_abs_median_deg_s=float(np.median(np.abs(sp))*180/np.pi) if len(sp) else np.nan,
                omega_y_speed_near_limit_fraction=float(np.mean(np.abs(sp)>=.9*.28)) if len(sp) else np.nan,
                torque_velocity_opposition_fraction=float(signok),
                torque_velocity_sign_valid_samples=int(signvalid.sum()),
                image_manual_review="unreviewed")
            events.append(e)
            # Before / onset / peak / end / after. No image synthesis or scaling
            # of individual intensities. Colored rectangle marks detected strip.
            ids=np.unique(np.clip([a-9,a,peak,b-1,b+8],0,len(t)-1))
            with h5py.File(meta['source'],'r') as h:
                fig,axs=plt.subplots(1,len(ids),figsize=(3.4*len(ids),4.6),squeeze=False)
                for ax,i in zip(axs.flat,ids):
                    im=decode(h['ultrasound/jpeg'][i]); hh,ww=im.shape
                    ax.imshow(im,cmap='gray',vmin=0,vmax=255)
                    lo,hi=spans[i]
                    if hi>lo:
                        from matplotlib.patches import Rectangle
                        x0=(.04+.92*lo/32)*ww; x1=(.04+.92*hi/32)*ww
                        ax.add_patch(Rectangle((x0,.04*hh),x1-x0,.41*hh,fill=False,edgecolor='tomato',linewidth=1.3))
                    ax.set_title(f"{t[i]:.2f} s   frame {i}"); ax.axis('off')
                fig.suptitle(f"{person}   {scan}   candidate {number}")
                fig.tight_layout(); fig.savefig(dest/f"event_{number:02d}.jpg",dpi=110); plt.close(fig)
        # Per-frame decisions are separate from raw features so thresholds can
        # be rerun without changing original measurements.
        write_csv(dest/'contact_candidates.csv',[dict(frame=i,time_s=float(x),candidate=bool(mask[i]),
            longest_dark_width_fraction=float(width[i]),side=str(side[i]),alignment_valid=bool(d['valid'][i])) for i,x in enumerate(t)])
        sensitivity={}
        for thr in (.65,.75,.85):
            ww,_,_=spatial_width(d['dark_occupancy'],thr)
            mm,_=episodes(t,ww>=.125)
            sensitivity[f"fraction_occupancy_{thr:.2f}"]=float(mm[d['valid']].mean())
        scans.append(dict(person=person,scan=scan,frames=len(t),valid_frames=nvalid,events=len(rr),
            candidate_fraction=float(mask[d['valid']].mean()),dark_width_mean=float(width[d['valid']].mean()),
            **sensitivity))
        fig,ax=plt.subplots(6,1,figsize=(12,13),sharex=True,gridspec_kw={'height_ratios':[1.35,1,1,1,1,1]})
        # Width-depth band occupancy displayed across all 32 aperture strips.
        near=d['dark_occupancy'][:,0,:]
        ax[0].imshow(near.T,extent=(t[0],t[-1],1,0),aspect='auto',cmap='magma',vmin=0,vmax=1)
        ax[0].set_ylabel('Aperture\nleft to right'); ax[0].set_title('Near-field dark occupancy')
        ax[1].plot(t,width*100,color='black',label='Continuous dark width')
        ax[1].axhline(12.5,color='grey',ls='--',lw=.8); ax[1].set_ylabel('Width %')
        ax[2].plot(ft,fz,label='Normal force',color='#1678a0'); ax[2].axhline(4,color='gray',ls='--',lw=.8)
        ax[2].set_ylabel('Force N')
        ax[3].plot(ft,w[:,4],label='My',color='#963dad'); ax[3].axhspan(-.025,.025,color='gray',alpha=.18,label='Coulomb threshold')
        ax[3].plot(ft,w[:,3],lw=.6,alpha=.45,label='Mx',color='#e29522'); ax[3].set_ylabel('Torque Nm')
        ax[4].plot(vt,om[:,1]*180/np.pi,color='gray',alpha=.25,lw=.5)
        ax[4].plot(vt,wy*180/np.pi,label='Actual body y',color='#269365')
        ax[4].axhline(.28*180/np.pi,color='gray',ls='--',lw=.8); ax[4].axhline(-.28*180/np.pi,color='gray',ls='--',lw=.8)
        ax[4].set_ylabel('Angular speed deg/s')
        ax[5].plot(d['tcp_time_s'],angle,label='Integrated body-y rotation',color='#d58318'); ax[5].set_ylabel('Rotation deg')
        ax[5].set_xlabel('Time from first ultrasound frame in seconds')
        for aa in ax[1:]:
            for a,b in rr: aa.axvspan(t[a],t[b] if b<len(t) else t[-1],color='tomato',alpha=.12)
            aa.grid(alpha=.2); aa.legend(loc='upper right',fontsize=8)
        ax[-1].set_xlim(t[0],t[-1]); fig.suptitle(f'{person}   {scan}')
        fig.tight_layout(); fig.savefig(dest/'timeline.png',dpi=130); plt.close(fig)
    write_csv(root/'events.csv',events); write_csv(root/'candidate_summary.csv',scans)
    delay_sensitivity(root,events)
    people=[]
    for name in sorted(set(s['person'] for s in scans)):
        selected=[s for s in scans if s['person']==name]
        n=sum(s['valid_frames'] for s in selected)
        people.append(dict(person=name,scans=len(selected),events=sum(s['events'] for s in selected),
            valid_frames=n,candidate_fraction=sum(s['candidate_fraction']*s['valid_frames'] for s in selected)/n,
            mean_dark_width=sum(s['dark_width_mean']*s['valid_frames'] for s in selected)/n))
    write_csv(root/'person_summary.csv',people)
    fig,ax=plt.subplots(figsize=(9,4))
    ax.bar([p['person'] for p in people],[100*p['candidate_fraction'] for p in people],color='#357b9d')
    ax.set_ylabel('Candidate frames %'); ax.set_title('Persistent near-field dropout candidates'); ax.grid(axis='y',alpha=.2)
    fig.tight_layout(); fig.savefig(root/'overview.png',dpi=150); plt.close(fig)
    params=json.loads((root/'parameters.json').read_text())
    lines=['<!doctype html><meta charset="utf-8"><title>超声接触审查</title>',
        '<style>body{max-width:1200px;margin:32px auto;font:16px sans-serif;line-height:1.6}img{max-width:100%}td,th{padding:8px;border-bottom:1px solid #ddd}details{margin:20px 0}</style>',
        '<h1>超声接触审查</h1><p>红色时段为无需训练的近场暗带候选，不能直接等同于物理脱离接触。时间零点为各原始 H5 第一帧超声。TCP 与力按体模延迟移到图像时钟，使用原始高频数据。</p>',
        f'<p>检测要求同一横向条带在近场和中场均有至少 75% 像素低于灰度 {params["dark_threshold"]}，连续宽度至少为有效孔径 12.5%，持续至少 0.2 秒。最多合并 0.1 秒间断。图像左右尚未与工具坐标标定。</p>',
        '<p>热图亮色表示暗像素比例高。灰色力矩带表示库仑阈值范围，不是实际门控日志。事件起点是本次特征越阈值时刻。</p>',
        '<p><a href="analysis_report.md">完整分析报告与 QP 设计</a> · <a href="ULTRASOUND_CONTACT_AUDIT.md">运行方法</a> · <a href="case_comparison.png">典型状态预览</a></p>',
        '<img src="overview.png"><p><a href="events.csv">事件表</a> · <a href="scan_summary.csv">采样与力矩统计</a> · <a href="candidate_summary.csv">阈值敏感性</a></p>']
    for s in scans:
        rel=f"{s['person']}/{s['scan']}"; title=html.escape(rel)
        lines.append(f'<details><summary>{title} · 候选 {s["candidate_fraction"]:.1%} · {s["events"]} 段</summary><img loading="lazy" src="{rel}/timeline.png"><img loading="lazy" src="{rel}/frames_overview.jpg">')
        for e in [e for e in events if e['person']==s['person'] and e['scan']==s['scan']]:
            lines.append(f'<p>{e["start_time_s"]:.2f}–{e["end_exclusive_time_s"]:.2f} s · My 低于库仑阈值占比 {e["my_deadzone_fraction"]:.1%}</p><img loading="lazy" src="{rel}/event_{e["event"]:02d}.jpg">')
        lines.append('</details>')
    (root/'index.html').write_text('\n'.join(lines),encoding='utf-8')
    if all((root/person/scan/'signals.npz').exists() for person,scan in
           [('zhongyao','RH_Per_C_DtP'),('chenwei','RH_Per_L_PtD'),('chenwei','RH_Per_C_DtP')]):
        plot_cases(root)
    print(f'Report complete: {len(scans)} scans, {len(events)} candidate intervals',flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('output'); report(p.parse_args().output)
