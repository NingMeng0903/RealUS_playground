"""Offline audit: exact feature/frame identity, final commands and measured TCP.

Run with the genesis interpreter (h5py, OpenCV, scipy and matplotlib).
Source files are opened read-only. The physical acoustic delay is not removed.
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/contact_scan_matplotlib')
import cv2
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session',type=Path,default=Path('/media/camp/yameng/icra 2027/uncalibrated/001'))
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    attempt=args.session/'attempts/RH_Per_L_PtD/001'
    samples=[];publications={}
    with (attempt/'contact_qp.jsonl').open() as stream:
        for line in stream:
            row=json.loads(line)
            if row['event']=='control_sample' and row['reference_s']>0:
                samples.append(row)
            elif row['event']=='publication' and row.get('success'):
                publications[row['control_id']]=row
    ref=np.array([s['reference_s'] for s in samples])
    mono=np.array([s['record_monotonic_s'] for s in samples])
    origin=mono[0]
    with_features=[s for s in samples if s['feature']]
    chosen=[min(with_features,key=lambda s:abs(s['reference_s']-t))
            for t in (1.4,6.7,11.9,17.2,22.46,27.7)]
    selections=[]
    with h5py.File(attempt/'raw.h5','r') as h5:
        us=h5['ultrasound'];indices=np.asarray(us['frame_index']).ravel()
        fig,axes=plt.subplots(2,3,figsize=(12,9))
        for ax,row in zip(axes.ravel(),chosen):
            feat=row['feature'];frame_id=feat['frame_seq']
            matches=np.flatnonzero(indices==frame_id)
            if len(matches)!=1:raise ValueError(f'Frame {frame_id}: {len(matches)} matches')
            index=int(matches[0]);image=cv2.imdecode(np.asarray(us['jpeg'][index],dtype=np.uint8),cv2.IMREAD_GRAYSCALE)
            capture=float(np.asarray(us['timestamp_mono_ns'][index]).item())/1e9
            quality=feat['quality'];diag=row['allocation_diagnostics']
            pub=publications.get(row['control_id'])
            final=None if pub is None else pub['final_command_model_tool']
            request=diag['differential_request_m_s'];direction=diag['differential_sign']
            signed_row=diag.get('differential_row')
            realized=None if final is None or signed_row is None else float(np.asarray(signed_row)@final)
            selections.append(dict(reference_s=row['reference_s'],frame_seq=frame_id,
                capture_mono_s=capture,effective_mono_s=feat['effective_time_s'],
                registered_delay_s=capture-feat['effective_time_s'],quality_lcr=quality,
                tank_j=row['energy']['balance_j'],request_m_s=request,direction=direction,
                nominal_differential_m_s=diag.get('differential_nominal_achieved_m_s'),
                final_differential_m_s=realized,
                nominal_omega_y_rad_s=row['nominal_twist_tool'][4],
                final_omega_y_rad_s=None if final is None else final[4]))
            ax.imshow(image,cmap='gray',vmin=0,vmax=255)
            ax.axhline(image.shape[0]*.22,color='cyan',lw=.7)
            ax.set_title(f"ref {row['reference_s']:.1f} s | frame {frame_id}\n"
                         f"cL/C/R {quality[0]:.3f}/{quality[1]:.3f}/{quality[2]:.3f}; E {row['energy']['balance_j']:.3f} J",fontsize=10)
            ax.set_axis_off()
        fig.suptitle('L PtD: exact frame_seq match; cyan = bottom of current confidence ROI',fontsize=12)
        fig.tight_layout();fig.savefig(args.output/'L_PtD_exact_frames.png',dpi=160);plt.close(fig)
        tcp=h5['tcp'];tm=np.asarray(tcp['timestamp_mono_ns']).ravel()/1e9
        pose=np.asarray(tcp['pose_m_rad'])
    mask=(tm>=mono[0])&(tm<=mono[-1]);tm=tm[mask];pose=pose[mask]
    rotations=Rotation.from_euler('xyz',pose[:,3:])
    # Measured orientation displacement, not a commanded or planned pose.
    relative_y=(rotations[0].inv()*rotations).as_rotvec()[:,1]*180/np.pi
    for evidence in selections:
        evidence['measured_relative_tcp_y_deg_at_image_effective_time']=float(
            np.interp(evidence['effective_mono_s'],tm,relative_y))
    quality=np.array([s['feature']['quality'] if s['feature'] else [np.nan]*3 for s in samples])
    request=np.array([s['allocation_diagnostics']['differential_request_m_s'] for s in samples])
    final_y=np.array([publications[s['control_id']]['final_command_model_tool'][4]
        if s['control_id'] in publications else np.nan for s in samples])*180/np.pi
    nominal_y=np.array([s['nominal_twist_tool'][4] for s in samples])*180/np.pi
    fig,axes=plt.subplots(5,1,sharex=True,figsize=(12,11))
    t=mono-origin
    for i,label in enumerate(('left','center','right')):axes[0].plot(t,quality[:,i],label=label,lw=.8)
    axes[0].axhline(.8,color='gray',ls='--');axes[0].set_ylabel('Confidence');axes[0].legend(ncol=3)
    axes[1].plot(t,request*1000,lw=.8);axes[1].set_ylabel('Visual minimum\nmm/s')
    axes[2].plot(t,nominal_y,label='nominal',lw=.8);axes[2].plot(t,final_y,label='published final',lw=.8,alpha=.8)
    axes[2].set_ylabel('Tool y deg/s');axes[2].legend(ncol=2)
    axes[3].plot(tm-origin,relative_y,lw=.9);axes[3].set_ylabel('Measured TCP\nrelative y deg')
    axes[4].plot(t,[s['energy']['balance_j'] for s in samples],lw=.9)
    axes[4].axhline(.05,color='gray',ls='--',label='reserve');axes[4].axhline(.1,color='gray',ls=':',label='initial')
    axes[4].set_ylabel('Tank J');axes[4].legend(ncol=2);axes[4].set_xlabel('Wall seconds since reference began')
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle('L PtD: cue at decision time, minimum request, command and measured pose (distinct quantities)')
    fig.tight_layout();fig.savefig(args.output/'L_PtD_trace.png',dpi=160);plt.close(fig)
    (args.output/'L_PtD_frame_evidence.json').write_text(json.dumps(selections,indent=2)+'\n')
    print(json.dumps(selections,indent=2))


if __name__=='__main__':main()
