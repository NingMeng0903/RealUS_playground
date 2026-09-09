#!/usr/bin/env python3
"""Read-only scan audit. Outputs are review candidates, not contact ground truth.

Uses JPEG intensities without trained weights, raw high-rate wrench and SE(3)
pose increments. All plots use the ultrasound receipt clock after applying the
phantom's effective delays. No controller API is imported or called.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import h5py
import numpy as np
from scipy.spatial.transform import Rotation


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def decode(blob):
    a = cv2.imdecode(np.asarray(blob, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if a is None:
        raise ValueError("JPEG decode failed")
    return a


def nearest(t, q):
    hi = np.clip(np.searchsorted(t, q), 0, len(t)-1)
    lo = np.maximum(hi-1, 0)
    return np.where(np.abs(q-t[lo]) <= np.abs(q-t[hi]), lo, hi)


def bands(im, threshold=40):
    """Direct dark-pixel occupancy in fixed, documented depth/lateral windows.

    Top 4% and side 4% excluded to avoid fixed interface/border artifacts.
    Each output column is a 1/32-width strip. Near: 4-22%, mid: 22-45%,
    deep: 45-80% of image depth. A dark deep band alone is NOT contact loss.
    """
    h, w = im.shape
    a = cv2.resize(im[:, int(.04*w):int(.96*w)], (192, 224), interpolation=cv2.INTER_AREA)
    near, mid, deep = [a[int(y0*224):int(y1*224)] for y0,y1 in ((.04,.22),(.22,.45),(.45,.80))]
    occ = []
    for patch in (near, mid, deep):
        strips = patch.reshape(patch.shape[0],32,6)
        occ.append(np.mean(strips < threshold, axis=(0,2)))
    # Each zone contains ~31% of active aperture. Keep the center separately.
    return np.stack(occ)


def run(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    summary = []
    for path in sorted(Path(args.input).glob("*/*.h5")):
        person = path.parent.name
        dest = out/person/path.stem
        dest.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "r") as f:
            if "alignment" in f or "alignment_schema" in f.attrs:
                raise ValueError(f"{path}: use raw uncalibrated H5 to avoid applying delays twice")
            usns = f["ultrasound/timestamp_ns"][:].astype(np.int64)
            origin = int(usns[0])
            ut = (usns-origin)*1e-9
            tn = f["tcp/timestamp_ns"][:].astype(np.int64)
            fn = f["force/timestamp_ns"][:].astype(np.int64)
            for label, stamps in (("ultrasound",usns),("TCP",tn),("force",fn)):
                if len(stamps)<2 or np.any(np.diff(stamps)<=0):
                    raise ValueError(f"{path}: {label} timestamps must strictly increase")
            tt = (tn-origin)*1e-9 + args.us_delay
            ft = (fn-origin)*1e-9 + args.us_delay - args.force_delay
            pose = f["tcp/pose_m_rad"][:]
            wrench = f["force/wrench_tcp"][:]
            force = f["force/contact_force_n"][:]
            # pose convention verified against recorder: fixed-axis xyz Euler.
            rot = Rotation.from_euler("xyz", pose[:,3:])
            dt = np.diff(tt)
            omega = (rot[:-1].inv()*rot[1:]).as_rotvec()/dt[:,None]
            vel = rot[:-1].inv().apply(np.diff(pose[:,:3],axis=0))/dt[:,None]
            vt = (tt[1:]+tt[:-1])/2
            valid = (ut >= max(tt[0],ft[0],vt[0])) & (ut <= min(tt[-1],ft[-1],vt[-1]))
            fi, ti, vi = nearest(ft,ut), nearest(tt,ut), nearest(vt,ut)
            valid &= (np.abs(ft[fi]-ut)<=1/60) & (np.abs(tt[ti]-ut)<=1/60)
            features = []
            sample_ids = np.linspace(0,len(ut)-1,6).round().astype(int)
            samples = {}
            for i,blob in enumerate(f["ultrasound/jpeg"]):
                im = decode(blob)
                features.append(bands(im, args.dark_threshold))
                if i in sample_ids:
                    samples[i] = im
            occ = np.stack(features)
            meta = json.loads(f.attrs["scan_metadata_json"])
            stages = meta.get("stages",{})
            # Stage stamps are on the TCP host clock. Put them on image clock.
            stage_t = {k:(int(v)-origin)*1e-9+args.us_delay for k,v in stages.items() if isinstance(v,(int,float))}
            raw = dict(us_timestamp_ns=usns, us_time_s=ut, tcp_time_s=tt, force_time_s=ft,
                       pose=pose,wrench=wrench,force=force,velocity_time_s=vt,
                       body_omega=omega,body_velocity=vel,dark_occupancy=occ,
                       valid=valid,force_index=fi,tcp_index=ti,velocity_index=vi)
            np.savez_compressed(dest/"signals.npz",**raw)
            (dest/"metadata.json").write_text(json.dumps(dict(source=str(path),
                origin_ns=origin, stages_on_image_clock_s=stage_t,
                force_overrides=meta.get("force_overrides"),
                us_delay_s=args.us_delay,force_delay_s=args.force_delay),indent=2))
            zones = np.stack([occ[:,:,s].mean(axis=2) for s in (slice(0,10),slice(11,21),slice(22,32))],axis=2)
            rows=[]
            for i,t in enumerate(ut):
                row=dict(frame=i,timestamp_ns=int(usns[i]),time_s=float(t),alignment_valid=bool(valid[i]))
                for b,bname in enumerate(("near","mid","deep")):
                    for z,zname in enumerate(("left","center","right")):
                        row[bname+"_dark_"+zname]=float(zones[i,b,z])
                for j,name in enumerate(("fx_n","fy_n","fz_n","mx_nm","my_nm","mz_nm")):
                    row[name]=float(wrench[fi[i],j]) if valid[i] else float("nan")
                row["omega_y_rad_s"]=float(omega[vi[i],1]) if valid[i] else float("nan")
                rows.append(row)
            write_csv(dest/"frames.csv",rows)
            fig, axs = plt.subplots(2,3,figsize=(12,9))
            for ax,i in zip(axs.flat,sample_ids):
                ax.imshow(samples[i],cmap="gray",vmin=0,vmax=255)
                ax.set_title(f"{ut[i]:.2f} s   frame {i}")
                ax.axis("off")
            fig.suptitle(f"{person}   {path.stem}")
            fig.tight_layout()
            fig.savefig(dest/"frames_overview.jpg",dpi=110)
            plt.close(fig)
            summary.append(dict(person=person,scan=path.stem,frames=len(ut),duration_s=float(ut[-1]),
                image_hz=(len(ut)-1)/ut[-1],tcp_hz=(len(tt)-1)/(tt[-1]-tt[0]),force_hz=(len(ft)-1)/(ft[-1]-ft[0]),
                near_dark_left=float(zones[:,0,0].mean()),near_dark_center=float(zones[:,0,1].mean()),near_dark_right=float(zones[:,0,2].mean()),
                my_median=float(np.median(wrench[:,4])),my_deadzone_fraction=float(np.mean(np.abs(wrench[:,4])<=.025)),
                fz_median=float(np.median(force)),fz_min=float(force.min()),fz_max=float(force.max()),
                max_abs_my=float(np.abs(wrench[:,4]).max())))
        print(person,path.stem,len(ut),flush=True)
    write_csv(out/"scan_summary.csv",summary)
    (out/"parameters.json").write_text(json.dumps(vars(args),indent=2))


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--us-delay",type=float,default=.15196365053143765)
    p.add_argument("--force-delay",type=float,default=.0006777471734364515)
    p.add_argument("--dark-threshold",type=int,default=40)
    p.add_argument("--features-only",action="store_true",help="Skip candidate plots")
    args=p.parse_args()
    run(args)
    if not args.features_only:
        from ultrasound_contact_report import report
        report(args.output)
