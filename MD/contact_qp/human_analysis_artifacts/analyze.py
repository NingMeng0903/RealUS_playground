"""Reproduce the descriptive human audit report artifacts; never changes H5."""
from pathlib import Path
import csv
import hashlib
import json
import sys

import h5py
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
AUDIT = OUT.parent / "human_audit"
summary = json.loads((AUDIT / "summary.json").read_text())
source = Path(summary["source"])
names = ("left", "center", "right")


def percentiles(a):
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    return {str(p): float(np.percentile(a, p)) for p in (0, 5, 25, 50, 75, 95, 100)}


def corr(a, b):
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    return float(spearmanr(a, b).statistic)


def nearest(t, query):
    hi = np.clip(np.searchsorted(t, query), 0, len(t)-1)
    lo = np.clip(hi-1, 0, len(t)-1)
    return np.where(np.abs(t[lo]-query) <= np.abs(t[hi]-query), lo, hi)


scan_rows, frame_rows = [], []
timing = {g: {"dt_ms": [], "match_ms": [], "counts": 0, "seq_gaps": 0,
              "seq_nonincreasing": 0} for g in ("ultrasound", "tcp", "force")}
raw_force, raw_force_err, same_time_diff, control_force_diff = [], [], [], []
meta = {"hflip": set(), "crop": set(), "shape": set(), "frame_id": set()}
event_candidates = []
per_scan = {}
for scan in summary["scans"]:
    rel = scan["file"]
    csv_path = AUDIT / Path(rel).with_suffix("") / "features.csv"
    rows = list(csv.DictReader(csv_path.open()))
    q = np.array([[float(r[n]) for n in names] for r in rows])
    covered = np.array([r["aligned_covered"] == "True" for r in rows])
    valid = np.array([r["valid"] == "True" for r in rows])
    fz = np.array([float(r["force_n"]) if r["force_n"] else np.nan for r in rows])
    my = np.array([float(r["torque_y_nm"]) if r["torque_y_nm"] else np.nan for r in rows])
    image_t = np.array([float(r["image_time_s"]) for r in rows])
    edge = q[:, (0, 2)].min(axis=1)
    asym = q[:, 2] - q[:, 0]
    low = covered & (np.abs(my) <= .025)
    eligible = covered & valid
    corrs = {"abs_my_vs_edge": corr(np.abs(my[eligible]), edge[eligible]),
             "my_vs_right_minus_left": corr(my[eligible], asym[eligible]),
             "fz_vs_edge": corr(fz[eligible], edge[eligible]),
             "abs_force_error_vs_edge": corr(np.abs(fz[eligible]-4), edge[eligible]),
             "center_vs_edge": corr(q[eligible, 1], edge[eligible])}
    with h5py.File(source / rel, "r") as f:
        ts = {g: np.asarray(f[g+"/timestamp_ns"], dtype=np.int64) for g in timing}
        for g in timing:
            timing[g]["counts"] += len(ts[g])
            timing[g]["dt_ms"].extend(np.diff(ts[g])*1e-6)
            seq = np.asarray(f[g+("/frame_index" if g == "ultrasound" else "/seq")], dtype=np.int64)
            timing[g]["seq_gaps"] += int(np.maximum(np.diff(seq)-1, 0).sum())
            timing[g]["seq_nonincreasing"] += int(np.sum(np.diff(seq) <= 0))
        for g, delay in (("tcp", .15196365053143765), ("force", .1512859033580012)):
            query = ts["ultrasound"]-int(round(delay*1e9))
            ni = nearest(ts[g], query)
            timing[g]["match_ms"].extend(np.abs(ts[g][ni[covered]]-query[covered])*1e-6)
        fw = np.asarray(f["force/wrench_tcp"])
        raw_force.extend(fw[:, 2]); raw_force_err.extend(np.abs(fw[:, 2]-4))
        control_force_diff.extend(np.asarray(f["force/contact_force_n"])-fw[:, 2])
        ni_now = nearest(ts["force"], ts["ultrasound"])
        now_covered = (ts["ultrasound"] >= ts["force"][0]) & (ts["ultrasound"] <= ts["force"][-1])
        same_time_diff.extend(np.abs(fw[ni_now[covered & now_covered], 2]-fz[covered & now_covered]))
        for raw in f["ultrasound/metadata_json"]:
            m = json.loads(raw)
            meta["hflip"].add(m.get("hflip")); meta["crop"].add(tuple(m.get("crop_box", [])))
            meta["shape"].add((m.get("width"), m.get("height"))); meta["frame_id"].add(m.get("frame_id"))
        raw_mae = float(np.mean(np.abs(fw[:, 2]-4)))
    low_q = percentiles(edge[low]) if low.any() else {}
    item = dict(file=rel, rows=len(rows), covered=int(covered.sum()), invalid=int((~valid).sum()),
                uncovered_frames=np.flatnonzero(~covered).tolist(),
                low_torque_rows=int(low.sum()), low_torque_fraction=float(low.sum()/covered.sum()),
                edge_median=float(np.median(edge)), mean_quality=q.mean(axis=0).tolist(),
                low_torque_edge_quantiles=low_q, raw_force_mae_n=raw_mae,
                duration_s=float(image_t[-1]-image_t[0]), correlations=corrs,
                csv_sha256=hashlib.sha256(csv_path.read_bytes()).hexdigest())
    scan_rows.append(item)
    per_scan[rel] = dict(q=q, image_t=image_t, fz=fz, my=my, low=low, edge=edge, covered=covered)
    for i in range(len(rows)):
        frame_rows.append(dict(file=rel, frame=i, image_time_s=float(image_t[i]),
                               q=q[i].tolist(), edge=float(edge[i]), center_minus_edge=float(q[i,1]-edge[i]),
                               center_minus_best_edge=float(q[i,1]-max(q[i,0],q[i,2])),
                               asym=float(asym[i]), fz=float(fz[i]), my=float(my[i]),
                               covered=bool(covered[i]), valid=bool(valid[i]), low=bool(low[i])))

allq = np.array([r["q"] for r in frame_rows]); edge = allq[:, (0,2)].min(axis=1)
mask = np.array([r["covered"] and r["valid"] for r in frame_rows]); low = np.array([r["low"] for r in frame_rows])
my = np.array([r["my"] for r in frame_rows]); fz = np.array([r["fz"] for r in frame_rows])
aggregate = dict(files=len(scan_rows), frames=len(frame_rows), covered=int(mask.sum()),
                 boundary_uncovered=int((~mask).sum()), low_torque_rows=int(low.sum()),
                 low_torque_fraction=float(low.sum()/mask.sum()), invalid=sum(s["invalid"] for s in scan_rows),
                 quality={name: percentiles(allq[:,i]) for i,name in enumerate(names)},
                 edge_quality=percentiles(edge), low_torque_edge=percentiles(edge[low]),
                 above_deadzone_edge=percentiles(edge[mask & ~low]),
                 center_minus_edge=percentiles(allq[:,1]-edge),
                 right_minus_left=percentiles(allq[:,2]-allq[:,0]),
                 center_lower_than_both_fraction=float(np.mean(allq[:,1] < allq[:,(0,2)].min(axis=1))),
                 left_lower_than_right_fraction=float(np.mean(allq[:,0] < allq[:,2])),
                 raw_current_fz_n=percentiles(raw_force), raw_current_abs_force_error_n=percentiles(raw_force_err),
                 aligned_image_fz_n=percentiles(fz[mask]), aligned_image_abs_my_nm=percentiles(np.abs(my[mask])),
                 raw_control_field_minus_wrench_z=percentiles(control_force_diff),
                 image_phase_vs_same_timestamp_force_absolute_difference_n=percentiles(same_time_diff),
                 summed_image_span_s=sum(s["duration_s"] for s in scan_rows),
                 image_span_s=percentiles([s["duration_s"] for s in scan_rows]),
                 scan_low_torque_fraction=percentiles([s["low_torque_fraction"] for s in scan_rows]),
                 scan_low_torque_edge_iqr=percentiles([s["low_torque_edge_quantiles"]["75"]-s["low_torque_edge_quantiles"]["25"] for s in scan_rows]),
                 pooled_correlations={"abs_my_vs_edge":corr(np.abs(my[mask]),edge[mask]),
                                      "my_vs_right_minus_left":corr(my[mask],allq[mask,2]-allq[mask,0]),
                                      "fz_vs_edge":corr(fz[mask],edge[mask]),
                                      "center_vs_edge":corr(allq[mask,1],edge[mask])},
                 within_scan_correlations={n: percentiles([s["correlations"][n] for s in scan_rows if s["correlations"][n] is not None]) for n in scan_rows[0]["correlations"]},
                 timing={g:dict(count=v["counts"], intersample_ms=percentiles(v["dt_ms"]),
                                matching_error_ms=percentiles(v["match_ms"]) if v["match_ms"] else None,
                                missing_sequence_numbers=v["seq_gaps"], nonincreasing_sequences=v["seq_nonincreasing"]) for g,v in timing.items()},
                 metadata={k:sorted(v) for k,v in meta.items()}, processing_ms=summary["processing_ms"],
                 audit_config_sha256=summary["config_sha256"])

# Rank-selected illustrations, not threshold-classified defects or frequencies.
eligible = [r for r in frame_rows if r["covered"] and r["valid"] and r["low"]]
events = [dict(max(eligible, key=lambda r:r["center_minus_edge"]), selection="maximum center-minus-worse-edge within torque deadzone"),
          dict(min(eligible, key=lambda r:r["center_minus_best_edge"]), selection="minimum center-minus-better-edge within torque deadzone"),
          dict(min(eligible, key=lambda r:r["center_minus_edge"]), selection="center lower than both edges, largest difference within torque deadzone")]
spread_scan = max(scan_rows, key=lambda s:s["low_torque_edge_quantiles"]["95"]-s["low_torque_edge_quantiles"]["5"])["file"]
subset = [r for r in eligible if r["file"] == spread_scan]
for rank, label in ((0, "minimum"),(-1,"maximum")):
    events.append(dict(sorted(subset,key=lambda r:r["edge"])[rank],
                       selection=f"{label} worse-edge score in the same scan within torque deadzone"))
aggregate["events"] = events
(OUT/"statistics.json").write_text(json.dumps(aggregate,indent=2,allow_nan=False)+"\n")
(OUT/"scan_statistics.json").write_text(json.dumps(scan_rows,indent=2,allow_nan=False)+"\n")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False})
fig,ax=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
for i,n in enumerate(names):
    v=np.sort(allq[:,i]); ax[0,0].plot(v,np.arange(1,len(v)+1)/len(v),label=n)
ax[0,0].set(xlabel="Random-walk window score",ylabel="Empirical CDF",title="All 12,588 frames; no quality threshold"); ax[0,0].legend()
for m,label in ((low,"|My| <= 0.025 Nm"),(mask & ~low,"|My| > 0.025 Nm")):
    v=np.sort(edge[m]); ax[0,1].plot(v,np.arange(1,len(v)+1)/len(v),label=label)
ax[0,1].set(xlabel="min(left, right) score",ylabel="Empirical CDF",title="Image-phase aligned torque groups"); ax[0,1].legend()
hb=ax[1,0].hexbin(np.abs(my[mask]),edge[mask],gridsize=40,mincnt=1,bins="log",cmap="viridis")
ax[1,0].axvline(.025,color="tab:red",ls="--",lw=1); ax[1,0].set(xlabel="|My| at image phase [Nm]",ylabel="min(left, right)",title="Pooled density, descriptive only")
fig.colorbar(hb,ax=ax[1,0],label="Frame count")
for i,n in enumerate(("abs_my_vs_edge","fz_vs_edge","center_vs_edge")):
    vals=[s["correlations"][n] for s in scan_rows]
    ax[1,1].plot(vals,np.full(len(vals),i)+np.linspace(-.18,.18,len(vals)),".",alpha=.65)
ax[1,1].axvline(0,color="gray",lw=1); ax[1,1].set(xlim=(-1,1),yticks=range(3),yticklabels=("|My| vs worse edge","Fz vs worse edge","center vs worse edge"),xlabel="Within-scan Spearman rho",title="One point per scan (68 scans)")
fig.savefig(OUT/"distributions.png",dpi=170); plt.close(fig)

import cv2
sys.path.insert(0,str(ROOT))
from peirastic.contact_qp.features import FeatureConfig,random_walk_confidence
cfg=FeatureConfig(**{k:(tuple(v) if k=="near_depth" else tuple(tuple(w) for w in v) if k=="lateral_windows" else v) for k,v in summary["config"].items()})
fig,axs=plt.subplots(len(events),3,figsize=(13,3.25*len(events)),constrained_layout=True)
for ri,e in enumerate(events):
    with h5py.File(source/e["file"],"r") as f:
        im=cv2.imdecode(np.asarray(f["ultrasound/jpeg"][e["frame"]],dtype=np.uint8),cv2.IMREAD_GRAYSCALE)
    c=random_walk_confidence(im,cfg)
    axs[ri,0].imshow(im,cmap="gray",vmin=0,vmax=255,aspect="auto")
    axs[ri,0].set_title(f"{e['file']}\nframe {e['frame']}, image t={e['image_time_s']:.3f} s",fontsize=9)
    axs[ri,0].set_axis_off()
    m=axs[ri,1].imshow(c,cmap="magma",vmin=0,vmax=.15,aspect="auto")
    from matplotlib.patches import Rectangle
    for (x0,x1),label in zip(cfg.lateral_windows,names):
        axs[ri,1].add_patch(Rectangle((x0*cfg.width,cfg.near_depth[0]*cfg.height),(x1-x0)*cfg.width,(cfg.near_depth[1]-cfg.near_depth[0])*cfg.height,fill=False,ec="cyan",lw=.8))
    axs[ri,1].set_title("Confidence display clipped at 0.15\nL/C/R="+"/".join(f"{v:.4f}" for v in e["q"]),fontsize=9)
    axs[ri,1].set_axis_off(); fig.colorbar(m,ax=axs[ri,1],fraction=.03)
    s=per_scan[e["file"]]
    for i,n in enumerate(names): axs[ri,2].plot(s["image_t"],s["q"][:,i],label=n)
    axs[ri,2].axvline(e["image_time_s"],color="k",ls="--",lw=.8)
    axs[ri,2].set(xlabel="Image timestamp relative to first frame [s]",ylabel="Window score",title=f"Image-phase Fz={e['fz']:.3f} N, My={e['my']:.4f} Nm")
    axs[ri,2].legend(loc="best",ncol=3,fontsize=8)
fig.savefig(OUT/"representative_frames.png",dpi=160); plt.close(fig)
print(json.dumps(aggregate,indent=2))
