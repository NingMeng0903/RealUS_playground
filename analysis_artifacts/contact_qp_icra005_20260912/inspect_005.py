"""Read-only audit of ICRA uncalibrated/005 ultrasound + logged visual requests."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from peirastic.contact_qp.features import (
    FeatureConfig,
    confidence_features,
    random_walk_confidence,
    window_quality,
)

ROOT = Path("/media/camp/yameng/icra 2027/uncalibrated/005")
OUT = Path(__file__).resolve().parent
C_MIN = 0.8
DEADBAND = 0.03


def quant(x, qs=(0, 0.05, 0.5, 0.95, 1)):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return [None] * len(qs)
    return np.quantile(x, qs).tolist()


def annotate(im, cfg, quality, request_mm_s, title):
    vis = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    h, w = im.shape
    y = int(round(cfg.near_depth[1] * h))
    cv2.line(vis, (0, y), (w - 1, y), (255, 255, 0), 1)
    for lo, hi in cfg.lateral_windows:
        cv2.line(vis, (int(lo * w), 0), (int(lo * w), h - 1), (0, 220, 0), 1)
        cv2.line(vis, (int(hi * w), 0), (int(hi * w), h - 1), (0, 220, 0), 1)
    label = f"{title}  qL/C/R={quality[0]:.2f}/{quality[1]:.2f}/{quality[2]:.2f}  req={request_mm_s:.2f}mm/s"
    cv2.putText(vis, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
    return vis


def main():
    session = json.loads((ROOT / "session.json").read_text())
    epoch = session["clock"]["anchor_time_ns"] - session["clock"]["anchor_monotonic_ns"]
    summaries = []
    OUT.mkdir(parents=True, exist_ok=True)

    for trial in session["trials"]:
        for attempt in trial["attempts"]:
            directory = ROOT / attempt["directory"]
            jsonl = directory / "contact_qp.jsonl"
            raw = directory / "raw.h5"
            if not jsonl.exists() or not raw.exists():
                continue
            cfg = FeatureConfig(**attempt["contact_qp"]["config"]["feature"]["config"])
            stages = attempt.get("stages") or {}
            start = (stages["tracking_observed"] - epoch) * 1e-9 if "tracking_observed" in stages else None
            end = (stages.get("path_done") or stages.get("retract_begin") or 0)
            end = (end - epoch) * 1e-9 if end else None

            samples = []
            pubs = {}
            events = {}
            for line in jsonl.open():
                rec = json.loads(line)
                events[rec["event"]] = events.get(rec["event"], 0) + 1
                if rec["event"] == "control_sample":
                    samples.append(rec)
                elif rec["event"] == "publication" and rec.get("success"):
                    pubs[rec["control_id"]] = rec

            t = np.array([r["record_monotonic_s"] for r in samples], dtype=float)
            if start is None and len(t):
                start = float(t[0])
            if end is None and len(t):
                end = float(t[-1])
            q = np.array([r["feature"]["quality"] if r.get("feature") else [np.nan] * 3 for r in samples], dtype=float)
            request = np.array([r.get("allocation_diagnostics", {}).get("differential_request_m_s", 0.0) for r in samples], dtype=float)
            sign = np.array([r.get("allocation_diagnostics", {}).get("differential_sign", 0.0) for r in samples], dtype=float)
            gate = np.array([r.get("allocation_diagnostics", {}).get("repair_force_gate", np.nan) for r in samples], dtype=float)
            image_ok = np.array([r.get("image_feedback_status") == "ok" for r in samples], dtype=bool)
            feat_none = np.array([r.get("feature") is None for r in samples], dtype=bool)
            nom = np.array([r["nominal_twist_tool"] for r in samples], dtype=float)
            can = np.array([r["candidate_twist_tool"] if r.get("candidate_twist_tool") else [np.nan] * 6 for r in samples], dtype=float)
            both_low = np.all(q[:, [0, 2]] < C_MIN, axis=1)
            side_low = np.any(q[:, [0, 2]] < C_MIN, axis=1)
            imbalance = q[:, 2] - q[:, 0]
            extra_wy = np.degrees(can[:, 4] - nom[:, 4])

            byframe = {}
            for i, rec in enumerate(samples):
                feat = rec.get("feature")
                if feat:
                    byframe.setdefault(int(feat["frame_seq"]), []).append(i)

            with h5py.File(raw, "r") as h:
                ids = np.asarray(h["ultrasound/frame_index"][:], dtype=int)
                ut = np.asarray(h["ultrasound/timestamp_mono_ns"][:], dtype=np.int64) * 1e-9
                force = np.asarray(h["force/contact_force_n"][:], dtype=float)
                ft = np.asarray(h["force/timestamp_mono_ns"][:], dtype=np.int64) * 1e-9
                lookup = {int(fid): i for i, fid in enumerate(ids)}
                if start is None:
                    start = float(ut[0])
                if end is None:
                    end = float(ut[-1])
                scan_mask = (ut >= start) & (ut <= end)
                scan_idx = np.flatnonzero(scan_mask)
                force_mask = (ft >= start) & (ft <= end)

                picks = []
                for label, rel in (("early", 0.08), ("mid", 0.50), ("late", 0.90)):
                    target = start + rel * (end - start)
                    picks.append((label, int(np.argmin(np.abs(ut - target)))))

                dark_scores = []
                for index in scan_idx[:: max(1, len(scan_idx) // 80)] if len(scan_idx) else []:
                    im = cv2.imdecode(np.asarray(h["ultrasound/jpeg"][index], np.uint8), cv2.IMREAD_GRAYSCALE)
                    if im is None:
                        continue
                    hh, ww = im.shape
                    top = im[: max(1, int(0.22 * hh))]
                    deep = im[int(0.22 * hh) :]
                    dark_scores.append((
                        float(np.mean(deep < 20)),
                        float(np.mean(top < 20)),
                        float(top.mean()),
                        int(index),
                    ))
                if dark_scores:
                    darkest = max(dark_scores, key=lambda z: (z[0], z[1]))
                    picks.append(("dark_deep", darkest[3]))

                if samples and np.isfinite(q).any():
                    worst_q_i = int(np.nanargmin(np.min(q[:, [0, 2]], axis=1)))
                    fid = int(samples[worst_q_i]["feature"]["frame_seq"]) if samples[worst_q_i].get("feature") else None
                    if fid in lookup:
                        picks.append(("worst_q", lookup[fid]))
                    max_req_i = int(np.nanargmax(request))
                    if samples[max_req_i].get("feature"):
                        fid = int(samples[max_req_i]["feature"]["frame_seq"])
                        if fid in lookup:
                            picks.append(("max_request", lookup[fid]))

                seen = set()
                unique_picks = []
                for label, index in picks:
                    key = (label, index)
                    if key in seen:
                        continue
                    seen.add(key)
                    unique_picks.append((label, index))

                n = len(unique_picks)
                fig, axes = plt.subplots(n, 3, figsize=(15, 4.2 * n))
                if n == 1:
                    axes = np.array([axes])
                frames = []
                for row, (label, index) in enumerate(unique_picks):
                    im = cv2.imdecode(np.asarray(h["ultrasound/jpeg"][index], np.uint8), cv2.IMREAD_GRAYSCALE)
                    conf = random_walk_confidence(im, cfg)
                    fq = window_quality(conf, cfg)
                    details = confidence_features(im, conf, cfg)
                    fid = int(ids[index])
                    online = byframe.get(fid, [])
                    req = 0.0 if not online else float(1000.0 * request[online[0]])
                    online_q = None if not online else q[online[0]].tolist()
                    hh, ww = im.shape
                    item = dict(
                        label=label,
                        frame_id=fid,
                        scan_time_s=float(ut[index] - start),
                        shape=list(im.shape),
                        recomputed_q=fq.tolist(),
                        online_q=online_q,
                        request_mm_s=req,
                        force_gate=None if not online else float(gate[online[0]]),
                        top22_mean_gray=float(im[: max(1, int(0.22 * hh))].mean()),
                        below22_frac_lt20=float(np.mean(im[int(0.22 * hh) :] < 20)),
                    )
                    frames.append(item)
                    vis = annotate(im, cfg, fq, req, f"{label} t={ut[index]-start:.1f}s")
                    cv2.imwrite(str(OUT / f"{trial['name'].replace('.h5','')}_{label}.jpg"), vis)
                    axes[row, 0].imshow(im, cmap="gray", vmin=0, vmax=255)
                    axes[row, 0].axhline(0.22 * hh, color="cyan")
                    for lo, hi in cfg.lateral_windows:
                        axes[row, 0].axvline(lo * ww, color="lime", lw=0.6)
                        axes[row, 0].axvline(hi * ww, color="lime", lw=0.6)
                    axes[row, 0].set_title(f"{label} t={ut[index]-start:.2f}s #{fid}")
                    axes[row, 1].imshow(conf, cmap="magma", vmin=0, vmax=1)
                    axes[row, 1].axhline(cfg.near_depth[1] * cfg.height, color="cyan")
                    axes[row, 1].set_title(f"map q={fq[0]:.3f}/{fq[1]:.3f}/{fq[2]:.3f} req={req:.2f}")
                    axes[row, 2].plot(np.arange(cfg.width) / cfg.width, details["column_confidence"])
                    axes[row, 2].axhline(C_MIN, color="red", ls="--")
                    axes[row, 2].set_ylim(0, 1.03)
                    axes[row, 2].set_title(f"R-L={fq[2]-fq[0]:+.3f} deadband={DEADBAND}")
                fig.suptitle(f"{trial['name']} / {attempt['directory']}  ({attempt['status']})")
                fig.tight_layout()
                fig.savefig(OUT / f"{trial['name'].replace('.h5','')}_board.png", dpi=120)
                plt.close(fig)

            summary = dict(
                name=trial["name"],
                status=attempt["status"],
                directory=attempt["directory"],
                scan_s=None if start is None or end is None else end - start,
                control_samples=len(samples),
                events=events,
                feature_none=int(feat_none.sum()),
                image_ok_frac=float(image_ok.mean()) if len(samples) else None,
                quality_lcr_p05_p50_p95=np.quantile(q, [0.05, 0.5, 0.95], axis=0).tolist() if len(samples) else None,
                side_below_cmin_frac=float(side_low.mean()) if len(samples) else None,
                both_sides_below_cmin_frac=float(both_low.mean()) if len(samples) else None,
                request_gt0_frac=float(np.mean(request > 0)) if len(samples) else None,
                request_mm_s=quant(1000.0 * request),
                force_gate=quant(gate),
                force_n=quant(force[force_mask]) if force_mask.any() else None,
                extra_wy_deg_s=quant(extra_wy),
                frames=frames,
            )
            summaries.append(summary)
            print(json.dumps({k: summary[k] for k in (
                "name", "status", "scan_s", "control_samples", "feature_none",
                "image_ok_frac", "quality_lcr_p05_p50_p95", "side_below_cmin_frac",
                "both_sides_below_cmin_frac", "request_gt0_frac", "request_mm_s",
                "force_n", "force_gate",
            )}, indent=2), flush=True)

    (OUT / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")


if __name__ == "__main__":
    main()
