#!/usr/bin/env python3
"""Cache force and contact series for the four-law comparison figure.

Force comes from the 100 Hz force_trace.csv; contact comes from the recorded
ultrasound frames re-run through the shipped Welleweerd confidence pipeline.
Both are expressed on the plan phase s = t_ref / T*, where T* is the longest
path prefix that every selected run actually executed.
"""
from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np

ROOT = Path("/media/camp/PEI_T7/icra 2027_contact/Comparison Study")
CONFIG = Path(
    "/media/camp/EXT_DRIVE/RealUS_playground/peirastic/config/contact_qp"
    "/active_probe50_delay_kf_cop.yaml"
)

# (law label, condition, run directory).  Selected best runs per README.
RUNS = (
    ("UltraPoC", "clean", "ultrapoc/002"),
    ("UltraPoC", "noisy", "ultrapoc/004noise"),
    ("AC2D", "clean", "ac2d/001"),
    ("AC2D", "noisy", "ac2d/003"),
    ("AC1D", "clean", "admittance_1d/001non"),
    ("AC1D", "noisy", "admittance_1d/001"),
    ("TAFAC", "clean", "tafac/001non"),
    ("TAFAC", "noisy", "tafac/001"),
)

_CFG = None


def _config():
    global _CFG
    if _CFG is None:
        from peirastic.contact_qp.features import load_feature_config

        _CFG = load_feature_config(str(CONFIG))
    return _CFG


def _frame_features(payload):
    """Confidence ROI mean and lateral centroid for one JPEG frame."""
    import cv2

    from peirastic.contact_qp.features import confidence_features, random_walk_confidence

    index, blob = payload
    image = cv2.imdecode(np.asarray(blob, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return index, float("nan"), float("nan")
    cfg = _config()
    feat = confidence_features(image, random_walk_confidence(image, cfg), cfg)
    centroid = feat["confidence_centroid_x"]
    if centroid is None or not feat["confidence_centroid_valid"]:
        centroid = float("nan")
    return index, float(feat["roi_mean"]), cfg.image_x_sign * float(centroid)


def _read_csv(path: Path, columns):
    rows = {name: [] for name in columns}
    with path.open() as handle:
        for row in csv.DictReader(handle):
            for name in columns:
                rows[name].append(row[name])
    return rows


def _path_clock(run: Path):
    """Monotonic-time to plan-phase-time map over the tracking stage."""
    raw = _read_csv(run / "motion_trace.csv", ("t_mono_s", "stage", "t_ref_s"))
    stage = np.asarray(raw["stage"])
    mono = np.asarray(raw["t_mono_s"], dtype=float)
    tref = np.asarray(raw["t_ref_s"], dtype=float)
    keep = (stage == "tracking") & np.isfinite(mono) & np.isfinite(tref)
    mono, tref = mono[keep], tref[keep]
    order = np.argsort(mono)
    return mono[order], tref[order]


def _force_series(run: Path, mono_ref, tref_ref):
    raw = _read_csv(
        run / "force_trace.csv",
        ("sample_t_mono_s", "stage", "fz_n", "desired_fz_n", "valid"),
    )
    stage = np.asarray(raw["stage"])
    mono = np.asarray(raw["sample_t_mono_s"], dtype=float)
    fz = np.asarray(raw["fz_n"], dtype=float)
    fd = np.asarray(raw["desired_fz_n"], dtype=float)
    valid = np.asarray(raw["valid"], dtype=float) > 0.5
    keep = (stage == "tracking") & valid & np.isfinite(mono) & np.isfinite(fz)
    mono, fz, fd = mono[keep], fz[keep], fd[keep]
    tref = np.interp(mono, mono_ref, tref_ref, left=np.nan, right=np.nan)
    keep = np.isfinite(tref)
    order = np.argsort(tref[keep])
    return tref[keep][order], fz[keep][order], fd[keep][order]


def _contact_series(run: Path, mono_ref, tref_ref, pool):
    with h5py.File(run / "raw.h5", "r") as handle:
        group = handle["ultrasound"]
        blobs = [np.asarray(group["jpeg"][i], dtype=np.uint8) for i in range(len(group["jpeg"]))]
        mono = np.asarray(group["timestamp_mono_ns"], dtype=float).ravel() * 1e-9
    # The controller consumes each frame one image-pipeline delay later.
    effective = mono - _config().effective_delay_s
    tref = np.interp(effective, mono_ref, tref_ref, left=np.nan, right=np.nan)
    wanted = np.flatnonzero(np.isfinite(tref))
    cbar = np.full(len(blobs), np.nan)
    mu_x = np.full(len(blobs), np.nan)
    jobs = [(int(i), blobs[i]) for i in wanted]
    for index, roi_mean, centroid in pool.map(_frame_features, jobs, chunksize=8):
        cbar[index] = roi_mean
        mu_x[index] = centroid
    keep = np.isfinite(tref) & np.isfinite(cbar)
    order = np.argsort(tref[keep])
    return tref[keep][order], cbar[keep][order], mu_x[keep][order]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("/tmp/comparison_figure_data.npz"))
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    series = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for law, condition, rel in RUNS:
            run = ROOT / rel
            mono_ref, tref_ref = _path_clock(run)
            t_force, fz, fd = _force_series(run, mono_ref, tref_ref)
            t_us, cbar, mu_x = _contact_series(run, mono_ref, tref_ref, pool)
            key = f"{law}|{condition}"
            series[key] = dict(
                law=law,
                condition=condition,
                run=rel,
                t_ref_max=float(tref_ref.max()),
                t_force=t_force,
                fz=fz,
                fd=fd,
                t_us=t_us,
                cbar=cbar,
                mu_x=mu_x,
            )
            print(
                f"{key:24s} {rel:22s} T={tref_ref.max():6.2f}s "
                f"force={len(fz):5d} frames={len(cbar):5d} "
                f"Cbar={np.nanmean(cbar):.3f}",
                flush=True,
            )

    # Common executed prefix: UltraPoC aborts a few percent early, so every
    # curve is compared over the same stretch of path rather than its own end.
    t_star = min(entry["t_ref_max"] for entry in series.values())
    payload = {"t_star": np.asarray(t_star)}
    meta = []
    for key, entry in series.items():
        tag = key.replace("|", "_").replace(" ", "")
        for name in ("t_force", "fz", "fd", "t_us", "cbar", "mu_x"):
            payload[f"{tag}/{name}"] = entry[name]
        meta.append(
            dict(
                key=key,
                tag=tag,
                law=entry["law"],
                condition=entry["condition"],
                run=entry["run"],
                t_ref_max=entry["t_ref_max"],
            )
        )
    payload["meta"] = np.asarray(json.dumps(meta))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **payload)
    print(f"wrote {args.out}  t_star={t_star:.2f} s")


if __name__ == "__main__":
    main()
