#!/usr/bin/env python3
"""Single-row time-averaged C_i heatmaps for the four biased comparison runs.

Each panel keeps the recorded B-mode aspect ratio. Frames are kept only while
the interpolated tool-z force is in contact (Fz >= 0.8 N on the tracking path).
Law order matches the comparison figure: UltraPoC, AC2D, AC1D, TAFAC.
"""
from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path("/media/camp/PEI_T7/icra 2027_contact/Comparison Study")
CONFIG = Path(
    "/media/camp/EXT_DRIVE/RealUS_playground/peirastic/config/contact_qp"
    "/active_probe50_delay_kf_cop.yaml"
)
DEFAULT_CACHE = Path("/tmp/biased_confidence_maps.npz")
DEFAULT_OUT = ROOT / "figures" / "biased_confidence"

BIASED_RUNS = (
    ("UltraPoC", "ultrapoc/004noise"),
    ("AC2D", "ac2d/003"),
    ("AC1D", "admittance_1d/001"),
    ("TAFAC", "tafac/001"),
)
FZ_CONTACT = 0.8
BMODE_ASPECT = 726.0 / 664.0
INK = "#272727"

RC = {
    "font.family": ["Nimbus Sans", "Liberation Sans", "DejaVu Sans", "sans-serif"],
    "font.size": 16,
    "axes.labelsize": 16,
    "axes.linewidth": 1.6,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.frameon": False,
    "svg.fonttype": "none",
}

_CFG = None


def _config():
    global _CFG
    if _CFG is None:
        from peirastic.contact_qp.features import load_feature_config

        _CFG = load_feature_config(str(CONFIG))
    return _CFG


def _read_force(run: Path):
    t, fz = [], []
    with (run / "force_trace.csv").open() as handle:
        for row in csv.DictReader(handle):
            if float(row["valid"]) <= 0.5:
                continue
            if row["stage"] != "tracking":
                continue
            t.append(float(row["sample_t_mono_s"]))
            fz.append(float(row["fz_n"]))
    return np.asarray(t, dtype=np.float64), np.asarray(fz, dtype=np.float64)


def _frame_map(blob):
    import cv2

    from peirastic.contact_qp.features import random_walk_confidence

    image = cv2.imdecode(np.asarray(blob, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    return random_walk_confidence(image, _config())


def average_run(run: Path, pool, *, fz_min=FZ_CONTACT):
    import h5py

    from peirastic.contact_qp.mean_confidence_map import contact_us_mask, mean_confidence_map

    force_t, force_fz = _read_force(run)
    with h5py.File(run / "raw.h5", "r") as handle:
        group = handle["ultrasound"]
        us_t = np.asarray(group["timestamp_mono_ns"], dtype=np.float64).ravel() * 1e-9
        keep = contact_us_mask(us_t, force_t, force_fz, fz_min=fz_min)
        blobs = [np.asarray(group["jpeg"][int(i)], dtype=np.uint8) for i in np.flatnonzero(keep)]
    maps = [cmap for cmap in pool.map(_frame_map, blobs, chunksize=8) if cmap is not None]
    if not maps:
        raise RuntimeError(f"no contact-force ultrasound frames in {run}")
    stacked = np.stack(maps, axis=0)
    return dict(
        mean=mean_confidence_map(stacked),
        n_frames=int(len(maps)),
        n_us=int(len(us_t)),
        spatial_mean=float(np.nanmean(stacked)),
    )


def compute(cache: Path, *, workers: int, recompute: bool):
    if cache.is_file() and not recompute:
        packed = np.load(cache, allow_pickle=False)
        meta = json.loads(str(packed["meta"]))
        return {
            entry["law"]: dict(
                mean=packed[entry["tag"]],
                n_frames=int(entry["n_frames"]),
                spatial_mean=float(entry["spatial_mean"]),
                run=entry["run"],
            )
            for entry in meta
        }
    series = {}
    meta = []
    with ProcessPoolExecutor(max_workers=int(workers)) as pool:
        for law, rel in BIASED_RUNS:
            print(f"averaging {law} {rel}", flush=True)
            out = average_run(ROOT / rel, pool)
            series[law] = dict(run=rel, **out)
            tag = law.replace(" ", "")
            meta.append(dict(
                law=law, tag=tag, run=rel,
                n_frames=out["n_frames"], spatial_mean=out["spatial_mean"],
            ))
            print(
                f"  frames={out['n_frames']} / {out['n_us']}  "
                f"<C>={out['spatial_mean']:.3f}",
                flush=True,
            )
    payload = {"meta": np.asarray(json.dumps(meta))}
    for entry in meta:
        payload[entry["tag"]] = series[entry["law"]]["mean"]
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **payload)
    return series


def build(series, out: Path, formats):
    plt.rcParams.update(RC)
    fig = plt.figure(figsize=(7.16, 2.72))
    gs = fig.add_gridspec(
        1, 5, width_ratios=[1.0, 1.0, 1.0, 1.0, 0.07],
        left=0.06, right=0.96, top=0.88, bottom=0.22, wspace=0.22,
    )
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    cax = fig.add_subplot(gs[0, 4])
    cfg = _config()
    mapped = None
    for ax, law in zip(axes, (entry[0] for entry in BIASED_RUNS)):
        cmap = np.asarray(series[law]["mean"], dtype=np.float64)
        h, w = cmap.shape
        y1 = max(1, int(h * float(cfg.near_depth[1])))
        mapped = ax.imshow(
            cmap, cmap="viridis", vmin=0.0, vmax=1.0,
            aspect=BMODE_ASPECT * w / h,
            interpolation="nearest",
        )
        ax.add_patch(Rectangle(
            (-0.5, -0.5), w, y1, fill=False, edgecolor="white", lw=1.1,
        ))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(law, fontsize=16, color=INK, pad=6)
        ax.text(
            0.04, 0.045,
            rf"$\bar{{C}}={float(np.nanmean(cmap[:y1])):.2f}$",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=12,
            color="white",
        )
        for spine in ax.spines.values():
            spine.set_linewidth(1.6)
            spine.set_color(INK)
    axes[0].set_ylabel("$v$", labelpad=2)
    for ax, tag in zip(axes, ("(a)", "(b)", "(c)", "(d)")):
        ax.set_xlabel("$u$\n" + tag, labelpad=1)
    cbar = fig.colorbar(mapped, cax=cax, ticks=(0.0, 0.5, 1.0))
    cbar.set_label(r"$C_i$", fontsize=16)
    cbar.ax.tick_params(labelsize=13)
    cbar.outline.set_linewidth(1.6)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    saved = []
    for suffix in formats:
        target = out.with_suffix(f".{suffix}")
        fig.savefig(target, dpi=300, bbox_inches="tight", pad_inches=0.04)
        saved.append(target)
    plt.close(fig)
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"])
    args = parser.parse_args()
    series = compute(args.cache, workers=args.workers, recompute=args.recompute)
    for path in build(series, args.out, args.formats):
        print("wrote", path)


if __name__ == "__main__":
    main()
