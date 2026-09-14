#!/usr/bin/env python3
"""One-row paper figure: B-mode, confidence, near-field profile + 10-region bars."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from peirastic.contact_qp.features import FeatureConfig, confidence_features, load_feature_config, random_walk_confidence

DEFAULT_H5 = Path("/media/camp/PEI_T7/icra 2027_contact/uncalibrated/016/RH_Per_C_PtD.h5")
DEFAULT_FRAME = 376663
DEFAULT_CONFIG = Path("peirastic/config/contact_qp/active_probe50_delay_kf_cop.yaml")
DEFAULT_OUT = Path("MD/contact_qp/welleweerd_v4_check/us_contact_pipeline_016_C_PtD")


def load_frame(path, frame_seq):
    with h5py.File(path, "r") as saved:
        index = np.flatnonzero(saved["ultrasound/frame_index"][:].astype(int) == int(frame_seq))
        if index.size != 1:
            raise ValueError(f"{path}: frame_seq {frame_seq} matched {index.size} rows")
        raw = cv2.imdecode(np.asarray(saved["ultrasound/jpeg"][int(index[0])], np.uint8), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise ValueError(f"{path}: undecodable JPEG for frame_seq {frame_seq}")
    return raw


def draw_geometry(ax, cfg, *, roi_color, line_color):
    y0, y1 = cfg.near_depth
    ax.axhspan(y0, y1, color=roi_color, alpha=0.16, zorder=2)
    ax.add_patch(Rectangle((0.004, y0 + 0.004), 0.992, max(y1 - y0 - 0.008, 0.01),
                           fill=False, lw=1.2, edgecolor=roi_color, zorder=3))
    for edge in cfg.region_edges[1:-1]:
        ax.plot([edge, edge], [0.0, 1.0], ls="--", lw=0.7, color=line_color, alpha=0.9, zorder=3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5)
    parser.add_argument("--frame-seq", type=int, default=DEFAULT_FRAME)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    cfg = load_feature_config(args.config)
    if not isinstance(cfg, FeatureConfig):
        raise TypeError("feature config must load as FeatureConfig")
    raw = load_frame(args.h5, args.frame_seq)
    confidence = random_walk_confidence(raw, cfg)
    feature = confidence_features(raw, confidence, cfg)
    columns = np.asarray(feature["column_confidence"], dtype=float)
    values = np.asarray(feature["region_confidence"], dtype=float)
    edges = np.asarray(feature["region_edges"], dtype=float)
    centers = 0.5 * (edges[:-1] + edges[1:])
    u = (np.arange(columns.size) + 0.5) / columns.size

    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
                         "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.03})
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.55),
                             gridspec_kw={"width_ratios": [1.0, 1.12, 1.18], "wspace": 0.28})

    ticks = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    axes[0].imshow(raw, cmap="gray", vmin=0, vmax=255, extent=(0, 1, 1, 0), aspect="auto")
    draw_geometry(axes[0], cfg, roi_color="cyan", line_color="white")
    axes[0].set_title("(a)  B-mode")
    axes[0].set_xlabel("Lateral coordinate  u")
    axes[0].set_ylabel("Depth  v")

    mapped = axes[1].imshow(confidence, vmin=0, vmax=1, cmap="viridis",
                            extent=(0, 1, 1, 0), aspect="auto")
    draw_geometry(axes[1], cfg, roi_color="white", line_color="white")
    axes[1].set_title("(b)  Confidence map")
    axes[1].set_xlabel("Lateral coordinate  u")
    axes[1].set_ylabel("Depth  v")
    fig.colorbar(mapped, ax=axes[1], fraction=0.046, pad=0.03, ticks=(0, 0.5, 1))

    axes[2].bar(centers, values, width=0.92 * np.diff(edges), color="#4C78A8", alpha=0.38,
                edgecolor="#1B4F72", linewidth=0.8, align="center",
                label="10-region Q25")
    axes[2].plot(u, columns, color="#1B4F72", lw=1.5, label="Near-field profile")
    axes[2].axhline(cfg.low_confidence_threshold, color="#C0392B", ls="--", lw=1.0,
                    label=f"c_min = {cfg.low_confidence_threshold:g}")
    for edge in edges[1:-1]:
        axes[2].axvline(edge, color="0.45", ls="--", lw=0.7)
    axes[2].set_xlim(0, 1)
    axes[2].set_ylim(0, 1.04)
    axes[2].set_title("(c)  Lateral profile and region scores")
    axes[2].set_xlabel("Lateral coordinate  u")
    axes[2].set_ylabel("Confidence")
    axes[2].set_xticks(ticks)
    axes[2].set_xticks(edges, minor=True)
    axes[2].grid(axis="y", alpha=0.25)
    axes[2].legend(loc="upper left", fontsize=6.5, framealpha=0.92)

    for ax in axes[:2]:
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)
        ax.set_xticks(ticks)
        ax.set_xticks(cfg.region_edges, minor=True)
        ax.set_yticks([0.0, 0.22, 0.5, 1.0])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png", "svg"):
        fig.savefig(args.output.with_suffix("." + extension))
    plt.close(fig)
    meta = {
        "h5": str(args.h5),
        "frame_seq": int(args.frame_seq),
        "region_confidence": values.tolist(),
        "region_edges": edges.tolist(),
        "c_min": cfg.low_confidence_threshold,
        "near_depth": list(cfg.near_depth),
        "region_count": int(cfg.region_count),
    }
    args.output.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(args.output.with_suffix(".png"))


if __name__ == "__main__":
    main()
