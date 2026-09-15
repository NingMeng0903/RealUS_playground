#!/usr/bin/env python3
"""Paper figure USCON: B-mode, C_j(u,v), near-field profile. Axes in millimetres."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from peirastic.contact_qp.features import load_feature_config, random_walk_confidence, region_quality, unknown_scanlines

DEFAULT_H5 = Path("/media/camp/PEI_T7/icra 2027_contact/calibrated/chenwei/RH_Per_C_PtD.h5")
DEFAULT_FRAME = 8255
DEFAULT_CONFIG = Path("peirastic/config/contact_qp/active_probe50_delay_kf_cop.yaml")
DEFAULT_OUT = Path("MD/contact_qp/welleweerd_v4_check/USCON")

RED = "#d62728"
BLUE = "#4c72b0"
INK = "#1f2d3d"
# 50 mm square field: u from the aperture centre, v from the probe face into tissue.
HALF_W_MM = 25.0
DEPTH_MM = 50.0
OUTLINE = [pe.Stroke(linewidth=1.7, foreground="black", alpha=0.75), pe.Normal()]


def load_frame(path, frame_seq):
    with h5py.File(path, "r") as saved:
        index = np.flatnonzero(saved["ultrasound/frame_index"][:].astype(int) == int(frame_seq))
        if index.size != 1:
            raise ValueError(f"{path}: frame_seq {frame_seq} matched {index.size} rows")
        raw = cv2.imdecode(np.asarray(saved["ultrasound/jpeg"][int(index[0])], np.uint8), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise ValueError(f"{path}: undecodable JPEG for frame_seq {frame_seq}")
    return raw


def frac_to_u(frac):
    return (2.0 * np.asarray(frac, dtype=float) - 1.0) * HALF_W_MM


def frac_to_v(frac):
    return np.asarray(frac, dtype=float) * DEPTH_MM


def draw_near_field(ax, u_edges, v_top, v_bot):
    ax.add_patch(Rectangle((u_edges[0], v_top), u_edges[-1] - u_edges[0], v_bot - v_top,
                           fill=False, edgecolor="white", lw=0.8, zorder=4, path_effects=OUTLINE))
    for edge in u_edges[1:-1]:
        ax.plot([edge, edge], [v_top, v_bot], color="white", ls=(0, (2.2, 1.8)), lw=0.7,
                zorder=4, path_effects=OUTLINE)


def style_image_axes(ax):
    ax.set_xlim(-HALF_W_MM, HALF_W_MM)
    ax.set_ylim(DEPTH_MM, 0.0)
    ax.set_xticks((-25, 0, 25))
    ax.set_yticks((0, 25, 50))
    ax.set_xlabel("$u$ (mm)", labelpad=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5)
    parser.add_argument("--frame-seq", type=int, default=DEFAULT_FRAME)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    cfg = load_feature_config(args.config)
    raw = load_frame(args.h5, args.frame_seq)
    confidence = random_walk_confidence(raw, cfg)
    region = region_quality(confidence, cfg, unknown_scanlines(raw, cfg))
    row0, row1 = region["region_roi_rows"]
    profile = confidence[row0:row1].mean(axis=0)
    values = np.asarray(region["region_confidence"], dtype=float)
    edges = np.asarray(region["region_edges"], dtype=float)
    threshold = float(cfg.low_confidence_threshold)
    low = values < threshold
    u_edges = frac_to_u(edges)
    v_top, v_bot = frac_to_v(row0 / cfg.height), frac_to_v(row1 / cfg.height)
    u = frac_to_u((np.arange(cfg.width) + 0.5) / cfg.width)
    extent = (-HALF_W_MM, HALF_W_MM, DEPTH_MM, 0.0)

    plt.rcParams.update({
        "font.family": "serif", "mathtext.fontset": "cm",
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "savefig.dpi": 300,
    })
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.35), layout="constrained",
                             gridspec_kw={"width_ratios": [1.05, 1.18, 1.22]})
    fig.get_layout_engine().set(wspace=0.10)

    ax = axes[0]
    ax.imshow(raw, cmap="gray", vmin=0, vmax=255, extent=extent, aspect="auto",
              interpolation="bilinear")
    for flagged, left, right in zip(low, u_edges[:-1], u_edges[1:]):
        if flagged:
            ax.add_patch(Rectangle((left, v_top), right - left, v_bot - v_top,
                                   facecolor="#ff6b6b", edgecolor="none", alpha=0.42, zorder=2))
    draw_near_field(ax, u_edges, v_top, v_bot)
    style_image_axes(ax)
    ax.set_ylabel("$v$ (mm)")
    ax.set_title("(a) B-mode frame $j$")

    ax = axes[1]
    mapped = ax.imshow(confidence, cmap="viridis", vmin=0, vmax=1, extent=extent,
                       aspect="auto", interpolation="nearest")
    draw_near_field(ax, u_edges, v_top, v_bot)
    style_image_axes(ax)
    ax.set_ylabel("$v$ (mm)")
    ax.set_title("(b) $C_j(u,v)$")
    bar = fig.colorbar(mapped, ax=ax, fraction=0.046, pad=0.04, ticks=(0, 0.5, 1))
    bar.outline.set_linewidth(0.6)
    bar.ax.tick_params(labelsize=7, width=0.6, length=2.5)

    ax = axes[2]
    centers = 0.5 * (u_edges[:-1] + u_edges[1:])
    ax.bar(centers, values, width=np.diff(u_edges), color=np.where(low, RED, BLUE), alpha=0.4,
           edgecolor="white", linewidth=0.8, zorder=1, align="center")
    ax.plot(u, profile, color=INK, lw=1.2, zorder=3)
    ax.axhline(threshold, color=RED, ls="--", lw=0.9, zorder=2)
    ax.text(-HALF_W_MM + 1.0, threshold + 0.03, r"$c_{\mathrm{th}}$", color=RED,
            ha="left", va="bottom", zorder=5)
    ax.set_xlim(-HALF_W_MM, HALF_W_MM)
    ax.set_ylim(0, 1.05)
    ax.set_xticks((-25, 0, 25))
    ax.set_yticks((0, 0.5, 1))
    ax.set_xlabel("$u$ (mm)", labelpad=1)
    ax.set_ylabel("confidence")
    ax.set_title(r"(c) $\bar{C}_j(u)$ and $\mathbf{c}_j$")

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png", "svg"):
        fig.savefig(args.output.with_suffix("." + extension))
    plt.close(fig)
    args.output.with_suffix(".json").write_text(json.dumps({
        "h5": str(args.h5), "frame_seq": int(args.frame_seq),
        "region_confidence": values.tolist(), "u_edges_mm": u_edges.tolist(),
        "below_threshold": low.tolist(), "c_th": threshold,
        "u_range_mm": [-HALF_W_MM, HALF_W_MM], "v_range_mm": [0.0, DEPTH_MM],
        "roi_v_mm": [float(v_top), float(v_bot)],
        "region_count": int(cfg.region_count),
    }, indent=2) + "\n")
    print(args.output.with_suffix(".png"))


if __name__ == "__main__":
    main()
