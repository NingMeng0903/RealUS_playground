#!/usr/bin/env python3
"""Coupling-aware progression in the representative UltraPoC scan (004noise).

Two panels, one claim each: (a) the establishment episode at cycle resolution,
where the speed factor collapses with the worst region and returns to nominal as
soon as the window is restored; (b) the near-field evidence at three instants of
that episode. Time is the scan clock: t=0 is the first tracking cycle.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
from matplotlib import gridspec
from matplotlib import patheffects as pe
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from contact_adaptation_data import DEFAULT_OUT as DEFAULT_DATA
from contact_adaptation_data import load
from contact_figure_style import (
    DEFICIT,
    EXECUTED,
    INK,
    MOMENT,
    RC,
    TEXT_IN,
    THRESHOLD,
    bin_edges_u,
    near_field_rows,
    scan_origin,
    shade_intervals,
    shift_intervals,
    time_axis,
)
from render_us_contact_pipeline import HALF_W_MM, load_frame

from peirastic.contact_qp.features import load_feature_config

ROOT = Path("/media/camp/PEI_T7/icra 2027_contact/Comparison Study")
DEFAULT_H5 = ROOT / "ultrapoc" / "004noise" / "raw.h5"
DEFAULT_CONFIG = REPO / "peirastic/config/contact_qp/active_probe50_delay_kf_cop.yaml"
DEFAULT_OUT = ROOT / "figures" / "coupling_progression"

# Scan clock: t=0 is the first tracking cycle.
EPISODE_XLIM = (0.0, 2.5)
EPISODE_XTICKS = (0, 0.5, 1.0, 1.5, 2.0, 2.5)
STRIP_KEYS = ("deficit_peak", "request_peak", "recovered")
# Offset of the on-plot badge from its point (dt, dc, ha).
BADGE_OFFSETS = ((0.10, -0.07, "left"), (-0.11, -0.05, "right"), (0.12, 0.05, "left"))
BADGE = [pe.Stroke(linewidth=1.9, foreground="black", alpha=0.75), pe.Normal()]
FS_LABEL = 14.5
FS_TICK = 12.5
FS_TEXT = 13.0
FS_BADGE = 11.5


def build(data, meta, h5: Path, config: Path, out: Path, formats):
    plt.rcParams.update(RC)
    plt.rcParams.update({
        "font.size": 13.0,
        "axes.labelsize": FS_LABEL,
        "axes.titlesize": FS_TEXT,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_TEXT,
    })
    cfg = load_feature_config(config)
    c_th = float(meta["c_th"])
    tracking = data["tracking"] > 0
    origin = scan_origin(data, meta)
    t = data["t"] - origin
    deficit = shift_intervals(meta["deficit_intervals"], origin)
    frames = {key: dict(value, t=float(value["t"]) - origin) for key, value in meta["frames"].items()}

    fig = plt.figure(figsize=(TEXT_IN, 3.40))
    grid = gridspec.GridSpec(
        1, 2, left=0.090, right=0.966, top=0.942, bottom=0.372,
        wspace=0.44, width_ratios=(1.24, 1.0),
    )
    ax_c = fig.add_subplot(grid[0, 0])
    ax_a = ax_c.twinx()
    strip_gs = grid[0, 1].subgridspec(3, 1, hspace=0.66)
    ax_strips = [fig.add_subplot(strip_gs[row, 0]) for row in range(3)]

    keep = tracking & (t >= EPISODE_XLIM[0] - 0.02) & (t <= EPISODE_XLIM[1] + 0.05)
    shade_intervals(ax_c, deficit, EPISODE_XLIM)
    ax_c.axhline(c_th, color=INK, lw=0.7, ls=(0, (2.6, 1.6)), alpha=0.55, zorder=2)
    ax_c.plot(t[keep], data["min_c"][keep], color=MOMENT, lw=1.2, zorder=4)
    ax_a.plot(t[keep], data["alpha"][keep], color=EXECUTED, lw=1.5, zorder=5)

    for ax in (ax_c, ax_a):
        ax.set_ylim(0.14, 1.08)
        ax.set_yticks((0.2, 0.4, 0.6, 0.8, 1.0))
        ax.set_yticklabels(("0.2", "0.4", "0.6", "0.8", "1"))
    ax_c.set_ylabel(r"confidence $c_i$", color=MOMENT, labelpad=3.0, fontsize=FS_LABEL)
    ax_a.set_ylabel(r"scan speed $\alpha^{*}$", color=EXECUTED, labelpad=3.0, fontsize=FS_LABEL)
    ax_c.tick_params(axis="y", colors=MOMENT, labelsize=FS_TICK)
    ax_a.tick_params(axis="y", colors=EXECUTED, labelsize=FS_TICK)
    ax_c.spines["left"].set_color(MOMENT)
    ax_a.spines["right"].set_color(EXECUTED)
    ax_a.spines["right"].set_visible(True)
    ax_a.spines["top"].set_visible(False)
    time_axis(ax_c, EPISODE_XLIM, EPISODE_XTICKS)
    ax_c.tick_params(axis="x", labelsize=FS_TICK)
    ax_c.xaxis.label.set_size(FS_LABEL)
    ax_c.text(EPISODE_XLIM[1] - 0.04, c_th + 0.03, r"$c_{\mathrm{th}}$", ha="right",
              va="bottom", fontsize=FS_TEXT, color=INK, alpha=0.8)

    for number, (key, offset) in enumerate(zip(STRIP_KEYS, BADGE_OFFSETS), start=1):
        frame = frames[key]
        ax_c.plot([frame["t"]], [frame["min_c"]], marker="o", ms=4.4, mfc="white",
                  mec=MOMENT, mew=1.2, zorder=8)
        ax_c.text(frame["t"] + offset[0], frame["min_c"] + offset[1], str(number),
                  ha=offset[2], va="center", fontsize=FS_BADGE, color=INK, zorder=9,
                  bbox=dict(boxstyle="circle,pad=0.16", fc="white", ec=INK, lw=0.75))

    alpha_min = float(np.nanmin(data["alpha"][keep]))
    t_alpha_min = float(t[keep][int(np.nanargmin(data["alpha"][keep]))])
    ax_a.annotate(rf"$\alpha^{{*}}\!=\!{alpha_min:.2f}$",
                  xy=(t_alpha_min, alpha_min - 0.015), xytext=(1.58, 0.30),
                  ha="center", va="center", fontsize=FS_TEXT, color=INK,
                  arrowprops=dict(arrowstyle="-", lw=0.7, color=INK, alpha=0.5,
                                  shrinkA=2.0, shrinkB=2.0))
    ax_a.annotate("window restored,\n" r"$\alpha^{*}\!\to\!1$", xy=(1.70, 0.985),
                  xytext=(1.70, 0.58), ha="center", va="center", fontsize=FS_TEXT, color=INK,
                  linespacing=1.25,
                  arrowprops=dict(arrowstyle="-", lw=0.7, color=INK, alpha=0.5,
                                  shrinkA=2.0, shrinkB=1.0))

    edges_u = bin_edges_u(10, HALF_W_MM)
    for number, (ax, key) in enumerate(zip(ax_strips, STRIP_KEYS), start=1):
        frame = frames[key]
        raw = load_frame(h5, frame["frame_seq"])
        row0, row1, frac0, frac1 = near_field_rows(cfg, raw.shape[0])
        depth0, depth1 = 50.0 * frac0, 50.0 * frac1
        ax.imshow(raw[row0:row1], cmap="gray", vmin=0, vmax=255, aspect="auto",
                  extent=(-HALF_W_MM, HALF_W_MM, depth1, depth0), interpolation="bilinear")
        index = int(np.argmin(np.abs(data["t"] - (frame["t"] + origin))))
        values = data["region_confidence"][index]
        ribbon = depth1 - 0.10 * (depth1 - depth0)
        for value, left, right in zip(values, edges_u[:-1], edges_u[1:]):
            if np.isfinite(value) and value < c_th:
                ax.add_patch(Rectangle((left + 0.25, ribbon), right - left - 0.5, depth1 - ribbon,
                                       facecolor=THRESHOLD, edgecolor="none", zorder=3))
        for edge in edges_u[1:-1]:
            ax.plot((edge, edge), (depth0, depth1), color="white", lw=0.4, alpha=0.45, zorder=2)
        ax.set_xlim(-HALF_W_MM, HALF_W_MM)
        ax.set_ylim(depth1, depth0)
        ax.set_yticks((4, 8))
        ax.set_yticklabels(("4", "8"))
        ax.set_xticks((-25, 0, 25))
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.6)
        ax.tick_params(length=2.2, pad=1.6, labelsize=FS_TICK)
        ax.text(-HALF_W_MM + 1.4, depth0 + 0.26 * (depth1 - depth0), str(number),
                ha="left", va="top", fontsize=FS_BADGE, color="white", zorder=5, path_effects=BADGE)
        ax.set_title(rf"$t\!=\!{frame['t']:.2f}\,$s", fontsize=FS_TEXT, pad=2.0, color=INK)
        if number != len(STRIP_KEYS):
            ax.set_xticklabels([])
    ax_strips[-1].set_xlabel("$u$ (mm)", labelpad=1.8, fontsize=FS_LABEL)
    ax_strips[1].set_ylabel("$v$ (mm)", labelpad=1.8, fontsize=FS_LABEL)

    box_a = ax_c.get_position()
    box_b = ax_strips[-1].get_position()
    y_caption = 0.165
    fig.text(0.5 * (box_a.x0 + box_a.x1), y_caption,
             r"(a)  Confidence and scan speed",
             ha="center", va="top", fontsize=FS_LABEL, color=INK)
    fig.text(0.5 * (box_b.x0 + box_b.x1), y_caption,
             r"(b)  Near-field B-mode",
             ha="center", va="top", fontsize=FS_LABEL, color=INK)

    handles = [
        Line2D([0], [0], color=MOMENT, lw=1.8, label=r"worst-region confidence $c_i$"),
        Line2D([0], [0], color=EXECUTED, lw=2.0, label=r"scan-speed factor $\alpha^{*}$"),
        Patch(facecolor=DEFICIT, edgecolor="none", alpha=0.35,
              label="contact-degraded interval"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.50, 0.015), ncols=3,
               fontsize=FS_TEXT, handlelength=1.7, columnspacing=1.8, handletextpad=0.5,
               borderaxespad=0.0)

    out.parent.mkdir(parents=True, exist_ok=True)
    saved = []
    for suffix in formats:
        target = out.with_suffix(f".{suffix}")
        fig.savefig(target)
        saved.append(target)
    plt.close(fig)
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"])
    args = parser.parse_args()
    data, meta = load(args.data)
    for path in build(data, meta, args.h5, args.config, args.out, args.formats):
        print("wrote", path)


if __name__ == "__main__":
    main()
