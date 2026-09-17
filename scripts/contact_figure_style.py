#!/usr/bin/env python3
"""Shared house style for the contact-adaptation paper figures.

Times-compatible serif with STIX math, drawn at the width it is used at, so
no LaTeX-side rescaling changes the effective font size.
"""
from __future__ import annotations

import numpy as np

COLUMN_IN = 3.5
TEXT_IN = 7.16

INK = "#272727"
MOMENT = "#0072B2"
IMAGE = "#E69F00"
EXECUTED = "#009E73"
THRESHOLD = "#D55E00"
UNMET = "#CC79A7"
DEFICIT = "#E69F00"
NEUTRAL = "#8C8C8C"

RC = {
    "font.family": "serif",
    # Nimbus Roman / Liberation Serif / Tinos are Times New Roman metric clones.
    "font.serif": ["Nimbus Roman", "Times New Roman", "Liberation Serif", "Tinos",
                   "DejaVu Serif", "serif"],
    "mathtext.fontset": "stix",
    "font.size": 9.0,
    "axes.labelsize": 9.5,
    "axes.titlesize": 9.5,
    "axes.linewidth": 0.8,
    "axes.labelcolor": INK,
    "axes.edgecolor": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "text.color": INK,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.major.size": 2.6,
    "ytick.major.size": 2.6,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "lines.solid_capstyle": "round",
    "legend.frameon": False,
    "legend.fontsize": 10.0,
    "legend.handlelength": 1.6,
    "legend.handletextpad": 0.5,
    "legend.columnspacing": 1.1,
    "legend.borderpad": 0.0,
    "legend.borderaxespad": 0.0,
    "svg.fonttype": "none",
    "savefig.dpi": 400,
}


def panel_label(ax, text, dx=0.0, dy=0.0):
    """Panel tag above the axes, left aligned with the y-label column."""
    ax.text(-0.02 + dx, 1.02 + dy, text, transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9.5, color=INK)


def shade_intervals(ax, intervals, xlim, color=DEFICIT, alpha=0.18, zorder=0):
    """Vertical bands, clipped to the panel, for episode or status intervals."""
    for t0, t1 in intervals:
        lo, hi = max(float(t0), xlim[0]), min(float(t1), xlim[1])
        if hi <= lo:
            continue
        # A zero-width band would vanish; give single-cycle events a hairline.
        if hi - lo < 0.004 * (xlim[1] - xlim[0]):
            hi = lo + 0.004 * (xlim[1] - xlim[0])
        ax.axvspan(lo, hi, color=color, alpha=alpha, lw=0, zorder=zorder)


def scan_origin(data, meta):
    """First tracking sample: t=0 on the published figures."""
    if "t_scan_start_s" in meta:
        return float(meta["t_scan_start_s"])
    tracking = data["tracking"] > 0
    return float(data["t"][tracking].min()) if tracking.any() else 0.0


def shift_intervals(intervals, origin):
    return [(float(t0) - origin, float(t1) - origin) for t0, t1 in intervals]


def time_axis(ax, xlim, ticks, label=True, minor=None):
    ax.set_xlim(*xlim)
    ax.set_xticks(ticks)
    if minor is not None:
        ax.set_xticks(minor, minor=True)
    if label:
        ax.set_xlabel("$t$ (s)", labelpad=1.5)


def near_field_rows(cfg, height):
    """Raw-image row span of the near-field ROI used by region_quality."""
    y0 = max(1, int(cfg.height * cfg.near_depth[0]))
    y1 = min(cfg.height, max(y0 + 1, int(cfg.height * cfg.near_depth[1])))
    scale = height / cfg.height
    return int(round(y0 * scale)), int(round(y1 * scale)), y0 / cfg.height, y1 / cfg.height


def bin_edges_u(count, half_width_mm):
    """Image-lateral bin edges in millimetres, left to right."""
    return (2.0 * np.linspace(0.0, 1.0, count + 1) - 1.0) * half_width_mm
