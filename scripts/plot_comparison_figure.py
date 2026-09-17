#!/usr/bin/env python3
"""Two-row comparison figure for the four force laws.

Row 1 (force): tool-z force versus time, including the press-in rise, then one
force-error box panel per law.  Row 2 (contact): confidence features versus the
path coordinate s. Both contact overlays use the biased plan: that is the
condition that separates the laws on C-bar and mu_x.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
from matplotlib import gridspec
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path("/media/camp/PEI_T7/icra 2027_contact/Comparison Study")

LAWS = ("UltraPoC", "AC2D", "AC1D", "TAFAC")
# Okabe–Ito: blue / orange / vermillion / green stay separable when overlaid.
LAW_COLOR = {
    "UltraPoC": "#0072B2",
    "AC2D": "#E69F00",
    "AC1D": "#D55E00",
    "TAFAC": "#009E73",
}
CONDITIONS = ("clean", "noisy")
CONDITION_COLOR = {"clean": "#56B4E9", "noisy": "#CC79A7"}
CONDITION_LABEL = {"clean": "unbiased", "noisy": "biased"}
INK = "#272727"

RC = {
    "font.family": "serif",
    # Nimbus Roman / Liberation Serif / Tinos are Times New Roman metric clones.
    "font.serif": ["Nimbus Roman", "Times New Roman", "Liberation Serif", "Tinos",
                   "DejaVu Serif", "serif"],
    "mathtext.fontset": "stix",
    "font.size": 33,
    "axes.labelsize": 35,
    "axes.linewidth": 2.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 30,
    "ytick.labelsize": 30,
    "xtick.major.size": 7,
    "ytick.major.size": 7,
    "xtick.major.width": 2.2,
    "ytick.major.width": 2.2,
    "legend.frameon": False,
    "svg.fonttype": "none",
}
TAG_SIZE = 33
LEGEND_SIZE = 30

FORCE_YLIM = (0.0, 6.0)
FORCE_XLIM = (0.0, 50.0)
FORCE_STRIDE = 2
FORCE_PREROLL_S = 0.5
ERROR_YLIM = (-0.85, 0.85)
CBAR_YLIM = (0.2, 1.03)
MU_YLIM = (-5.5, 5.5)
# geometry.half_length_m of the probe50 calibration: the stored centroid is
# normalised to the half aperture, so this puts mu_x back into millimetres.
HALF_APERTURE_MM = 25.0


def smooth(y, window):
    """Centred moving average that keeps the original sample count."""
    y = np.asarray(y, dtype=float)
    if window < 2 or y.size < window:
        return y
    kernel = np.ones(window)
    finite = np.isfinite(y)
    total = np.convolve(np.where(finite, y, 0.0), kernel, mode="same")
    count = np.convolve(finite.astype(float), kernel, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / count, np.nan)


def bin_quantiles(s, y, n_bins, quantiles):
    """Per-path-coordinate-bin quantiles, so the ripple envelope survives decimation."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    index = np.clip(np.searchsorted(edges, s, side="right") - 1, 0, n_bins - 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    out = np.full((len(quantiles), n_bins), np.nan)
    for b in range(n_bins):
        values = y[index == b]
        values = values[np.isfinite(values)]
        if values.size:
            out[:, b] = np.quantile(values, quantiles)
    keep = np.isfinite(out).all(axis=0)
    return centers[keep], out[:, keep]


def _force_time_series(run: str):
    """Clock time aligned at the press-in from Fz ≈ 0, with a short pre-roll."""
    path = ROOT / run / "force_trace.csv"
    t, fz = [], []
    with path.open() as handle:
        for row in csv.DictReader(handle):
            if float(row["valid"]) <= 0.5:
                continue
            t.append(float(row["sample_t_mono_s"]))
            fz.append(float(row["fz_n"]))
    t = np.asarray(t, dtype=float)
    fz = np.asarray(fz, dtype=float)
    reached = np.flatnonzero(fz >= 3.5)
    if not reached.size:
        t0 = t[0]
    else:
        prefix = fz[: int(reached[0]) + 1]
        below = np.flatnonzero(prefix < 0.4)
        t0 = t[int(below[-1])] if below.size else t[int(reached[0])]
    clock = t - t0 + FORCE_PREROLL_S
    keep = (clock >= 0.0) & (clock <= FORCE_XLIM[1])
    return clock[keep][::FORCE_STRIDE], fz[keep][::FORCE_STRIDE]


def load(path: Path):
    blob = np.load(path, allow_pickle=False)
    t_star = float(blob["t_star"])
    series = {}
    for entry in json.loads(str(blob["meta"])):
        tag = entry["tag"]
        law = _display_law(entry["law"])
        s_force = blob[f"{tag}/t_force"] / t_star
        s_us = blob[f"{tag}/t_us"] / t_star
        keep_f = s_force <= 1.0
        keep_u = s_us <= 1.0
        t_clock, fz_clock = _force_time_series(entry["run"])
        series[(law, entry["condition"])] = dict(
            run=entry["run"],
            t_clock=t_clock,
            fz_clock=fz_clock,
            e_force=blob[f"{tag}/fz"][keep_f] - blob[f"{tag}/fd"][keep_f],
            s_us=s_us[keep_u],
            cbar=blob[f"{tag}/cbar"][keep_u],
            mu_x=blob[f"{tag}/mu_x"][keep_u] * HALF_APERTURE_MM,
        )
    return t_star, series


def _display_law(name):
    return {"Admittance 1D": "AC1D"}.get(name, name)


def tag_panel(ax, text):
    ax.text(0.5, 1.04, text, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=TAG_SIZE, color=INK)


def path_axis(ax):
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "", "0.5", "", "1"])
    ax.set_xlabel("$s$", labelpad=1)


def time_axis(ax):
    ax.set_xlim(*FORCE_XLIM)
    ax.set_xticks([0.0, 25.0, 50.0])
    ax.set_xticklabels(["", "25", "50"])
    ax.set_xlabel("t (s)", labelpad=2)


def force_overlay(ax, series, condition, bands=True):
    """Raw tool-z force versus time.  UltraPoC is drawn last."""
    del bands
    weight = {
        "TAFAC": (0.55, 0.45),
        "AC1D": (0.55, 0.50),
        "AC2D": (0.60, 0.55),
        "UltraPoC": (1.10, 0.95),
    }
    for law in reversed(LAWS):
        width, alpha = weight[law]
        entry = series[(law, condition)]
        ax.plot(
            entry["t_clock"],
            entry["fz_clock"],
            color=LAW_COLOR[law],
            linewidth=width,
            alpha=alpha,
            solid_capstyle="butt",
            zorder=10 - LAWS.index(law),
        )


def contact_overlay(ax, series, condition, key, window):
    # Thickest first, thinnest last: UltraPoC rides the top of the C-bar panel
    # almost everywhere, so later curves must stay visible where they coincide.
    widths = (3.4, 2.5, 1.8, 1.2)
    for law, width in zip(LAWS, widths):
        entry = series[(law, condition)]
        ax.plot(entry["s_us"], smooth(entry[key], window), color=LAW_COLOR[law],
                linewidth=width, alpha=0.9, solid_capstyle="round",
                zorder=2 + LAWS.index(law))


def box_panel(ax, series, law, key):
    data = [series[(law, condition)][key] for condition in CONDITIONS]
    data = [values[np.isfinite(values)] for values in data]
    artists = ax.boxplot(data, positions=[0.32, 0.68], widths=0.12, whis=(5, 95),
                         showfliers=False, patch_artist=True,
                         medianprops=dict(color=INK, linewidth=2.2))
    for patch, condition in zip(artists["boxes"], CONDITIONS):
        color = CONDITION_COLOR[condition]
        patch.set(facecolor=color, alpha=0.5, edgecolor=color, linewidth=2.0)
    for part in ("whiskers", "caps"):
        for line, condition in zip(artists[part], np.repeat(CONDITIONS, 2)):
            line.set(color=CONDITION_COLOR[condition], linewidth=2.0)
    ax.set_xlim(0.0, 1.0)


def condition_pair(ax, series, law, key, window):
    # Noisy first and thicker so the flat noise-free reference stays visible
    # on top where the two conditions coincide near C-bar = 1.
    width = {"noisy": 3.6, "clean": 2.4}
    for condition in reversed(CONDITIONS):
        entry = series[(law, condition)]
        ax.plot(entry["s_us"], smooth(entry[key], window),
                color=CONDITION_COLOR[condition], linewidth=width[condition],
                solid_capstyle="round")


def build(series, out: Path, formats, force_bands=True):
    plt.rcParams.update(RC)
    fig = plt.figure(figsize=(31.5, 8.8))
    row_ratios = [1.0, 0.66, 1.0, 0.38, 0.20]
    left = gridspec.GridSpec(
        5, 2,
        width_ratios=[1.0, 1.0],
        height_ratios=row_ratios,
        left=0.042,
        right=0.356,
        top=0.925,
        bottom=0.030,
        wspace=0.40,
        hspace=0.05,
    )
    right = gridspec.GridSpec(
        5, 4,
        width_ratios=[1.0, 1.0, 1.0, 1.0],
        height_ratios=row_ratios,
        left=0.404,
        right=0.997,
        top=0.925,
        bottom=0.030,
        wspace=0.18,
        hspace=0.05,
    )
    axes = [
        [fig.add_subplot(left[0, 0]), fig.add_subplot(left[0, 1])]
        + [fig.add_subplot(right[0, i]) for i in range(4)],
        [fig.add_subplot(left[2, 0]), fig.add_subplot(left[2, 1])]
        + [fig.add_subplot(right[2, i]) for i in range(4)],
    ]
    law_legend_ax = fig.add_subplot(left[4, :])
    condition_legend_ax = fig.add_subplot(right[4, :])
    law_legend_ax.set_axis_off()
    condition_legend_ax.set_axis_off()

    for column, condition in enumerate(CONDITIONS):
        ax = axes[0][column]
        ax.axhline(4.0, color=INK, linewidth=1.5, linestyle="--", alpha=0.4, zorder=1)
        force_overlay(ax, series, condition, bands=force_bands)
        ax.set_ylim(*FORCE_YLIM)
        ax.set_yticks([0.0, 2.0, 4.0, 6.0])
        ax.set_yticklabels(["0", "2", "4", "6"])
        time_axis(ax)
        tag_panel(ax, CONDITION_LABEL[condition])
        ax.set_ylabel("$F_z$  (N)", labelpad=1)

    for column, law in enumerate(LAWS, start=2):
        ax = axes[0][column]
        ax.axhline(0.0, color=INK, linewidth=1.5, linestyle="--", alpha=0.4, zorder=1)
        box_panel(ax, series, law, "e_force")
        ax.set_ylim(*ERROR_YLIM)
        ax.set_yticks([-0.8, -0.4, 0.0, 0.4, 0.8])
        ax.set_xticks([])
        ax.set_xlabel("")
        tag_panel(ax, law)
        if column > 2:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel("$e_F$  (N)", labelpad=6)

    ax = axes[1][0]
    contact_overlay(ax, series, "noisy", "cbar", window=3)
    ax.set_ylim(*CBAR_YLIM)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    path_axis(ax)
    ax.set_ylabel(r"$\bar{C}$", labelpad=1)
    tag_panel(ax, CONDITION_LABEL["noisy"])

    ax = axes[1][1]
    ax.axhline(0.0, color=INK, linewidth=1.5, linestyle="--", alpha=0.4, zorder=1)
    contact_overlay(ax, series, "noisy", "mu_x", window=3)
    ax.set_ylim(*MU_YLIM)
    ax.set_yticks([-5, 0, 5])
    path_axis(ax)
    ax.set_ylabel(r"$\mu_x$ (mm)", labelpad=2)
    tag_panel(ax, CONDITION_LABEL["noisy"])

    for column, law in enumerate(LAWS, start=2):
        ax = axes[1][column]
        condition_pair(ax, series, law, "cbar", window=3)
        ax.set_ylim(*CBAR_YLIM)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        path_axis(ax)
        tag_panel(ax, law)
        if column > 2:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel(r"$\bar{C}$", labelpad=1)

    law_handles = [Line2D([0], [0], color=LAW_COLOR[law], linewidth=3.2, label=law)
                   for law in LAWS]
    condition_handles = [
        Patch(facecolor=CONDITION_COLOR[c], edgecolor=CONDITION_COLOR[c], alpha=0.6,
              label=CONDITION_LABEL[c])
        for c in CONDITIONS
    ]
    law_legend_ax.legend(
        handles=law_handles,
        loc="center",
        ncols=4,
        fontsize=LEGEND_SIZE,
        handlelength=2.2,
        columnspacing=1.2,
        handletextpad=0.5,
    )
    condition_legend_ax.legend(
        handles=condition_handles,
        loc="center left",
        ncols=2,
        fontsize=LEGEND_SIZE,
        handlelength=1.4,
        columnspacing=1.8,
        handletextpad=0.65,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    saved = []
    for suffix in formats:
        target = out.with_suffix(f".{suffix}")
        fig.savefig(target, dpi=300)
        saved.append(target)
    plt.close(fig)
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("/tmp/comparison_figure_data.npz"))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/media/camp/PEI_T7/icra 2027_contact/Comparison Study/figures/comparison"),
    )
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"])
    parser.add_argument("--no-force-band", action="store_true")
    args = parser.parse_args()

    t_star, series = load(args.data)
    for path in build(series, args.out, args.formats, not args.no_force_band):
        print("wrote", path)
    print(f"path coordinate s normalised by T* = {t_star:.2f} s")


if __name__ == "__main__":
    main()
