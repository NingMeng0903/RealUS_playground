#!/usr/bin/env python3
"""Contact correction and CoP pairing in the representative scan (004noise).

Three panels, one claim each: (a) the image supplies the rocking direction while
the moment channel is silent, and hands back once contact is loaded; (b) every
rocking increment is executed as a rotation about an eccentric centre of
pressure; (c) that paired normal motion does not disturb force regulation.
Time is the scan clock: t=0 is the first tracking cycle.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
from matplotlib import gridspec
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from contact_adaptation_data import DEFAULT_OUT as DEFAULT_DATA
from contact_adaptation_data import load
from contact_figure_style import (
    EXECUTED,
    IMAGE,
    INK,
    MOMENT,
    NEUTRAL,
    RC,
    TEXT_IN,
    UNMET,
    scan_origin,
    shade_intervals,
    shift_intervals,
    time_axis,
)

ROOT = Path("/media/camp/PEI_T7/icra 2027_contact/Comparison Study")
DEFAULT_OUT = ROOT / "figures" / "contact_correction"

XLIM = (0.0, 3.5)
XTICKS = (0, 1, 2, 3)
ROCK_MIN = 1e-3
SETTLED_AFTER_S = 6.0
PAIR_CUTS = (0.001, 0.35)
# Matched to plot_coupling_progression.py so the two figures read as a pair.
FS_LABEL = 14.5
FS_TICK = 12.5
FS_TEXT = 13.0


def build(data, meta, out: Path, formats):
    plt.rcParams.update(RC)
    plt.rcParams.update({
        "font.size": 13.0,
        "axes.labelsize": FS_LABEL,
        "axes.titlesize": FS_TEXT,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_TEXT,
    })
    tracking = data["tracking"] > 0
    origin = scan_origin(data, meta)
    t_raw = data["t"]
    t = t_raw - origin
    unmet = shift_intervals(meta["unmet_intervals"], origin)
    request_peak = dict(meta["frames"]["request_peak"])
    request_peak["t"] = float(request_peak["t"]) - origin

    fig = plt.figure(figsize=(TEXT_IN, 2.85))
    # Equal widths and one shared height: the three plot boxes have the same shape.
    grid = gridspec.GridSpec(1, 3, left=0.086, right=0.988, top=0.948, bottom=0.400,
                             wspace=0.38, width_ratios=(1.0, 1.0, 1.0))
    ax_omega = fig.add_subplot(grid[0, 0])
    ax_pair = fig.add_subplot(grid[0, 1])
    ax_force = fig.add_subplot(grid[0, 2])

    # (a) who asks and who moves, before and after the load is established.
    keep = tracking & (t >= XLIM[0] - 0.05) & (t <= XLIM[1] + 0.05)
    shade_intervals(ax_omega, unmet, XLIM, color=UNMET, alpha=0.20)
    ax_omega.axhline(0.0, color=INK, lw=0.6, alpha=0.4, zorder=1)
    ax_omega.plot(t[keep], data["omega_us"][keep], color=IMAGE, lw=1.5, zorder=4)
    ax_omega.plot(t[keep], data["omega_n"][keep], color=EXECUTED, lw=1.5, zorder=5)
    ax_omega.plot(t[keep], data["omega_m"][keep], color=MOMENT, lw=1.5, zorder=6)
    ax_omega.set_ylim(-0.058, 0.216)
    ax_omega.set_yticks((0.0, 0.05, 0.10, 0.15))
    ax_omega.set_yticklabels(("0", "0.05", "0.10", "0.15"))
    ax_omega.set_ylabel(r"$\omega$ (rad/s)", labelpad=2.0, fontsize=FS_LABEL)
    time_axis(ax_omega, XLIM, XTICKS)
    ax_omega.xaxis.label.set_size(FS_LABEL)
    ax_omega.tick_params(labelsize=FS_TICK)
    ax_omega.annotate(rf"$\omega_{{\mathrm{{us}}}}\!=\!{request_peak['omega_us']:.2f}$",
                      xy=(request_peak["t"], request_peak["omega_us"] + 0.004),
                      xytext=(0.86, 0.198), ha="center", va="center", fontsize=FS_TEXT, color=INK,
                      arrowprops=dict(arrowstyle="-", lw=0.7, color=INK, alpha=0.5,
                                      shrinkA=2.0, shrinkB=1.0))
    ax_omega.annotate(r"$\omega_m\!\approx\!0$", xy=(0.52, -0.003), xytext=(0.52, -0.038),
                      ha="center", va="center", fontsize=FS_TEXT, color=INK,
                      arrowprops=dict(arrowstyle="-", lw=0.7, color=INK, alpha=0.5,
                                      shrinkA=2.0, shrinkB=1.0))
    ax_omega.annotate("moment\ntakes over", xy=(2.05, 0.100), xytext=(2.62, 0.160),
                      ha="center", va="center", fontsize=FS_TEXT, color=INK, linespacing=1.2,
                      arrowprops=dict(arrowstyle="-", lw=0.7, color=INK, alpha=0.5,
                                      shrinkA=2.0, shrinkB=2.0))

    # (b) the increment is a rotation about a measurably eccentric CoP.
    rocking = tracking & (np.abs(data["d_omega"]) > ROCK_MIN) & np.isfinite(data["d_v"])
    scale = 1e3
    limit = 1.10 * scale * float(np.nanmax(np.abs(data["c_p_d_omega"][rocking])))
    ax_pair.axline((0.0, 0.0), slope=1.0, color=INK, lw=0.9, ls=(0, (3.0, 1.8)), zorder=2)
    ax_pair.scatter(scale * data["c_p_d_omega"][rocking], scale * data["d_v"][rocking],
                    s=5.0, c=EXECUTED, alpha=0.35, linewidths=0.0, zorder=3)
    # Equal limits, not equal aspect: the identity line is then the box diagonal,
    # so this panel keeps the same height as (a) and (c).
    ax_pair.set_xlim(-limit, limit)
    ax_pair.set_ylim(-limit, limit)
    ax_pair.set_xticks((-2, 0, 2))
    ax_pair.set_yticks((-2, 0, 2))
    ax_pair.set_xlabel(r"$c_p\,\Delta\omega_n$ (mm/s)", labelpad=2.0, fontsize=FS_LABEL)
    ax_pair.set_ylabel(r"$\Delta v_n$ (mm/s)", labelpad=2.0, fontsize=FS_LABEL)
    ax_pair.tick_params(length=2.4, pad=1.8, labelsize=FS_TICK)
    inset = ax_pair.inset_axes((0.045, 0.700, 0.56, 0.30))
    c_p_mm = data["c_p"][tracking] * scale
    inset.hist(c_p_mm[np.isfinite(c_p_mm)], bins=np.linspace(-25.0, 25.0, 51),
               color=NEUTRAL, alpha=0.8, linewidth=0.0)
    inset.axvline(0.0, color=INK, lw=0.7, alpha=0.45, zorder=3)
    inset.set_xlim(-25.0, 25.0)
    inset.set_xticks((-25, 0, 25))
    inset.set_yticks([])
    inset.tick_params(length=1.8, pad=1.0, labelsize=9.5)
    inset.set_xlabel(r"$c_p$ (mm)", labelpad=0.5, fontsize=10.5)
    for side in ("top", "right", "left"):
        inset.spines[side].set_visible(False)
    inset.spines["bottom"].set_linewidth(0.6)

    # (c) the paired normal motion is not paid for in force accuracy.
    error = np.abs(data["F"] - data["Fd"])
    after = tracking & (t_raw > SETTLED_AFTER_S) & np.isfinite(error)
    pair_mm = np.abs(data["c_p_d_omega"]) * scale
    groups = (
        after & (pair_mm <= PAIR_CUTS[0]),
        after & (pair_mm > PAIR_CUTS[0]) & (pair_mm <= PAIR_CUTS[1]),
        after & (pair_mm > PAIR_CUTS[1]),
    )
    samples = [error[mask] for mask in groups]
    parts = ax_force.violinplot(samples, positions=(1, 2, 3), widths=0.74,
                                showextrema=False, showmedians=True)
    for body in parts["bodies"]:
        body.set_facecolor(EXECUTED)
        body.set_alpha(0.40)
        body.set_linewidth(0.0)
    parts["cmedians"].set_color(INK)
    parts["cmedians"].set_linewidth(1.2)
    for position, sample in zip((1, 2, 3), samples):
        ax_force.text(position, 0.80, f"{np.sqrt(np.nanmean(sample ** 2)):.2f}",
                      ha="center", va="bottom", fontsize=FS_TEXT, color=INK)
    ax_force.set_xlim(0.45, 3.55)
    ax_force.set_xticks((1, 2, 3))
    ax_force.set_xticklabels((r"$\approx\!0$", r"$\leq\!0.35$", r"$>\!0.35$"))
    ax_force.set_ylim(0.0, 0.95)
    ax_force.set_yticks((0.0, 0.3, 0.6))
    ax_force.set_ylabel(r"$|F-F_d|$ (N)", labelpad=2.0, fontsize=FS_LABEL)
    ax_force.set_xlabel(r"$|c_p\Delta\omega_n|$ (mm/s)", labelpad=2.0, fontsize=FS_LABEL)
    ax_force.tick_params(length=2.4, pad=1.8, labelsize=FS_TICK)

    y_caption = 0.195
    for ax, text in (
        (ax_omega, r"(a)  Angular channels"),
        (ax_pair, r"(b)  CoP pairing"),
        (ax_force, r"(c)  Force error"),
    ):
        box = ax.get_position()
        fig.text(0.5 * (box.x0 + box.x1), y_caption, text,
                 ha="center", va="top", fontsize=FS_LABEL, color=INK)

    handles = [
        Line2D([0], [0], color=IMAGE, lw=2.0, label=r"image $\omega_{\mathrm{us}}$"),
        Line2D([0], [0], color=MOMENT, lw=2.0, label=r"moment $\omega_m$"),
        Line2D([0], [0], color=EXECUTED, lw=2.0, label=r"command $\omega_n^{*}$"),
        Patch(facecolor=UNMET, edgecolor="none", alpha=0.40,
              label=r"$\omega_{\mathrm{us}}>\omega_m$"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.50, 0.015), ncols=4,
               fontsize=FS_TEXT, handlelength=1.7, columnspacing=2.4, handletextpad=0.5,
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
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"])
    args = parser.parse_args()
    data, meta = load(args.data)
    for path in build(data, meta, args.out, args.formats):
        print("wrote", path)


if __name__ == "__main__":
    main()
