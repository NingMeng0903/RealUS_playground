#!/usr/bin/env python3
"""Three-controller display with a disclosed late-bias adjustment of UltraPoC.

This is a derived visualization, not a replacement for the measured force or
its 4 N tracking-error statistics. AC1D and TAFAC retain the preceding figure's
exact filtered curves and original error samples. Only UltraPoC's estimated
2.5--5 s mean offset is removed, using a smooth 2.0--2.5 s onset. All source
files and original derived columns are preserved.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
from scipy.stats import gaussian_kde

import plot_proposal_force as source

OUT = Path(__file__).resolve().parents[1] / "outputs/proposal_force_20261015_bias_adjusted"
COLORS = {"UltraPoC": "#0055E8", "AC1D": "#B000C8", "TAFAC": "#008600"}
LABELS = {"UltraPoC": "UltraPoC* (late-bias adjusted)", "AC1D": source.DISPLAY_NAMES["AC1D"],
          "TAFAC": source.DISPLAY_NAMES["TAFAC"]}


def correction_weight(t, transition_start, late_start):
    u = np.clip((np.asarray(t) - transition_start) / (late_start - transition_start), 0, 1)
    return u * u * (3.0 - 2.0 * u)


def apply_adjustment(series, duration):
    late_start = duration / 2
    transition_start = late_start - 0.5
    for law, entry in series.items():
        raw, plotted = entry["data"], entry["plotted"]
        late = raw.loc[raw.display_time_s >= late_start]
        mean_error = float(late.error_n.mean())
        removed_offset = mean_error if law == "UltraPoC" else 0.0
        for table in (raw, plotted):
            w = correction_weight(table.display_time_s, transition_start, late_start)
            table["removed_baseline_n"] = removed_offset * w
        raw["adjusted_fz_n"] = raw.fz_n - raw.removed_baseline_n
        raw["adjusted_deviation_n"] = raw.error_n - raw.removed_baseline_n
        plotted["adjusted_fz_n"] = plotted.filtered_fz_n - plotted.removed_baseline_n
        plotted["adjusted_deviation_n"] = plotted.filtered_error_n - plotted.removed_baseline_n
        entry["adjusted_post"] = raw.loc[raw.contact_time_s >= entry["actual_rise_duration_s"]].copy()
        adjusted_stats = entry["adjusted_post"].assign(error_n=entry["adjusted_post"].adjusted_deviation_n)
        entry["bias_adjustment"] = dict(
            applied=law == "UltraPoC", estimated_late_mean_error_n=mean_error,
            removed_late_offset_n=removed_offset,
            mean_estimation_display_interval_s=[late_start, duration],
            transition_display_interval_s=[transition_start, late_start],
            definition="F_adjusted = F_measured - mean_late(F_measured-4 N) * smoothstep(tau)",
            zero_outside_ultrapoc=True,
            post_rise_adjusted=source.stats(adjusted_stats),
        )
    return series


def configure():
    source.configure()
    plt.rcParams.update({
        "font.family": ["Nimbus Sans", "Liberation Sans", "DejaVu Sans"],
        "font.size": 13,
        "axes.labelsize": 16,
        "axes.titlesize": 14,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "mathtext.fontset": "dejavusans",
    })


def axis_style(ax):
    ax.set_axisbelow(True)
    ax.grid(False)
    ax.grid(axis="x", color="#DADADA", linewidth=0.65)
    ax.tick_params(direction="out", length=4, width=0.8)


def violin(ax, samples):
    # A shared density-to-width scale gives equal-area violins. A narrower
    # error distribution becomes shorter and wider through its actual density,
    # without any per-controller width manipulation or amplitude compression.
    densities = []
    for values in samples:
        y = np.linspace(0, np.max(values), 700)
        mirrored = np.r_[values, -values]
        kernel = gaussian_kde(mirrored, bw_method=0.04 / np.std(mirrored, ddof=1))
        density = 2 * kernel(y)
        density /= np.trapezoid(density, y)
        densities.append((y, density))
    common_peak = max(float(d.max()) for _, d in densities)
    for i, (values, (y, density)) in enumerate(zip(samples, densities), start=1):
        width = 0.46 * density / common_peak
        ax.fill_betweenx(y, i - width, i + width, facecolor="#78A9C7",
                         edgecolor="#4C5960", linewidth=1.0, alpha=0.9)
        q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
        ax.plot([i, i], [q1, q3], color="#42484B", lw=3.0, solid_capstyle="round")
        ax.scatter(i, median, color="white", edgecolors="#42484B", s=15,
                   linewidths=0.7, zorder=5)
    ax.set_xlim(0.4, 3.6)
    ax.set_xticks([1, 2, 3], ["UltraPoC*", source.DISPLAY_NAMES["AC1D"],
                             source.DISPLAY_NAMES["TAFAC"]], fontsize=11.5)
    max_val = max(float(np.max(v)) for v in samples)
    ax.set_ylim(0, max(1.25, np.ceil(max_val / 0.25) * 0.25) + 0.015)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_ylabel("Absolute deviation (N)", labelpad=6, fontsize=14)
    ax.grid(False)


def build(series, duration, out):
    fig = plt.figure(figsize=(10.4, 6.5))
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1.55, 1],
                           left=0.095, right=0.985, bottom=0.12, top=0.94,
                           hspace=0.49, wspace=0.31)
    ax_f, ax_e, ax_v = (fig.add_subplot(grid[0, :]), fig.add_subplot(grid[1, 0]),
                        fig.add_subplot(grid[1, 1]))
    for ax in (ax_f, ax_e, ax_v):
        axis_style(ax)
    for law in reversed(source.LAWS):
        data = series[law]["plotted"]
        ax_f.plot(data.display_time_s, data.adjusted_fz_n, color=COLORS[law],
                  lw=1.8, solid_capstyle="round")
        ax_e.plot(data.display_time_s, data.adjusted_deviation_n, color=COLORS[law],
                  lw=1.8, solid_capstyle="round")
    ax_f.axhline(4, color="#242424", lw=1.3, ls=(0, (5, 3)), zorder=6)
    ax_f.set_ylim(0, 5.15)
    ax_f.set_yticks([0, 1, 2, 3, 4, 5])
    ax_e.set_ylim(-1.15, 1.15)
    ax_e.set_yticks([-1, -0.5, 0, 0.5, 1])
    for ax in (ax_f, ax_e):
        ax.set_xlim(0, duration)
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.set_xlabel("Rise-normalized time (s)", labelpad=6)
    ax_f.set_ylabel("Contact force (N)", labelpad=6)
    ax_e.set_ylabel("Force deviation (N)", labelpad=6)
    ax_f.set_title("(a) Contact force", loc="left", pad=9)
    ax_e.set_title("(b) Post-rise force deviation", loc="left", pad=9)
    ax_v.set_title("(c) Absolute deviation", loc="left", pad=9)
    handles = [Line2D([], [], color=COLORS[law], lw=2, label=LABELS[law]) for law in source.LAWS]
    handles.append(Line2D([], [], color="#242424", lw=1.3, ls=(0, (5, 3)), label="Desired force (4 N)"))
    ax_f.legend(handles=handles, loc="lower right", fontsize=11.7,
                borderaxespad=0.65, handlelength=2.1, labelspacing=0.4)
    violin(ax_v, [np.abs(series[law]["adjusted_post"].adjusted_deviation_n.to_numpy())
                  for law in source.LAWS])
    # The asterisk keeps the derived UltraPoC distribution identifiable without
    # a long figure footer; the force legend explicitly names the adjustment.
    paths = []
    for suffix in ("png", "pdf", "svg"):
        path = out.with_suffix(f".{suffix}")
        fig.savefig(path, dpi=600)
        paths.append(str(path))
    plt.close(fig)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=source.STUDY)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--rise-display", type=float, default=0.30)
    parser.add_argument("--cutoff-hz", type=float, default=8.0)
    args = parser.parse_args()
    if args.duration <= 1 or not 0 < args.rise_display < args.duration:
        parser.error("Require duration > 1 s and 0 < rise-display < duration")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "data").mkdir(exist_ok=True)
    configure()
    series, sources = source.load_runs(args.study, args.duration, args.rise_display, args.cutoff_hz)
    apply_adjustment(series, args.duration)
    metadata = {}
    for law, entry in series.items():
        # The exact preceding measured columns remain in this export; the new
        # derived values and corrections occupy separate, clearly named fields.
        entry["data"].to_csv(args.out / "data" / f"{law}_measured_and_adjusted.csv",
                             index=False, float_format="%.17g")
        entry["plotted"].to_csv(args.out / "data" / f"{law}_display.csv",
                                index=False, float_format="%.17g")
        metadata[law] = {key: value for key, value in entry.items()
                         if key not in ("data", "plotted", "post", "adjusted_post")}
    figures = build(series, args.duration, args.out / "force_comparison_bias_adjusted")
    manifest = dict(
        description="Disclosed UltraPoC late-bias-adjusted visualization; not raw force-tracking accuracy",
        runs=source.RUNS, paper_controller_names=source.DISPLAY_NAMES,
        source_files=sources, per_controller=metadata,
        correction_scope="UltraPoC only; AC1D and TAFAC have zero baseline correction and unchanged previous filtering",
        original_processing="100 Hz grid; identical 4th-order 8 Hz Butterworth zero-phase low-pass; no decimation; rise normalized to 0.30 s",
        error_processing="Derived UltraPoC deviation uses measured error minus the explicitly exported late-baseline correction. AC1D and TAFAC use measured errors. All exclude rise. No clipping or amplitude compression.",
        violin="Unfiltered post-rise deviations under the stated correction; mirrored 0.04 N KDE, equal area and one common density-to-width scale for all controllers",
        source_integrity="Original CSV/H5/log recordings are unchanged. Measured and corrected columns are separately exported.",
        interpretation="This bias-adjusted figure compares fluctuations. Its UltraPoC deviation is not the original Fz-4 N tracking error and is not a QP force-tolerance violation metric.",
        qp_evidence="The recorded QP slack is in velocity/angular-velocity units, not N. The log does not specify a force-N tolerance to subtract. The estimated 0.2 N late offset is therefore a display bias adjustment, not a certified allowed QP force error.",
        figures=figures,
    )
    (args.out / "force_adjustment_provenance.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    shutil.copy2(Path(__file__), args.out / Path(__file__).name)
    shutil.copy2(Path(source.__file__), args.out / Path(source.__file__).name)
    (args.out / "README.txt").write_text(
        'Three-controller display: UltraPoC late-bias-adjusted; F-AC/TAF-AC unchanged.\n'
        'Paper names: F-AC is force-only admittance control; TAF-AC is trajectory-aware feedforward admittance control.\n'
        'Historical run IDs AC1D and TAFAC remain in data filenames for source traceability.\n'
        'This is a derived fluctuation comparison, not raw force-tracking accuracy.\n'
        'Only UltraPoC receives a baseline correction. Its measured mean error over display time 2.5-5 s is removed.\n'
        'The correction turns on with a cubic smoothstep over 2.0-2.5 s and remains constant afterwards.\n'
        'Curve amplitudes are not rescaled or clipped; only the disclosed additive correction is applied.\n'
        'Input recordings remain unchanged. Both measured and adjusted values and per-sample corrections are in data/.\n'
        'The QP log does not define a force-N tolerance; this correction is not labeled as an allowed QP force error.\n'
        'The three violins share KDE bandwidth, area normalization, vertical axis, and density-to-width scale.\n'
        'No gray rise shading, Rise excluded text, or horizontal gridlines.\n'
        'Reproduce: python plot_proposal_force_adjusted.py --out regenerated\n',encoding='utf-8')
    print(json.dumps(dict(figures=figures, bias_adjustments={law:entry['bias_adjustment']
                                                           for law,entry in series.items()}),indent=2))


if __name__ == "__main__":
    main()
