#!/usr/bin/env python3
"""English force-only comparison with an explicitly normalized rise phase.

Source recordings are never modified. All curves use the same fourth-order
8 Hz zero-phase Butterworth low-pass on a 100 Hz grid, without decimation.
For display only, each controller's rise to the filtered 4 N target is mapped
to 0.30 s. Subsequent elapsed time is unscaled. Both error panels exclude the
rise; distributions and statistics use original, unfiltered post-rise samples.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
from scipy.signal import butter, sosfiltfilt
from scipy.stats import gaussian_kde

STUDY = Path("/media/camp/PEI_T7/icra 2027_contact/Comparison Study")
OUT = Path(__file__).resolve().parents[1] / "outputs/proposal_force_20261015_3controllers"
LAWS = ("UltraPoC", "AC1D", "TAFAC")
# Internal run IDs remain unchanged; paper-facing names follow the manuscript.
DISPLAY_NAMES = {"UltraPoC": "UltraPoC", "AC1D": "F-AC", "TAFAC": "TAF-AC"}
COLORS = {"UltraPoC": "#0072B2", "AC1D": "#D55E00", "TAFAC": "#009E73"}
RUNS = {"UltraPoC": "ultrapoc/004noise", "AC1D": "admittance_1d/001", "TAFAC": "tafac/001"}
INK = "#252B32"
SAMPLE_HZ = 100.0
FILTER_ORDER = 4


def configure():
    plt.rcParams.update({
        "font.family": ["Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
        "font.size": 13, "axes.labelsize": 15, "axes.titlesize": 15,
        "axes.linewidth": 1.0, "axes.edgecolor": INK, "axes.labelcolor": INK,
        "text.color": INK, "xtick.color": INK, "ytick.color": INK,
        "xtick.labelsize": 12, "ytick.labelsize": 12,
        "mathtext.fontset": "stix", "axes.spines.top": False,
        "axes.spines.right": False, "legend.frameon": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "savefig.facecolor": "white", "path.simplify": False,
    })


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(data):
    e = data.error_n.to_numpy()
    return dict(n=int(len(e)), mean_error_n=float(np.mean(e)),
                std_error_n=float(np.std(e)), rmse_n=float(np.sqrt(np.mean(e * e))),
                mae_n=float(np.mean(np.abs(e))),
                p95_absolute_error_n=float(np.quantile(np.abs(e), 0.95)),
                max_absolute_error_n=float(np.max(np.abs(e))))


def display_clock(real_time, actual_rise_s, display_rise_s):
    """Invertible piecewise linear time map; all force amplitudes untouched."""
    t = np.asarray(real_time)
    return np.where(t <= actual_rise_s,
                    t * display_rise_s / actual_rise_s,
                    display_rise_s + t - actual_rise_s)


def real_clock(display_time, actual_rise_s, display_rise_s):
    t = np.asarray(display_time)
    return np.where(t <= display_rise_s,
                    t * actual_rise_s / display_rise_s,
                    actual_rise_s + t - display_rise_s)


def first_target_crossing(t, fz, target):
    crossings = np.flatnonzero((t[1:] >= 0) & (fz[:-1] < target) & (fz[1:] >= target)) + 1
    if not len(crossings):
        raise ValueError("Filtered force never reaches target after contact")
    i = int(crossings[0])
    fraction = (target - fz[i - 1]) / (fz[i] - fz[i - 1])
    return float(t[i - 1] + fraction * (t[i] - t[i - 1]))


def load_runs(study, duration, display_rise, cutoff):
    series, sources = {}, []
    sos = butter(FILTER_ORDER, cutoff, fs=SAMPLE_HZ, output="sos")
    for law, rel in RUNS.items():
        folder = study / rel
        force_path, motion_path = folder / "force_trace.csv", folder / "motion_trace.csv"
        force = pd.read_csv(force_path)
        force = force.loc[(force.valid > 0.5) & np.isfinite(force.sample_t_mono_s) &
                          np.isfinite(force.fz_n) & np.isfinite(force.desired_fz_n)]
        force = force.sort_values("sample_t_mono_s").reset_index(drop=True)
        t_mono = force.sample_t_mono_s.to_numpy()
        raw_fz = force.fz_n.to_numpy()
        reached = np.flatnonzero(raw_fz >= 3.5)
        if not len(reached):
            raise ValueError(f"No contact rise to 3.5 N in {rel}")
        j = int(reached[0])
        below = np.flatnonzero(raw_fz[:j + 1] < 0.4)
        if not len(below):
            raise ValueError(f"No pre-contact sample below 0.4 N in {rel}")
        origin = float(t_mono[int(below[-1])])
        t = t_mono - origin
        if np.any(np.diff(t) <= 0):
            raise ValueError(f"Non-increasing source timestamps in {rel}")
        force["contact_time_s"] = t
        force["error_n"] = force.fz_n - force.desired_fz_n
        # Filter the full recorded trace before cropping the displayed window;
        # this avoids endpoint artifacts at first contact and at the figure end.
        grid = np.arange(t.min(), t.max(), 1.0 / SAMPLE_HZ)
        interpolated = np.interp(grid, t, raw_fz)
        filtered = sosfiltfilt(sos, interpolated)
        actual_rise = first_target_crossing(grid, filtered, target=4.0)
        actual_end = actual_rise + duration - display_rise
        if actual_end > t.max():
            raise ValueError(f"Incomplete post-rise data in {rel}")
        motion = pd.read_csv(motion_path)
        tracking_start = float(motion.loc[motion.stage == "tracking", "t_mono_s"].min())
        if not origin <= tracking_start <= origin + 1.0:
            raise ValueError(f"Contact threshold not near tracking onset for {rel}")
        force["display_time_s"] = display_clock(t, actual_rise, display_rise)
        data = force.loc[force.contact_time_s.between(0, actual_end)].copy()
        if not np.array_equal(np.unique(data.desired_fz_n), [4.0]):
            raise ValueError(f"Unexpected target force in {rel}")
        post = data.loc[data.contact_time_s >= actual_rise].copy()
        # Retain the 100 Hz grid. Add exactly evaluated boundary points so the
        # normalized rise ends at exactly 4 N and error starts exactly at zero.
        wanted = grid[(grid > 0) & (grid < actual_end)]
        wanted = np.unique(np.r_[0.0, wanted, actual_rise, actual_end])
        f_plot = np.interp(wanted, grid, filtered)
        f_plot[np.isclose(wanted, actual_rise, rtol=0, atol=1e-12)] = 4.0
        plotted = pd.DataFrame(dict(
            contact_time_s=wanted,
            display_time_s=display_clock(wanted, actual_rise, display_rise),
            filtered_fz_n=f_plot,
            filtered_error_n=np.where(wanted >= actual_rise, f_plot - 4.0, np.nan),
        ))
        nominal_dt = float(np.median(np.diff(t)))
        visible_t = t[(t >= -0.5) & (t <= actual_end + 0.5)]
        series[law] = dict(
            data=data, post=post, plotted=plotted,
            run=rel, contact_origin_mono_s=origin,
            actual_rise_duration_s=actual_rise,
            displayed_rise_duration_s=display_rise,
            rise_display_scale=display_rise / actual_rise,
            source_window_s=[0.0, actual_end],
            source_error_interval_s=[actual_rise, actual_end],
            source_nominal_sample_hz=1 / nominal_dt,
            maximum_local_valid_sample_gap_s=float(np.diff(visible_t).max()),
            tracking_start_after_contact_s=tracking_start - origin,
            full_contact_raw=stats(data), post_rise_raw=stats(post),
        )
        for path in (force_path, motion_path, folder / "meta.json"):
            sources.append(dict(path=str(path), sha256=digest(path)))
    return series, sources


def style_axis(ax):
    ax.set_axisbelow(True)
    ax.grid(color="#DEE2E6", linewidth=0.65, alpha=0.85)
    ax.tick_params(direction="out", length=4, width=0.8)


def draw_violin(ax, samples):
    # Reflection corrects the density at the nonnegative error boundary.
    max_error = max(float(np.max(v)) for v in samples)
    bandwidth = 0.04
    for position, law, values in zip(range(1, len(LAWS) + 1), LAWS, samples):
        grid = np.linspace(0, np.max(values), 500)
        reflected = np.r_[values, -values]
        kde = gaussian_kde(reflected, bw_method=bandwidth / np.std(reflected, ddof=1))
        density = 2 * kde(grid)
        width = 0.37 * density / density.max()
        color = COLORS[law]
        ax.fill_betweenx(grid, position - width, position + width,
                         facecolor=color, alpha=0.43, edgecolor=color, linewidth=1.0)
        q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
        ax.plot([position, position], [q1, q3], color=INK, lw=3.3, solid_capstyle="round")
        ax.scatter(position, median, s=16, color="white", edgecolors=INK,
                   linewidth=0.7, zorder=5)
    ax.set_xticks(range(1, len(LAWS) + 1), [DISPLAY_NAMES[law] for law in LAWS], fontsize=11)
    ax.set_xlim(0.45, len(LAWS) + 0.55)
    ax.set_ylim(0, np.ceil(max_error / 0.25) * 0.25 + 0.025)
    ax.yaxis.set_major_locator(MultipleLocator(0.25))
    ax.set_ylabel(r"Absolute error $|e_F|$ (N)", labelpad=6)
    ax.grid(axis="x", visible=False)


def build(series, duration, display_rise, cutoff, out):
    fig = plt.figure(figsize=(10.4, 6.9))
    layout = fig.add_gridspec(2, 2, width_ratios=[1.5, 1.0], height_ratios=[1.0, 1.0],
                             left=0.09, right=0.985, bottom=0.175, top=0.865,
                             hspace=0.56, wspace=0.34)
    ax_f, ax_e, ax_v = (fig.add_subplot(layout[0, :]), fig.add_subplot(layout[1, 0]),
                        fig.add_subplot(layout[1, 1]))
    for ax in (ax_f, ax_e, ax_v):
        style_axis(ax)
    for ax in (ax_f, ax_e):
        ax.axvspan(0, display_rise, facecolor="#EEF0F3", zorder=0)
    for law in reversed(LAWS):
        data = series[law]["plotted"]
        ax_f.plot(data.display_time_s, data.filtered_fz_n, color=COLORS[law],
                  lw=1.65, solid_capstyle="round")
        ax_e.plot(data.display_time_s, data.filtered_error_n, color=COLORS[law],
                  lw=1.65, solid_capstyle="round")
    ax_f.axhline(4.0, color=INK, lw=1.15, ls=(0, (5, 3)), zorder=6)
    ax_e.axhline(0.0, color=INK, lw=1.0, ls=(0, (5, 3)), zorder=6)
    ax_f.set_ylim(0.0, 5.1)
    ax_f.set_yticks([0, 1, 2, 3, 4, 5])
    max_curve_error = max(np.nanmax(np.abs(series[law]["plotted"].filtered_error_n)) for law in LAWS)
    error_lim = max(1.0, np.ceil(max_curve_error / 0.25) * 0.25)
    ax_e.set_ylim(-error_lim - 0.04, error_lim + 0.04)
    ax_e.yaxis.set_major_locator(MultipleLocator(0.5))
    for ax in (ax_f, ax_e):
        ax.set_xlim(0, duration)
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.set_xlabel("Rise-normalized time (s)", labelpad=5)
    ax_f.set_ylabel(r"Contact force $F_z$ (N)", labelpad=6)
    ax_e.set_ylabel(r"Force error $e_F$ (N)", labelpad=5)
    ax_f.set_title("(a) Contact force", loc="left", pad=10)
    ax_e.set_title("(b) Post-rise force error", loc="left", pad=10)
    ax_v.set_title("(c) Post-rise absolute error", loc="left", pad=10)
    ax_e.text(display_rise / 2, 0, "Rise excluded", rotation=90,
              ha="center", va="center", fontsize=8.5, color="#717780")
    samples = [np.abs(series[law]["post"].error_n.to_numpy()) for law in LAWS]
    draw_violin(ax_v, samples)
    handles = [Line2D([], [], color=COLORS[law], lw=2.1, label=DISPLAY_NAMES[law]) for law in LAWS]
    handles.append(Line2D([], [], color=INK, lw=1.2, ls=(0, (5, 3)), label="Desired force (4 N)"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.54, 0.987),
               ncol=len(handles), fontsize=12.5, handlelength=1.9,
               columnspacing=1.25, handletextpad=0.5)
    fig.text(0.09, 0.056,
             f"Rise to 4 N normalized to {display_rise:.2f} s for display only; post-rise timing is unscaled.",
             fontsize=10.2, color="#525A65", ha="left")
    fig.text(0.09, 0.029,
             f"Curves: {cutoff:g} Hz zero-phase low-pass, 100 Hz grid. Errors exclude rise; violins use raw post-rise samples.",
             fontsize=10.2, color="#525A65", ha="left")
    saved = []
    for ext in ("png", "pdf", "svg"):
        target = out.with_suffix(f".{ext}")
        fig.savefig(target, dpi=600)
        saved.append(str(target))
    plt.close(fig)
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=STUDY)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--duration", type=float, default=5.0,
                        help="Extent of the explicitly rise-normalized display clock")
    parser.add_argument("--rise-display", type=float, default=0.30)
    parser.add_argument("--cutoff-hz", type=float, default=8.0)
    args = parser.parse_args()
    if not 0 < args.rise_display < args.duration or not 0 < args.cutoff_hz < SAMPLE_HZ / 2:
        parser.error("Require 0 < rise-display < duration, and 0 < cutoff-hz < 50")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "data").mkdir(exist_ok=True)
    configure()
    series, sources = load_runs(args.study, args.duration, args.rise_display, args.cutoff_hz)
    rows = []
    for law, entry in series.items():
        columns = ["sample_t_mono_s", "contact_time_s", "display_time_s", "fz_n", "desired_fz_n", "error_n", "stage"]
        if "force_seq" in entry["data"]:
            columns.append("force_seq")
        entry["data"][columns].to_csv(args.out / "data" / f"{law}_original_samples.csv", index=False, float_format="%.17g")
        entry["plotted"].to_csv(args.out / "data" / f"{law}_filtered_display.csv", index=False, float_format="%.17g")
        for interval in ("full_contact_raw", "post_rise_raw"):
            rows.append(dict(controller=law, interval=interval, **entry[interval]))
    saved = build(series, args.duration, args.rise_display, args.cutoff_hz,
                  args.out / "force_contact_comparison")
    manifest = dict(
        description="English force-only figure; explicitly rise-normalized display; original source files unchanged",
        study=str(args.study.resolve()), runs=RUNS, paper_controller_names=DISPLAY_NAMES,
        source_files=sources,
        contact_origin="Last valid Fz < 0.4 N before first Fz >= 3.5 N, matching the original paper plotting script",
        rise_end="First crossing of 4 N in the identically filtered force after contact, interpolated between 100 Hz grid points",
        display_clock="tau=t*T_display/T_rise when t<=T_rise; tau=T_display+t-T_rise afterwards",
        inverse_display_clock="t=tau*T_rise/T_display when tau<=T_display; t=T_rise+tau-T_display afterwards",
        display_window_s=[0, args.duration], displayed_rise_duration_s=args.rise_display,
        display_error_interval_s=[args.rise_display, args.duration],
        error_definition="Fz - 4 N; force buildup excluded, no amplitude rescaling or offset removal",
        filtering=dict(kind="Butterworth SOS forward-backward (zero phase)",
                       single_pass_order=FILTER_ORDER, single_pass_cutoff_hz=args.cutoff_hz,
                       grid_hz=SAMPLE_HZ, decimation=False,
                       processing="Full valid trace linearly interpolated to 100 Hz, then sosfiltfilt before window crop and time mapping"),
        violin="Original unfiltered post-rise absolute errors, same elapsed post-rise duration for every law; reflected KDE with common 0.04 N Gaussian bandwidth and equal maximum width; median dot and interquartile bar",
        interpretation="The force panel cannot be used to compare actual rise times because its rise phase is explicitly normalized. Post-rise elapsed seconds are unscaled. Full-window raw error metrics are retained separately.",
        caveat="One previously selected biased-phantom run per controller; time samples are autocorrelated, not independent repeated trials.",
        per_controller={law: {k: v for k, v in entry.items() if k not in ("data", "post", "plotted")} for law, entry in series.items()},
        figures=saved,
    )
    (args.out / "force_provenance.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    pd.DataFrame(rows).to_csv(args.out / "force_metrics.csv", index=False)
    print(json.dumps(dict(figures=saved, per_controller=manifest["per_controller"]), indent=2))


if __name__ == "__main__":
    main()
