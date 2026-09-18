#!/usr/bin/env python3
"""Cache UltraPoC 004noise mechanism traces for the contact-adaptation figures.

Clock stored on the arrays is seconds after the first control_sample (includes
the press-in). Figures shift that clock by t_scan_start_s so t=0 is the first
tracking cycle. Tracking cycles are those whose proposal time falls in
motion_trace stage == tracking.
Force and moment are interpolated from force_trace.csv onto that clock.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path("/media/camp/PEI_T7/icra 2027_contact/Comparison Study")
DEFAULT_RUN = ROOT / "ultrapoc" / "004noise"
DEFAULT_OUT = Path("/tmp/contact_adaptation_data.npz")
C_TH = 0.8
MERGE_GAP_S = 0.30
MIN_EPISODE_S = 0.20
# alpha is a continuous QP output; this is the "not at nominal speed" threshold.
ALPHA_NOMINAL = 0.99
# alpha_us = clip(1 - 0.75 max_i d_i / c_th, 0.25, 1) in region_visual.py.
ALPHA_GAIN = 0.75
SETTLED_AFTER_S = 2.0
# Scan-clock time for the "restored window" B-mode (frame 3). The first
# full-window sample is ~1.00 s with min c_i still 0.88; 1.10 s is after
# alpha has returned to 1 and every region is ~0.98.
RECOVERED_SCAN_S = 1.10
# A rocking increment worth counting, and the paired-increment bins (mm/s) used
# to test whether that increment disturbs force regulation.
ROCK_MIN = 1e-3
FORCE_SETTLED_S = 6.0
PAIR_CUTS_MM_S = (0.001, 0.35)


def _read_csv(path: Path, columns):
    rows = {name: [] for name in columns}
    with path.open() as handle:
        for row in csv.DictReader(handle):
            for name in columns:
                rows[name].append(row[name])
    return rows


def _intervals(mask, t):
    edges = np.diff(mask.astype(np.int8), prepend=0, append=0)
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    out = []
    for lo, hi in zip(starts, ends):
        if hi <= lo:
            continue
        out.append((float(t[lo]), float(t[hi - 1]), int(lo), int(hi)))
    return out


def _merge(intervals, gap):
    if not intervals:
        return []
    merged = [list(intervals[0])]
    for item in intervals[1:]:
        if item[0] - merged[-1][1] <= gap:
            merged[-1][1] = item[1]
            merged[-1][3] = item[3]
        else:
            merged.append(list(item))
    return [tuple(item) for item in merged]


def _finite(value):
    if value is None:
        return np.nan
    value = float(value)
    return value if np.isfinite(value) else np.nan


def extract(run: Path):
    motion = _read_csv(run / "motion_trace.csv", ("t_mono_s", "stage", "t_ref_s"))
    mono = np.asarray(motion["t_mono_s"], dtype=float)
    t_ref = np.asarray(motion["t_ref_s"], dtype=float)
    tracking_flag = np.asarray(motion["stage"]) == "tracking"
    order = np.argsort(mono)
    mono, t_ref, tracking_flag = mono[order], t_ref[order], tracking_flag[order]

    force = _read_csv(
        run / "force_trace.csv",
        ("sample_t_mono_s", "fz_n", "my_nm", "desired_fz_n", "valid", "stage"),
    )
    force_t = np.asarray(force["sample_t_mono_s"], dtype=float)
    force_f = np.asarray(force["fz_n"], dtype=float)
    force_m = np.asarray(force["my_nm"], dtype=float)
    force_fd = np.asarray(force["desired_fz_n"], dtype=float)
    force_ok = np.asarray(force["valid"], dtype=float) > 0.5
    force_ok &= np.isfinite(force_t) & np.isfinite(force_f) & np.isfinite(force_m)

    records = []
    t0 = None
    with (run / "contact_qp.jsonl").open() as handle:
        for line in handle:
            event = json.loads(line)
            if event.get("event") != "control_sample":
                continue
            t_mono = float(event["record_monotonic_s"])
            if t0 is None:
                t0 = t_mono
            diag = event.get("allocation_diagnostics") or {}
            feature = event.get("feature") or {}
            confidence = np.asarray(feature.get("region_confidence") or [], dtype=float)
            if confidence.size != 10:
                confidence = np.full(10, np.nan)
            solution = diag.get("solution") or [np.nan, np.nan, np.nan, np.nan]
            records.append(dict(
                t=t_mono - t0,
                t_prop=float(event["proposal_time_s"]) - t0,
                t_ref=_finite(event.get("reference_s")),
                source_t=float((event.get("source") or {}).get("source_t_s") or event["proposal_time_s"]) - t0,
                v_m=_finite(diag.get("mechanical_normal_m_s")),
                omega_m=_finite(diag.get("mechanical_omega_rad_s")),
                v_n=_finite(diag.get("U_m_s")),
                omega_n=_finite(diag.get("Omega_rad_s")),
                alpha=_finite(solution[2] if len(solution) > 2 else event.get("alpha")),
                alpha_us=_finite(diag.get("alpha_preferred", event.get("alpha_preferred"))),
                omega_us=_finite(diag.get("visual_request_rad_s")),
                omega_us_gated=_finite(diag.get("visual_request_gated_rad_s")),
                visual_gamma=_finite(diag.get("visual_gamma")),
                c_p=_finite(diag.get("cop_m")),
                d_omega=_finite(diag.get("qp_delta_omega_y_rad_s")),
                r_cop=_finite(diag.get("cop_increment_residual_m_s")),
                pairing=str(diag.get("cop_pairing") or ""),
                rows=tuple(diag.get("active_hard_rows") or ()),
                frame_seq=-1 if feature.get("frame_seq") is None else int(feature["frame_seq"]),
                confidence=confidence,
            ))

    n = len(records)
    t = np.asarray([row["t"] for row in records], dtype=float)
    t_prop = np.asarray([row["t_prop"] for row in records], dtype=float)
    source_t = np.asarray([row["source_t"] for row in records], dtype=float)
    tracking = np.interp(t_prop + t0, mono, tracking_flag.astype(float), left=0.0, right=0.0) > 0.5
    confidence = np.vstack([row["confidence"] for row in records])
    has_c = np.isfinite(confidence).any(axis=1)
    min_c = np.full(n, np.nan)
    max_d = np.zeros(n)
    if has_c.any():
        min_c[has_c] = np.nanmin(confidence[has_c], axis=1)
        max_d[has_c] = np.nanmax(np.maximum(C_TH - confidence[has_c], 0.0), axis=1)
    omega_us = np.asarray([row["omega_us"] for row in records], dtype=float)
    omega_m = np.asarray([row["omega_m"] for row in records], dtype=float)
    s_us = np.sign(np.nan_to_num(omega_us, nan=0.0))
    unmet = tracking & np.isfinite(omega_us) & np.isfinite(omega_m) & (s_us * omega_m < np.abs(omega_us))
    deficit = tracking & (max_d > 0.0)
    v_m = np.asarray([row["v_m"] for row in records], dtype=float)
    v_n = np.asarray([row["v_n"] for row in records], dtype=float)
    d_omega = np.asarray([row["d_omega"] for row in records], dtype=float)
    c_p = np.asarray([row["c_p"] for row in records], dtype=float)
    d_v = v_n - v_m
    c_p_d_omega = c_p * d_omega

    force_clock = force_t[force_ok] - t0
    force_f = force_f[force_ok]
    force_m = force_m[force_ok]
    force_fd = force_fd[force_ok]
    F = np.interp(source_t, force_clock, force_f, left=np.nan, right=np.nan)
    M = np.interp(source_t, force_clock, force_m, left=np.nan, right=np.nan)
    Fd = np.interp(source_t, force_clock, force_fd, left=np.nan, right=np.nan)

    raw_deficit = _intervals(deficit, t)
    raw_unmet = _intervals(unmet, t)
    merged = [item for item in _merge(raw_deficit, MERGE_GAP_S)
              if item[1] - item[0] >= MIN_EPISODE_S]
    named = []
    for label, item in zip(("A", "B"), merged):
        lo, hi, i0, i1 = item
        window = slice(i0, i1)
        named.append(dict(
            label=label,
            t0=lo,
            t1=hi,
            min_c=float(np.nanmin(min_c[window])),
            alpha_min=float(np.nanmin(np.asarray([row["alpha"] for row in records], float)[window])),
            max_abs_omega_us=float(np.nanmax(np.abs(omega_us[window]))),
        ))

    episode_a = next(item for item in named if item["label"] == "A")
    in_a = tracking & (t >= episode_a["t0"]) & (t <= episode_a["t1"])
    peak = int(np.nanargmin(np.where(in_a, min_c, np.nan)))
    recovered = np.flatnonzero(tracking & (t > episode_a["t1"]) & (max_d <= 0.0))
    scan0 = float(t[tracking].min()) if tracking.any() else float(t[peak])
    recovery = int(np.argmin(np.abs(t - (scan0 + RECOVERED_SCAN_S))))
    if recovered.size == 0:
        recovery = int(np.argmin(np.abs(t - (t[peak] + 1.0))))
    request_peak = int(np.nanargmax(np.where(in_a, np.abs(omega_us), np.nan)))
    settled = int(np.argmin(np.abs(t - (t[recovery] + SETTLED_AFTER_S))))

    alpha = np.asarray([row["alpha"] for row in records], dtype=float)
    alpha_us = np.asarray([row["alpha_us"] for row in records], dtype=float)
    slow = tracking & (alpha < ALPHA_NOMINAL)
    slow_intervals = []
    for lo, hi, i0, i1 in _intervals(slow, t):
        window = slice(i0, i1)
        slow_intervals.append(dict(
            t0=lo,
            t1=hi,
            n=int(i1 - i0),
            alpha_min=float(np.nanmin(alpha[window])),
            max_d=float(np.nanmax(max_d[window])),
            min_c=float(np.nanmin(min_c[window])) if np.isfinite(min_c[window]).any() else float("nan"),
        ))
    deficit_times = t[deficit]
    coupled_slow = [
        bool(np.any(np.abs(deficit_times - time) <= MERGE_GAP_S)) for time in t[slow]
    ]

    # Arc length: the reference clock advances at alpha, so L = v0 * max t_ref.
    run_meta = json.loads((run / "meta.json").read_text())
    v0 = float(run_meta.get("speed_m_s") or 0.01)
    t_ref = np.asarray([row["t_ref"] for row in records], dtype=float)
    d_t_ref = np.clip(np.diff(np.nan_to_num(t_ref), prepend=0.0), 0.0, None)
    d_t = np.clip(np.diff(t, prepend=t[0]), 0.0, None)
    path_mm = v0 * t_ref * 1e3

    rocking = np.array(["published_contact_rocking_speed_acc_jerk" in row["rows"] for row in records])
    normal_bound = np.array([
        any(str(label).startswith("velocity_2") or str(label) == "velocity_2" for label in row["rows"])
        for row in records
    ])
    track = tracking
    pair = np.abs(c_p_d_omega)
    residual = np.asarray([row["r_cop"] for row in records], dtype=float)

    # CoP pairing: the increment the QP would otherwise inject at the loaded point.
    rocks = track & (np.abs(d_omega) > ROCK_MIN) & np.isfinite(c_p_d_omega)
    excursions = [0.0]
    for _, _, i0, i1 in _intervals(rocks, t):
        window = slice(i0, i1)
        cumulative = np.nancumsum(c_p_d_omega[window] * d_t[window])
        excursions.append(float(np.nanmax(np.abs(cumulative))))

    # Force regulation after the load is established, binned by the paired increment.
    steady = track & (t > FORCE_SETTLED_S) & np.isfinite(F) & np.isfinite(Fd)
    f_error = np.abs(F - Fd)
    pair_mm = pair * 1e3
    bins = (
        steady & (pair_mm <= PAIR_CUTS_MM_S[0]),
        steady & (pair_mm > PAIR_CUTS_MM_S[0]) & (pair_mm <= PAIR_CUTS_MM_S[1]),
        steady & (pair_mm > PAIR_CUTS_MM_S[1]),
    )
    requesting = steady & (np.abs(omega_us) > 0.0)

    # Indentation that carries the nominal load, measured from the press-in command.
    contact = np.flatnonzero(np.isfinite(F) & (F > 0.2))
    press_in_mm = np.nan
    if contact.size:
        start = int(contact[0])
        reached = np.flatnonzero(np.isfinite(F) & (F > 0.9 * np.nanmax(Fd[track])) & (t > t[start]))
        if reached.size:
            stop = int(reached[0])
            press_in_mm = float(np.nansum(v_n[start:stop] * d_t[start:stop]) * 1e3)
    meta = dict(
        run=str(run),
        t0_monotonic_s=float(t0),
        t_scan_start_s=float(t[track].min()) if track.any() else 0.0,
        c_th=C_TH,
        n_cycles=int(n),
        n_tracking=int(track.sum()),
        deficit_frac=float(deficit[track].mean()) if track.any() else 0.0,
        unmet_frac=float(unmet[track].mean()) if track.any() else 0.0,
        residual_max_m_s=float(np.nanmax(np.abs(residual[track]))) if track.any() else np.nan,
        pair_rms_mm_s=float(np.sqrt(np.nanmean(pair[track] ** 2)) * 1e3) if track.any() else np.nan,
        pair_max_mm_s=float(np.nanmax(pair[track]) * 1e3) if track.any() else np.nan,
        v_m_rms_mm_s=float(np.sqrt(np.nanmean(v_m[track] ** 2)) * 1e3) if track.any() else np.nan,
        c_p_median_mm=float(np.nanmedian(np.abs(c_p[track])) * 1e3) if track.any() else np.nan,
        rocking_frac=float(rocking[track].mean()) if track.any() else 0.0,
        normal_bound_frac=float(normal_bound[track].mean()) if track.any() else 0.0,
        deficit_intervals=[[a, b] for a, b, *_ in raw_deficit],
        unmet_intervals=[[a, b] for a, b, *_ in raw_unmet],
        episodes=named,
        peak_request_omega_m=float(omega_m[request_peak]),
        peak_request_abs_d_omega=float(np.nanmax(np.abs(d_omega[unmet & in_a]))) if np.any(unmet & in_a) else np.nan,
        # Coupling-aware progression.
        v0_m_s=v0,
        path_length_mm=float(v0 * t_ref[track].max() * 1e3),
        path_under_deficit_mm=float(v0 * d_t_ref[deficit].sum() * 1e3),
        time_under_deficit_s=float(d_t[deficit].sum()),
        time_slow_s=float(d_t[slow].sum()),
        alpha_mean=float(np.nanmean(alpha[track])),
        alpha_min=float(np.nanmin(alpha[track])),
        alpha_slow_frac=float(slow[track].mean()),
        alpha_gain=ALPHA_GAIN,
        alpha_slow_near_deficit_frac=float(np.mean(coupled_slow)) if coupled_slow else 0.0,
        alpha_follows_request_frac=float(np.mean(np.abs(alpha[track] - alpha_us[track]) <= 0.01)),
        slow_intervals=[item for item in slow_intervals if item["n"] > 1],
        # Contact correction.
        request_frac=float((np.abs(omega_us[track]) > 0).mean()),
        unmet_window=[float(t[unmet].min()), float(t[unmet].max())] if unmet.any() else None,
        unmet_abs_omega_m_max=float(np.nanmax(np.abs(omega_m[unmet]))) if unmet.any() else np.nan,
        unmet_abs_omega_m_mean=float(np.nanmean(np.abs(omega_m[unmet]))) if unmet.any() else np.nan,
        unmet_abs_omega_us_max=float(np.nanmax(np.abs(omega_us[unmet]))) if unmet.any() else np.nan,
        unmet_abs_d_omega_max=float(np.nanmax(np.abs(d_omega[unmet]))) if unmet.any() else np.nan,
        unmet_sign_agreement=float(np.mean(np.sign(d_omega[unmet]) == np.sign(omega_us[unmet]))) if unmet.any() else np.nan,
        # CoP pairing.
        pairing_active_cycles=int(rocks.sum()),
        pairing_active_frac=float(rocks[track].mean()) if track.any() else 0.0,
        pair_rms_rocking_mm_s=float(np.sqrt(np.nanmean(pair_mm[rocks] ** 2))) if rocks.any() else np.nan,
        pair_p90_rocking_mm_s=float(np.nanpercentile(pair_mm[rocks], 90)) if rocks.any() else np.nan,
        unpaired_excursion_max_mm=float(np.nanmax(excursions) * 1e3),
        press_in_travel_mm=press_in_mm,
        c_p_p10_mm=float(np.nanpercentile(np.abs(c_p[track]), 10) * 1e3) if track.any() else np.nan,
        c_p_p90_mm=float(np.nanpercentile(np.abs(c_p[track]), 90) * 1e3) if track.any() else np.nan,
        c_p_max_mm=float(np.nanmax(np.abs(c_p[track])) * 1e3) if track.any() else np.nan,
        c_p_over_5mm_frac=float(np.nanmean(np.abs(c_p[track]) > 5e-3)) if track.any() else np.nan,
        # Force regulation once the load is established.
        force_settled_s=FORCE_SETTLED_S,
        f_mean_settled_n=float(np.nanmean(F[steady])) if steady.any() else np.nan,
        f_rms_error_settled_n=float(np.sqrt(np.nanmean(f_error[steady] ** 2))) if steady.any() else np.nan,
        f_max_error_settled_n=float(np.nanmax(f_error[steady])) if steady.any() else np.nan,
        pair_cuts_mm_s=list(PAIR_CUTS_MM_S),
        f_rms_error_by_pair_n=[
            float(np.sqrt(np.nanmean(f_error[mask] ** 2))) if mask.any() else np.nan
            for mask in bins
        ],
        f_cycles_by_pair=[int(mask.sum()) for mask in bins],
        f_rms_error_requesting_n=float(np.sqrt(np.nanmean(f_error[requesting] ** 2))) if requesting.any() else np.nan,
        f_rms_error_idle_n=float(np.sqrt(np.nanmean(f_error[steady & ~requesting] ** 2))) if steady.any() else np.nan,
        # Frames for the B-mode panels.
        frames=dict(
            deficit_peak=dict(frame_seq=int(records[peak]["frame_seq"]), t=float(t[peak]),
                              min_c=float(min_c[peak]), alpha=float(alpha[peak])),
            request_peak=dict(frame_seq=int(records[request_peak]["frame_seq"]), t=float(t[request_peak]),
                              min_c=float(min_c[request_peak]), alpha=float(alpha[request_peak]),
                              omega_us=float(omega_us[request_peak]), omega_m=float(omega_m[request_peak]),
                              d_omega=float(d_omega[request_peak])),
            recovered=dict(frame_seq=int(records[recovery]["frame_seq"]), t=float(t[recovery]),
                           min_c=float(min_c[recovery]), alpha=float(alpha[recovery])),
            settled=dict(frame_seq=int(records[settled]["frame_seq"]), t=float(t[settled]),
                         min_c=float(min_c[settled]), alpha=float(alpha[settled])),
        ),
        frame_peak=int(records[peak]["frame_seq"]),
        t_peak=float(t[peak]),
        min_c_peak=float(min_c[peak]),
        frame_recovery=int(records[recovery]["frame_seq"]),
        t_recovery=float(t[recovery]),
        min_c_recovery=float(min_c[recovery]),
    )
    arrays = dict(
        t=t,
        t_prop=t_prop,
        t_ref=t_ref,
        path_mm=path_mm,
        tracking=track.astype(np.uint8),
        v_m=v_m,
        omega_m=omega_m,
        v_n=v_n,
        omega_n=np.asarray([row["omega_n"] for row in records], dtype=float),
        alpha=alpha,
        alpha_us=alpha_us,
        omega_us=omega_us,
        omega_us_gated=np.asarray([row["omega_us_gated"] for row in records], dtype=float),
        visual_gamma=np.asarray([row["visual_gamma"] for row in records], dtype=float),
        s_us=s_us,
        c_p=c_p,
        d_omega=d_omega,
        d_v=d_v,
        r_cop=residual,
        c_p_d_omega=c_p_d_omega,
        min_c=min_c,
        max_d=max_d,
        deficit=deficit.astype(np.uint8),
        unmet=unmet.astype(np.uint8),
        frame_seq=np.asarray([row["frame_seq"] for row in records], dtype=np.int32),
        region_confidence=confidence,
        F=F,
        M=M,
        Fd=Fd,
        t_force=force_clock,
        F_force=force_f,
        M_force=force_m,
        Fd_force=force_fd,
        rocking_active=rocking.astype(np.uint8),
        meta=np.asarray(json.dumps(meta)),
    )
    return arrays, meta


def load(path: Path):
    blob = np.load(path, allow_pickle=False)
    data = {key: blob[key] for key in blob.files if key != "meta"}
    return data, json.loads(str(blob["meta"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    arrays, meta = extract(args.run)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **arrays)
    print(f"wrote {args.out}")
    print(
        f"tracking={meta['n_tracking']} deficit={100 * meta['deficit_frac']:.1f}% "
        f"unmet={100 * meta['unmet_frac']:.1f}% "
        f"|r|_max={meta['residual_max_m_s']:.2e} "
        f"|cp dw| rms={meta['pair_rms_mm_s']:.2f} peak={meta['pair_max_mm_s']:.2f} mm/s"
    )
    for episode in meta["episodes"]:
        print(
            f"  episode {episode['label']}: {episode['t0']:.2f}-{episode['t1']:.2f} s "
            f"min c={episode['min_c']:.2f} alpha={episode['alpha_min']:.2f} "
            f"|w_us|={episode['max_abs_omega_us']:.3f}"
        )
    print(
        f"  progression: L={meta['path_length_mm']:.0f} mm, "
        f"D_under={meta['path_under_deficit_mm']:.1f} mm, "
        f"alpha mean={meta['alpha_mean']:.3f} min={meta['alpha_min']:.2f}, "
        f"slow cycles={100 * meta['alpha_slow_frac']:.1f}% "
        f"({100 * meta['alpha_slow_near_deficit_frac']:.0f}% near a deficit)"
    )
    print(
        f"  correction: unmet {meta['unmet_window'][0]:.2f}-{meta['unmet_window'][1]:.2f} s, "
        f"|w_m|<={meta['unmet_abs_omega_m_max']:.4f}, "
        f"|w_us|<={meta['unmet_abs_omega_us_max']:.3f}, "
        f"|dw|<={meta['unmet_abs_d_omega_max']:.3f}, "
        f"sign agree={100 * meta['unmet_sign_agreement']:.0f}%"
    )
    print(
        f"  pairing: active {100 * meta['pairing_active_frac']:.0f}% of cycles, "
        f"|cp dw| rms={meta['pair_rms_rocking_mm_s']:.2f} p90={meta['pair_p90_rocking_mm_s']:.2f} mm/s, "
        f"worst unpaired excursion={meta['unpaired_excursion_max_mm']:.2f} mm "
        f"against {meta['press_in_travel_mm']:.1f} mm press-in travel"
    )
    print(
        f"  cop eccentricity: |cp| median={meta['c_p_median_mm']:.1f} p90={meta['c_p_p90_mm']:.1f} "
        f"max={meta['c_p_max_mm']:.1f} mm, >5 mm in {100 * meta['c_p_over_5mm_frac']:.0f}% of cycles"
    )
    print(
        f"  force (t>{meta['force_settled_s']:.0f}s): {meta['f_mean_settled_n']:.2f} N, "
        f"rms err={meta['f_rms_error_settled_n']:.3f} N, max={meta['f_max_error_settled_n']:.2f} N; "
        f"by |cp dw| bin={[round(v, 3) for v in meta['f_rms_error_by_pair_n']]} N "
        f"(n={meta['f_cycles_by_pair']}); "
        f"image-active={meta['f_rms_error_requesting_n']:.3f} vs idle={meta['f_rms_error_idle_n']:.3f} N"
    )
    for name, frame in meta["frames"].items():
        print(f"  frame {name:13s} seq={frame['frame_seq']} t={frame['t']:.2f}s "
              f"min c={frame['min_c']:.2f} alpha={frame['alpha']:.2f}")


if __name__ == "__main__":
    main()
