"""Air-only plant campaign: dense steps, multi-amp chirps, required plots.

Window A still owns the servo (SERVO_TWIST).  This module writes its own
200 Hz CSV from the command we send + MotionBus + Fz snapshot.  Do not
analyse the Window A hybrid log — force-law columns are empty by design
because the admittance loop stays off (otherwise it contaminates the
vel_ff → vz plant).

Pairs ``vel_ff_vz → vz_achieved_tool``.  Same-tick ``Fz × v_cmd`` is
enough to shadow De Stefano §IV bookkeeping in air; it is not a rigid-
press sign check.  The sign check is ``--tdpa-press``: open-loop seek
then a constant +8 mm/s press, force loop off.  Hybrid F* tracking is
the wrong excitation.  Identification is incomplete until the seven
plots exist.  Does not run stop-reverse or backup-replay.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

AIR_STEPS_MM_S = (4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0, 40.0, 60.0, 80.0)
AIR_CHIRP_AMPS_MM_S = (8.0, 15.0, 25.0)
AIR_CHIRP_F0_HZ = 0.2
AIR_CHIRP_F1_HZ = 8.0
AIR_CHIRP_S = 60.0
AIR_HOLD_S = 1.0
AIR_REST_S = 0.5
COHERENCE_MIN = 0.6
T0_SPREAD_WARN_S = 0.008
LINEAR_ACCEL_M_S2 = 2.0
GROUP_DELAY_HZ = 3.0
TDPA_SEEK_M_S = 0.010
TDPA_PRESS_M_S = 0.003
TDPA_CONTACT_N = 0.40
TDPA_TARGET_N = 2.50
TDPA_ABORT_N = 4.50
TDPA_SEEK_MAX_S = 12.0
TDPA_PRESS_S = 2.0
TDPA_MIN_WINDOW_S = 1.0
TDPA_V_PRESS_M_S = 0.002
TDPA_MIN_PRESS_TICKS = 40
TDPA_MIN_OPEN_LOOP_TICKS = 20

REQUIRED_PLOTS = (
    "01_delay_vs_accel.png",
    "02_delay_vs_speed.png",
    "03_rise_vs_accel.png",
    "04_kp_vs_speed.png",
    "05_chirp_bode.png",
    "06_group_delay_3hz.png",
    "07_jitter.png",
)

AIR_LOG_FIELDS = (
    "t_wall_s",
    "t_mono_s",
    "dt_actual_s",
    "phase",
    "vel_ff_vz",
    "v_cmd_z",
    "vz_achieved_tool",
    "fz",
    "feedback_age_s",
    "sensor_age_s",
    "a_tcp_z_plus",
    "motion_seq",
    "tdpa_e_obs_j",
    "tdpa_alpha",
    "tdpa_clamped",
)


@dataclass
class EdgeRow:
    cmd_mm_s: float
    sign: float
    t_edge_s: float
    delay_s: float
    rise_s: float
    peak_accel_m_s2: float
    kp: float


@dataclass
class ChirpFit:
    amp_mm_s: float
    t0_s: float
    tp_s: float
    gain: float
    group_delay_3hz_s: float
    freq_hz: np.ndarray
    mag: np.ndarray
    phase_rad: np.ndarray
    coherence: np.ndarray


@dataclass
class JitterStats:
    n: int
    feedback_age: dict[str, float]
    sensor_age: dict[str, float]
    dt_actual: dict[str, float]


@dataclass
class AirCampaignResult:
    edges: list[EdgeRow] = field(default_factory=list)
    chirps: list[ChirpFit] = field(default_factory=list)
    jitter: JitterStats | None = None
    linear_speed_mm_s: float = float("nan")
    t0_spread_s: float = float("nan")
    out_dir: Path | None = None


def _col(rows: list[dict[str, str]], *keys: str) -> np.ndarray:
    out = np.full(len(rows), np.nan, dtype=float)
    for i, row in enumerate(rows):
        for key in keys:
            raw = row.get(key)
            if raw in (None, ""):
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                out[i] = value
                break
    return out


def _moments(arr: np.ndarray) -> dict[str, float]:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"n": 0, "mean": float("nan"), "p95": float("nan"), "max": float("nan")}
    return {
        "n": int(finite.size),
        "mean": float(np.mean(finite)),
        "p95": float(np.percentile(finite, 95)),
        "max": float(np.max(finite)),
    }


def load_tool_z_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_tool_z_arrays(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = load_tool_z_rows(path)
    t = _col(rows, "t_wall_s")
    u = _col(rows, "vel_ff_vz")
    y = _col(rows, "vz_achieved_tool")
    fb = _col(rows, "feedback_age_s")
    sn = _col(rows, "sensor_age_s")
    dt = _col(rows, "dt_actual_s")
    mask = np.isfinite(t) & np.isfinite(u) & np.isfinite(y)
    if int(np.count_nonzero(mask)) < 8:
        raise ValueError(f"{path} has no finite vel_ff_vz/vz_achieved_tool pair")
    order = np.argsort(t[mask], kind="stable")
    return (
        t[mask][order],
        u[mask][order],
        y[mask][order],
        fb[mask][order],
        sn[mask][order],
        dt[mask][order],
    )


def _xcorr_delay_s(cmd: np.ndarray, ach: np.ndarray, dt: float) -> float:
    if cmd.size < 8 or float(np.std(cmd)) < 1e-8:
        return float("nan")
    c0 = cmd - np.mean(cmd)
    a0 = ach - np.mean(ach)
    corr = np.correlate(a0, c0, mode="full")
    lags = np.arange(-c0.size + 1, a0.size)
    lag = int(lags[int(np.argmax(corr))])
    return max(lag, 0) * float(dt)


def _rise_s(cmd: np.ndarray, ach: np.ndarray, t: np.ndarray) -> float:
    if cmd.size < 4:
        return float("nan")
    target = float(cmd[-1])
    start = float(ach[0])
    span = target - start
    if abs(span) < 1e-5:
        return float("nan")
    lo = start + 0.10 * span
    hi = start + 0.90 * span
    t10 = t90 = float("nan")
    if span > 0.0:
        for ti, yi in zip(t, ach):
            if not math.isfinite(t10) and yi >= lo:
                t10 = float(ti)
            if math.isfinite(t10) and yi >= hi:
                t90 = float(ti)
                break
    else:
        for ti, yi in zip(t, ach):
            if not math.isfinite(t10) and yi <= lo:
                t10 = float(ti)
            if math.isfinite(t10) and yi <= hi:
                t90 = float(ti)
                break
    if not (math.isfinite(t10) and math.isfinite(t90)):
        return float("nan")
    return max(t90 - t10, 0.0)


def _peak_accel(ach: np.ndarray, t: np.ndarray) -> float:
    if ach.size < 3:
        return float("nan")
    dt = np.diff(t)
    dv = np.diff(ach)
    ok = dt > 1e-4
    if not np.any(ok):
        return float("nan")
    return float(np.max(np.abs(dv[ok] / dt[ok])))


def detect_step_edges(
    t: np.ndarray,
    cmd: np.ndarray,
    ach: np.ndarray,
    *,
    min_hold_s: float = 0.25,
    jump_m_s: float = 0.002,
) -> list[EdgeRow]:
    dt = float(np.median(np.diff(t))) if t.size > 2 else 0.005
    if not math.isfinite(dt) or dt <= 1e-4:
        dt = 0.005
    edges: list[EdgeRow] = []
    i = 1
    while i < cmd.size:
        if abs(float(cmd[i] - cmd[i - 1])) < jump_m_s:
            i += 1
            continue
        t0 = float(t[i])
        cmd_now = float(cmd[i])
        j = i
        while j < cmd.size and abs(float(cmd[j]) - cmd_now) < jump_m_s:
            j += 1
        hold = float(t[min(j, cmd.size) - 1] - t0)
        if hold < min_hold_s:
            i = j
            continue
        sl = slice(i, j)
        pre = max(int(round(0.080 / dt)), 4)
        sl_x = slice(max(0, i - pre), j)
        delay = _xcorr_delay_s(cmd[sl_x], ach[sl_x], dt)
        rise = _rise_s(cmd[sl], ach[sl], t[sl])
        accel = _peak_accel(ach[sl], t[sl])
        tail = ach[sl]
        n_tail = max(int(0.2 / dt), 4)
        settled = float(np.mean(tail[-n_tail:])) if tail.size >= n_tail else float(np.mean(tail))
        kp = settled / cmd_now if abs(cmd_now) > 1e-5 else float("nan")
        sign = 1.0 if cmd_now >= 0.0 else -1.0
        edges.append(
            EdgeRow(
                cmd_mm_s=abs(cmd_now) * 1000.0,
                sign=sign,
                t_edge_s=t0,
                delay_s=delay,
                rise_s=rise,
                peak_accel_m_s2=accel,
                kp=kp,
            )
        )
        i = j
    return edges


def _coherence_frf(
    cmd: np.ndarray,
    ach: np.ndarray,
    dt: float,
    *,
    nperseg: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = int(cmd.size)
    if n < 64:
        return (
            np.zeros(0),
            np.zeros(0),
            np.zeros(0),
            np.zeros(0),
        )
    seg = int(nperseg or min(2048, max(256, n // 8)))
    seg = min(seg, n)
    hop = max(seg // 2, 1)
    window = np.hanning(seg)
    u_acc = []
    y_acc = []
    uy_acc = []
    for start in range(0, n - seg + 1, hop):
        u = np.fft.rfft((cmd[start : start + seg] - np.mean(cmd[start : start + seg])) * window)
        y = np.fft.rfft((ach[start : start + seg] - np.mean(ach[start : start + seg])) * window)
        u_acc.append(u * np.conj(u))
        y_acc.append(y * np.conj(y))
        uy_acc.append(y * np.conj(u))
    if not u_acc:
        return np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0)
    puu = np.mean(np.stack(u_acc, axis=0), axis=0)
    pyy = np.mean(np.stack(y_acc, axis=0), axis=0)
    puy = np.mean(np.stack(uy_acc, axis=0), axis=0)
    freq = np.fft.rfftfreq(seg, dt)
    eps = 1e-18
    gyy = np.abs(puy) ** 2 / (np.abs(puu) * np.abs(pyy) + eps)
    h = puy / (puu + eps)
    return freq, np.abs(h), np.angle(h), np.real(gyy)


def fit_chirp_segment(
    t: np.ndarray,
    cmd: np.ndarray,
    ach: np.ndarray,
    *,
    amp_mm_s: float,
    f0: float = AIR_CHIRP_F0_HZ,
    f1: float = AIR_CHIRP_F1_HZ,
) -> ChirpFit:
    from peirastic.apps.identify_plant import _fit_fopdt, _resample_uniform

    tu, uu, yy, dt = _resample_uniform(t, cmd, ach)
    delay, tp, gain, _rmse = _fit_fopdt(uu, yy, dt)
    freq, mag, phase, coh = _coherence_frf(uu, yy, dt)
    band = (freq >= f0) & (freq <= f1) & (coh >= COHERENCE_MIN)
    t0 = float(delay) * float(dt)
    group = float("nan")
    if np.any(np.abs(freq - GROUP_DELAY_HZ) < 0.25) and np.any(band):
        idx = int(np.argmin(np.abs(freq - GROUP_DELAY_HZ)))
        if 1 <= idx < freq.size - 1 and coh[idx] >= COHERENCE_MIN:
            dph = phase[idx + 1] - phase[idx - 1]
            dw = 2.0 * math.pi * (freq[idx + 1] - freq[idx - 1])
            if abs(dw) > 1e-9:
                group = float(-dph / dw)
    if not math.isfinite(group):
        # FOPDT group delay at 3 Hz: T0 + Tp / (1 + (w Tp)^2)
        w = 2.0 * math.pi * GROUP_DELAY_HZ
        group = t0 + tp / (1.0 + (w * tp) ** 2)
    return ChirpFit(
        amp_mm_s=float(amp_mm_s),
        t0_s=t0,
        tp_s=float(tp),
        gain=float(gain),
        group_delay_3hz_s=float(group),
        freq_hz=freq,
        mag=mag,
        phase_rad=phase,
        coherence=coh,
    )


def split_chirp_windows(
    t: np.ndarray,
    cmd: np.ndarray,
    *,
    amps_mm_s: tuple[float, ...] = AIR_CHIRP_AMPS_MM_S,
    chirp_s: float = AIR_CHIRP_S,
) -> list[tuple[float, slice]]:
    """Isolate oscillating chirp bursts; ignore piecewise-constant steps."""
    del chirp_s
    if cmd.size < 64 or t.size != cmd.size:
        return []
    dt = float(np.median(np.diff(t))) if t.size > 2 else 0.005
    if not math.isfinite(dt) or dt <= 1e-4:
        dt = 0.005
    win = max(int(0.5 / dt), 16)
    step = max(win // 4, 1)
    unique_count = np.zeros(cmd.size)
    env_peak = np.zeros(cmd.size)
    for i in range(0, cmd.size - win + 1, step):
        slw = slice(i, i + win)
        nuniq = int(np.unique(np.round(cmd[slw], decimals=5)).size)
        unique_count[slw] = np.maximum(unique_count[slw], nuniq)
        peak = float(np.max(np.abs(cmd[slw])))
        env_peak[slw] = np.maximum(env_peak[slw], peak)
    min_n = int(4.0 / dt)
    windows: list[tuple[float, slice]] = []
    for amp in amps_mm_s:
        target = amp / 1000.0
        mask = (unique_count >= 12) & (
            np.abs(env_peak - target) <= 0.22 * max(target, 0.002)
        )
        padded = np.concatenate(([False], mask, [False]))
        edges = np.diff(padded.astype(int))
        starts = np.flatnonzero(edges == 1)
        ends = np.flatnonzero(edges == -1)
        best: tuple[int, int] | None = None
        for start, end in zip(starts, ends):
            if int(end) - int(start) < min_n:
                continue
            if best is None or int(end) - int(start) > best[1] - best[0]:
                best = (int(start), int(end))
        if best is not None:
            windows.append((float(amp), slice(best[0], best[1])))
    return windows


def jitter_from_columns(
    feedback_age: np.ndarray,
    sensor_age: np.ndarray,
    dt_actual: np.ndarray,
) -> JitterStats:
    return JitterStats(
        n=int(feedback_age.size),
        feedback_age=_moments(feedback_age),
        sensor_age=_moments(sensor_age),
        dt_actual=_moments(dt_actual),
    )


def shadow_tdpa_from_rows(rows: list[dict[str, str]]) -> dict:
    """De Stefano §IV on same-tick Fz × v_cmd.  Air residual is bookkeeping only."""
    from rm75_control.control.admittance_common.tdpa import (
        TdpaConfig,
        TimeDomainPassivityObserver,
    )

    f = _col(rows, "fz", "f_ext_z", "fz_raw_comp")
    v = _col(rows, "v_cmd_z", "vel_ff_vz")
    dt_col = _col(rows, "dt_actual_s")
    n = min(f.size, v.size)
    obs = TimeDomainPassivityObserver(TdpaConfig(enabled=True))
    e_hist = np.zeros(n)
    a_hist = np.zeros(n)
    clamped = 0
    for i in range(n):
        fi = float(f[i]) if math.isfinite(float(f[i])) else 0.0
        vi = float(v[i]) if math.isfinite(float(v[i])) else 0.0
        dti = float(dt_col[i]) if i < dt_col.size and math.isfinite(float(dt_col[i])) else 0.005
        dti = max(dti, 1e-6)
        obs.preview(fi, vi, dti)
        obs.commit(fi, vi, dti, in_contact=False)
        e_hist[i] = float(obs.e_obs_j)
        a_hist[i] = float(obs.alpha)
        clamped += int(bool(obs.alpha_clamped))
    fz_ok = f[np.isfinite(f)]
    return {
        "n": int(n),
        "port": "Fz x v_cmd, same tick",
        "air_only": True,
        "rigid_press_sign_check": False,
        "e_obs_final_j": float(e_hist[-1]) if n else float("nan"),
        "e_obs_min_j": float(np.min(e_hist)) if n else float("nan"),
        "e_obs_max_j": float(np.max(e_hist)) if n else float("nan"),
        "alpha_max": float(np.max(a_hist)) if n else float("nan"),
        "alpha_clamped_ticks": int(clamped),
        "fz_mean_n": float(np.mean(fz_ok)) if fz_ok.size else float("nan"),
        "fz_p95_n": float(np.quantile(np.abs(fz_ok), 0.95)) if fz_ok.size else float("nan"),
        "note": (
            "Air Fz is residual / bias.  This shadows observer leak and "
            "bookkeeping.  E_obs rising on a rigid press needs contact."
        ),
    }


def _tdpa_contact_mask(rows: list[dict[str, str]], f: np.ndarray) -> np.ndarray:
    phases = [str(row.get("phase") or "") for row in rows]
    press_phase = np.array(
        [p.startswith("tdpa_press") or p == "press" for p in phases],
        dtype=bool,
    )
    if np.any(press_phase):
        return press_phase.astype(float)
    contact = _col(rows, "contact_present", "force_barrier_contact_active")
    if np.any(np.isfinite(contact)):
        return contact
    return (np.isfinite(f) & (f > TDPA_CONTACT_N)).astype(float)


def _bool_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    flagged = np.asarray(mask, dtype=bool)
    if flagged.size == 0:
        return []
    delta = np.diff(flagged.astype(np.int8))
    starts = list(np.flatnonzero(delta == 1) + 1)
    ends = list(np.flatnonzero(delta == -1) + 1)
    if flagged[0]:
        starts = [0] + starts
    if flagged[-1]:
        ends = ends + [int(flagged.size)]
    return list(zip(starts, ends))


def analyze_tdpa_contact(path: Path) -> dict:
    """Go/no-go for the open-loop press sign check.

    Scores the longest *continuous* press window.  Concatenating bounce
    fragments is not a sign check — hybrid F* tracking is the wrong
    excitation for this certificate.
    """
    rows = load_tool_z_rows(path)
    if not rows:
        raise ValueError(f"{path} is empty")
    if "tdpa_e_obs_j" not in rows[0]:
        raise ValueError(
            f"{path} has no tdpa_e_obs_j.  Use --tdpa-press (SERVO_TWIST, "
            "force loop off) or a hybrid CSV from this build."
        )
    e = _col(rows, "tdpa_e_obs_j")
    v = _col(rows, "v_force_cmd_z", "v_cmd_z", "vel_ff_vz")
    f = _col(rows, "fz", "fz_raw_comp")
    t = _col(rows, "t_wall_s")
    dt = _col(rows, "dt_actual_s")
    contact = _tdpa_contact_mask(rows, f)
    alpha = _col(rows, "tdpa_alpha")
    clamped = _col(rows, "tdpa_clamped")
    press = (
        (contact > 0.5)
        & np.isfinite(e)
        & np.isfinite(v)
        & np.isfinite(f)
        & (f > TDPA_CONTACT_N)
        & (v > TDPA_V_PRESS_M_S)
    )
    n_press = int(np.count_nonzero(press))
    n_contact = int(np.count_nonzero(contact > 0.5))
    phases = [str(row.get("phase") or "") for row in rows]
    open_loop = any(p.startswith("tdpa_press") or p == "press" for p in phases)
    min_ticks = TDPA_MIN_OPEN_LOOP_TICKS if open_loop else TDPA_MIN_PRESS_TICKS
    if n_press < min_ticks:
        return {
            "ok": False,
            "sign_ok": False,
            "reason": (
                "not enough press-in-contact ticks "
                f"(need ≥{min_ticks} at F>{TDPA_CONTACT_N} N, "
                f"v>{1e3 * TDPA_V_PRESS_M_S:.0f} mm/s)"
            ),
            "n_press": n_press,
            "n_contact": n_contact,
            "window_s": 0.0,
        }

    def _window_s(start: int, end: int) -> float:
        if end <= start:
            return 0.0
        t0 = float(t[start]) if start < t.size and np.isfinite(t[start]) else float("nan")
        t1 = float(t[end - 1]) if end - 1 < t.size and np.isfinite(t[end - 1]) else float("nan")
        if math.isfinite(t0) and math.isfinite(t1) and t1 >= t0:
            return float(t1 - t0)
        dts = dt[start:end]
        finite = dts[np.isfinite(dts)]
        if finite.size:
            return float(np.nansum(finite))
        return float(end - start) * 0.005

    windows = []
    for start, end in _bool_segments(press):
        sl = slice(start, end)
        e_w = e[sl]
        if e_w.size < 2:
            continue
        risen = float(e_w[-1] - e_w[0])
        frac_up = float(np.mean(np.diff(e_w) > 0.0))
        clamp_frac = (
            float(np.mean(clamped[sl] > 0.5))
            if np.any(np.isfinite(clamped[sl]))
            else float("nan")
        )
        windows.append(
            {
                "start": start,
                "end": end,
                "window_s": _window_s(start, end),
                "n": int(end - start),
                "e_obs_press_start_j": float(e_w[0]),
                "e_obs_press_end_j": float(e_w[-1]),
                "e_obs_press_delta_j": risen,
                "e_obs_up_frac": frac_up,
                "alpha_p95": (
                    float(np.nanquantile(alpha[sl], 0.95))
                    if np.any(np.isfinite(alpha[sl]))
                    else float("nan")
                ),
                "clamp_frac_press": clamp_frac,
            }
        )
    if not windows:
        return {
            "ok": False,
            "sign_ok": False,
            "reason": "press ticks are isolated (no continuous window)",
            "n_press": n_press,
            "n_contact": n_contact,
            "window_s": 0.0,
        }
    best = max(windows, key=lambda row: row["window_s"])
    risen = float(best["e_obs_press_delta_j"])
    frac_up = float(best["e_obs_up_frac"])
    clamp_frac = float(best["clamp_frac_press"])
    window_s = float(best["window_s"])
    sign_ok = risen > 0.0 and frac_up >= 0.55
    long_enough = window_s + 1e-9 >= TDPA_MIN_WINDOW_S
    # One commanded open-loop ramp is the sign check.  The 1 s floor is
    # only to reject hybrid bounce fragments, not a second plant ID.
    ok = bool(sign_ok and (open_loop or long_enough))
    if not sign_ok:
        reason = (
            "E_obs fell inside the longest press window "
            "(sign or port pairing is wrong)"
        )
    elif open_loop and not long_enough:
        reason = (
            f"E_obs rose on a {window_s:.2f}s open-loop press "
            "(sign ok; stroke ended before 1 s)"
        )
    elif not long_enough:
        reason = (
            f"E_obs rose on a {window_s:.2f}s window "
            f"(need ≥{TDPA_MIN_WINDOW_S:.1f}s continuous open-loop press; "
            "hybrid F* tracking is the wrong excitation)"
        )
    else:
        reason = f"E_obs rose on a {window_s:.2f}s continuous press"
    return {
        "ok": ok,
        "sign_ok": bool(sign_ok),
        "open_loop": bool(open_loop),
        "reason": reason,
        "n_press": n_press,
        "n_contact": n_contact,
        "n_windows": len(windows),
        "window_s": window_s,
        "e_obs_press_start_j": float(best["e_obs_press_start_j"]),
        "e_obs_press_end_j": float(best["e_obs_press_end_j"]),
        "e_obs_press_delta_j": risen,
        "e_obs_up_frac": frac_up,
        "alpha_p95": float(best["alpha_p95"]),
        "clamp_frac_press": clamp_frac,
        "passivity_claimed": bool(
            ok and (not math.isfinite(clamp_frac) or clamp_frac < 0.05)
        ),
    }


def fit_contact_press(path: Path) -> dict:
    """Contact stiffness from one open-loop ``--tdpa-press`` CSV."""
    rows = load_tool_z_rows(path)
    if not rows:
        raise ValueError(f"{path} is empty")
    sign = analyze_tdpa_contact(path)
    f = _col(rows, "fz", "fz_raw_comp")
    v = _col(rows, "v_cmd_z", "vel_ff_vz", "v_force_cmd_z")
    dt = _col(rows, "dt_actual_s")
    phases = np.array([str(row.get("phase") or "") for row in rows])
    press = np.array(
        [p.startswith("tdpa_press") or p == "press" for p in phases],
        dtype=bool,
    )
    if not np.any(press):
        contact = _tdpa_contact_mask(rows, f)
        press = (
            (contact > 0.5)
            & np.isfinite(f)
            & np.isfinite(v)
            & (f > TDPA_CONTACT_N)
            & (v > TDPA_V_PRESS_M_S)
        )
    if int(np.count_nonzero(press)) < 2:
        raise ValueError(f"{path} has no press ticks")
    fp = f[press]
    vp = v[press]
    dtp = dt[press]
    dtp = np.where(np.isfinite(dtp) & (dtp > 0.0), dtp, 0.005)
    dx = np.cumsum(vp * dtp)
    window_s = float(np.sum(dtp))
    dF = float(fp[-1] - fp[0])
    dx_end = float(dx[-1])
    ke = float(dF / dx_end) if abs(dx_end) > 1e-6 else float("nan")
    settle = window_s > 0.20
    if settle:
        t_acc = np.cumsum(dtp)
        tail = t_acc >= 0.10
        if int(np.count_nonzero(tail)) >= 4:
            dF_t = float(fp[tail][-1] - fp[tail][0])
            dx_t = float(dx[tail][-1] - dx[tail][0])
            ke_settled = float(dF_t / dx_t) if abs(dx_t) > 1e-6 else float("nan")
        else:
            ke_settled = ke
    else:
        ke_settled = ke
    return {
        "path": str(path),
        "n_press": int(np.count_nonzero(press)),
        "window_s": window_s,
        "f0_n": float(fp[0]),
        "f1_n": float(fp[-1]),
        "dF_n": dF,
        "dx_mm": 1e3 * dx_end,
        "v_cmd_mm_s": float(1e3 * np.nanmedian(vp)),
        "ke_n_m": ke,
        "ke_settled_n_m": ke_settled,
        "sign_ok": bool(sign.get("sign_ok")),
        "passivity_claimed": bool(sign.get("passivity_claimed")),
        "e_obs_press_delta_j": float(sign.get("e_obs_press_delta_j", float("nan"))),
        "e_obs_up_frac": float(sign.get("e_obs_up_frac", float("nan"))),
        "alpha_p95": float(sign.get("alpha_p95", float("nan"))),
    }


def write_identification_report(
    *,
    air_dir: Path,
    press_paths: list[Path],
    out_dir: Path,
) -> dict:
    """Bundle air FOPDT + contact Ke + TDPA sign.  Does not refit chirps."""
    air_dir = Path(air_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    chirp_path = air_dir / "chirp_fits.csv"
    jitter_path = air_dir / "jitter.json"
    if not chirp_path.is_file() or not jitter_path.is_file():
        raise FileNotFoundError(
            f"{air_dir} is not a finished air campaign (need chirp_fits.csv, jitter.json)"
        )
    chirps: list[dict[str, float]] = []
    with chirp_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            chirps.append({key: float(row[key]) for key in row})
    jitter = json.loads(jitter_path.read_text())
    t0s = [c["t0_s"] for c in chirps if math.isfinite(c["t0_s"])]
    tps = [c["tp_s"] for c in chirps if math.isfinite(c["tp_s"])]
    gds = [c["group_delay_3hz_s"] for c in chirps if math.isfinite(c["group_delay_3hz_s"])]
    t0 = float(np.median(t0s)) if t0s else float("nan")
    tp = float(np.median(tps)) if tps else float("nan")
    t0_spread = float(max(t0s) - min(t0s)) if t0s else float("nan")
    gd_spread = float(max(gds) - min(gds)) if gds else float("nan")
    write_t0 = bool(math.isfinite(t0_spread) and t0_spread < T0_SPREAD_WARN_S)
    contacts = [fit_contact_press(Path(p)) for p in press_paths]
    report = {
        "air_dir": str(air_dir),
        "plant": {
            "t0_s": t0,
            "tp_s": tp,
            "t0_spread_s": t0_spread,
            "group_delay_3hz_s": gds,
            "group_delay_3hz_spread_s": gd_spread,
            "write_single_fopdt": write_t0,
            "do_not_use_group_delay_as_t0": True,
            "td_is_band": bool(jitter.get("td_is_band")),
            "feedback_age_p95_s": float(
                (jitter.get("jitter") or {}).get("feedback_age", {}).get("p95", float("nan"))
            ),
            "linear_speed_mm_s": float(jitter.get("linear_speed_mm_s", float("nan"))),
            "first_touch_m_s": 0.010,
            "note": (
                "Chirp T0 is the delay to write.  3 Hz group delay disagrees "
                "across amplitudes — not a second FOPDT.  Age p95>5 ms so Td "
                "is a band around T0, not a point.  Force-loop FRF not identified."
            ),
        },
        "tdpa_sign": {
            "ok": all(c["sign_ok"] for c in contacts) if contacts else False,
            "n_surfaces": len(contacts),
            "passivity_claimed": all(c["passivity_claimed"] for c in contacts)
            if contacts
            else False,
        },
        "contact": contacts,
        "not_identified": [
            "force-loop F→v FRF (hybrid tracking still the wrong excitation)",
            "single tissue Ke (these are fixture stiffnesses)",
            "stop-reverse Δx_b^ub",
        ],
    }
    (out_dir / "identification.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    if contacts:
        with (out_dir / "contact_ke.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(contacts[0].keys()))
            writer.writeheader()
            writer.writerows(contacts)
        _plot_contact_presses(press_paths, out_dir)
    write_controller_ref(report, out_dir)
    return report


def write_controller_ref(report: dict, out_dir: Path) -> None:
    """Local CSV+LOG the controller can re-read.  Does not patch yaml."""
    plant = report.get("plant") or {}
    contacts = list(report.get("contact") or [])
    ke_vals = [
        float(c["ke_settled_n_m"])
        for c in contacts
        if math.isfinite(float(c.get("ke_settled_n_m", float("nan"))))
    ]
    t0 = float(plant.get("t0_s", float("nan")))
    tp = float(plant.get("tp_s", float("nan")))
    age_p95 = float(plant.get("feedback_age_p95_s", float("nan")))
    linear = float(plant.get("linear_speed_mm_s", float("nan")))
    rows = [
        {
            "key": "plant.t0_s",
            "identified": f"{t0:.6f}",
            "unit": "s",
            "yaml_now": "shield.plant.t0_s=0.028; system_delay_s=0.055; barrier.t_react_s=0.055; cdyob.t0_s=0.028",
            "consume": "use T0=0.028 as the delay centre; do not bake 0.055 into certificates",
            "note": "Td is a band (age p95).  55 ms is a placeholder aligned to t_react.",
        },
        {
            "key": "plant.tp_s",
            "identified": f"{tp:.6f}",
            "unit": "s",
            "yaml_now": "shield.plant.tp_s=0.014; cdyob.tp_s=0.014",
            "consume": "0.014",
            "note": "chirp median",
        },
        {
            "key": "plant.td_band_p95_s",
            "identified": f"{age_p95:.6f}",
            "unit": "s",
            "yaml_now": "not a yaml key",
            "consume": "treat Td as [T0, T0+p95], never a single FOPDT from 3 Hz group delay",
            "note": "group_delay_3hz disagreed 65/49/20 ms",
        },
        {
            "key": "plant.linear_speed_m_s",
            "identified": f"{linear / 1000.0:.6f}" if math.isfinite(linear) else "",
            "unit": "m/s",
            "yaml_now": "max_force_axis=0.025; labeled linear ≤40 mm/s but +20 already ~2.4 m/s²",
            "consume": "force-axis saturation 0.025; first-touch 0.010",
            "note": "do not command 80 mm/s on the force axis",
        },
        {
            "key": "press.first_touch_m_s",
            "identified": "0.010",
            "unit": "m/s",
            "yaml_now": "0.010",
            "consume": "air seek + first/recontact latch only",
            "note": "unknown Ke.  8–12 mm/s from delay×Ke first-touch.",
        },
        {
            "key": "press.soft_approach_m_s",
            "identified": "0.020",
            "unit": "m/s",
            "yaml_now": "0.020",
            "consume": "confirmed-contact underforce chase (tissue)",
            "note": "not an air seek",
        },
        {
            "key": "press.max_force_axis_m_s",
            "identified": "0.025",
            "unit": "m/s",
            "yaml_now": "0.025",
            "consume": "linear press sat only; retract escape is u_retract (0.080)",
            "note": "leaving tissue does not re-indent through Td",
        },
        {
            "key": "press.v_delay_safe_m_s",
            "identified": "approx 0.0028 from (F_max-F_enter-ΔF)/(K_ub T_stop)",
            "unit": "m/s",
            "yaml_now": "runtime; recontact_vz_cap=0.012 clips the formula",
            "consume": "first/recontact latch press only",
            "note": "K_ub=8000 engineering start, not a proven 3 N guarantee",
        },
        {
            "key": "tdpa.sign",
            "identified": str(bool((report.get("tdpa_sign") or {}).get("ok"))),
            "unit": "",
            "yaml_now": "tdpa.enabled=true",
            "consume": "port is F_meas × v_cmd same tick",
            "note": "open-loop press only; α clamp means passivity not claimed",
        },
        {
            "key": "contact.ke_settled_n_m",
            "identified": (
                f"{min(ke_vals):.1f}–{max(ke_vals):.1f}" if ke_vals else ""
            ),
            "unit": "N/m",
            "yaml_now": "ke_initial=80; ke_cap_ub=10000; k_ub=8000",
            "consume": "feel barrier uses K̂e; K_ub only sizes first-touch press speed",
            "note": "fixture pads, not a single tissue Ke.  Hybrid ke_est often sits near the floor.",
        },
        {
            "key": "not.force_loop_frf",
            "identified": "missing",
            "unit": "",
            "yaml_now": "cdyob.mode=off",
            "consume": "do not turn CDYOB on",
            "note": "hybrid tracking is the wrong excitation until chatter stops",
        },
        {
            "key": "not.stop_dx_ub",
            "identified": "missing",
            "unit": "",
            "yaml_now": "stop_dx certified=false",
            "consume": "do not set certified true",
            "note": "shield stays observe",
        },
    ]
    out_dir = Path(out_dir)
    csv_path = out_dir / "controller_ref.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["key", "identified", "unit", "yaml_now", "consume", "note"],
        )
        writer.writeheader()
        writer.writerows(rows)
    ke_txt = (
        ", ".join(f"{v:.0f}" for v in ke_vals) + " N/m (fixture)"
        if ke_vals
        else "no --tdpa-press yet"
    )
    log = (
        f"id_reference  air={report.get('air_dir', '')}\n"
        f"plant  T0={1e3 * t0:.1f} ms  Tp={1e3 * tp:.1f} ms  "
        f"Td_band_p95={1e3 * age_p95:.1f} ms  linear_label={linear:.0f} mm/s\n"
        f"do_not_write  system_delay_s=0.055 as if it were T0; "
        f"3 Hz group delay; CDYOB t0=30 ms\n"
        f"envelope  air/first_touch=10 mm/s  "
        f"confirmed_chase=soft_approach 20 mm/s  "
        f"overforce_retract=max_force_axis 25 mm/s  "
        f"recontact_press=v_delay_safe ~2.8 mm/s\n"
        f"tdpa_sign  {bool((report.get('tdpa_sign') or {}).get('ok'))}  "
        f"surfaces={int((report.get('tdpa_sign') or {}).get('n_surfaces') or 0)}  "
        f"ke_settled={ke_txt}\n"
        f"missing  force-loop F→v FRF; single tissue Ke; stop-reverse Δx_b^ub\n"
        f"certificates  Nyquist=filter+CDYOB off  "
        f"TDPA=enabled (clamp ⇒ no passivity claim)  "
        f"corridor=enabled shield=observe\n"
    )
    (out_dir / "id_reference.log").write_text(log)


def _plot_contact_presses(press_paths: list[Path], out_dir: Path) -> None:
    plt = _require_mpl()
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    for path in press_paths:
        rows = load_tool_z_rows(Path(path))
        f = _col(rows, "fz")
        v = _col(rows, "v_cmd_z", "vel_ff_vz")
        e = _col(rows, "tdpa_e_obs_j")
        dt = _col(rows, "dt_actual_s")
        phases = np.array([str(row.get("phase") or "") for row in rows])
        press = np.array([p.startswith("tdpa_press") for p in phases], dtype=bool)
        if not np.any(press):
            continue
        dtp = np.where(np.isfinite(dt[press]) & (dt[press] > 0.0), dt[press], 0.005)
        x_mm = 1e3 * np.cumsum(v[press] * dtp)
        t_s = np.cumsum(dtp)
        label = Path(path).parent.name
        axes[0].plot(x_mm, f[press], label=label)
        axes[1].plot(t_s, e[press], label=label)
    axes[0].set_xlabel("commanded indentation (mm)")
    axes[0].set_ylabel("Fz (N)")
    axes[0].set_title("contact F vs x")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("press time (s)")
    axes[1].set_ylabel("E_obs (J)")
    axes[1].set_title("TDPA energy on press")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    _save(fig, out_dir, "08_contact_press.png")
    plt.close(fig)


def linear_speed_mm_s(edges: list[EdgeRow], *, accel_cap: float = LINEAR_ACCEL_M_S2) -> float:
    linear = [
        e.cmd_mm_s
        for e in edges
        if math.isfinite(e.peak_accel_m_s2) and e.peak_accel_m_s2 < accel_cap
    ]
    return float(max(linear)) if linear else float("nan")


def _require_mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save(fig, out_dir: Path, name: str) -> None:
    png = out_dir / name
    svg = out_dir / name.replace(".png", ".svg")
    fig.savefig(png, dpi=140, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")


def plot_campaign(
    result: AirCampaignResult,
    out_dir: Path,
    *,
    sweep_f1_hz: float = AIR_CHIRP_F1_HZ,
) -> None:
    plt = _require_mpl()
    out_dir.mkdir(parents=True, exist_ok=True)
    edges = result.edges
    pos = [e for e in edges if e.sign >= 0.0]
    neg = [e for e in edges if e.sign < 0.0]
    v_lin = result.linear_speed_mm_s

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for group, marker, label in ((pos, "o", "+cmd"), (neg, "s", "-cmd")):
        ax.plot(
            [e.peak_accel_m_s2 for e in group],
            [1e3 * e.delay_s for e in group],
            marker,
            label=label,
        )
    ax.axvline(LINEAR_ACCEL_M_S2, color="0.4", ls="--", label="linear accel cap")
    ax.set_xlabel("peak |dv/dt| (m/s²)")
    ax.set_ylabel("xcorr delay (ms)")
    ax.set_title("01 delay vs acceleration")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, out_dir, "01_delay_vs_accel.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for group, marker, label in ((pos, "o", "+cmd"), (neg, "s", "-cmd")):
        ax.plot(
            [e.cmd_mm_s for e in group],
            [1e3 * e.delay_s for e in group],
            marker,
            label=label,
        )
    if math.isfinite(v_lin):
        ax.axvline(v_lin, color="0.2", ls="--", label=f"linear ≤ {v_lin:.0f} mm/s")
    ax.set_xlabel("command speed (mm/s)")
    ax.set_ylabel("xcorr delay (ms)")
    ax.set_title("02 delay vs speed")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, out_dir, "02_delay_vs_speed.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(
        [e.peak_accel_m_s2 for e in edges],
        [1e3 * e.rise_s for e in edges],
        "o",
    )
    ax.axvline(LINEAR_ACCEL_M_S2, color="0.4", ls="--")
    ax.set_xlabel("peak |dv/dt| (m/s²)")
    ax.set_ylabel("t10–t90 (ms)")
    ax.set_title("03 rise vs acceleration")
    ax.grid(True, alpha=0.3)
    _save(fig, out_dir, "03_rise_vs_accel.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot([e.cmd_mm_s for e in edges], [e.kp for e in edges], "o")
    ax.axhline(0.892, color="0.4", ls="--", label="yaml K=0.892")
    ax.set_xlabel("command speed (mm/s)")
    ax.set_ylabel("settled Kp")
    ax.set_title("04 Kp vs speed")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, out_dir, "04_kp_vs_speed.png")
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.2), sharex=True)
    for fit in result.chirps:
        axes[0].semilogx(fit.freq_hz, 20.0 * np.log10(np.maximum(fit.mag, 1e-6)), label=f"{fit.amp_mm_s:.0f} mm/s")
        axes[1].semilogx(fit.freq_hz, np.degrees(fit.phase_rad))
        axes[2].semilogx(fit.freq_hz, fit.coherence)
    axes[2].axhline(COHERENCE_MIN, color="0.4", ls="--")
    axes[0].set_ylabel("|G| (dB)")
    axes[1].set_ylabel("phase (deg)")
    axes[2].set_ylabel("coherence γ²")
    axes[2].set_xlabel("frequency (Hz)")
    axes[0].legend(title="velocity chirp amplitude")
    f1 = max(float(sweep_f1_hz), 1.0)
    xmax = max(9.0, f1 + 1.0)
    ticks = [0.2, 0.5, 1.0, 2.0, 5.0, 8.0]
    if f1 > 8.01:
        ticks.append(10.0 if abs(f1 - 10.0) < 0.2 else f1)
    for ax in axes:
        ax.grid(True, which="both", alpha=0.3)
        ax.set_xlim(0.15, xmax)
        ax.axvline(8.0, color="0.35", ls="--", lw=1)
        if f1 > 8.01:
            ax.axvline(f1, color="0.55", ls=":", lw=1)
    axes[2].set_xticks(ticks)
    axes[2].set_xticklabels(
        ["0.2", "0.5", "1", "2", "5", "8"]
        + ([f"{ticks[-1]:.0f}"] if len(ticks) > 6 else [])
    )
    _save(fig, out_dir, "05_chirp_bode.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(
        [c.amp_mm_s for c in result.chirps],
        [1e3 * c.group_delay_3hz_s for c in result.chirps],
        "o-",
    )
    ax.set_xlabel("chirp amplitude (mm/s)")
    ax.set_ylabel("3 Hz group delay (ms)")
    ax.set_title("06 group delay at 3 Hz")
    ax.grid(True, alpha=0.3)
    _save(fig, out_dir, "06_group_delay_3hz.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.4))
    jitter = result.jitter
    if jitter is not None:
        for ax, title, stats in (
            (axes[0], "feedback_age_s", jitter.feedback_age),
            (axes[1], "sensor_age_s", jitter.sensor_age),
            (axes[2], "dt_actual_s", jitter.dt_actual),
        ):
            ax.bar(["mean", "p95", "max"], [stats["mean"], stats["p95"], stats["max"]])
            ax.axhline(0.005, color="0.4", ls="--", label="1 tick")
            ax.set_title(title)
            ax.set_ylabel("s")
            ax.legend()
    fig.suptitle("07 state-stream jitter")
    fig.tight_layout()
    _save(fig, out_dir, "07_jitter.png")
    plt.close(fig)


def write_tables(result: AirCampaignResult, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "edges.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "cmd_mm_s",
                "sign",
                "t_edge_s",
                "delay_s",
                "rise_s",
                "peak_accel_m_s2",
                "kp",
            ],
        )
        writer.writeheader()
        for edge in result.edges:
            writer.writerow(
                {
                    "cmd_mm_s": f"{edge.cmd_mm_s:.4f}",
                    "sign": f"{edge.sign:.0f}",
                    "t_edge_s": f"{edge.t_edge_s:.6f}",
                    "delay_s": f"{edge.delay_s:.6f}",
                    "rise_s": f"{edge.rise_s:.6f}",
                    "peak_accel_m_s2": f"{edge.peak_accel_m_s2:.6f}",
                    "kp": f"{edge.kp:.6f}",
                }
            )
    with (out_dir / "chirp_fits.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["amp_mm_s", "t0_s", "tp_s", "gain", "group_delay_3hz_s"],
        )
        writer.writeheader()
        for fit in result.chirps:
            writer.writerow(
                {
                    "amp_mm_s": f"{fit.amp_mm_s:.3f}",
                    "t0_s": f"{fit.t0_s:.6f}",
                    "tp_s": f"{fit.tp_s:.6f}",
                    "gain": f"{fit.gain:.6f}",
                    "group_delay_3hz_s": f"{fit.group_delay_3hz_s:.6f}",
                }
            )
    payload = {
        "linear_speed_mm_s": result.linear_speed_mm_s,
        "t0_spread_s": result.t0_spread_s,
        "write_yaml_allowed": bool(
            math.isfinite(result.t0_spread_s) and result.t0_spread_s < T0_SPREAD_WARN_S
        ),
        "td_is_band": bool(
            result.jitter is not None
            and math.isfinite(result.jitter.feedback_age.get("p95", float("nan")))
            and result.jitter.feedback_age["p95"] > 0.005
        ),
        "jitter": None
        if result.jitter is None
        else {
            "n": result.jitter.n,
            "feedback_age": result.jitter.feedback_age,
            "sensor_age": result.jitter.sensor_age,
            "dt_actual": result.jitter.dt_actual,
        },
    }
    (out_dir / "jitter.json").write_text(json.dumps(payload, indent=2) + "\n")


def assert_plots_complete(out_dir: Path) -> None:
    missing = [name for name in REQUIRED_PLOTS if not (out_dir / name).is_file()]
    missing += [
        name.replace(".png", ".svg")
        for name in REQUIRED_PLOTS
        if not (out_dir / name.replace(".png", ".svg")).is_file()
    ]
    tables = ["edges.csv", "chirp_fits.csv", "jitter.json"]
    # tdpa_shadow.json is written after write_tables; do not require it here.
    missing += [name for name in tables if not (out_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "air identification incomplete; missing " + ", ".join(missing)
        )


def analyze_air_paths(
    paths: list[Path],
    *,
    out_dir: Path,
    chirp_amps_mm_s: tuple[float, ...] = AIR_CHIRP_AMPS_MM_S,
    chirp_f0_hz: float = AIR_CHIRP_F0_HZ,
    chirp_f1_hz: float = AIR_CHIRP_F1_HZ,
) -> AirCampaignResult:
    result = AirCampaignResult(out_dir=out_dir)
    fb_all: list[np.ndarray] = []
    sn_all: list[np.ndarray] = []
    dt_all: list[np.ndarray] = []
    f0 = max(float(chirp_f0_hz), 1e-3)
    f1 = max(float(chirp_f1_hz), f0 + 1e-3)
    for path in paths:
        t, u, y, fb, sn, dt = load_tool_z_arrays(path)
        result.edges.extend(detect_step_edges(t, u, y))
        for amp, sl in split_chirp_windows(t, u, amps_mm_s=chirp_amps_mm_s):
            result.chirps.append(
                fit_chirp_segment(
                    t[sl],
                    u[sl],
                    y[sl],
                    amp_mm_s=amp,
                    f0=f0,
                    f1=f1,
                )
            )
        fb_all.append(fb)
        sn_all.append(sn)
        dt_all.append(dt)
    result.jitter = jitter_from_columns(
        np.concatenate(fb_all) if fb_all else np.zeros(0),
        np.concatenate(sn_all) if sn_all else np.zeros(0),
        np.concatenate(dt_all) if dt_all else np.zeros(0),
    )
    result.linear_speed_mm_s = linear_speed_mm_s(result.edges)
    if len(result.chirps) >= 2:
        t0s = [c.t0_s for c in result.chirps if math.isfinite(c.t0_s)]
        result.t0_spread_s = float(max(t0s) - min(t0s)) if t0s else float("nan")
    write_tables(result, out_dir)
    plot_campaign(result, out_dir, sweep_f1_hz=f1)
    tdpa = shadow_tdpa_from_rows(load_tool_z_rows(paths[0]))
    (out_dir / "tdpa_shadow.json").write_text(json.dumps(tdpa, indent=2) + "\n")
    assert_plots_complete(out_dir)
    return result


def synthesize_air_csv(
    path: Path,
    *,
    chirp_s: float = AIR_CHIRP_S,
    steps_mm_s: tuple[float, ...] = AIR_STEPS_MM_S,
) -> None:
    """Unit-test plant: FOPDT that saturates above ~25 mm/s."""
    from peirastic.apps.identify_plant import _simulate_fopdt_k

    dt = 0.005
    rows: list[dict[str, str]] = []
    t = 0.0
    chirp_s = max(float(chirp_s), 4.0)

    def push(cmd: float, n: int, *, t0: float, tp: float, k: float) -> None:
        nonlocal t
        u = np.full(n, cmd, dtype=float)
        y = _simulate_fopdt_k(u, delay_steps=max(int(round(t0 / dt)), 0), tp_s=tp, gain=k, dt=dt)
        for i in range(n):
            t += dt
            rows.append(
                {
                    "t_wall_s": f"{t:.6f}",
                    "t_mono_s": f"{t:.6f}",
                    "phase": "synth_step",
                    "vel_ff_vz": f"{cmd:.6f}",
                    "v_cmd_z": f"{cmd:.6f}",
                    "vz_achieved_tool": f"{float(y[i]):.6f}",
                    "fz": "0.700000",
                    "feedback_age_s": "0.003000",
                    "sensor_age_s": "0.002800",
                    "dt_actual_s": f"{dt:.6f}",
                    "a_tcp_z_plus": "0.000000",
                    "motion_seq": str(len(rows)),
                    "tdpa_e_obs_j": "",
                    "tdpa_alpha": "",
                    "tdpa_clamped": "0",
                }
            )

    for mm in steps_mm_s:
        cmd = mm / 1000.0
        sat = mm >= 40.0
        t0 = 0.050 if sat else 0.030
        tp = 0.026 if sat else 0.012
        k = 0.85 if sat else 0.892
        for sign in (1.0, -1.0):
            push(0.0, int(AIR_REST_S / dt), t0=t0, tp=tp, k=k)
            push(sign * cmd, int(AIR_HOLD_S / dt), t0=t0, tp=tp, k=k)
        push(0.0, int(AIR_REST_S / dt), t0=t0, tp=tp, k=k)

    for amp in AIR_CHIRP_AMPS_MM_S:
        n = int(chirp_s / dt)
        f0, f1 = AIR_CHIRP_F0_HZ, AIR_CHIRP_F1_HZ
        kk = math.log(f1 / f0) / chirp_s
        u = np.zeros(n)
        for i in range(n):
            phase = 2.0 * math.pi * f0 * (math.exp(kk * i * dt) - 1.0) / kk
            u[i] = (amp / 1000.0) * math.sin(phase)
        sat = amp >= 25.0
        t0 = 0.042 if sat else 0.030
        y = _simulate_fopdt_k(
            u,
            delay_steps=max(int(round(t0 / dt)), 0),
            tp_s=0.014 if sat else 0.012,
            gain=0.89,
            dt=dt,
        )
        for i in range(n):
            t += dt
            rows.append(
                {
                    "t_wall_s": f"{t:.6f}",
                    "t_mono_s": f"{t:.6f}",
                    "phase": f"synth_chirp_{amp:.0f}",
                    "vel_ff_vz": f"{float(u[i]):.6f}",
                    "v_cmd_z": f"{float(u[i]):.6f}",
                    "vz_achieved_tool": f"{float(y[i]):.6f}",
                    "fz": "0.700000",
                    "feedback_age_s": "0.003100",
                    "sensor_age_s": "0.002900",
                    "dt_actual_s": f"{dt:.6f}",
                    "a_tcp_z_plus": "0.000000",
                    "motion_seq": str(len(rows)),
                    "tdpa_e_obs_j": "",
                    "tdpa_alpha": "",
                    "tdpa_clamped": "0",
                }
            )
        push(0.0, int(AIR_REST_S / dt), t0=0.030, tp=0.012, k=0.892)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: float, digits: int = 6) -> str:
    if not math.isfinite(float(value)):
        return ""
    return f"{float(value):.{digits}f}"


def run_air_campaign(
    *,
    prefix: str,
    hold_s: float,
    rest_s: float,
    chirp_s: float,
    chirp_amps_m_s: tuple[float, ...],
    steps_mm_s: tuple[float, ...],
    hz: float,
    log_csv: Path,
    out_dir: Path,
    chirp_f0_hz: float = AIR_CHIRP_F0_HZ,
    chirp_f1_hz: float = AIR_CHIRP_F1_HZ,
) -> int:
    """Drive SERVO_TWIST and write the campaign CSV.  Window A only servos."""
    import time

    from peirastic.core.ipc import CommandClient, MotionBus, Status, TwistBus
    from peirastic.core.modes import Mode, ModeRequest
    from rm75_control.control.admittance_common.tdpa import (
        TdpaConfig,
        TimeDomainPassivityObserver,
    )

    dt_nom = 1.0 / max(float(hz), 1.0)
    client = CommandClient(prefix=prefix)
    bus = TwistBus(prefix=prefix, create=False)
    try:
        motion = MotionBus(prefix=prefix, create=False)
    except Exception as exc:
        bus.close()
        client.close()
        print(
            f"[ERR] peirastic_motion SHM missing ({exc}).  "
            "Window A must be running this build.  "
            "Air campaign will not fall back to the Window A CSV.",
            flush=True,
        )
        return 2

    log_csv = Path(log_csv)
    out_dir = Path(out_dir)
    log_csv.parent.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    handle = log_csv.open("w", newline="")
    writer = csv.DictWriter(handle, fieldnames=list(AIR_LOG_FIELDS))
    writer.writeheader()
    obs = TimeDomainPassivityObserver(TdpaConfig(enabled=True))
    last_wall = float("nan")
    n_rows = 0

    def _tick(vz_m_s: float, phase: str) -> bool:
        nonlocal last_wall, n_rows
        tw = np.zeros(6, dtype=float)
        tw[2] = float(vz_m_s)
        bus.write(tw, hz=hz, connected=True)
        tel = client.snapshot()
        if int(tel["status"]) == int(Status.ESTOP):
            print("[ESTOP] " + str(tel["msg"]), flush=True)
            return False
        row_m = motion.read()
        t_wall = float(row_m.get("t_wall_s", float("nan")))
        if not math.isfinite(t_wall):
            t_wall = time.time()
        if math.isfinite(last_wall):
            dt_act = max(t_wall - last_wall, 1e-6)
        else:
            dt_act = dt_nom
        last_wall = t_wall
        fz = float(tel.get("f_ext_z", float("nan")))
        v_cmd = float(vz_m_s)
        y = float(row_m.get("v_tcp_z", float("nan")))
        age = float(row_m.get("feedback_age_s", float("nan")))
        if math.isfinite(fz) and math.isfinite(v_cmd):
            obs.preview(fz, v_cmd, dt_act)
            obs.commit(fz, v_cmd, dt_act, in_contact=False)
        writer.writerow(
            {
                "t_wall_s": _fmt(t_wall),
                "t_mono_s": _fmt(time.monotonic()),
                "dt_actual_s": _fmt(dt_act),
                "phase": phase,
                "vel_ff_vz": _fmt(v_cmd),
                "v_cmd_z": _fmt(v_cmd),
                "vz_achieved_tool": _fmt(y),
                "fz": _fmt(fz),
                "feedback_age_s": _fmt(age),
                "sensor_age_s": _fmt(age),
                "a_tcp_z_plus": _fmt(float(row_m.get("a_tcp_z_plus", float("nan")))),
                "motion_seq": str(int(row_m.get("seq", 0))),
                "tdpa_e_obs_j": _fmt(obs.e_obs_j),
                "tdpa_alpha": _fmt(obs.alpha),
                "tdpa_clamped": "1" if obs.alpha_clamped else "0",
            }
        )
        n_rows += 1
        if n_rows % 200 == 0:
            handle.flush()
        return True

    def _hold(vz_m_s: float, seconds: float, phase: str) -> bool:
        t0 = time.monotonic()
        while time.monotonic() - t0 < seconds:
            if not _tick(vz_m_s, phase):
                return False
            time.sleep(dt_nom)
        return True

    client.set_mode(
        ModeRequest(Mode.SERVO_TWIST, {"filter": False, "secondary": "payload_id"})
    )
    print(
        f"[MODE] SERVO_TWIST identify_air  filter OFF  secondary=payload_id  log={log_csv}  "
        "force loop stays off; Window A is servo only",
        flush=True,
    )
    try:
        if not _hold(0.0, rest_s, "rest"):
            return 130
        for mm_s in steps_mm_s:
            vz = float(mm_s) / 1000.0
            for sign in (1.0, -1.0):
                cmd = sign * vz
                print(f"[STEP] vz={cmd:+.4f} m/s hold={hold_s:.2f}s", flush=True)
                if not _hold(cmd, hold_s, f"step_{mm_s:.0f}"):
                    return 130
                if not _hold(0.0, rest_s, "rest"):
                    return 130
        f0 = max(float(chirp_f0_hz), 1e-3)
        f1 = max(float(chirp_f1_hz), f0 + 1e-3)
        for amp in chirp_amps_m_s:
            print(
                f"[CHIRP] {amp*1000:.1f} mm/s  {f0:.2f}–{f1:.1f} Hz  {chirp_s:.1f}s",
                flush=True,
            )
            t0 = time.monotonic()
            k = math.log(f1 / f0) / max(float(chirp_s), 1e-6)
            while True:
                t = time.monotonic() - t0
                if t >= float(chirp_s):
                    break
                phase = 2.0 * math.pi * f0 * (math.exp(k * t) - 1.0) / k
                if not _tick(float(amp) * math.sin(phase), f"chirp_{amp*1000:.0f}"):
                    return 130
                time.sleep(dt_nom)
            if not _hold(0.0, rest_s, "rest"):
                return 130
        _tick(0.0, "done")
        handle.flush()
        print(f"[OK] air campaign wrote {n_rows} rows → {log_csv}", flush=True)
        result = analyze_air_paths(
            [log_csv],
            out_dir=out_dir,
            chirp_f0_hz=f0,
            chirp_f1_hz=f1,
        )
        print(
            f"[ID-AIR] out={out_dir} edges={len(result.edges)} "
            f"chirps={len(result.chirps)} "
            f"linear≤{result.linear_speed_mm_s:.1f} mm/s "
            f"T0_spread={1e3 * result.t0_spread_s:.1f} ms",
            flush=True,
        )
        if math.isfinite(result.t0_spread_s) and result.t0_spread_s > T0_SPREAD_WARN_S:
            print(
                "[ID-AIR] T0 spread > 8 ms: do not write a single FOPDT into yaml",
                flush=True,
            )
        return 0
    except KeyboardInterrupt:
        tw = np.zeros(6, dtype=float)
        bus.write(tw, hz=hz, connected=True)
        client.stop()
        print("[STOP] interrupted", flush=True)
        return 0
    finally:
        handle.close()
        bus.close()
        client.close()
        motion.close()


TDPA_PRESS_LOG_FIELDS = AIR_LOG_FIELDS + ("contact_present",)


def run_tdpa_press_campaign(
    *,
    prefix: str,
    hz: float,
    log_csv: Path,
    press_m_s: float = TDPA_PRESS_M_S,
    press_s: float = TDPA_PRESS_S,
    seek_m_s: float = TDPA_SEEK_M_S,
    contact_n: float = TDPA_CONTACT_N,
    target_n: float = TDPA_TARGET_N,
    abort_n: float = TDPA_ABORT_N,
    seek_max_s: float = TDPA_SEEK_MAX_S,
) -> int:
    """Open-loop seek + constant press.  Force loop stays off.

    This is the De Stefano §IV sign check.  Do not run TRACK_HYBRID.
    """
    import time

    from peirastic.core.ipc import CommandClient, MotionBus, Status, TwistBus
    from peirastic.core.modes import Mode, ModeRequest
    from rm75_control.control.admittance_common.tdpa import (
        TdpaConfig,
        TimeDomainPassivityObserver,
    )

    dt_nom = 1.0 / max(float(hz), 1.0)
    client = CommandClient(prefix=prefix)
    bus = TwistBus(prefix=prefix, create=False)
    try:
        motion = MotionBus(prefix=prefix, create=False)
    except Exception as exc:
        bus.close()
        client.close()
        print(
            f"[ERR] peirastic_motion SHM missing ({exc}).  "
            "Window A must be running this build.",
            flush=True,
        )
        return 2

    log_csv = Path(log_csv)
    log_csv.parent.mkdir(parents=True, exist_ok=True)
    handle = log_csv.open("w", newline="")
    writer = csv.DictWriter(handle, fieldnames=list(TDPA_PRESS_LOG_FIELDS))
    writer.writeheader()
    obs = TimeDomainPassivityObserver(TdpaConfig(enabled=True))
    last_wall = float("nan")
    n_rows = 0
    last_fz = float("nan")
    latched = False
    force_abort = False

    def _tick(vz_m_s: float, phase: str) -> bool:
        nonlocal last_wall, n_rows, last_fz, latched, force_abort
        tw = np.zeros(6, dtype=float)
        tw[2] = float(vz_m_s)
        bus.write(tw, hz=hz, connected=True)
        tel = client.snapshot()
        if int(tel["status"]) == int(Status.ESTOP):
            print("[ESTOP] " + str(tel["msg"]), flush=True)
            return False
        row_m = motion.read()
        t_wall = float(row_m.get("t_wall_s", float("nan")))
        if not math.isfinite(t_wall):
            t_wall = time.time()
        if math.isfinite(last_wall):
            dt_act = max(t_wall - last_wall, 1e-6)
        else:
            dt_act = dt_nom
        last_wall = t_wall
        fz = float(tel.get("f_ext_z", float("nan")))
        last_fz = fz
        v_cmd = float(vz_m_s)
        y = float(row_m.get("v_tcp_z", float("nan")))
        age = float(row_m.get("feedback_age_s", float("nan")))
        if math.isfinite(fz) and fz >= float(contact_n):
            latched = True
        in_contact = bool(latched)
        if math.isfinite(fz) and math.isfinite(v_cmd):
            obs.preview(fz, v_cmd, dt_act)
            obs.commit(fz, v_cmd, dt_act, in_contact=in_contact)
        writer.writerow(
            {
                "t_wall_s": _fmt(t_wall),
                "t_mono_s": _fmt(time.monotonic()),
                "dt_actual_s": _fmt(dt_act),
                "phase": phase,
                "vel_ff_vz": _fmt(v_cmd),
                "v_cmd_z": _fmt(v_cmd),
                "vz_achieved_tool": _fmt(y),
                "fz": _fmt(fz),
                "feedback_age_s": _fmt(age),
                "sensor_age_s": _fmt(age),
                "a_tcp_z_plus": _fmt(float(row_m.get("a_tcp_z_plus", float("nan")))),
                "motion_seq": str(int(row_m.get("seq", 0))),
                "tdpa_e_obs_j": _fmt(obs.e_obs_j),
                "tdpa_alpha": _fmt(obs.alpha),
                "tdpa_clamped": "1" if obs.alpha_clamped else "0",
                "contact_present": "1" if in_contact else "0",
            }
        )
        n_rows += 1
        if n_rows % 200 == 0:
            handle.flush()
        if math.isfinite(fz) and fz >= float(abort_n):
            force_abort = True
            print(f"[ABORT] Fz={fz:.2f} N ≥ {abort_n:.1f} N — stop motion, score", flush=True)
            return False
        return True

    def _zero() -> None:
        tw = np.zeros(6, dtype=float)
        bus.write(tw, hz=hz, connected=True)

    client.set_mode(
        ModeRequest(Mode.SERVO_TWIST, {"filter": False, "secondary": "payload_id"})
    )
    print(
        f"[MODE] SERVO_TWIST tdpa-press  filter OFF  secondary=payload_id  log={log_csv}  "
        "force loop OFF — do not start hybrid / F* tracking",
        flush=True,
    )
    print(
        f"[TDPA-PRESS] seek {1e3 * seek_m_s:.0f} mm/s until F>{contact_n:.1f} N "
        f"(max {seek_max_s:.0f}s), then hold {1e3 * press_m_s:.0f} mm/s for "
        f"{press_s:.1f}s (do not stop at {target_n:.1f} N)  "
        f"abort motion F>{abort_n:.1f} N then score",
        flush=True,
    )
    try:
        t_seek = time.monotonic()
        while time.monotonic() - t_seek < float(seek_max_s):
            if not _tick(float(seek_m_s), "tdpa_seek"):
                _zero()
                if n_rows > 0 and (latched or force_abort):
                    handle.flush()
                    print(f"[OK] tdpa-press wrote {n_rows} rows → {log_csv}", flush=True)
                    verdict = analyze_tdpa_contact(log_csv)
                    print(f"[TDPA] {verdict}", flush=True)
                    return 0 if verdict.get("ok") else 2
                return 130
            if latched:
                print(f"[TDPA-PRESS] contact Fz={last_fz:.2f} N — holding press", flush=True)
                break
            time.sleep(dt_nom)
        else:
            _zero()
            print(
                "[TDPA-PRESS] no contact.  Move the probe closer to the pad "
                "and rerun --tdpa-press.  Do not start hybrid.",
                flush=True,
            )
            handle.flush()
            return 2

        t_press = time.monotonic()
        while time.monotonic() - t_press < float(press_s) and not force_abort:
            if not _tick(float(press_m_s), "tdpa_press"):
                break
            time.sleep(dt_nom)
        if force_abort:
            print("[TDPA-PRESS] force abort — zeroing, then scoring the press", flush=True)
        elif math.isfinite(last_fz):
            print(f"[TDPA-PRESS] press hold done Fz={last_fz:.2f} N", flush=True)

        t_rest = time.monotonic()
        while time.monotonic() - t_rest < 0.40 and not force_abort:
            if not _tick(0.0, "tdpa_rest"):
                break
            time.sleep(dt_nom)
        _zero()
        handle.flush()
        print(f"[OK] tdpa-press wrote {n_rows} rows → {log_csv}", flush=True)
        verdict = analyze_tdpa_contact(log_csv)
        print(f"[TDPA] {verdict}", flush=True)
        return 0 if verdict.get("ok") else 2
    except KeyboardInterrupt:
        _zero()
        client.stop()
        print("[STOP] interrupted", flush=True)
        return 0
    finally:
        handle.close()
        bus.close()
        client.close()
        motion.close()
