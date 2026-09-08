#!/usr/bin/env python3
"""Offline directional tilt-law/robot-response audit; never opens hardware.

CSV angular twists are in base coordinates. Transform them to tool coordinates
before comparing them with tilt_omega_y_rad_s. A candidate quiet interval is
not proof that the probe had no external load.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def load_log(path):
    text_names = {"phase", "tilt_stop_reason", "phi_source", "phi_sha8", "qpik_fallback_reason"}
    names = ["t_wall_s", "dt_actual_s", "tilt_engaged", "tilt_frozen", "tilt_capped",
             "tilt_stalled", "tilt_tau_error_y_nm", "tilt_omega_y_rad_s", "tilt_deadband_nm",
             "contact_present", "fz", "ty", "ty_raw_comp", "f_des_z_eff", "slack_norm",
             "joint_limited", "pos_clamped", "qpik_solver_fault_latched"]
    names += ["pose_" + a for a in ("x", "y", "z", "rx", "ry", "rz")]
    names += ["q_meas_" + str(i) for i in range(8)]
    names += [p + "_" + a for p in ("twist_achieved", "task_model", "twist_requested")
              for a in ("wx", "wy", "wz")]
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        missing = (set(names) | text_names) - set(reader.fieldnames)
        if missing:
            raise ValueError(f"CSV lacks required telemetry: {sorted(missing)}")
        data = {key: [] for key in [*names, *text_names]}
        for row in reader:
            for key in names:
                data[key].append(float(row[key]) if row[key] else np.nan)
            for key in text_names:
                data[key].append(row[key])
    return {key: np.asarray(value) for key, value in data.items()}


def counts(values):
    keys, values = np.unique(values, return_counts=True)
    return {str(k): int(v) for k, v in zip(keys, values)}


def stats(values):
    x = np.asarray(values)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"samples": 0}
    return {"samples": len(x), "mean": float(np.mean(x)), "std": float(np.std(x)),
            "p01": float(np.percentile(x, 1)), "p50": float(np.median(x)),
            "p99": float(np.percentile(x, 99))}


def analyze(d, baseline):
    t, dt = d["t_wall_s"], d["dt_actual_s"]
    eng = d["tilt_engaged"] == 1
    w, drive = d["tilt_omega_y_rad_s"], d["tilt_tau_error_y_nm"]
    previous = np.r_[np.nan, w[:-1]]
    accel = (w - previous) / dt
    rotation = Rotation.from_euler("xyz", np.column_stack(
        [d["pose_" + a] for a in ("rx", "ry", "rz")])).as_matrix()
    for key, prefix in (("actual", "twist_achieved"), ("model", "task_model"),
                        ("requested", "twist_requested")):
        base = np.column_stack([d[prefix + "_" + a] for a in ("wx", "wy", "wz")])
        d["wy_" + key] = np.einsum("nij,nj->ni", rotation.transpose(0, 2, 1), base)[:, 1]
    valid = eng & np.r_[False, eng[:-1]] & (np.r_[0., np.diff(t)] < .008)
    valid &= np.isfinite(w) & np.isfinite(drive) & np.isfinite(previous) & (dt > 0)
    valid &= (d["tilt_frozen"] == 0) & (d["tilt_stalled"] == 0) & (d["tilt_capped"] == 0)
    report = {"rows": len(t), "time_range_s": [float(t[0]), float(t[-1])],
              "contact_samples": int(eng.sum()), "directions": {},
              "baseline_is_confirmed_unloaded": False,
              "fit_selection": {"min_speed_rad_s": .01, "max_speed_rad_s": .215,
                                "max_accel_rad_s2": 4.4, "max_adjacent_log_gap_s": .008},
              "notes": ["Signs refer to tool omega_y; physical up/down is not identified by this CSV.",
                        "A loaded contact trace alone does not identify TCP or gravity calibration error.",
                        "The fit estimates the recorded command law, not closed-loop contact stiffness."]}
    for key in ("phase", "tilt_stop_reason", "tilt_frozen", "tilt_capped", "tilt_stalled",
                "joint_limited", "pos_clamped", "qpik_solver_fault_latched", "qpik_fallback_reason",
                "phi_source", "phi_sha8"):
        report[key] = counts(d[key])
    report["contact_deadband_fraction"] = float(np.mean(d["tilt_stop_reason"][eng] == "torque_deadband"))
    stopped = eng & (d["tilt_stop_reason"] == "torque_deadband") & (abs(w) < 1e-6)
    edges = np.diff(np.r_[False, stopped, False].astype(int))
    spans = list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))
    if spans:
        start, end = max(spans, key=lambda ij: t[ij[1] - 1] - t[ij[0]])
        report["longest_contact_deadband_stop"] = {
            "range_s": [float(t[start]), float(t[end - 1])],
            "my_nm": stats(d["ty"][start:end]), "fz_n": stats(d["fz"][start:end]),
            "command_omega_rad_s": stats(w[start:end]),
        }
    report["slack_norm"] = stats(d["slack_norm"])
    for sign in (-1, 1):
        moving = valid & (sign * w > .01)
        fit = moving & (abs(w) < .215) & (abs(accel) < 4.4)
        A = np.column_stack([accel, w, np.sign(w)])[fit]
        coeff, _, _, _ = np.linalg.lstsq(A, drive[fit], rcond=None)
        residual = drive[fit] - A @ coeff
        item = {"samples": int(moving.sum()), "moving_time_s": float(dt[moving].sum()),
                "abs_omega_rad_s": stats(abs(w[moving])), "abs_drive_nm": stats(abs(drive[moving])),
                "fitted_mass_damping_coulomb": coeff.tolist(),
                "law_fit_residual_rms_nm": float(np.sqrt(np.mean(residual ** 2))),
                "qp_tool_y_error_rad_s": stats(abs(d["wy_model"][moving] - w[moving]))}
        # Estimate a scalar delay and gain independently in each direction.
        best = None
        for delay in np.arange(0., .201, .005):
            command = np.interp(t - delay, t, np.nan_to_num(w))
            m = eng & (sign * command > .02) & np.isfinite(d["wy_actual"])
            a, b = d["wy_actual"][m], command[m]
            gain = float(np.dot(a, b) / np.dot(b, b))
            rms = float(np.sqrt(np.mean((a - gain * b) ** 2)))
            if best is None or rms < best["rms_rad_s"]:
                best = {"delay_s": float(delay), "gain": gain, "rms_rad_s": rms, "samples": int(m.sum())}
        item["measured_response"] = best
        item["matched_steady_bins"] = []
        for lo, hi in ((.03, .04), (.04, .05), (.05, .06), (.06, .08)):
            m = moving & (sign * drive >= lo) & (sign * drive < hi) & (abs(accel) < .05)
            item["matched_steady_bins"].append({"drive_range_nm": [lo, hi],
                "omega_rad_s": stats(abs(w[m])), "drive_nm": stats(abs(drive[m]))})
        report["directions"][str(sign)] = item
    if baseline:
        mask = (t >= baseline[0]) & (t < baseline[1])
        report["candidate_quiet_window"] = {"range_s": list(baseline), "my_nm": stats(d["ty"][mask]),
            "raw_comp_my_nm": stats(d["ty_raw_comp"][mask]), "fz_n": stats(d["fz"][mask]),
            "measured_omega_y_rad_s": stats(d["wy_actual"][mask]),
            "phase": counts(d["phase"][mask]), "tilt_engaged": counts(d["tilt_engaged"][mask]),
            "note": "Servo-only samples have no active hybrid contact classification; zero contact flag is not proof of no physical contact."}
    d["law_fit_mask"] = valid & (abs(w) > .01) & (abs(w) < .215) & (abs(accel) < 4.4)
    fitted_mass = np.mean([v["fitted_mass_damping_coulomb"][0] for v in report["directions"].values()])
    d["inertia_removed_drive"] = drive - fitted_mass * accel
    return report


def check_tcp_binding(d, path):
    from rm75_control.control.joint_admittance_8dof.model import RobotKinematics

    binding = json.loads(path.read_text())["tool_binding"]
    stored = np.asarray(binding["T_link7_tcp"])
    kin = RobotKinematics()
    observed = []
    for i in np.linspace(0, len(d["t_wall_s"]) - 1, 24, dtype=int):
        q = np.asarray([d["q_meas_" + str(j)][i] for j in range(8)])
        flange = kin.frame_placement(q, "link_7").homogeneous
        tcp = np.eye(4)
        tcp[:3, 3] = [d["pose_" + a][i] for a in ("x", "y", "z")]
        tcp[:3, :3] = Rotation.from_euler("xyz", [d["pose_" + a][i] for a in ("rx", "ry", "rz")]).as_matrix()
        observed.append(np.linalg.inv(flange) @ tcp)
    observed = np.asarray(observed)
    angles = [np.rad2deg(Rotation.from_matrix(stored[:3, :3].T @ v[:3, :3]).magnitude()) for v in observed]
    return {"binding_file": str(path.resolve()), "active_tool_name": binding.get("active_tool_name"),
            "samples": len(observed), "reconstructed_translation_m": np.median(observed[:, :3, 3], axis=0).tolist(),
            "translation_delta_mm": (1000 * (np.median(observed[:, :3, 3], axis=0) - stored[:3, 3])).tolist(),
            "max_rotation_delta_deg": float(max(angles)),
            "interpretation": "Checks FK/software binding consistency; does not establish physical TCP calibration accuracy."}


def compare_deadbands(d):
    """Replay recorded torque; this does not simulate changed contact forces."""
    t, dt = d["t_wall_s"], d["dt_actual_s"]
    drive, recorded = d["tilt_tau_error_y_nm"], d["tilt_omega_y_rad_s"]
    eng = d["tilt_engaged"] == 1
    indices = np.flatnonzero(np.isfinite(drive))
    quiet = eng & (abs(recorded) < 1e-6) & (d["tilt_stop_reason"] == "torque_deadband")
    edges = np.diff(np.r_[False, quiet, False].astype(int))
    windows = [(i, j) for i, j in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))
               if t[j - 1] - t[i] >= .15]
    result = {"mass": .065, "damping": .28, "vmax_rad_s": .22, "amax_rad_s2": 4.5,
              "quiet_windows": len(windows), "quiet_total_s": float(sum(dt[i:j].sum() for i, j in windows)),
              "major_reversal_definition": "Opposite speed exceeds 0.01 rad/s for at least 50 ms; pauses do not count as reversals.",
              "interpretation": "Same measured torque and contact flags, with full velocity state carried through the trace. Changed pose/contact reaction is not simulated; this is not a closed-loop stability guarantee.",
              "cases": []}
    for fc in (.025, .0225, .020, .0175, .015):
        w = np.full(len(t), np.nan)
        previous = 0.
        for i in indices:
            target = 0.
            if eng[i] and not (d["tilt_frozen"][i] or d["tilt_stalled"][i] or d["tilt_capped"][i]):
                denominator = .065 / dt[i] + .28
                free = (.065 / dt[i] * previous + drive[i]) / denominator
                target = np.sign(free) * max(abs(free) - fc / denominator, 0.)
                target = float(np.clip(target, -.22, .22))
            previous += np.clip(target - previous, -4.5 * dt[i], 4.5 * dt[i])
            w[i] = previous
        last = candidate = reversals = 0
        elapsed = 0.
        for i in indices:
            if not eng[i]:
                last = candidate = 0
                elapsed = 0.
                continue
            sign = 1 if w[i] > .01 else (-1 if w[i] < -.01 else 0)
            if not sign:
                candidate = 0
                elapsed = 0.
                continue
            if sign != candidate:
                candidate, elapsed = sign, 0.
            elapsed += dt[i]
            if elapsed >= .05 and last != sign:
                reversals += int(last != 0)
                last = sign
        excursions = [float(np.max(abs(np.cumsum(w[i:j] * dt[i:j])))) for i, j in windows]
        item = {"deadband_nm": fc, "contact_in_deadband_percent": float(100 * np.mean(abs(drive[eng]) <= fc)),
                "moving_seconds_over_001_rad_s": float(dt[eng & (abs(w) > .01)].sum()),
                "total_absolute_command_rotation_deg": float(np.rad2deg(np.sum(abs(w[eng]) * dt[eng]))),
                "p99_speed_deg_s": float(np.rad2deg(np.percentile(abs(w[eng]), 99))),
                "peak_speed_deg_s": float(np.rad2deg(np.max(abs(w[eng])))),
                "speed_cap_seconds": float(dt[eng & (abs(w) >= .22 - 1e-9)].sum()),
                "major_reversals": reversals,
                "quiet_max_window_excursion_deg": float(np.rad2deg(max(excursions, default=0.))),
                "quiet_total_absolute_rotation_deg": float(np.rad2deg(sum(np.sum(abs(w[i:j]) * dt[i:j]) for i, j in windows)))}
        if fc == .025:
            error = w[indices] - recorded[indices]
            item["recorded_baseline_rms_error_rad_s"] = float(np.sqrt(np.mean(error ** 2)))
            item["recorded_baseline_max_error_rad_s"] = float(np.max(abs(error)))
        result["cases"].append(item)
    return result


def plot(d, prefix, baseline, report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = d["t_wall_s"]
    fig, ax = plt.subplots(4, 1, figsize=(12, 9), sharex=True, layout="constrained")
    ax[0].plot(t, d["fz"], lw=.7, label="Measured Fz")
    ax[0].plot(t, d["f_des_z_eff"], lw=1, label="Target Fz")
    ax[0].set_ylabel("Normal force (N)")
    ax[1].plot(t, d["ty"], lw=.7, label="Compensated TCP My")
    ax[1].axhspan(-.025, .025, color="grey", alpha=.18, label="Torque deadband")
    ax[1].set_ylabel("Moment (Nm)")
    ax[2].plot(t, d["wy_actual"], lw=.5, alpha=.55, label="Measured tool wy")
    ax[2].plot(t, d["tilt_omega_y_rad_s"], lw=.9, label="Tilt command wy")
    ax[2].set_ylabel("Angular speed (rad/s)")
    ax[3].plot(t, np.rad2deg(d["pose_ry"]), lw=1, label="Measured Euler pitch")
    ax[3].set_ylabel("Pitch (deg)")
    ax[3].set_xlabel("Time since logging started (s)")
    for a in ax:
        if baseline:
            a.axvspan(*baseline, color="orange", alpha=.17)
        a.grid(alpha=.2)
        a.legend(loc="upper right", fontsize=9)
    fig.suptitle(f"{prefix.name}: command, motion and torque | orange = candidate quiet interval")
    fig.savefig(str(prefix) + "_timeline.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    delay = np.mean([v["measured_response"]["delay_s"] for v in report["directions"].values()])
    gain = np.mean([v["measured_response"]["gain"] for v in report["directions"].values()])
    for sign, color in ((-1, "#286aa6"), (1, "#b74f2f")):
        mask = d["law_fit_mask"] & (sign * d["tilt_omega_y_rad_s"] > 0)
        ax[0].scatter(d["tilt_omega_y_rad_s"][mask][::4], d["inertia_removed_drive"][mask][::4],
                      s=5, alpha=.35, color=color, label=f"tool wy sign {sign:+d}")
        command = np.interp(t - delay, t, np.nan_to_num(d["tilt_omega_y_rad_s"]))
        m = (d["tilt_engaged"] == 1) & (sign * command > .02)
        ax[1].scatter(command[m][::4], d["wy_actual"][m][::4], s=5, alpha=.25, color=color)
    ax[0].set(xlabel="Command wy (rad/s)", ylabel="Drive minus inertia term (Nm)",
              title="Same damping and deadband in both directions")
    ax[0].legend()
    ax[1].plot([-.2, .22], [-.2, .22], "k--", lw=1)
    ax[1].set(xlabel=f"Command wy, delayed {1000 * delay:.0f} ms (rad/s)", ylabel="Measured tool wy (rad/s)",
              title=f"Measured following: pooled gain about {100 * gain:.1f}%")
    for a in ax:
        a.grid(alpha=.2)
    fig.savefig(str(prefix) + "_direction_fit.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--baseline", type=float, nargs=2, metavar=("START_S", "END_S"))
    parser.add_argument("--binding-json", type=Path)
    parser.add_argument("--compare-deadbands", action="store_true")
    args = parser.parse_args()
    d = load_log(args.csv)
    report = analyze(d, args.baseline)
    report["csv"] = str(args.csv.resolve())
    if args.binding_json:
        report["tcp_binding_check"] = check_tcp_binding(d, args.binding_json)
    if args.compare_deadbands:
        report["deadband_replay"] = compare_deadbands(d)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    Path(str(args.output_prefix) + ".json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    plot(d, args.output_prefix, args.baseline, report)
    print(f"Wrote {args.output_prefix}.json and two PNG figures")


if __name__ == "__main__":
    main()
