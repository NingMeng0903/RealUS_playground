"""Independently recompute shifted-probe witnesses from measured native trace files."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.integrate import quad
from scipy.spatial.transform import Rotation
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ab_closed_loop_reviewed_20261111"))
from acoustics import ProbeAction, P_CLUTTER, detection_probability, geometry_value, sensor_sd


def info(row):
    payload = dict(row["public"]["planning_action"])
    payload.pop("frame_role")
    payload["xy_m"] = tuple(payload["xy_m"])
    action = ProbeAction(**payload)
    pt = detection_probability(action)
    pf = (1.0 - pt) * P_CLUTTER
    mu, sigma = geometry_value(action), sensor_sd(action)

    def integrand(z):
        y = mu + sigma*z
        real = pt*norm.pdf(y, mu, sigma)
        mixture = real + pf*norm.pdf(y, 0.12, 0.08)
        return real*real*z*z/(mixture*sigma) if mixture > 0.0 else 0.0

    kappa = quad(integrand, -12, 12, epsabs=1e-9)[0]
    assert -1e-9 <= kappa <= pt/(sigma*sigma) + 1e-8
    return {
        "p_target": pt, "p_false": pf, "p_missing": 1-pt-pf, "kappa": kappa,
        "s": float(action.as_feature()[1]), "nominal_model_mean": mu, "sensor_sd": sigma,
    }


def audit(run):
    layouts = {}
    for file in sorted(run.glob("layouts/*/TRANSITIONS.json")):
        payload = json.loads(file.read_text())
        rows = payload["transitions"]
        exits = {tuple(row["exit_state_signature"]["key"]): row["exit_state_signature"]
                 for row in rows.values() if row["valid"]}
        per_layout = {}
        for key, row in rows.items():
            if row["sequence"]:
                assert row["entry_state_signature"] == exits[tuple(row["sequence"])], key
            attempts = []
            for attempt in row["dird"]["attempts"]:
                errs, qmin, ticks = [], math.inf, 0
                for link, step in zip(attempt["linkage"]["trace_links"], attempt["steps"]):
                    with np.load(run/link["path"]) as npz:
                        desired, measured = npz["desired"], npz["measured_before"]
                        pos = np.linalg.norm(desired[:, :3, 3]-measured[:, :3, 3], axis=1).max()
                        mat = np.swapaxes(measured[:, :3, :3], 1, 2) @ desired[:, :3, :3]
                        rot = Rotation.from_matrix(mat).magnitude().max()
                        qmin = min(qmin, float(npz["self_clearance_m"].min()))
                        assert np.array_equal(npz["q"][-1], step["end_q"]), link["path"]
                        assert np.array_equal(npz["qdot"][-1], step["end_qdot"]), link["path"]
                        assert abs(pos-step["same_tick_position_error_max_m"]) < 1e-10
                        assert abs(rot-step["same_tick_rotation_error_max_rad"]) < 1e-10
                        errs.append((float(pos), float(rot), step["synthetic_contact_gap_max_m"]))
                        ticks += len(desired)
                attempts.append({
                    "route": attempt["route_id"], "valid": attempt["valid"],
                    "duration_s": attempt["duration_s"], "ticks": ticks,
                    "max_position_mm": max(e[0] for e in errs)*1000,
                    "max_rotation_deg": math.degrees(max(e[1] for e in errs)),
                    "max_synthetic_gap_mm": max(e[2] for e in errs)*1000,
                    "min_self_clearance_mm": qmin*1000,
                    "dird_mean": attempt["dird_mean_calibrated_margin"],
                })
            entry = {
                "valid": row["valid"], "seconds": row["actual_elapsed_s"],
                "actual_exit_local": row["public"]["measured_exit_local"],
                "attempts": attempts, "selected_route": row["selected_route_id"],
            }
            if row["capture_kind"] == "fork" and row["valid"]:
                entry["fisher"] = info(row)
            per_layout[key] = entry
        first = per_layout["START|capture=fork_minus"]["fisher"]
        for name in ("fork_plus", "fork_plus_shift3"):
            second = per_layout[f"fork_minus|capture={name}"]["fisher"]
            second["pair_slope_info"] = (
                first["kappa"]*second["kappa"]/(first["kappa"]+second["kappa"])
                * (first["s"]-second["s"])**2
            )
        layouts[payload["layout"]] = per_layout
    return {"run_root": str(run.resolve()), "layouts": layouts}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", type=Path)
    ap.add_argument("output", type=Path)
    args = ap.parse_args()
    with args.output.open("x") as file:
        json.dump(audit(args.run), file, indent=2, allow_nan=False)
        file.write("\n")
