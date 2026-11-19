"""Pair spline and frozen-tet scans with the same physical world and start."""

import argparse
import json
import lzma
from pathlib import Path

import numpy as np
from scipy.stats import beta as beta_distribution

from analyze_dynamic import COHORTS, COST_MARGINS, PRIMARY, bootstrap, interval, outcome_rows
from atlas_scan import PAIRED_ARMS, sha

EXPECTED_CASES = dict(zip(COHORTS, (32, 32, 32, 16, 16, 32, 12), strict=True))
NONINFERIORITY_MARGIN = .05
METRICS = ("success", "wrong", "false_complete", "tcp_mm", "turn_deg", "duration_s",
           "run_failure", "acquisitions", "correct_frames", "mean_abs_lateral_mm", "loss_recovery")


def read_run(directory):
    report_path = directory / "SUMMARY.json"
    if not report_path.exists():
        report_path = directory / "SUMMARY.json.xz"
        with lzma.open(report_path, "rt") as stream:
            report = json.load(stream)
    else:
        report = json.loads(report_path.read_text())
    traces = directory / "TRACES.npz"
    return report, traces, {path.name: sha(path) for path in (report_path, traces)}


def paired_runs(base_folder, new_folder, name):
    base, base_path, base_hash = read_run(base_folder)
    new, new_path, new_hash = read_run(new_folder)
    count = EXPECTED_CASES[name]
    assert base["case_count"] == new["case_count"] == len(base["cases"]) == len(new["cases"]) == count
    assert base["geometry"].get("chart", "tet") == "tet"
    assert new["geometry"]["chart"] == "spline"
    assert new["observation_world"] == new["physical_start"] == "tet"
    for key in ("seed", "motion", "robot_input_sha256", "model_input_sha256", "sample_dt_s", "frame_dt_s"):
        assert base[key] == new[key], key
    for key in ("beta", "knee_deg", "material_binding", "input_sha256"):
        assert base["geometry"][key] == new["geometry"][key], key
    with np.load(base_path, allow_pickle=False) as old, np.load(new_path, allow_pickle=False) as smooth:
        for old_key, new_key in (
            ("nominal_vessels_m", "world_vessels_m"), ("offset_axes", "world_offset_axes"),
            ("vessel_material_xi", "vessel_legacy_xi" if "vessel_legacy_xi" in smooth else "vessel_material_xi"),
            ("station_ref_mm", "station_ref_mm"),
        ):
            np.testing.assert_array_equal(old[old_key], smooth[new_key])
        for i, (br, sr) in enumerate(zip(base["cases"], new["cases"], strict=True)):
            assert br["case"] == sr["case"] == i
            for key in ("hidden_static_offset_mm", "hidden_calibration_bias_mm"):
                assert br[key] == sr[key]
            for key in ("clear", "noise_mm", "z_mm"):
                np.testing.assert_array_equal(old[f"{i}_{key}"], smooth[f"{i}_{key}"])
            for arm in PAIRED_ARMS:
                key = f"{i}_{arm}"
                for suffix in ("q", "fk_m", "x_v"):
                    np.testing.assert_array_equal(old[f"{key}_{suffix}"][0], smooth[f"{key}_{suffix}"][0])
                before = br["arms"][arm]["frames"]
                after = sr["arms"][arm]["frames"]
                if before and after:
                    assert before[0]["packet_mm"] == after[0]["packet_mm"]
        result = {"tet": outcome_rows(base, old), "spline": outcome_rows(new, smooth)}
    return result, {"tet": base_hash, "spline": new_hash}


def summarize(runs):
    count = max(len(run["tet"]["mpc_ou"]["success"]) for run in runs)
    draws = bootstrap(count)
    arms = {}
    for arm in PAIRED_ARMS:
        fields = {}
        for field in METRICS:
            total = np.zeros(count)
            available = np.zeros(count)
            wins = losses = 0
            base_values, new_values = [], []
            for run in runs:
                base, new = (run[label][arm][field] for label in ("tet", "spline"))
                valid = np.isfinite(new) & np.isfinite(base)
                difference = new[valid] - base[valid]
                indices = np.flatnonzero(valid)
                total[indices] += difference
                available[indices] += 1.
                wins += int(np.count_nonzero(difference > 0))
                losses += int(np.count_nonzero(difference < 0))
                base_values.extend(base[valid])
                new_values.extend(new[valid])
            if not available.sum():
                fields[field] = None
                continue
            numerator = total[draws].sum(axis=1)
            denominator = available[draws].sum(axis=1)
            contrast = np.full(len(draws), np.nan)
            np.divide(numerator, denominator, out=contrast, where=denominator > 0)
            fields[field] = {"tet_mean": float(np.mean(base_values)), "spline_mean": float(np.mean(new_values)),
                             "difference": float(total.sum() / available.sum()),
                             "paired_ci95": interval(contrast), "n_pairs": int(available.sum())}
            if field in ("success", "wrong", "false_complete", "run_failure", "loss_recovery"):
                fields[field].update(tet_count=int(sum(base_values)), spline_count=int(sum(new_values)),
                                     spline_only=wins, tet_only=losses)
        fields["cost_equivalent"] = all(
            fields[key]["paired_ci95"][0] >= -margin and fields[key]["paired_ci95"][1] <= margin
            for key, margin in COST_MARGINS.items()
        )
        arms[arm] = fields
    return arms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--spline", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    runs, results, hashes = {}, {}, {}
    for name in COHORTS:
        runs[name], hashes[name] = paired_runs(args.baseline / name, args.spline / name, name)
        results[name] = summarize([runs[name]])
        with np.load(args.spline / name / "TRACES.npz", allow_pickle=False) as traces:
            if name == COHORTS[0]:
                canonical_labels = traces["vessel_material_xi"]
            else:
                np.testing.assert_array_equal(traces["vessel_material_xi"], canonical_labels)
    primary = summarize([runs[name] for name in PRIMARY])
    pooled = summarize(list(runs.values()))
    primary_losses = np.column_stack([
        runs[name]["spline"]["mpc_ou"]["success"] < runs[name]["tet"]["mpc_ou"]["success"]
        for name in PRIMARY
    ]).any(axis=1)
    loss_count = int(primary_losses.sum())
    clusters = len(primary_losses)
    # A zero-discordance bootstrap can collapse to zero. The exact
    # independent-tape bound is a conservative supplemental diagnostic.
    loss_upper = 1. if loss_count == clusters else float(beta_distribution.ppf(
        .95, loss_count + 1, clusters - loss_count,
    ))
    main_arm = primary["mpc_ou"]
    gate = {
        "primary_ou_point_estimate_not_lower": main_arm["success"]["difference"] >= 0.,
        "primary_ou_paired_lower_bound_above_minus_5pp": main_arm["success"]["paired_ci95"][0] > -NONINFERIORITY_MARGIN,
        "primary_ou_cost_equivalence": main_arm["cost_equivalent"],
        "no_new_wrong_identity_in_full_suite": all(pooled[arm]["wrong"]["spline_only"] == 0 for arm in PAIRED_ARMS),
        "full_suite_success_not_lower_in_any_arm": all(pooled[arm]["success"]["difference"] >= 0. for arm in PAIRED_ARMS),
    }
    report = {"comparison": "spline minus tet; identical physical world, robot entry and disturbance tapes",
              "noninferiority_margin": NONINFERIORITY_MARGIN, "cost_margins": COST_MARGINS,
              "cohort_case_counts": EXPECTED_CASES, "hashes": hashes,
              "acceptance": {**gate, "pass": all(gate.values())},
              "primary_ou_loss_cluster_sensitivity": {
                  "tapes": clusters, "tapes_with_any_tet_only_success": loss_count,
                  "one_sided_95pct_upper_loss_tape_probability": loss_upper,
              },
              "primary": primary, "full_suite_descriptive": pooled, "cohorts": results}
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report["acceptance"], indent=2), flush=True)


if __name__ == "__main__":
    main()
