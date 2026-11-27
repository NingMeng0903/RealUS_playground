"""Compare paired scan outcomes and causal forecasts on a common packet history."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from atlas_scan import FRAME_DT_S, PAIRED_ARMS, RESIDUAL_ARMS, project, sha
from primitive import FRAME_STRIDE
from residual import GRID_MM, ResidualFilter, offset_stats
from sensor import Sensor


HORIZONS = (0, 1, 3, 5, 10)
PRIMARY = ("reference", "minus-plus", "plus-minus")
COHORTS = (*PRIMARY, "beta2", "pose25", "transient", "static-world")
COMPARISONS = (("mpc_ou", "mpc_stream"), ("mpc_ou", "mpc_iid"), ("mpc_iid", "mpc_stream"))
COST_MARGINS = {"tcp_mm": 2., "turn_deg": 2., "duration_s": .1}
FIELDS = ("success", "wrong", "false_complete", "coverage", "tcp_mm", "turn_deg", "duration_s", "revisions")
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 2026111406


def bootstrap(n):
    return np.random.default_rng(BOOTSTRAP_SEED).integers(0, n, size=(BOOTSTRAP_DRAWS, n))


def interval(draws):
    lo, hi = np.nanquantile(draws, [.025, .975])
    return [float(lo), float(hi)] if np.isfinite(lo + hi) else None


def recovery_window(clear):
    """The first naturally sampled loss interval meeting the protocol."""
    starts = np.flatnonzero(~clear & np.r_[True, clear[:-1]])
    for start in starts[starts >= 5]:
        after = np.flatnonzero(clear[start:])
        if len(after):
            end = int(start + after[0])
            if end - start >= 3 and end + 5 <= len(clear):
                return slice(end, end + 5)
    return None


def cluster_sign_flip(delta):
    """Exact two-sided sign-flip sensitivity check on integer case contrasts."""
    magnitudes = np.abs(delta.sum(axis=1)).astype(int)
    mass = np.array([1.])
    for step in magnitudes[magnitudes > 0]:
        signs = np.zeros(2 * step + 1)
        signs[[0, -1]] = .5
        mass = np.convolve(mass, signs)
    grid = np.arange(len(mass)) - len(mass) // 2
    return {"nonzero_clusters": int(np.count_nonzero(magnitudes)),
            "two_sided_p": float(mass[np.abs(grid) >= abs(delta.sum())].sum())}


def outcome_rows(report, traces):
    output = {arm: [] for arm in PAIRED_ARMS}
    for row in report["cases"]:
        window = recovery_window(traces[f'{row["case"]}_clear'][:, 0])
        for arm, records in output.items():
            result = row["arms"][arm]
            frames = result["frames"]
            tracked = [abs(frame["physical_lateral_mm"]) for frame in frames
                       if frame["physical_lateral_mm"] is not None]
            metrics = {key: float(result[key]) for key in FIELDS}
            metrics.update(run_failure=float(result["failure"] is not None),
                           motion_rejection=float(result["failure"] is not None and
                                                  result["failure"].startswith("no feasible first segment")),
                           acquisitions=float(len(frames)), physical_crossings=float(len(tracked)),
                           correct_frames=float(sum(f["correct"] for f in frames)),
                           mean_abs_lateral_mm=float(np.mean(tracked)) if tracked else np.nan,
                           loss_recovery=float(any(f["correct"] for f in frames[window])) if window else np.nan)
            records.append(metrics)
    return {arm: {key: np.array([row[key] for row in rows]) for key in rows[0]}
            for arm, rows in output.items()}


def scan_summary(cohorts):
    """Average within each shared case ID before resampling the IDs."""
    per_arm = {arm: {field: np.stack([cohort[arm][field] for cohort in cohorts], axis=1)
                     for field in cohorts[0][arm]} for arm in PAIRED_ARMS}
    draws = bootstrap(len(per_arm["mpc_stream"]["success"]))
    arms = {}
    for arm, data in per_arm.items():
        arms[arm] = {field: float(np.nanmean(values)) if np.isfinite(values).any() else None
                     for field, values in data.items()}
        for field in ("success", "wrong", "false_complete", "run_failure", "motion_rejection"):
            arms[arm][field + "_count"] = int(data[field].sum())
        arms[arm]["loss_recovered_count"] = int(np.nansum(data["loss_recovery"]))
        arms[arm]["loss_eligible_count"] = int(np.isfinite(data["loss_recovery"]).sum())
    paired = {}
    for arm, base in COMPARISONS:
        comparisons = {}
        for field in per_arm[arm]:
            delta = per_arm[arm][field] - per_arm[base][field]
            valid = np.isfinite(delta)
            if valid.any():
                total = np.nansum(delta, axis=1)
                counts = valid.sum(axis=1)
                resampled = counts[draws].sum(axis=1)
                estimate = np.divide(total[draws].sum(axis=1), resampled,
                                     out=np.full(len(draws), np.nan), where=resampled > 0)
                comparisons[field] = {"difference": float(np.nanmean(delta)), "ci95": interval(estimate)}
        success = per_arm[arm]["success"] - per_arm[base]["success"]
        comparisons["success"].update(left_only=int((success > 0).sum()), right_only=int((success < 0).sum()))
        comparisons["success"]["cluster_sign_flip"] = cluster_sign_flip(success)
        comparisons["cost_equivalent_at_frozen_margins"] = all(
            -margin <= comparisons[field]["ci95"][0] <= comparisons[field]["ci95"][1] <= margin
            for field, margin in COST_MARGINS.items())
        paired[f"{arm}_vs_{base}"] = comparisons
    return {"case_count": int(per_arm["mpc_stream"]["success"].size), "shared_case_ids": int(draws.shape[1]),
            "arms": arms, "paired": paired}


def replay(report, traces, grid_mm=GRID_MM):
    """Forecast on the static controller's public history, at its actual FK."""
    sensor = Sensor()
    models = {"mpc_stream": sensor}
    models.update({arm: ResidualFilter(sensor, FRAME_DT_S, model, grid_mm)
                   for arm, model in RESIDUAL_ARMS.items()})
    geometry = SimpleNamespace(vessels_m=traces["nominal_vessels_m"], offset_axes=traces["offset_axes"])
    # Categories are all origins and origins without a visible artery.
    sums = {arm: np.zeros((len(report["cases"]), 2, len(HORIZONS), 5)) for arm in models}
    max_edge_mass = {arm: 0. for arm in RESIDUAL_ARMS}
    for i, row in enumerate(report["cases"]):
        case_id = row["case"]
        frames = row["arms"]["mpc_stream"]["frames"]
        poses = traces[f"{case_id}_mpc_stream_fk_m"][::FRAME_STRIDE]
        effective_truth = row["hidden_static_offset_mm"][0] + traces[f"{case_id}_z_mm"]
        beliefs = {arm: model.prior() for arm, model in models.items()}
        for k, saved in enumerate(frames):
            packet = np.array(saved["packet_mm"]).reshape(-1, 2)
            without_artery = 0 not in saved["role_truth"]
            projected = {arm: project(geometry, poses[k], model.offsets_mm)
                         for arm, model in models.items() if arm != "mpc_iid"}
            projected["mpc_iid"] = projected["mpc_ou"]
            for arm, model in models.items():
                belief, _ = model.update_projected(beliefs[arm], packet, projected[arm])
                beliefs[arm] = belief
                if arm in max_edge_mass:
                    edge = belief[:, [0, -1], :].sum()
                    max_edge_mass[arm] = max(max_edge_mass[arm], float(edge))
                for index, horizon in enumerate(HORIZONS):
                    target = k + horizon
                    if target >= len(effective_truth):
                        break
                    mean, bounds = offset_stats(model, belief, horizon)
                    error = mean[0] - effective_truth[target]
                    measures = (1., error**2, error, bounds[0] <= effective_truth[target] <= bounds[1],
                                bounds[1] - bounds[0])
                    sums[arm][i, 0, index] += measures
                    if without_artery:
                        sums[arm][i, 1, index] += measures
    return sums, max_edge_mass


def forecast_summary(cohorts):
    values = {arm: sum(cohort[arm] for cohort in cohorts) for arm in PAIRED_ARMS}
    draws = bootstrap(len(values["mpc_stream"]))
    output = {}
    for category, label in enumerate(("all_origins", "artery_missing_at_origin")):
        output[label] = {}
        for j, horizon in enumerate(HORIZONS):
            rows = {}
            rmses = {}
            for arm, data in values.items():
                parts = data[:, category, j]
                count, error2, error, covered, width = parts.sum(axis=0)
                if count == 0:
                    rows[arm] = None
                    continue
                rows[arm] = {"count": int(count), "rmse_mm": float(np.sqrt(error2 / count)),
                             "mean_error_mm": float(error / count), "interval90_coverage": float(covered / count),
                             "interval90_mean_width_mm": float(width / count)}
                sampled = parts[draws].sum(axis=1)
                rmses[arm] = np.sqrt(np.divide(sampled[:, 1], sampled[:, 0],
                                              out=np.full(len(draws), np.nan), where=sampled[:, 0] > 0))
            paired = {}
            for arm, base in COMPARISONS:
                if rows[arm] is not None and rows[base] is not None:
                    paired[f"{arm}_vs_{base}"] = {
                        "rmse_difference_mm": rows[arm]["rmse_mm"] - rows[base]["rmse_mm"],
                        "ci95": interval(rmses[arm] - rmses[base]),
                    }
            output[label][f"{horizon * FRAME_DT_S:.1f}"] = {"arms": rows, "paired": paired}
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True, help="Directory containing the named cohorts")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summary = {"cost_margins": COST_MARGINS, "bootstrap_draws": BOOTSTRAP_DRAWS, "bootstrap_seed": BOOTSTRAP_SEED,
               "cohorts": {}}
    scans, forecasts = {}, {}
    for name in COHORTS:
        folder = args.runs / name
        report_path, trace_path = folder / "SUMMARY.json", folder / "TRACES.npz"
        report = json.loads(report_path.read_text())
        with np.load(trace_path, allow_pickle=False) as traces:
            scans[name] = outcome_rows(report, traces)
            forecasts[name], edge_mass = replay(report, traces)
            if name == "reference":
                fine, fine_edge = replay(report, traces, GRID_MM / 2.)
                summary["reference_half_grid"] = {"grid_mm": GRID_MM / 2., "max_edge_posterior_mass": fine_edge,
                                                  "common_packet_forecasts": forecast_summary([fine])}
        summary["cohorts"][name] = {
            "input_sha256": {path.name: sha(path) for path in (report_path, trace_path)},
            "motion": report["motion"], "geometry": {"beta": report["geometry"]["beta"],
                                                      "knee_deg": report["geometry"]["knee_deg"]},
            "max_edge_posterior_mass": edge_mass,
            "closed_loop": scan_summary([scans[name]]),
            "common_packet_forecasts": forecast_summary([forecasts[name]]),
        }
        print(name, flush=True)
    summary["primary_clustered"] = {"cohorts": PRIMARY,
                                     "closed_loop": scan_summary([scans[name] for name in PRIMARY]),
                                     "common_packet_forecasts": forecast_summary([forecasts[name] for name in PRIMARY])}
    args.out.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
