"""Merge paired knee-response runs and resample independent case IDs."""

import argparse
import json
import lzma
from pathlib import Path

import numpy as np

from knee_response_benchmark import ACQUISITIONS, ARMS, HORIZONS
from sensor import CENTER_TOL_MM, IDENTITY_MIN


def paired(base, response, *, root=False):
    values = np.asarray([base, response], dtype=float)
    if values.shape[1] == 0:
        return None
    indices = np.random.default_rng(2026111517).integers(0, values.shape[1], (8192, values.shape[1]))
    point = values.mean(axis=-1)
    draws = values[:, indices].mean(axis=-1)
    if root:
        point, draws = np.sqrt(point), np.sqrt(draws)
    intervals = np.quantile(draws, [.025, .975], axis=-1).T
    return dict(base=float(point[0]), response=float(point[1]), difference=float(point[1] - point[0]),
                base_ci95=intervals[0].tolist(), response_ci95=intervals[1].tolist(),
                ci95=np.quantile(draws[1] - draws[0], [.025, .975]).tolist())


def closed_loop(cases, history, frames_scheduled):
    runs = [[case["histories"][history]["arms"][arm] for case in cases] for arm in ARMS]
    result = {"arms": {}}
    for name, histories in zip(ARMS, runs, strict=True):
        frames = [frame for run in histories for frame in run["frames"]]
        result["arms"][name] = dict(
            cases=len(histories), frames=len(frames), correct_frames=sum(frame["correct"] for frame in frames),
            accepted_frames=sum(frame["accepted"] for frame in frames),
            wrong_frames=sum(frame["wrong"] for frame in frames),
            false_accepts=sum(frame["accepted"] and not frame["correct"] for frame in frames),
            completions=sum(run["success"] for run in histories),
            wrong_cases=sum(run["wrong"] for run in histories),
            false_completions=sum(run["false_complete"] for run in histories),
            execution_failures=sum(run["failure"] is not None for run in histories),
            mean_revisions=float(np.mean([run["revisions"] for run in histories])),
        )
    for field in ("success", "wrong", "false_complete", "tcp_mm", "turn_deg", "duration_s"):
        result[field] = paired(*[[run[field] for run in histories] for histories in runs])
    result["correct_frame_rate"] = paired(*[
        [sum(frame["correct"] for frame in run["frames"]) / frames_scheduled for run in histories]
        for histories in runs])
    result["wrong_frames_per_run"] = paired(*[
        [sum(frame["wrong"] for frame in run["frames"]) for run in histories] for histories in runs])
    outcome = np.asarray([[run["success"] for run in group] for group in runs], dtype=int)
    result["completion_discordance"] = dict(response_only=int(np.sum(outcome[1] > outcome[0])),
                                           baseline_only=int(np.sum(outcome[0] > outcome[1])))
    margins = {"tcp_mm": 2., "turn_deg": 2., "duration_s": .10}
    result["cost_equivalent"] = all(max(abs(x) for x in result[field]["ci95"]) <= margin
                                    for field, margin in margins.items())
    return result


def slope_metrics(cases, history, *, after_holdout=False):
    summaries = [(case["replay"][history]["heldout_posterior"] if after_holdout
                  else case["histories"][history]["posterior"]) for case in cases]
    truth = np.asarray([case["true_slope_mm_per_deg"] for case in cases])
    mean = np.asarray([summary["mean"] for summary in summaries])
    bounds = np.asarray([summary["interval90"] for summary in summaries])
    return dict(
        rmse=float(np.sqrt(np.mean((mean - truth) ** 2))),
        mean_sd=float(np.mean([summary["sd"] for summary in summaries])),
        mean_width90=float(np.mean(bounds[:, 1] - bounds[:, 0])),
        coverage90=float(np.mean((bounds[:, 0] <= truth) & (truth <= bounds[:, 1]))),
        sign_agrees_and_excludes_zero=int(np.sum(((bounds[:, 0] > 0.) & (truth > 0.)) |
                                                ((bounds[:, 1] < 0.) & (truth < 0.)))),
        excludes_zero=int(np.sum((bounds[:, 0] > 0.) | (bounds[:, 1] < 0.))),
    )


def forecast_samples(forecasts, horizon, subset):
    rows = [row for row in forecasts if row["horizon"] == horizon and (subset == "all" or row[subset])]
    if not rows:
        return None
    errors = np.asarray([row["mean"] - row["truth"] for row in rows])
    bounds = np.asarray([row["interval90"] for row in rows])
    truth = np.asarray([row["truth"] for row in rows])
    return dict(squared=float(np.mean(errors ** 2)), bias=float(np.mean(errors)),
                coverage90=float(np.mean((bounds[:, 0] <= truth) & (truth <= bounds[:, 1]))),
                width90=float(np.mean(bounds[:, 1] - bounds[:, 0])), forecasts=len(rows))


def forecast_metrics(cases, history):
    summaries = {}
    for horizon in HORIZONS:
        for subset in ("all", "active", "missing_artery"):
            cases_by_arm = [[forecast_samples(case["replay"][history][arm]["forecasts"], horizon, subset)
                             for case in cases] for arm in ARMS]
            available = [pair for pair in zip(*cases_by_arm, strict=True) if all(x is not None for x in pair)]
            if not available:
                continue
            arms = list(zip(*available, strict=True))
            result = {key: paired(*[[row[key] for row in arm] for arm in arms], root=key == "squared")
                      for key in ("squared", "bias", "coverage90", "width90")}
            result["rmse"] = result.pop("squared")
            result["cases"] = len(available)
            result["predictions_per_arm"] = sum(row["forecasts"] for row in arms[0])
            summaries[f"{horizon}_{subset}"] = result
    return summaries


def association_metrics(cases, history):
    """The same packets and poses isolate the effect of g on association."""
    reports = {arm: [] for arm in ARMS}
    maximum_change, different_slots = 0., 0
    for case in cases:
        frames = case["histories"]["multiple_angles"]["arms"][ARMS[0]]["frames"]
        selections = []
        for arm in ARMS:
            errors, accepted, correct = [], 0, 0
            probabilities = case["replay"][history][arm]["association"]
            selected = []
            for row, probs in zip(frames, probabilities, strict=True):
                truth = np.asarray(row["role_truth"]) == 0
                errors.extend((np.asarray(probs) - truth) ** 2)
                chosen = -1
                if probs:
                    slot = int(np.argmax(probs))
                    if probs[slot] >= IDENTITY_MIN and abs(row["packet_mm"][slot][0]) <= CENTER_TOL_MM:
                        chosen = slot
                        accepted += 1
                        correct += int(truth[slot])
                selected.append(chosen)
            selections.append(selected)
            reports[arm].append(dict(brier=float(np.mean(errors)) if errors else None,
                                     accepted=accepted, wrong=accepted - correct))
        different_slots += sum(a != b for a, b in zip(*selections, strict=True))
        for a, b in zip(*(case["replay"][history][arm]["association"] for arm in ARMS), strict=True):
            maximum_change = max(maximum_change, max((abs(x - y) for x, y in zip(a, b, strict=True)), default=0.))
    common = list(zip(*(reports[arm] for arm in ARMS), strict=True))
    both_observed = [pair for pair in common if all(row["brier"] is not None for row in pair)]
    return dict(
        accepted=[sum(row["accepted"] for row in reports[arm]) for arm in ARMS],
        wrong=[sum(row["wrong"] for row in reports[arm]) for arm in ARMS],
        different_accepted_slots=different_slots, max_probability_difference=maximum_change,
        brier=paired(*[[pair[i]["brier"] for pair in both_observed] for i in range(2)]),
    )


def angle_diversity(cases):
    histories = ("same_angle", "multiple_angles")
    slope_errors, slope_sd, forecasts = [], [], []
    for history in histories:
        summaries = [case["histories"][history]["posterior"] for case in cases]
        slope_errors.append([(summary["mean"] - case["true_slope_mm_per_deg"]) ** 2
                             for summary, case in zip(summaries, cases, strict=True)])
        slope_sd.append([summary["sd"] for summary in summaries])
        forecasts.append([forecast_samples(case["replay"][history][ARMS[1]]["forecasts"], 5, "active")
                          for case in cases])
    common = [pair for pair in zip(*forecasts, strict=True) if all(row is not None for row in pair)]
    groups = list(zip(*common, strict=True)) if common else ([], [])
    prediction = {key: paired(*[[row[key] for row in group] for group in groups], root=key == "squared")
                  for key in ("squared", "coverage90", "width90")}
    prediction["rmse"] = prediction.pop("squared")
    return dict(histories=histories, estimator=ARMS[1], slope_rmse=paired(*slope_errors, root=True),
                slope_posterior_sd=paired(*slope_sd), forecast_0_5_s_active=prediction)


def summarize(report):
    results = {}
    cases = report["cases"]
    for history, visit_names in ACQUISITIONS.items():
        visits = [case["visits"][name] for case in cases for name in visit_names]
        results[history] = dict(
            slope_after_acquisition=slope_metrics(cases, history),
            slope_after_common_holdout_replay=slope_metrics(cases, history, after_holdout=True),
            calibration=dict(visits=len(visits), frames=sum(len(visit["frames"]) for visit in visits),
                             failures=sum(visit["failure"] is not None for visit in visits),
                             mean_tcp_per_visit_mm=float(np.mean([visit["tcp_mm"] for visit in visits])),
                             mean_turn_per_visit_deg=float(np.mean([visit["turn_deg"] for visit in visits])),
                             mean_duration_per_visit_s=float(np.mean([visit["duration_s"] for visit in visits]))),
            forecast=forecast_metrics(cases, history),
            same_frames_identity=association_metrics(cases, history),
            closed_loop=closed_loop(cases, history, report["acquisition_frames"]),
        )
    return dict(cases=len(cases), interpolation=report["interpolation"], histories=results,
                angle_diversity=angle_diversity(cases))


def merge(runs, destination):
    destination.mkdir(parents=True, exist_ok=True)
    merged = {}
    traces = {}
    for directory in runs:
        with lzma.open(directory / "CASES.json.xz", "rt") as file:
            report = json.load(file)
        cohort = report["cohort"]
        rows = report.pop("cases")
        if cohort not in merged:
            merged[cohort] = report | {"cases": []}
            traces[cohort] = {}
        elif report != {key: val for key, val in merged[cohort].items() if key != "cases"}:
            raise ValueError(f"shards have different study inputs: {directory}")
        merged[cohort]["cases"].extend(rows)
        storage = traces[cohort]
        with np.load(directory / "TRACES.npz", allow_pickle=False) as source:
            for name in source.files:
                if name in storage:
                    if name.startswith(f"{cohort}_"):
                        raise ValueError(f"duplicate case trace: {name}")
                    np.testing.assert_array_equal(source[name], storage[name])
                else:
                    storage[name] = source[name]
    summary = {}
    for cohort, report in sorted(merged.items()):
        report["cases"].sort(key=lambda row: row["case"])
        expected = 12 if cohort == "null" else 24
        if [row["case"] for row in report["cases"]] != list(range(expected)):
            raise ValueError(f"{cohort} must contain the frozen set of {expected} case IDs")
        directory = destination / cohort
        directory.mkdir(exist_ok=True)
        with lzma.open(directory / "CASES.json.xz", "wt", preset=4) as file:
            json.dump(report, file, allow_nan=False, separators=(",", ":"))
            file.write("\n")
        np.savez_compressed(directory / "TRACES.npz", **traces[cohort])
        summary[cohort] = summarize(report)
    if sorted(summary) != ["null", "response"]:
        raise ValueError("both frozen cohorts are required")
    (destination / "ANALYSIS.json").write_text(json.dumps(summary, allow_nan=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    merge(args.runs, args.out)


if __name__ == "__main__":
    main()
