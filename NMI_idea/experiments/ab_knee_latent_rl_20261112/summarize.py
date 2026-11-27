"""Paired case metrics and offline interactive 3D views."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


MAIN = ("rule", "fqi_512_7", "cql_512_7", "sac_512_7")
NAMES = ("Lookahead rule", "FQI", "Discrete CQL", "Offline discrete SAC")
COLORS = ("#206f83", "#c88828", "#8055a4", "#b94261")


def collect_rows(out, split):
    rows = []
    for path in sorted(out.glob(f"{split}_[0-9]*.jsonl")):
        rows.extend(json.loads(line) for line in path.read_text().splitlines())
    rows.sort(key=lambda row: row["index"])
    if not rows or len({r["index"] for r in rows}) != len(rows):
        raise ValueError("missing or duplicated test records")
    if not np.array_equal([r["index"] for r in rows], np.arange(len(rows))):
        raise ValueError("case indices not contiguous")
    return rows


def average(rows, extract):
    return float(np.mean([extract(row) for row in rows]))


def pooled_rmse(rows, extract):
    return float(np.sqrt(average(rows, lambda row: extract(row) ** 2)))


def arm_metrics(rows):
    return {
        "n": len(rows),
        "strict": sum(r["strict"] for r in rows),
        "both_correct": sum(r["both_correct"] for r in rows),
        "wrong_accept": sum(r["wrong_accept"] for r in rows),
        "wrong_clutter": sum(r["wrong_clutter"] for r in rows),
        "wrong_branch": sum(r["wrong_branch"] for r in rows),
        "partial_or_no_confirmation": sum(r["confirmed_labels"] < 2 for r in rows),
        "material_within_3mm": sum(r["material_within_3mm"] for r in rows),
        "prefix_provisional_wrong": sum(r["prefix_provisional_wrong"] for r in rows),
        "prefix_confident_wrong": sum(r["prefix_confident_wrong"] for r in rows),
        "prefix_reassociated": sum(r["prefix_reassociated"] for r in rows),
        "prefix_wrong_replaced": sum(r["prefix_wrong_replaced"] for r in rows),
        "prefix_clutter_replaced": sum(r["prefix_clutter_replaced"] for r in rows),
        "prefix_branch_replaced": sum(r["prefix_branch_replaced"] for r in rows),
        "prefix_confident_wrong_replaced": sum(r["prefix_confident_wrong_replaced"] for r in rows),
        "new_confident_wrong": sum(r["new_confident_wrong"] for r in rows),
        "current_rmse_mm": pooled_rmse(rows, lambda r: r["current"]["rmse_mm"]),
        "future_half_s_rmse_mm": pooled_rmse(rows, lambda r: r["future"][0]["rmse_mm"]),
        "future_one_s_rmse_mm": pooled_rmse(rows, lambda r: r["future"][1]["rmse_mm"]),
        "fixed_clock_rmse_mm": pooled_rmse(rows, lambda r: r["common_clock"]["rmse_mm"]),
        "future_half_s_90_coverage": average(rows, lambda r: r["future"][0]["coordinate_90_coverage"]),
        "future_one_s_90_coverage": average(rows, lambda r: r["future"][1]["coordinate_90_coverage"]),
        "fixed_clock_90_coverage": average(rows, lambda r: r["common_clock"]["coordinate_90_coverage"]),
        "z_rmse_mm": pooled_rmse(rows, lambda r: r["z_error_mm"]),
        "mean_duration_s": average(rows, lambda r: r["elapsed_s"]),
        "mean_travel_mm": average(rows, lambda r: r["travel_mm"]),
        "mean_captures": average(rows, lambda r: r["captures"]),
    }


def paired_metrics(rule, variant, random):
    n = len(rule)
    bootstrap = random.integers(0, n, size=(4000, n))
    def delta(key):
        difference = np.array([float(b[key]) - float(a[key]) for a, b in zip(rule, variant)])
        low, high = np.quantile(difference[bootstrap].mean(axis=1), [.025, .975])
        return dict(mean=float(difference.mean()), ci95=[float(low), float(high)],
                    wins=int(np.sum(difference > 0)), losses=int(np.sum(difference < 0)))
    x = np.square([a["common_clock"]["rmse_mm"] for a in rule])
    y = np.square([b["common_clock"]["rmse_mm"] for b in variant])
    interval = np.quantile(np.sqrt(y[bootstrap].mean(axis=1)) - np.sqrt(x[bootstrap].mean(axis=1)), [.025, .975])
    return {
        "strict": delta("strict"), "both_correct": delta("both_correct"), "wrong_accept": delta("wrong_accept"),
        "fixed_clock_rmse_delta_mm": float(np.sqrt(y.mean()) - np.sqrt(x.mean())),
        "fixed_clock_rmse_delta_ci95_mm": interval.tolist(),
    }


def data_diagnostics(out):
    reports = [r for p in sorted(out.glob("train_[0-9]*.json")) for r in json.loads(p.read_text())]
    if not reports:
        return {}
    prefixes = [r["branches"][0] for r in reports]
    candidates = [arm for r in reports for arm in r["branches"]]
    opportunity = [r for r in reports if r["branches"][0]["prefix_provisional_wrong"]]
    confident = [r for r in reports if r["branches"][0]["prefix_confident_wrong"]]
    return {
        "cases": len(reports),
        "candidate_sequences": len(candidates),
        "prefix_provisional_error_cases": len(opportunity),
        "prefix_confident_error_cases": len(confident),
        "prefix_provisional_error_records": sum(r["prefix_provisional_wrong"] for r in prefixes),
        "prefix_confident_error_records": sum(r["prefix_confident_wrong"] for r in prefixes),
        "provisional_error_cases_with_correct_replacement": sum(
            any(a["prefix_wrong_replaced"] for a in r["branches"]) for r in opportunity),
        "confident_error_cases_with_correct_replacement": sum(
            any(a["prefix_confident_wrong_replaced"] for a in r["branches"]) for r in confident),
        "identity_pass_and_fail_by_action_cases": sum(
            len({a["both_correct"] for a in r["branches"]}) > 1 for r in reports),
        "cases_with_clutter_replacement": sum(any(a["prefix_clutter_replaced"] for a in r["branches"]) for r in reports),
        "cases_with_branch_replacement": sum(any(a["prefix_branch_replaced"] for a in r["branches"]) for r in reports),
    }


def xyz_trace(xyz, title, color, *, dash=None, width=4, markers=False, text=None, group=None):
    xyz = np.asarray(xyz)
    return go.Scatter3d(
        x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], name=title,
        mode="lines+markers" if markers else "lines",
        line={"color": color, "width": width, "dash": dash}, marker={"size": 3, "color": color},
        text=text, hovertemplate="%{text}<extra>%{fullData.name}</extra>" if text else None,
        legendgroup=group,
    )


def witness_xyz(world, frame, ordinal):
    q, site, view = frame["q"], frame["site"], frame["view"]
    pose = world._blend(world.scan.view_frames[:, site, view], q)
    lateral = pose[:, 1] / np.linalg.norm(pose[:, 1])
    inward = pose[:, 2] - lateral * (pose[:, 2] @ lateral)
    inward /= np.linalg.norm(inward)
    x, depth = frame["centroids_mm"][ordinal]
    return world.probe_world(q, site, view) + x * lateral + depth * inward


def comparison_3d(rows, summary, world, out):
    from experiment import prepared
    selected = next((r for r in rows if r["runs"]["rule"]["prefix_provisional_wrong"]
                     and len({r["runs"][a]["both_correct"] for a in MAIN}) > 1),
                    next((r for r in rows if r["runs"]["rule"]["prefix_provisional_wrong"]), rows[0]))
    runs = selected["runs"]
    fig = make_subplots(
        rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "xy"}]], column_widths=[.63, .37],
        subplot_titles=("Acquisition poses and fixed-time vessel forecasts", "Position prediction on all held-out cases"),
    )
    prefix = runs["rule"]["trace"][:selected["prefix_acquisitions"]]
    sweep = xyz_trace([f["probe_xyz_mm"] for f in prefix], "Shared rule sweep", "#a8b4b8",
                      dash="dot", width=3)
    sweep.visible = "legendonly"
    fig.add_trace(sweep, row=1, col=1)
    original = prepared(world, world.case(selected["seed"]))
    keys, _ = original.belief.decisions()
    wrong_xyz = [witness_xyz(world, prefix[key[0]], key[1]) for label, key in enumerate(keys)
                 if key is not None and original.labels[key] != label]
    if wrong_xyz:
        points = np.asarray(wrong_xyz)
        fig.add_trace(go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2],
                                   mode="markers", name="Misassigned prefix detection",
                                   marker={"symbol": "x", "size": 7, "color": "#c23532"}),
                      row=1, col=1)
    truth = np.asarray(runs["rule"]["common_clock"]["truth_xyz_mm"])
    groups = [[i for i, (_, label) in enumerate(world.query_sites) if label == l] for l in range(2)]
    for i, group in enumerate(groups):
        fig.add_trace(xyz_trace(truth[group], ("True main vessel", "True branch")[i], "#242a30",
                                width=5, markers=True), row=1, col=1)
    for arm, label, color in zip(MAIN, NAMES, COLORS):
        run = runs[arm]
        path = [prefix[-1], *run["trace"][selected["prefix_acquisitions"]:]]
        status = "both verified" if run["both_correct"] else "wrong confirmation" if run["wrong_accept"] else "incomplete"
        hover = [f"{frame['t'] - prefix[-1]['t']:.2f} s after sweep" for frame in path]
        fig.add_trace(xyz_trace([f["probe_xyz_mm"] for f in path], f"{label} · {status}", color,
                                dash="dot", markers=True, width=4, text=hover, group=arm), row=1, col=1)
        predict = np.asarray(run["common_clock"]["mean_xyz_mm"])
        for group in groups:
            line = xyz_trace(predict[group], f"{label} forecast", color, width=3, group=arm)
            line.showlegend = False
            fig.add_trace(line, row=1, col=1)
        metrics = summary["arms"][arm]
        fig.add_trace(go.Bar(x=["+0.5 s", "+1.0 s", "Shared clock"],
                             y=[metrics["future_half_s_rmse_mm"], metrics["future_one_s_rmse_mm"],
                                metrics["fixed_clock_rmse_mm"]],
                             name=label, marker_color=color, legendgroup=arm, showlegend=False),
                      row=1, col=2)
    fig.update_scenes(aspectmode="data", xaxis_title="World X (mm)", yaxis_title="World Y (mm)",
                      zaxis_title="World Z (mm)", camera={"eye": {"x": 2.0, "y": 1.8, "z": 1.25}})
    fig.update_yaxes(title_text="3D position RMSE (mm)", rangemode="tozero", row=1, col=2)
    fig.update_layout(template="plotly_white", height=780,
                      title="Knee-only recheck policies<br><sup>Dotted lines connect capture poses · Solid lines show vessel locations</sup>",
                      legend={"orientation": "h", "y": -.13}, margin={"l": 24, "r": 26, "t": 112, "b": 110})
    fig.write_html(out / "RECHECK_3D.html", include_plotlyjs=True)
    return selected["index"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--split", default="test")
    parser.add_argument("--data-only", action="store_true")
    args = parser.parse_args()
    data = data_diagnostics(args.out)
    if args.data_only:
        print(json.dumps(data, indent=2))
        return
    from knee_world import World
    rows = collect_rows(args.out, args.split)
    if any(set(row["runs"]) != set(rows[0]["runs"]) for row in rows):
        raise ValueError("inconsistent model arms")
    arms = {arm: [r["runs"][arm] for r in rows] for arm in rows[0]["runs"]}
    random = np.random.default_rng(93681)
    summary = {"cases": len(rows), "training_support": data,
               "arms": {arm: arm_metrics(values) for arm, values in arms.items()},
               "paired": {arm: paired_metrics(arms["rule"], values, random)
                          for arm, values in arms.items() if arm != "rule"}}
    summary["visualized_test_index"] = comparison_3d(rows, summary, World(), args.out)
    (args.out / "SUMMARY.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print("cases", len(rows), "main", {arm: summary["arms"][arm] for arm in MAIN}, flush=True)


if __name__ == "__main__":
    main()
