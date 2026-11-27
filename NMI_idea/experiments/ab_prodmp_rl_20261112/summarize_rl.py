"""Evaluate frozen suffix policies on paired closed-loop rechecks."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import binomtest

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "ab_b_refinement_20261112"
sys.path.insert(0, str(BASE))
from plot_counterfactual import _load_geometry, _path_points  # noqa: E402

ARMS = {
    "rule": ("观测状态规则", "#397c9d"),
    "immediate": ("当前后验评分", "#8e829e"),
    "mc": ("回报回归", "#548261"),
    "rl": ("离线 RL", "#c34c67"),
}
CONTROL_NAMES = {"locked_seen": "rule", "B_seen": "immediate"}


def load_rows(paths, mc_paths):
    cases = defaultdict(dict)
    inputs = [(p, False) for p in paths] + [(p, True) for p in mc_paths]
    for path, force_mc in inputs:
        for line in (path / "RUNS.jsonl").open():
            row = json.loads(line)
            name = row.get("arm", row.get("policy"))
            row["arm"] = "mc" if force_mc else CONTROL_NAMES.get(name, name)
            key = row["seed"]
            if row["arm"] in cases[key]:
                raise ValueError(f"duplicate case and arm: {key}, {row['arm']}")
            cases[key][row["arm"]] = row
    return dict(sorted(cases.items()))


def length_m(row):
    return row["recheck_path_mm"] / 1000.


def arm_metrics(rows):
    return {
        "n": len(rows),
        "success": sum(r["complete"] for r in rows),
        "false_stops": sum(r["online_stop"] and not r["complete"] for r in rows),
        "no_stops": sum(not r["online_stop"] for r in rows),
        "mean_rechecks": float(np.mean([len(r["steps"]) for r in rows])),
        "mean_travel_m": float(np.mean([length_m(r) for r in rows])),
        "mean_undiscounted_return": float(np.mean([
            float(r["complete"]) - .002 * len(r["steps"]) - .05 * length_m(r) for r in rows
        ])),
        "suffix_changes": sum(s["changed_suffix"] for r in rows for s in r["steps"]),
        "executed_tails": sum(bool(s["tail_choice"]) for r in rows for s in r["steps"]),
        "cases_changed_suffix": sum(any(s["changed_suffix"] for s in r["steps"]) for r in rows),
        "success_by_40_captures_and_2_5_m": sum(
            r["complete"] and len(r["steps"]) <= 40 and length_m(r) <= 2.5 for r in rows
        ),
    }


def same_actions(a, b):
    return [(s["pose"], s["action"]) for s in a["steps"]] == [
        (s["pose"], s["action"]) for s in b["steps"]
    ]


def paired(cases, other):
    rows = [(case["rl"], case[other]) for case in cases.values()]
    success = np.array([int(a["complete"]) - int(b["complete"]) for a, b in rows])
    travel = np.array([length_m(a) - length_m(b) for a, b in rows])
    frames = np.array([len(a["steps"]) - len(b["steps"]) for a, b in rows])
    wins, losses = int((success > 0).sum()), int((success < 0).sum())
    draw = np.random.default_rng(356).integers(0, len(rows), size=(8000, len(rows)))
    return {
        "wins": wins, "losses": losses,
        "success_delta_pp": float(100. * success.mean()),
        "exact_p_two_sided": float(binomtest(wins, wins + losses, .5).pvalue) if wins + losses else 1.,
        "success_bootstrap_95_pp": np.percentile(100. * success[draw].mean(axis=1), [2.5, 97.5]).tolist(),
        "travel_delta_m": float(travel.mean()),
        "travel_bootstrap_95_m": np.percentile(travel[draw].mean(axis=1), [2.5, 97.5]).tolist(),
        "recheck_delta": float(frames.mean()),
        "same_action_sequences": sum(same_actions(a, b) for a, b in rows),
    }


def choose_example(cases):
    differing = [key for key, case in cases.items() if case["rl"]["complete"] != case["rule"]["complete"]]
    if differing:
        return min(differing)
    return max(cases, key=lambda k: abs(length_m(cases[k]["rl"]) - length_m(cases[k]["rule"])))


def plot(cases, seed, output):
    figure = make_subplots(
        rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "xy"}]],
        column_widths=[.58, .42], horizontal_spacing=.08,
        subplot_titles=("补扫位置参考（每段 9 点）", "累计严格完成率"),
    )
    geometry = _load_geometry()
    for key, name in (("trunk", "主干模板"), ("branch", "侧支模板")):
        coords = np.asarray(geometry[key])
        if coords.size:
            figure.add_trace(go.Scatter3d(
                x=coords[:, 0], y=coords[:, 1], z=coords[:, 2], name=name,
                mode="lines", line={"color": "#a4b2b7", "width": 4},
            ), row=1, col=1)
    case = cases[seed]
    for arm in ("rule", "rl"):
        name, color = ARMS[arm]
        row = case[arm]
        xyz = np.asarray(_path_points(row))
        if xyz.size:
            figure.add_trace(go.Scatter3d(
                x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="lines",
                name=f"{name} · {len(row['steps'])} 次 · {length_m(row):.2f} m",
                legendgroup=arm, line={"color": color, "width": 5},
            ), row=1, col=1)
            endpoints = 1000. * np.asarray([s["path_m"][-1] for s in row["steps"]])
            custom = np.asarray([[i + 1, s["action"][0]] for i, s in enumerate(row["steps"])])
            figure.add_trace(go.Scatter3d(
                x=endpoints[:, 0], y=endpoints[:, 1], z=endpoints[:, 2], mode="markers",
                name=name, marker={"color": color, "size": 3}, customdata=custom,
                hovertemplate="第 %{customdata[0]} 次补采 · 站点 %{customdata[1]}<extra></extra>",
                legendgroup=arm, showlegend=False,
            ), row=1, col=1)
    caps = np.arange(81)
    for arm in next(iter(cases.values())):
        name, color = ARMS[arm]
        counts = [
            sum(case[arm]["complete"] and len(case[arm]["steps"]) <= n for case in cases.values())
            for n in caps
        ]
        figure.add_trace(go.Scatter(
            x=caps, y=np.asarray(counts) / len(cases), customdata=counts, name=name, mode="lines",
            legendgroup=arm, showlegend=arm not in {"rule", "rl"},
            line={"shape": "hv", "width": 2.5, "color": color},
            hovertemplate=f"{name}<br>≤ %{{x}} 次 · %{{customdata}} / {len(cases)}<extra></extra>",
        ), row=1, col=2)
    figure.update_xaxes(title_text="最多补采次数", range=[20, 80], row=1, col=2)
    figure.update_yaxes(title_text="严格完成率", range=[0, 1.02], tickformat=".0%", row=1, col=2)
    figure.update_scenes(aspectmode="data", xaxis_title="X (mm)", yaxis_title="Y (mm)", zaxis_title="Z (mm)")
    figure.update_layout(
        template="plotly_white", height=765, margin={"l": 24, "r": 35, "t": 125, "b": 124},
        title={"text": "离线 RL 与规则选择 ProDMP 续行<br>"
                       f"<sup>{len(cases)} 个留出模拟病例 · 示例 {seed} · 未运行控制器 · 15° 图册参考</sup>",
               "x": .04, "y": .98},
        legend={"orientation": "h", "x": 0., "y": -.15, "groupclick": "togglegroup"},
    )
    figure.write_html(output, include_plotlyjs=True, auto_open=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval", type=Path, nargs="+", required=True)
    parser.add_argument("--mc", type=Path, nargs="+", default=[])
    parser.add_argument("--train", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cases", type=int, default=48)
    args = parser.parse_args()
    cases = load_rows(args.eval, args.mc)
    arms = [name for name in ARMS if name != "mc" or args.mc]
    if len(cases) != args.cases or any(set(row) != set(arms) for row in cases.values()):
        raise ValueError(f"expected {args.cases} complete paired cases")
    for row in cases.values():
        if len({r["prefix_acquisitions"] for r in row.values()}) != 1:
            raise ValueError("shared prefix counts disagree")
    train_seeds, transitions, alternatives = set(), 0, 0
    for path in args.train:
        with np.load(path / "TRANSITIONS.npz", allow_pickle=False) as d:
            train_seeds.update(d["episode_seed"].tolist())
            transitions += len(d["actions"])
            alternatives += int(np.count_nonzero(d["actions"] != 0))
    if train_seeds.intersection(cases):
        raise ValueError("train and evaluation cases overlap")
    sample = choose_example(cases)
    result = {
        "training_cases_with_transitions": len(train_seeds), "transitions": transitions,
        "training_nonrule_actions": alternatives, "evaluation_cases": len(cases),
        "arms": {a: arm_metrics([r[a] for r in cases.values()]) for a in arms},
        "paired": {a: paired(cases, a) for a in arms if a != "rl"},
        "visualization_seed": sample,
        "sample": {
            a: {key: cases[sample][a][key]
                for key in ("complete", "online_stop", "map_max_error_mm", "recheck_path_mm")}
            for a in arms
        },
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "SUMMARY.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    plot(cases, sample, args.out / "RL_RECHECK_3D.html")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
