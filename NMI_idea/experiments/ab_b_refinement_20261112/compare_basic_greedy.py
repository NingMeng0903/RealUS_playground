"""Compare the historical one-step rule and the refined policies on paired cases."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from plot_counterfactual import _load_geometry, _path_points
from summarize_refinement import collect, _arm_summary, _paired_difference


ARMS = {
    "basic_A": ("基础贪心 A", "#a76736"),
    "basic_A_predictive_gate": ("基础 A ＋共同验收", "#a983bd"),
    "greedy_no_fisher": ("新版单步 A · 无 Fisher", "#519483"),
    "A": ("新版单步 A", "#5379b2"),
    "locked_seen": ("按观测状态续行", "#a15c87"),
    "B_seen": ("B · 完整后验续行", "#c73d64"),
}


def example(cases):
    eligible = [
        (int(seed), rows) for seed, rows in cases.items()
        if rows["B_seen"]["complete"] and not rows["basic_A"]["complete"]
    ]
    if eligible:
        return min(eligible, key=lambda v: (v[1]["basic_A"]["online_stop"], v[0]))
    return min((int(seed), rows) for seed, rows in cases.items())


def plot(cases, seed, rows, out):
    geometry = _load_geometry()
    figure = make_subplots(
        rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "xy"}]],
        column_widths=[.57, .43], horizontal_spacing=.09,
        subplot_titles=("同一首扫之后的补扫路径", "严格完成随补采次数变化"),
    )
    for key, label, color in (("trunk", "主干模板", "#648b96"), ("branch", "侧支模板", "#adbaba")):
        xyz = np.array(geometry[key])
        if len(xyz):
            figure.add_trace(go.Scatter3d(
                x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], name=label,
                mode="lines", opacity=.55, line={"color": color, "width": 5},
            ), row=1, col=1)
    for arm in ("basic_A", "B_seen"):
        row = rows[arm]
        xyz = np.array(_path_points(row))
        if not len(xyz):
            continue
        label, color = ARMS[arm]
        figure.add_trace(go.Scatter3d(
            x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
            name=f"{label} · {len(row['steps'])} 次 · {row['recheck_path_mm']/1000.:.2f} m",
            mode="lines", legendgroup=arm,
            line={"color": color, "width": 5, "dash": "dot" if arm == "basic_A" else "solid"},
        ), row=1, col=1)
        end = np.array([s["path_m"][-1] for s in row["steps"]]) * 1000.
        steps = np.arange(1, len(end) + 1)
        sites = [s["action"][0] for s in row["steps"]]
        figure.add_trace(go.Scatter3d(
            x=end[:, 0], y=end[:, 1], z=end[:, 2],
            customdata=np.column_stack([steps, sites]),
            hovertemplate="第 %{customdata[0]} 次 · 站点 %{customdata[1]}<extra></extra>",
            marker={"color": color, "size": 3}, mode="markers", legendgroup=arm, showlegend=False,
        ), row=1, col=1)
    start = np.array(rows["B_seen"]["steps"][0]["path_m"][0]) * 1000.
    figure.add_trace(go.Scatter3d(
        x=[start[0]], y=[start[1]], z=[start[2]], mode="markers",
        marker={"size": 5, "color": "#233146"}, name="共同补扫起点",
    ), row=1, col=1)
    cap = np.arange(81)
    for arm, (label, color) in ARMS.items():
        counts = [sum(v[arm]["complete"] and len(v[arm]["steps"]) <= k
                      for v in cases.values()) for k in cap]
        figure.add_trace(go.Scatter(
            x=cap, y=np.asarray(counts) / len(cases), customdata=counts, name=label,
            mode="lines", line={"shape": "hv", "color": color, "width": 2.3},
            legendgroup=arm, showlegend=arm not in {"basic_A", "B_seen"},
            hovertemplate=f"{label}<br>≤ %{{x}} 次 · %{{customdata}} / {len(cases)}<extra></extra>",
        ), row=1, col=2)
    figure.update_xaxes(title_text="最多补采次数", range=[20, 80], row=1, col=2)
    figure.update_yaxes(title_text="严格完成率", tickformat=".0%", range=[0, 1.02], row=1, col=2)
    figure.update_scenes(
        aspectmode="data", xaxis_title="X (mm)", yaxis_title="Y (mm)", zaxis_title="Z (mm)",
        camera={"eye": {"x": 1.75, "y": 1.1, "z": 1.2}},
    )
    figure.update_layout(
        template="plotly_white", height=770, margin={"l": 24, "r": 34, "t": 123, "b": 122},
        title={"text": "基础贪心 A 与 B 的配对补扫<br>"
               f"<sup>{len(cases)} 个模拟病例 · 示例种子 {seed} · 15° 图册参考</sup>", "x": .04, "y": .98},
        legend={"orientation": "h", "x": 0., "y": -.15, "groupclick": "togglegroup"},
    )
    figure.write_html(out, include_plotlyjs=True, auto_open=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    _, configs, cases = collect(args.runs)
    if len(cases) != 72 or any(not set(ARMS).issubset(rows) for rows in cases.values()):
        raise ValueError("expected 72 complete pairs of all comparison arms")
    for rows in cases.values():
        assert len({rows[arm]["prefix_acquisitions"] for arm in ARMS}) == 1
        assert len({rows[arm]["truth_rejections"] for arm in ARMS}) == 1
    seed, picked = example(cases)
    result = {
        "cases": len(cases), "shards": sorted(configs),
        "arms": {name: _arm_summary([rows[name] for rows in cases.values()]) for name in ARMS},
        "paired": {name: _paired_difference(cases, "B_seen", name) for name in ARMS if name != "B_seen"},
        "example": {
            "seed": seed,
            "arms": {
                name: {
                    "complete": picked[name]["complete"], "online_stop": picked[name]["online_stop"],
                    "ready": picked[name]["certified"], "captures": len(picked[name]["steps"]),
                    "travel_mm": picked[name]["recheck_path_mm"],
                    "most_visited_sites": Counter(s["action"][0] for s in picked[name]["steps"]).most_common(3),
                }
                for name in ("basic_A", "B_seen")
            },
        },
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "SUMMARY.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    plot(cases, seed, picked, args.out / "BASIC_GREEDY_3D.html")
    print({k: v["strict_successes"] for k, v in result["arms"].items()})


if __name__ == "__main__":
    main()
