"""Audit the frozen data-quality holdout and show paired recheck references."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from compare_data_quality import _arm_metrics, _paired
from fit_q import load_transition_shards
from summarize_rl import _load_geometry, _path_points


HERE = Path(__file__).resolve().parent
OUT = HERE / "data_quality_20261112"
ARMS = {
    "rule": ("观测状态规则", "#277083"),
    "original": ("原 RL", "#8374a7"),
    "uniform": ("RL + 均匀数据", "#c87931"),
    "fisher": ("RL + Fisher 数据", "#b2486d"),
    "uniform_matched": ("RL + 等训练行数均匀数据", "#4e8266"),
}


def load_cases(path):
    records = [json.loads(line) for shard in path.glob("eval_*.jsonl") for line in shard.open()]
    indices = sorted(row["seed_index"] for row in records)
    if indices != list(range(48)) or len({row["episode_seed"] for row in records}) != 48:
        raise ValueError("expected 48 complete paired holdout records")
    if any(set(row["arms"]) != set(ARMS) or row["seed_base"] != 94286251 for row in records):
        raise ValueError("unexpected holdout arms or seeds")
    return sorted(records, key=lambda row: row["seed_index"])


def metrics(records):
    cases = {row["episode_seed"]: row["arms"] for row in records}
    result = {
        "arms": {arm: _arm_metrics([case[arm] for case in cases.values()]) for arm in ARMS},
        "paired": {f"{arm}_vs_rule": _paired(cases, arm, "rule") for arm in ARMS if arm != "rule"},
    }
    for first, second in (("uniform", "original"), ("fisher", "uniform"),
                          ("fisher", "uniform_matched")):
        result["paired"][f"{first}_vs_{second}"] = _paired(cases, first, second)
    for name, info in result["arms"].items():
        rows = [row[name] for row in cases.values() if not row[name]["strict_complete"]]
        if all("map_max_error_mm" in row and "incorrect_witnesses" in row for row in rows):
            info["identity_failures"] = sum(bool(row["incorrect_witnesses"]) for row in rows)
            info["map_error_failures"] = sum(row["map_max_error_mm"] > 3. for row in rows)
    return result


def plot(records, example, path):
    fig = make_subplots(
        rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "xy"}]], column_widths=[.55, .45],
        subplot_titles=("同一病例的补采位置参考", "按补采次数累计完成"),
    )
    geometry = _load_geometry()
    for key, title in (("trunk", "主干模板"), ("branch", "分支模板")):
        xyz = np.asarray(geometry[key])
        fig.add_trace(go.Scatter3d(
            x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="lines", name=title,
            line={"color": "#a3adb2", "width": 4},
        ), row=1, col=1)
    for arm, (name, color) in ARMS.items():
        case = example["arms"][arm]
        xyz = np.asarray(_path_points(case))
        if xyz.size:
            fig.add_trace(go.Scatter3d(
                x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="lines", name=name, legendgroup=arm,
                line={"color": color, "width": 5},
                visible=True if arm in ("rule", "uniform", "fisher") else "legendonly",
            ), row=1, col=1)
        caps = np.arange(81)
        complete_counts = [sum(row["arms"][arm]["strict_complete"] and
                               row["arms"][arm]["captures"] <= cap for row in records) for cap in caps]
        fig.add_trace(go.Scatter(
            x=caps, y=np.asarray(complete_counts) / len(records), name=name, legendgroup=arm,
            showlegend=False, line={"shape": "hv", "color": color, "width": 3},
            customdata=complete_counts, hovertemplate=f"%{{x}} 次 · %{{customdata}}/48<extra>{name}</extra>",
        ), row=1, col=2)
    fig.update_scenes(aspectmode="data", camera={"eye": {"x": 1.65, "y": 1.65, "z": 1.65}},
                      xaxis_title="X (mm)", yaxis_title="Y (mm)", zaxis_title="Z (mm)")
    fig.update_xaxes(title_text="最多补采次数", range=[20, 80], row=1, col=2)
    fig.update_yaxes(title_text="严格完成率", tickformat=".0%", range=[0., 1.02], row=1, col=2)
    fig.update_layout(
        template="plotly_white", height=745, title={
            "text": "规则首扫后的训练数据对照<br>"
                    f"<sup>48 个新留出模拟病例 · 3D 示例为留出序号 {example['seed_index'] + 1} · 未运行控制器</sup>",
            "x": .035,
        },
        legend={"orientation": "h", "y": -.12, "groupclick": "toggleitem"},
        margin={"l": 25, "r": 32, "t": 115, "b": 115},
    )
    fig.write_html(path, include_plotlyjs=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    records = load_cases(args.out)
    paths = [*HERE.glob("train_[0-9]*/TRANSITIONS.npz"), *args.out.glob("*_[0-9]*/TRANSITIONS.npz")]
    train_seeds = set()
    for path in paths:
        train_seeds.update(load_transition_shards(path).episode_seed.tolist())
    if train_seeds.intersection(row["episode_seed"] for row in records) or len(train_seeds) != 192:
        raise ValueError("training and holdout seed audit failed")
    example = next((row for row in records
                    if row["arms"]["fisher"]["strict_complete"] != row["arms"]["uniform"]["strict_complete"]),
                   records[0])
    result = {
        "evaluation_cases": len(records), "unique_training_seeds": len(train_seeds),
        "example_index": example["seed_index"],
        **metrics(records), "blocks": [metrics(records[:24]), metrics(records[24:])],
    }
    (args.out / "EVALUATION_SUMMARY.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    plot(records, example, args.out / "DATA_QUALITY_3D.html")
    print(json.dumps(result["arms"], indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
