"""Render the locked/B_once counterfactual for the requested seed.

The report is intentionally a small, reproducible Plotly artifact.  It uses
only the public run traces for the route comparison; evaluation-only labels
are kept out of hover text and plotting decisions.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
GEOMETRY = HERE.parent / "ab_leg_dynamic_20261112" / "DYNAMIC_GEOMETRY_FROZEN.npz"
DEFAULT_SEED = 2998748339
HEAD_COUNT = 15


def _plotly_import():
    """Load Plotly from an available workspace environment if needed."""

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        return go, make_subplots
    except ModuleNotFoundError:
        candidates = [
            "/media/camp/EXT_DRIVE/envs/genesis/lib/python3.10/site-packages",
            "/media/camp/EXT_DRIVE/envs/rm75/lib/python3.10/site-packages",
            "/media/camp/EXT_DRIVE/envs/camera_calib/lib/python3.10/site-packages",
        ]
        for candidate in candidates:
            if candidate not in sys.path:
                sys.path.insert(0, candidate)
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        return go, make_subplots


def _collect(run_dirs: Sequence[Path]) -> dict[str, dict[str, dict]]:
    """Reuse the shard collector without importing any plotting code."""

    from summarize_refinement import collect

    _, _, cases = collect(run_dirs)
    return cases


def _load_geometry() -> dict[str, Any]:
    sys.path.insert(0, str(GEOMETRY.parent))
    from dynamic_scan import DynamicScan

    with np.load(GEOMETRY) as data:
        scan = DynamicScan(data)
        pose = int(scan.bent)
        return {
            "pose": pose,
            "angle_deg": float(scan.angles[pose]),
            "trunk": (1000.0 * np.asarray(scan.trunk[pose])).tolist(),
            "branch": (1000.0 * np.asarray(scan.dense_branch[pose])).tolist(),
        }


def _steps(row: Mapping[str, Any]) -> list[dict]:
    return [step for step in row.get("steps", []) if isinstance(step, dict)]


def _path_points(row: Mapping[str, Any], start: int = 0, stop: int | None = None) -> list[list[float]]:
    result: list[list[float]] = []
    for step in _steps(row)[start:stop]:
        for point in step.get("path_m", []) or []:
            if len(point) != 3:
                continue
            result.append([1000.0 * float(v) for v in point])
    return result


def _endpoints(row: Mapping[str, Any], start: int = 0, stop: int | None = None) -> list[list[float]]:
    result = []
    for step in _steps(row)[start:stop]:
        path = step.get("path_m", []) or []
        if path and len(path[-1]) == 3:
            result.append([1000.0 * float(v) for v in path[-1]])
    return result


def _common_head(left: Mapping[str, Any], right: Mapping[str, Any], count: int) -> tuple[bool, dict[str, Any]]:
    left_steps, right_steps = _steps(left), _steps(right)
    same = True
    checks = []
    for index in range(min(count, len(left_steps), len(right_steps))):
        a, b = left_steps[index], right_steps[index]
        actions_same = a.get("pose") == b.get("pose") and a.get("action") == b.get("action")
        observations_same = np.allclose(a.get("centroids_mm", []), b.get("centroids_mm", []), atol=1.0e-12, rtol=0.0)
        cumulative_same = np.isclose(float(a.get("cumulative_m", 0.0)), float(b.get("cumulative_m", 0.0)), atol=1.0e-12, rtol=0.0)
        checks.append({"acquisition": index + 1, "actions": actions_same, "centroids": bool(observations_same), "cumulative_m": bool(cumulative_same)})
        same &= actions_same and bool(observations_same) and bool(cumulative_same)
    if len(left_steps) < count or len(right_steps) < count:
        same = False
    return same, {"count": count, "all_public_fields_equal": bool(same), "checks": checks}


def _path_m(row: Mapping[str, Any]) -> float:
    if "recheck_path_mm" in row:
        return float(row["recheck_path_mm"]) / 1000.0
    steps = _steps(row)
    if not steps:
        return 0.0
    return float(steps[-1].get("cumulative_m", sum(float(s.get("length_m", 0.0)) for s in steps)))


def _common_measured(cases: Mapping[str, Mapping[str, dict]], arms: Sequence[str]) -> list[tuple[str, dict[str, dict]]]:
    return [(seed, {arm: cases[seed][arm] for arm in arms}) for seed in sorted(cases) if all(arm in cases[seed] for arm in arms)]


def _success_distance_curves(cases: Mapping[str, Mapping[str, dict]], arms: Sequence[str]) -> tuple[dict[str, Any], list[str]]:
    paired = _common_measured(cases, arms)
    seeds = [seed for seed, _ in paired]
    if not paired:
        return {arm: {"x": [], "y": [], "n": 0} for arm in arms}, seeds
    max_distance = max(_path_m(rows[arm]) for _, rows in paired for arm in arms)
    x = np.linspace(0.0, max(3.0, max_distance) * 1.02, 31)
    curves = {}
    for arm in arms:
        values = []
        for limit in x:
            values.append(float(np.mean([
                bool(rows[arm].get("complete", False)) and _path_m(rows[arm]) <= limit + 1.0e-12
                for _, rows in paired
            ])))
        curves[arm] = {"x": x.tolist(), "y": values, "n": len(paired)}
    return curves, seeds


def _capture_text(row: Mapping[str, Any]) -> str:
    steps = _steps(row)
    centroids = sum(len(step.get("centroids_mm", []) or []) for step in steps)
    return f"{len(steps)} 次补充采集；{centroids} 个公开质心；路径 {_path_m(row):.3f} m"


def build_figure(cases: Mapping[str, Mapping[str, dict]], seed: int = DEFAULT_SEED):
    go, make_subplots = _plotly_import()
    key = str(seed)
    if key not in cases:
        raise KeyError(f"seed {seed} not found in supplied shards")
    case = cases[key]
    for arm in ("locked", "B_once"):
        if arm not in case:
            raise KeyError(f"seed {seed} lacks required arm {arm!r}")
    locked, b_once = case["locked"], case["B_once"]
    geometry = _load_geometry()
    head_equal, head_audit = _common_head(locked, b_once, HEAD_COUNT)

    figure = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "xy"}]],
        column_widths=[0.68, 0.32],
        horizontal_spacing=0.06,
        subplot_titles=(
            "第 16 次采集的两条续行",
            "严格完成率随路径长度变化",
        ),
    )

    def add_line(points, name, color, width, *, dash="solid", legendgroup=None, showlegend=True, hover=None):
        if not points:
            return
        values = np.asarray(points, dtype=float)
        kwargs = {
            "x": values[:, 0], "y": values[:, 1], "z": values[:, 2],
            "mode": "lines", "name": name,
            "line": {"color": color, "width": width, "dash": dash},
            "legendgroup": legendgroup or name, "showlegend": showlegend,
            "hoverinfo": "text" if hover else "skip",
        }
        if hover:
            kwargs["text"] = hover
        figure.add_trace(go.Scatter3d(**kwargs), row=1, col=1)

    add_line(geometry["trunk"], "主干模板（模拟）", "#11a8a1", 7, legendgroup="geometry")
    add_line(geometry["branch"], "分支模板（模拟）", "#f2a65a", 7, legendgroup="geometry")

    # The first 15 acquisitions are required to be a public-identical head.
    # Draw it once, then emphasize the two different immediate tails at #16.
    head = _path_points(b_once, 0, HEAD_COUNT)
    add_line(head, "共同头部 · 1–15（空观测在第15次）", "#8899a6", 5, legendgroup="head")
    add_line(_path_points(b_once, HEAD_COUNT, HEAD_COUNT + 1), "B_once 即时尾段 · #16", "#df416c", 14, legendgroup="B_once_tail")
    add_line(_path_points(locked, HEAD_COUNT, HEAD_COUNT + 1), "locked 即时尾段 · #16", "#3b78c8", 14, legendgroup="locked_tail")
    add_line(_path_points(b_once, HEAD_COUNT + 1), "B_once 最终路线", "#df416c", 4, dash="solid", legendgroup="B_once")
    add_line(_path_points(locked, HEAD_COUNT + 1), "locked 最终路线", "#3b78c8", 4, dash="dash", legendgroup="locked")

    for row, arm, color in ((b_once, "B_once", "#df416c"), (locked, "locked", "#3b78c8")):
        points = _endpoints(row)
        if points:
            values = np.asarray(points, dtype=float)
            figure.add_trace(go.Scatter3d(
                x=values[:, 0], y=values[:, 1], z=values[:, 2],
                mode="markers", name=f"{arm} 采集端点",
                marker={"size": 3.5, "color": color, "opacity": 0.7},
                legendgroup=arm, showlegend=True,
                text=[f"第 {index + 1} 次" for index in range(len(points))],
                hovertemplate="%{text}<extra>" + arm + "</extra>",
            ), row=1, col=1)

    curve_arms = ("old_fixed", "locked", "A", "B")
    curves, common_seeds = _success_distance_curves(cases, curve_arms)
    curve_style = {
        "old_fixed": ("固定旧策略", "#7e8791", "dash"),
        "locked": ("锁定", "#3b78c8", "dot"),
        "A": ("新 A", "#d08b38", "dashdot"),
        "B": ("新 B", "#df416c", "solid"),
    }
    for arm in curve_arms:
        name, color, dash = curve_style[arm]
        curve = curves[arm]
        figure.add_trace(go.Scatter(
            x=curve["x"], y=curve["y"], mode="lines+markers", name=name,
            line={"color": color, "dash": dash, "width": 2.2}, marker={"size": 4},
            customdata=np.full(len(curve["x"]), curve["n"]),
            hovertemplate="路径 ≤ %{x:.2f} m<br>严格成功率 %{y:.1%}<br>共同案例 %{customdata}<extra>" + name + "</extra>",
            legendgroup=arm, showlegend=True,
        ), row=1, col=2)

    all_xyz = np.asarray(
        geometry["trunk"] + geometry["branch"] + _path_points(b_once) + _path_points(locked),
        dtype=float,
    )
    if len(all_xyz):
        low, high = all_xyz.min(axis=0), all_xyz.max(axis=0)
        padding = np.maximum((high - low) * 0.04, 1.0)
        axis_ranges = [
            [float(low[index] - padding[index]), float(high[index] + padding[index])]
            for index in range(3)
        ]
    else:
        axis_ranges = [[-1.0, 1.0]] * 3
    figure.update_scenes(
        xaxis={"title": "X（mm）", "range": axis_ranges[0], "autorange": False},
        yaxis={"title": "Y（mm）", "range": axis_ranges[1], "autorange": False},
        zaxis={"title": "Z（mm）", "range": axis_ranges[2], "autorange": False},
        aspectmode="data",
        camera={"eye": {"x": 1.8, "y": 1.2, "z": 1.25}, "up": {"x": 0, "y": 0, "z": 1}},
        uirevision="fixed-scale-15deg",
        row=1, col=1,
    )
    figure.update_xaxes(title_text="路径长度（m）", range=[0, max(3.0, max(v for arm in curve_arms for v in curves[arm]["x"]))], row=1, col=2)
    figure.update_yaxes(title_text="严格成功率", range=[0, 1.04], tickformat=".0%", row=1, col=2)
    figure.update_layout(
        title={"text": f"反事实路线 · seed {seed} · 模拟几何 · 15°参考位姿", "x": 0.5, "y": .982},
        template="plotly_white",
        height=865,
        margin={"l": 24, "r": 26, "t": 122, "b": 150},
        legend={"orientation": "h", "y": -.16, "x": 0.01, "groupclick": "togglegroup"},
        hovermode="closest",
        uirevision="fixed-scale-15deg",
    )
    figure.add_annotation(
        text=f"前 15 次采集{'相同' if head_equal else '存在差异'}。第 15 次失检。加入有无检出标志后，固定组也选择就地补采。",
        x=.5, y=1.085, xref="paper", yref="paper", showarrow=False,
        font={"size": 12, "color": "#52616e"},
    )
    figure.add_annotation(
        text=f"{len(common_seeds)} 个共同病例 · 以已走路径回算",
        x=.84, y=-.104, xref="paper", yref="paper", showarrow=False,
        font={"size": 11, "color": "#52616e"},
    )
    audit = {
        "seed": seed,
        "geometry_simulated": True,
        "reference_angle_deg": geometry["angle_deg"],
        "common_head": head_audit,
        "head_acquisitions": HEAD_COUNT,
        "immediate_tail_acquisition": HEAD_COUNT + 1,
        "locked_capture_text": _capture_text(locked),
        "B_once_capture_text": _capture_text(b_once),
        "common_measured_cases": len(common_seeds),
        "common_measured_seed_sample": common_seeds[:5],
    }
    return figure, audit


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, default=None, help="paired_* shard directories")
    parser.add_argument("--out", type=Path, default=HERE / "primary" / "COUNTERFACTUAL_3D.html")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)
    run_dirs = args.runs or sorted(path for path in HERE.glob("paired_*") if path.is_dir())
    if not run_dirs:
        raise FileNotFoundError("no paired_* shards found; pass --runs explicitly")
    cases = _collect(run_dirs)
    figure, audit = build_figure(cases, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # include_plotlyjs=True gives a portable offline artifact with the full
    # Plotly runtime embedded in the HTML; legend toggles and fixed uirevision
    # remain available without a network connection.
    figure.write_html(args.out, include_plotlyjs=True, full_html=True, auto_open=False)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()

