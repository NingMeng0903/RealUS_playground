"""Diagnose the displayed route from frozen choices and ProDMP path samples."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from summarize_rl import ARMS, load_rows

HERE = Path(__file__).resolve().parent
DYNAMIC = HERE.parent / "ab_leg_dynamic_20261112"
sys.path[:0] = [str(DYNAMIC), str(HERE.parent / "ab_fisher_continuation_20261112")]
from certification import Edges  # noqa: E402
from dynamic_scan import DynamicScan  # noqa: E402


def chart_samples(scan, previous, goal, skill, phase):
    """Use the same skill and fallback as the recorded nine-point segment."""
    _, a, b = previous
    _, i, j = goal
    start = np.array([scan.s[a], scan.offsets[b]])
    end = np.array([scan.s[i], scan.offsets[j]])
    if skill == "in_chart_fallback":
        blend = phase**3 * (10. + phase * (-15. + 6. * phase))
        return start + blend[:, None] * (end - start)
    if skill == "endpoint_only":
        return scan.primitive.endpoint_only_path(start, end, samples=len(phase))
    return scan.primitive.path(start, end, skill=skill, samples=len(phase))


def surface_samples(scan, pose, chart):
    """Evaluate the existing piecewise linear surface lookup at one pose."""
    points = []
    order = scan.surface_order
    for s, x in chart:
        right = int(np.clip(np.searchsorted(scan.s, s), 1, scan.n - 1))
        left = right - 1
        fraction = np.clip((s - scan.s[left]) / (scan.s[right] - scan.s[left]), 0., 1.)
        pair = [
            [np.interp(x, scan.offsets[order], scan.views[pose, k, order, c])
             for c in range(3)]
            for k in (left, right)
        ]
        points.append((1. - fraction) * np.asarray(pair[0]) + fraction * np.asarray(pair[1]))
    return np.asarray(points)


def turn_angles(before, after):
    norms = np.linalg.norm(before, axis=-1) * np.linalg.norm(after, axis=-1)
    valid = (np.linalg.norm(before, axis=-1) > 1.e-8) & (np.linalg.norm(after, axis=-1) > 1.e-8)
    cosines = np.sum(before[valid] * after[valid], axis=-1) / norms[valid]
    return np.rad2deg(np.arccos(np.clip(cosines, -1., 1.)))


def turn_metrics(paths):
    difference = np.diff(paths, axis=1)
    inside = turn_angles(difference[:, :-1], difference[:, 1:])
    joins = turn_angles(difference[:-1, -1], difference[1:, 0])
    return {
        "tested_joins": len(joins),
        "joins_over_90_deg": int(np.count_nonzero(joins > 90.)),
        "joins_over_120_deg": int(np.count_nonzero(joins > 120.)),
        "median_join_deg": float(np.median(joins)) if joins.size else None,
        "max_join_deg": float(np.max(joins)) if joins.size else None,
        "interior_angles_over_90_deg": int(np.count_nonzero(inside > 90.)),
        "interior_max_deg": float(np.max(inside)) if inside.size else None,
    }


def reconstruct(scan, row, previous):
    # 257 samples include the original nine phase values exactly.
    phase = np.linspace(0., 1., 257)
    goal = [(step["pose"], *step["action"]) for step in row["steps"]]
    states = [previous, *goal]
    logged = np.asarray([step["path_m"] for step in row["steps"]])
    dense, material, incoming, outgoing = [], [], [], []
    for start, end, step in zip(states, states[1:], row["steps"]):
        chart = chart_samples(scan, start, end, step["path_skill"], phase)
        p0, p1 = start[0], end[0]
        at_start_pose = surface_samples(scan, p0, chart)
        at_end_pose = surface_samples(scan, p1, chart)
        drift = at_end_pose - at_start_pose
        points = at_start_pose + phase[:, None] * drift
        points[0], points[-1] = scan.views[start], scan.views[end]
        dense.append(points)
        material.append(surface_samples(scan, scan.bent, chart))
        # At the endpoints the chart derivative is zero. The derivative
        # with respect to phase comes from the linear pose interpolation.
        incoming.append(drift[-1])
        outgoing.append(drift[0])
    dense, material = np.asarray(dense), np.asarray(material)
    incoming, outgoing = np.asarray(incoming[:-1]), np.asarray(outgoing[1:])
    phase_turn = turn_angles(incoming, outgoing)
    sampled_error = np.max(np.linalg.norm(dense[:, ::32] - logged, axis=-1))
    if sampled_error > 1.e-10:
        raise ValueError(f"reconstruction differs from frozen samples by {sampled_error} m")
    station_steps = np.diff(scan.s[[state[1] for state in states]])
    nonzero = station_steps[np.abs(station_steps) > 1.e-8]
    biggest = int(np.argmax(np.abs(station_steps)))
    metrics = {
        "segments": len(goal),
        "reproduction_error_mm": float(1000. * sampled_error),
        "max_join_position_gap_mm": float(
            1000. * np.linalg.norm(logged[:-1, -1] - logged[1:, 0], axis=-1).max(initial=0.)
        ),
        "world_logged": turn_metrics(logged),
        "world_dense": turn_metrics(dense),
        "fixed_15_deg_dense": turn_metrics(material),
        "body_phase_tangent": {
            "joins_over_120_deg": int(np.count_nonzero(phase_turn > 120.)),
            "max_change_mm_per_phase": float(1000. * np.linalg.norm(incoming - outgoing, axis=-1).max(initial=0.)),
        },
        "material_longitudinal_direction_changes": int(np.count_nonzero(nonzero[:-1] * nonzero[1:] < 0)),
        "longest_station_jump": {
            "acquisition": biggest + 1,
            "from_station": states[biggest][1],
            "to_station": states[biggest + 1][1],
            "atlas_arclength_mm": float(abs(station_steps[biggest])),
        },
    }
    return metrics, (logged, dense, material), goal


def compare_checked_path(edge, transform, previous, goals, dense):
    """Compare TCP FK at the gate's joint samples to the same displayed edge."""
    states = [previous, *goals]
    same_phase, nearest, passed = [], [], []
    for start, goal, points in zip(states, states[1:], dense):
        if start == goal:
            continue
        path = points @ transform[:3, :3].T + transform[:3, 3]
        direction = np.diff(path, axis=0)
        length_squared = np.sum(direction * direction, axis=1)
        for phase in (.25, .50, .75):
            q = (1. - phase) * edge.q[start] + phase * edge.q[goal]
            xyz = edge.robot.fk(q)[:3, 3]
            passed.append(edge.robot.check_q(q)["valid"])
            same_phase.append(np.linalg.norm(xyz - path[int(phase * (len(path) - 1))]))
            delta = xyz - path[:-1]
            projection = np.clip(np.sum(delta * direction, axis=1) / np.maximum(length_squared, 1.e-18), 0., 1.)
            nearest.append(np.linalg.norm(delta - projection[:, None] * direction, axis=1).min())
    return {
        "passing_samples": sum(passed),
        "joint_samples": len(passed),
        "same_phase_max_mm": float(1000. * max(same_phase, default=0.)),
        "distance_to_curve_p95_mm": float(1000. * np.percentile(nearest, 95.)) if nearest else None,
        "distance_to_curve_max_mm": float(1000. * max(nearest, default=0.)),
    }


def render(output, scan, seed, paths_by_arm, goals_by_arm):
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}], [{"type": "scene"}, {"type": "xy"}]],
        subplot_titles=("原记录 · 9 点 / 段", "同一参考 · 257 点 / 段",
                        "同一采集顺序投到固定 15° 体表", "采集顺序与材料站点"),
        horizontal_spacing=.07, vertical_spacing=.1,
    )
    for arm, paths in paths_by_arm.items():
        name, color = ARMS[arm]
        goals = goals_by_arm[arm]
        for panel, ((r, c), segments) in enumerate(zip(((1, 1), (1, 2), (2, 1)), paths)):
            xyz = 1000. * segments.reshape(-1, 3)
            endpoint = 1000. * segments[:, -1]
            count = len(segments)
            fig.add_trace(go.Scatter3d(
                x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="lines",
                line={"color": color, "width": 4}, name=name, legendgroup=arm,
                showlegend=panel == 0, hoverinfo="skip",
            ), row=r, col=c)
            fig.add_trace(go.Scatter3d(
                x=endpoint[:, 0], y=endpoint[:, 1], z=endpoint[:, 2], mode="markers",
                marker={"color": color, "size": 3}, legendgroup=arm,
                showlegend=False, name=name,
                customdata=np.column_stack((np.arange(1, count + 1), np.asarray(goals)[:, 1])),
                hovertemplate="第 %{customdata[0]} 次补采 · 站点 %{customdata[1]}<extra></extra>",
            ), row=r, col=c)
        fig.add_trace(go.Scatter(
            x=np.arange(1, len(goals) + 1), y=scan.s[np.asarray(goals)[:, 1]],
            mode="lines+markers", line={"color": color, "width": 2},
            name=name, legendgroup=arm, showlegend=False,
        ), row=2, col=2)
    fig.update_scenes(
        aspectmode="data", xaxis_title="X (mm)", yaxis_title="Y (mm)", zaxis_title="Z (mm)",
        camera={"eye": {"x": 1.65, "y": 1.65, "z": 1.65}},
    )
    fig.update_xaxes(title_text="补采序号", row=2, col=2)
    fig.update_yaxes(title_text="材料弧长 (mm)", row=2, col=2)
    fig.update_layout(
        template="plotly_white", height=925, margin={"l": 40, "r": 42, "t": 110, "b": 68},
        title=f"病例 {seed} 的参考轨迹检查<br>"
              "<sup>两种策略各自固定原采集顺序 · 曲线为位置参考 · 圆点为采集端点</sup>",
        legend={"orientation": "h", "y": -.055, "groupclick": "togglegroup"},
    )
    fig.write_html(output, include_plotlyjs=True, auto_open=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=201568239)
    parser.add_argument("--eval", type=Path, nargs="+", default=[HERE / "control_24", HERE / "eval_q_24"])
    parser.add_argument("--out", type=Path, default=HERE / "trajectory_audit")
    args = parser.parse_args()

    case = load_rows(args.eval, [])[args.seed]
    prefix_end = case["rl"]["prefix_steps"][-1]
    previous = (prefix_end["pose"], *prefix_end["action"])
    with np.load(DYNAMIC / "DYNAMIC_GEOMETRY_FROZEN.npz", allow_pickle=False) as data:
        scan = DynamicScan(data)
        transform = data["robot_from_atlas"]
    edge = Edges(scan)
    results, paths, goals = {}, {}, {}
    for arm in ("rule", "rl"):
        results[arm], paths[arm], goals[arm] = reconstruct(scan, case[arm], previous)
        results[arm]["checked_joint_interpolation"] = compare_checked_path(
            edge, transform, previous, goals[arm], paths[arm][1],
        )
    args.out.mkdir(parents=True, exist_ok=True)
    summary = {
        "seed": args.seed,
        "trace_scope": "reference positions and sampled kinematics; no controller or physical time",
        "arms": results,
    }
    (args.out / "SUMMARY.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    render(args.out / "TRAJECTORY_CHECK_3D.html", scan, args.seed, paths, goals)
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
