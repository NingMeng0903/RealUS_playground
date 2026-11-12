"""Draw the sampled kinematic paths and paired cohort outcomes."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from verify_study import ARMS, RESERVES, load_lines


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ab_map_recheck_20261112"))
from map_model import MapWorld


LABELS = {
    "online_B": ("Online map + B", "#15706e"),
    "online_MPC": ("Online map + MPC", "#2d62ad"),
    "frozen_B": ("Frozen map + B", "#b75844"),
}


def choose_example(rows, common):
    for subject in common:
        for reserve in RESERVES:
            b, a = (rows[subject["seed"], reserve, name] for name in ARMS[:2])
            if b.get("tail_switched") and b["strict"] and not a["strict"]:
                return subject, "First paired case with a successful contingent switch"
    for subject in common:
        for reserve in RESERVES:
            b, a = (rows[subject["seed"], reserve, name] for name in ARMS[:2])
            if (b.get("first_view"), b.get("actual_tail_view")) != (a.get("first_view"), a.get("actual_tail_view")):
                return subject, "First paired case with different online planner routes"
    for subject in common:
        if any(rows[subject["seed"], reserve, ARMS[0]]["strict"] !=
               rows[subject["seed"], reserve, ARMS[-1]]["strict"] for reserve in RESERVES):
            return subject, "First paired case where map updating changed the outcome"
    return common[0], "First prespecified paired case"


def trajectory(fig, xyz, name, color, col, *, legend, width=5, dash="solid", samples=None, group=None):
    if len(xyz) == 0:
        return
    fig.add_trace(go.Scatter3d(
        x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="lines", name=name,
        legendgroup=group or name, showlegend=legend, customdata=samples,
        line=dict(color=color, width=width, dash=dash),
        hovertemplate="X %{x:.1f} mm<br>Y %{y:.1f} mm<br>Z %{z:.1f} mm<extra>%{fullData.name}</extra>",
    ), row=1, col=col)


def main(out):
    common = load_lines(out / "COMMON.jsonl")
    raw = load_lines(out / "RUNS.jsonl")
    rows = {(r["seed"], r["reserve"], r["arm"]): r for r in raw}
    summary = json.loads((out / "VERIFIED_SUMMARY.json").read_text())["primary"]
    subject, selection = choose_example(rows, common)
    world = MapWorld(mode="stress")
    case = world.case(subject["seed"])
    t = subject["probe_motion"]["t_end"]
    trunk = world.truth_world(case, t, [(i, 0) for i in range(8)])
    branch = world.truth_world(case, t, [(i, 1) for i in world.branch_sites])
    probe = np.asarray(subject["probe_motion"]["samples"]["atlas_xyz_mm"])
    fig = make_subplots(
        rows=1, cols=3, specs=[[{"type": "scene"}, {"type": "scene"}, {"type": "xy"}]],
        column_widths=[.36, .36, .28], horizontal_spacing=.05,
        subplot_titles=("Narrow stroke · 10 mm", "Wide stroke · 26 mm", "Strict completion"),
    )
    for col, reserve in enumerate(RESERVES, 1):
        legend = col == 1
        for name, xyz in (("Main vessel", trunk), ("Daughter branch", branch)):
            trajectory(fig, xyz, name, "#8d979f", col, legend=legend, width=8)
        trajectory(fig, probe, "Shared probe approach", "#474f58", col, legend=legend, dash="dot")
        fig.add_trace(go.Scatter3d(
            x=probe[-1:, 0], y=probe[-1:, 1], z=probe[-1:, 2], mode="markers",
            marker=dict(color="#474f58", size=5, symbol="diamond"),
            name="Shared probe", showlegend=legend, legendgroup="probe",
        ), row=1, col=col)
        for j, arm in enumerate(ARMS):
            row = rows[subject["seed"], reserve, arm]
            name, color = LABELS[arm]
            if row["stages"]:
                segments = [np.asarray(s["motion"]["samples"]["atlas_xyz_mm"]) for s in row["stages"]]
                points = np.concatenate(segments)
                trajectory(fig, points, name, color, col, legend=legend, width=7 - j,
                           dash=("solid", "dash", "dot")[j], group=arm)
                finish = np.stack([s[-1] for s in segments])
                final = "Strict success" if row["strict"] else "Strict failure"
                note = f"{final}<br>{row['travel_mm']:.1f} mm · {row['turn_deg']:.1f}° · {row['duration_s']:.2f} s"
                fig.add_trace(go.Scatter3d(
                    x=finish[:, 0], y=finish[:, 1], z=finish[:, 2], mode="markers",
                    marker=dict(size=5 - j, color=color, symbol=("circle", "square-open", "diamond-open")[j]),
                    name=name, legendgroup=arm, showlegend=False,
                    text=[note] * len(finish), hovertemplate="%{text}<extra>%{fullData.name}</extra>",
                ), row=1, col=col)
    for arm in ARMS:
        name, color = LABELS[arm]
        counts = [summary["arms"][arm][r] for r in RESERVES]
        fig.add_trace(go.Bar(
            x=["Narrow", "Wide"], y=[100 * c["success"] / c["total"] for c in counts],
            text=[f"{c['success']}/{c['total']}" for c in counts],
            textposition="outside", cliponaxis=False, marker_color=color,
            name=name, legendgroup=arm, showlegend=False,
            hovertemplate="%{y:.1f}%<extra>%{fullData.name}</extra>",
        ), row=1, col=3)
    fig.update_scenes(
        xaxis_title="X (mm)", yaxis_title="Y (mm)", zaxis_title="Z (mm)",
        aspectmode="data", camera_eye=dict(x=1.4, y=1.55, z=1.15),
    )
    fig.update_yaxes(title_text="Cases (%)", range=[0, 106], row=1, col=3)
    fig.update_layout(
        title=dict(text=f"Moving Y-bifurcation continuation<br><sup>{selection} · vessel snapshot at shared probe</sup>"),
        template="plotly_white", font=dict(family="Inter, Arial, sans-serif"),
        height=620, margin=dict(l=25, r=40, t=115, b=90),
        legend=dict(orientation="h", x=.02, y=-.14, groupclick="togglegroup"),
        barmode="group",
    )
    fig.write_html(out / "TRO_CONTINUATION_3D.html", include_plotlyjs=True)
    (out / "PLOT_CASE.json").write_text(json.dumps({
        "index": subject["index"], "seed": subject["seed"], "selection": selection,
        "description": "Atlas-mm FK samples in the ideal simulated actuator model. Vessel geometry is a snapshot.",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "holdout")
