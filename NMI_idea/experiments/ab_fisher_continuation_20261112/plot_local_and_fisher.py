"""Create the standalone local-continuation/Fisher contact visualization.

The figure is descriptive only.  The left scene shows one matched held-out
local replay for the frozen local study; the right scene shows the checked
DynamicScan contact coordinates selected by the 48-contact Fisher, uniform,
and center-only teacher rules.  Neither scene is a hardware result or a claim
of a conditional B gain.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


HERE = Path(__file__).resolve().parent
LOCAL_DIR = HERE / "local_study_v1_frozen"
LOCAL_RUNS = LOCAL_DIR / "HELDOUT_RUNS.jsonl"
LOCAL_SUMMARY = LOCAL_DIR / "SUMMARY.json"
FISHER_JSON = HERE / "FISHER_OBLIQUE_ABLATION.json"
GEOMETRY = HERE.parent / "ab_leg_dynamic_20261112" / "DYNAMIC_GEOMETRY_FROZEN.npz"
POSE_INDEX = 2
POSE_DEG = 15.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load() -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, np.ndarray]]:
    if not all(path.exists() for path in (LOCAL_RUNS, LOCAL_SUMMARY, FISHER_JSON, GEOMETRY)):
        missing = [str(path) for path in (LOCAL_RUNS, LOCAL_SUMMARY, FISHER_JSON, GEOMETRY) if not path.exists()]
        raise FileNotFoundError("missing visualization input(s): " + ", ".join(missing))
    local_rows = [json.loads(line) for line in LOCAL_RUNS.read_text().splitlines() if line.strip()]
    local_summary = json.loads(LOCAL_SUMMARY.read_text())
    fisher = json.loads(FISHER_JSON.read_text())
    if fisher["source"]["source_sha256"] != _sha256(GEOMETRY):
        raise ValueError("FISHER_OBLIQUE_ABLATION source hash does not match frozen DynamicScan geometry")
    with np.load(GEOMETRY, allow_pickle=False) as payload:
        geometry = {key: np.asarray(payload[key]) for key in payload.files}
    return local_rows, local_summary, fisher, geometry


def _contact_features(geometry: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Return actual posed contact points, depth, and obliquity in millimetres."""

    views = np.asarray(geometry["posed_views_m"], dtype=float)
    frames = np.asarray(geometry["posed_frames"], dtype=float)
    vessel = np.asarray(geometry["posed_vessel_m"], dtype=float)
    ray = vessel[:, :, None, :] - views
    depth_mm = np.einsum("pskc,pskc->psk", ray, frames[:, :, :, 2]) * 1000.0
    unit_ray = ray / np.maximum(np.linalg.norm(ray, axis=-1, keepdims=True), 1.0e-12)
    obliquity_deg = np.degrees(
        np.arccos(np.clip(np.einsum("pskc,pskc->psk", unit_ray, frames[:, :, :, 2]), -1.0, 1.0))
    )
    return {
        "views_mm": views * 1000.0,
        "vessel_mm": vessel * 1000.0,
        "depth_mm": depth_mm,
        "obliquity_deg": obliquity_deg,
        "pose_deg": np.asarray(geometry["pose_knee_deg"], dtype=float),
        "offset_mm": np.asarray(geometry["view_offsets_mm"], dtype=float),
        "tilt_deg": np.asarray(geometry["view_tilts_deg"], dtype=float),
    }


def _fisher_contact(
    index: int,
    features: Mapping[str, np.ndarray],
    n_site: int,
    n_view: int,
) -> dict[str, Any]:
    pose, remainder = divmod(int(index), n_site * n_view)
    site, view = divmod(remainder, n_view)
    point = features["views_mm"][pose, site, view]
    return {
        "index": int(index),
        "pose": int(pose),
        "pose_deg": float(features["pose_deg"][pose]),
        "site": int(site),
        "view": int(view),
        "x_mm": float(point[0]),
        "y_mm": float(point[1]),
        "z_mm": float(point[2]),
        "offset_mm": float(features["offset_mm"][view]),
        "tilt_deg": float(features["tilt_deg"][view]),
        "depth_mm": float(features["depth_mm"][pose, site, view]),
        "obliquity_deg": float(features["obliquity_deg"][pose, site, view]),
    }


def _trace_hover(row: Mapping[str, Any], action_index: int) -> list[Any]:
    return [
        int(row["source_station"]),
        int(row["target_station"]),
        str(row["method"]),
        int(action_index + 1),
    ]


def build_figure() -> go.Figure:
    local_rows, local_summary, fisher, geometry = _load()
    features = _contact_features(geometry)
    views = features["views_mm"]
    vessel = features["vessel_mm"]
    n_site, n_view = views.shape[1:3]

    heldout_pairs = local_summary["heldout"]["pair_indices"]
    if not heldout_pairs:
        raise ValueError("frozen local held-out summary has no pair")
    pair_index = int(heldout_pairs[0])
    pair_rows = [row for row in local_rows if int(row["pair_index"]) == pair_index]
    if not pair_rows:
        raise ValueError(f"no frozen local rows for held-out pair index {pair_index}")
    seed = min(int(row["seed"]) for row in pair_rows)
    pair_rows = [row for row in pair_rows if int(row["seed"]) == seed]
    source = int(pair_rows[0]["source_station"])
    target = int(pair_rows[0]["target_station"])
    method_order = ("A_reactive", "Fisher_MI", "Fixed", "B3")
    method_labels = {
        "A_reactive": "A reactive · simulated",
        "Fisher_MI": "Fisher/MI · simulated",
        "Fixed": "Fixed continuation · simulated",
        "B3": "B3 look-ahead · simulated",
    }
    method_colors = {
        "A_reactive": "#1f77b4",
        "Fisher_MI": "#d62728",
        "Fixed": "#2ca02c",
        "B3": "#9467bd",
    }
    method_symbols = {
        "A_reactive": "circle",
        "Fisher_MI": "diamond",
        "Fixed": "square",
        "B3": "cross",
    }

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        column_widths=[0.50, 0.50],
        horizontal_spacing=0.02,
        subplot_titles=(
            f"Local matched replay · stations {source}→{target}, seed {seed}",
            "Checked contacts used by the Fisher oblique teacher",
        ),
    )

    # Left: actual 15-degree vessel segment and the four public-policy traces.
    route = vessel[POSE_INDEX, source : target + 1]
    fig.add_trace(
        go.Scatter3d(
            x=route[:, 0],
            y=route[:, 1],
            z=route[:, 2],
            mode="lines+markers",
            name="15° vessel centerline",
            legendgroup="geometry",
            line=dict(color="#707070", width=7),
            marker=dict(size=3, color="#707070"),
            hovertemplate="vessel centerline<extra></extra>",
            showlegend=True,
        ),
        row=1,
        col=1,
    )
    # Show the actual checked contact bank at the two sites, without implying
    # that every grey candidate was selected by any policy.
    bank_points = np.vstack((views[POSE_INDEX, source], views[POSE_INDEX, target]))
    fig.add_trace(
        go.Scatter3d(
            x=bank_points[:, 0],
            y=bank_points[:, 1],
            z=bank_points[:, 2],
            mode="markers",
            name="checked candidate bank",
            legendgroup="geometry",
            marker=dict(size=3, color="#c8c8c8", opacity=0.45),
            hovertemplate="checked candidate bank<extra></extra>",
            showlegend=True,
        ),
        row=1,
        col=1,
    )
    for method in method_order:
        selected = [row for row in pair_rows if row["method"] == method]
        if not selected:
            continue
        row = selected[0]
        trace = row["trace"]
        points = np.asarray([views[POSE_INDEX, int(step["site"]), int(step["candidate"])] for step in trace])
        outcomes = [str(step["outcome"]) for step in trace]
        customdata = np.asarray(
            [
                _trace_hover(row, i) + [outcomes[i], float(trace[i]["depth_mm"]), float(trace[i]["tilt_deg"])]
                for i in range(len(trace))
            ],
            dtype=object,
        )
        fig.add_trace(
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="lines+markers+text",
                name=method_labels[method],
                legendgroup=method,
                text=[f"{i + 1}:{outcome}" for i, outcome in enumerate(outcomes)],
                textposition="top center",
                line=dict(color=method_colors[method], width=5),
                marker=dict(size=7, color=method_colors[method], symbol=method_symbols[method]),
                customdata=customdata,
                hovertemplate=(
                    "<b>%{customdata[2]}</b><br>"
                    "acquisition %{customdata[3]} · %{customdata[4]}<br>"
                    "station %{customdata[0]}→%{customdata[1]}<br>"
                    "depth %{customdata[5]:.2f} mm · tilt %{customdata[6]:.1f}°"
                    "<extra></extra>"
                ),
                showlegend=True,
            ),
            row=1,
            col=1,
        )
    fig.add_trace(
        go.Scatter3d(
            x=[route[0, 0], route[-1, 0]],
            y=[route[0, 1], route[-1, 1]],
            z=[route[0, 2], route[-1, 2]],
            mode="text",
            text=[f"source {source}", f"target {target}"],
            textfont=dict(color="#333333", size=11),
            name="pair sites",
            legendgroup="geometry",
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )

    # Right: map the Fisher ablation's stable flattened contact indices back
    # to the actual checked 3D DynamicScan contact coordinates.
    route_all = vessel[POSE_INDEX]
    fig.add_trace(
        go.Scatter3d(
            x=route_all[:, 0],
            y=route_all[:, 1],
            z=route_all[:, 2],
            mode="lines",
            name="15° vessel atlas",
            legendgroup="fisher_geometry",
            line=dict(color="#8a8a8a", width=4),
            hovertemplate="15° vessel atlas<extra></extra>",
            showlegend=True,
        ),
        row=1,
        col=2,
    )
    metrics = fisher["metrics"]
    fisher48 = [_fisher_contact(index, features, n_site, n_view) for index in metrics["fisher_teacher_48"]["selection_indices"]]
    fisher_oblique = [row for row in fisher48 if abs(row["offset_mm"]) > 1.0e-6 or abs(row["tilt_deg"]) > 1.0e-6]
    fisher_center = [row for row in fisher48 if abs(row["offset_mm"]) <= 1.0e-6 and abs(row["tilt_deg"]) <= 1.0e-6]
    def add_contact_trace(rows: list[dict[str, Any]], name: str, color: str, symbol: str, showlegend: bool = True) -> None:
        if not rows:
            return
        customdata = np.asarray(
            [
                [
                    item["pose_deg"],
                    item["site"],
                    item["view"],
                    item["offset_mm"],
                    item["tilt_deg"],
                    item["depth_mm"],
                    item["obliquity_deg"],
                    item["index"],
                ]
                for item in rows
            ],
            dtype=float,
        )
        points = np.asarray([[item["x_mm"], item["y_mm"], item["z_mm"]] for item in rows])
        fig.add_trace(
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="markers",
                name=name,
                legendgroup="teacher",
                marker=dict(size=6, color=color, symbol=symbol, line=dict(width=0.5, color="#333333")),
                customdata=customdata,
                hovertemplate=(
                    f"<b>{name}</b><br>"
                    "pose %{customdata[0]:.0f}° · site %{customdata[1]:.0f} · window %{customdata[2]:.0f}<br>"
                    "offset %{customdata[3]:.1f} mm · tilt %{customdata[4]:.1f}°<br>"
                    "depth %{customdata[5]:.2f} mm · obliquity %{customdata[6]:.1f}°<br>"
                    "flat contact %{customdata[7]:.0f}<extra></extra>"
                ),
                showlegend=showlegend,
            ),
            row=1,
            col=2,
        )

    add_contact_trace(fisher_oblique, "Fisher teacher 48 · oblique", "#d62728", "diamond")
    add_contact_trace(fisher_center, "Fisher teacher 48 · center", "#ff9896", "circle")
    uniform48 = [_fisher_contact(index, features, n_site, n_view) for index in metrics["uniform_diverse_48"]["selection_indices"]]
    center48 = [_fisher_contact(index, features, n_site, n_view) for index in metrics["center_only_48"]["selection_indices"]]
    add_contact_trace(uniform48, "uniform diverse 48", "#1f77b4", "circle")
    add_contact_trace(center48, "center-only 48", "#2ca02c", "square")

    fisher_rmse = float(metrics["fisher_teacher_48"]["summary"]["oblique_rmse_mean"])
    uniform_rmse = float(metrics["uniform_diverse_48"]["summary"]["oblique_rmse_mean"])
    center_rmse = float(metrics["center_only_48"]["summary"]["oblique_rmse_mean"])
    fig.update_layout(
        title=(
            "Local continuation + Fisher oblique contacts"
            "<br><sup>15° checked DynamicScan coordinates · synthetic diagnostics only · "
            "no conditional B-gain claim</sup>"
        ),
        template="plotly_white",
        height=760,
        width=1500,
        margin=dict(l=0, r=0, t=100, b=120),
        legend=dict(orientation="h", y=-0.08, x=0.01, tracegroupgap=10),
        annotations=[
            dict(
                x=0.5,
                y=-0.17,
                xref="paper",
                yref="paper",
                showarrow=False,
                align="center",
                text=(
                    "Left: one matched held-out local replay (frozen, simulated public outcomes). "
                    "Right: 48-contact teacher selections mapped to actual checked 3D contacts. "
                    f"Synthetic oblique RMSE (48): Fisher {fisher_rmse:.3f}, uniform {uniform_rmse:.3f}, center-only {center_rmse:.3f}."
                ),
                font=dict(size=11, color="#444444"),
            )
        ],
    )
    fig.update_scenes(
        xaxis_title="x (mm)",
        yaxis_title="y (mm)",
        zaxis_title="z (mm)",
        aspectmode="data",
        camera=dict(eye=dict(x=1.65, y=1.25, z=1.20)),
        row=1,
        col=1,
    )
    fig.update_scenes(
        xaxis_title="x (mm)",
        yaxis_title="y (mm)",
        zaxis_title="z (mm)",
        aspectmode="data",
        camera=dict(eye=dict(x=1.65, y=1.25, z=1.20)),
        row=1,
        col=2,
    )
    return fig


def main() -> None:
    figure = build_figure()
    output = HERE / "LOCAL_FISHER_3D.html"
    figure.write_html(output, include_plotlyjs=True, full_html=True)
    print(output)
    print("traces", len(figure.data))


if __name__ == "__main__":
    main()

