"""Render held-knee capture geometry, matched clocks, and parameter errors."""
from pathlib import Path
import json

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


ROOT = Path(__file__).resolve().parent / "separation"
ARMS = ("burst_one_site", "separated_one_site", "burst_two_response", "separated_two_response")
LABELS = ("Early · one station", "Late · one station", "Early · two stations", "Late · two stations")
COLORS = ("#2b7990", "#2559a6", "#d19644", "#ab5337")


def locations(xyz, label, *, color, symbol="circle", group=None, legend=True):
    xyz = np.asarray(xyz, dtype=float)
    return go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], name=label,
                        mode="markers+lines", marker={"size": 4, "symbol": symbol},
                        line={"color": color, "dash": "dot", "width": 3},
                        legendgroup=group, showlegend=legend)


def main():
    case = json.loads((ROOT / "TRACE.json").read_text())
    summary = json.loads((ROOT / "VERIFIED_SUMMARY.json").read_text())
    trace = case["trace"]
    stop = case["truth"]["stop_t_s"]
    fig = make_subplots(
        rows=2, cols=2, specs=[[{"type": "scene", "rowspan": 2}, {"type": "xy"}],
                                [None, {"type": "xy"}]], column_widths=[.56, .44],
        row_heights=[.43, .57], vertical_spacing=.16, horizontal_spacing=.07,
        subplot_titles=("Probe acquisition poses and vessel sites", "Three matched-budget capture times",
                        "Parameter RMSE across 128 paired cases"),
    )
    prefix = locations([p["probe_xyz_mm"] for p in trace["prefix_frames"]],
                       "Shared rule sweep", color="#a5b1b5")
    prefix.visible = "legendonly"
    fig.add_trace(prefix, row=1, col=1)
    truth = trace["chosen_truth"]
    targets = [truth["i_xyz_mm"][2], truth["j_xyz_mm"][2]]
    fig.add_trace(locations(targets, "True vessel at T + 0.5 s", color="#27313a", symbol="diamond"),
                  row=1, col=1)
    for index, (arm, label, color) in enumerate(zip(ARMS, LABELS, COLORS)):
        item = trace["arms"][arm]
        frames = item["frames"]
        probe = locations([f["probe_xyz_mm"] for f in frames], label, color=color,
                          group=arm, symbol="square" if "two_response" in arm else "circle")
        probe.text = [f"Station {f['site']} · {f['t'] - stop:.2f} s after stop" for f in frames]
        probe.hovertemplate = "%{text}<extra>%{fullData.name}</extra>"
        fig.add_trace(probe, row=1, col=1)
        estimate = locations(item["latent_forecast"]["0.5s"]["selected_mean_xyz_mm"],
                             f"{label} forecast", color=color, symbol="x", group=arm, legend=False)
        estimate.mode = "markers"
        fig.add_trace(estimate, row=1, col=1)
        fig.add_trace(go.Scatter(x=[f["t"] - stop for f in frames], y=[index]*3, name=label,
                                 mode="markers+lines", line={"color": color},
                                 marker={"symbol": ["diamond" if f["site"] == 11 else "circle" for f in frames],
                                         "size": 8}, legendgroup=arm, showlegend=False),
                      row=1, col=2)
        m = summary["pooled"][f"{arm}/latent"]
        fig.add_trace(go.Bar(x=["Persistent offset", "Lag at stop", "Lag at end"],
                             y=[m["w_rmse_mm"], m["z_stop_smoothed_rmse_mm"], m["z_end_rmse_mm"]],
                             name=label, marker_color=color, legendgroup=arm, showlegend=False),
                      row=2, col=2)
    m = summary["pooled"]["burst_one_site/latent_no_post_stop_US"]
    fig.add_trace(go.Bar(x=["Persistent offset", "Lag at stop", "Lag at end"],
                         y=[m["w_rmse_mm"], m["z_stop_smoothed_rmse_mm"], m["z_end_rmse_mm"]],
                         name="Propagate after shared stop frame", marker_color="#a6aeb3"),
                  row=2, col=2)
    fig.update_xaxes(title_text="Seconds after knee stops", row=1, col=2)
    fig.update_yaxes(tickvals=list(range(4)), ticktext=["Early · 1", "Late · 1", "Early · 2", "Late · 2"],
                     range=[-.5, 3.5], row=1, col=2)
    fig.update_yaxes(title_text="RMSE (mm)", rangemode="tozero", row=2, col=2)
    fig.update_scenes(xaxis_title="World X (mm)", yaxis_title="World Y (mm)",
                      zaxis_title="World Z (mm)", aspectmode="data",
                      camera={"eye": {"x": 2.25, "y": 1.28, "z": 1.22}})
    fig.update_layout(template="plotly_white", height=780, barmode="group",
                      title="Held-knee test of persistent offset and transient motion<br>"
                            "<sup>Station 8 response 0.807 · Station 11 response 0.541 · Lag at stop uses later data</sup>",
                      margin={"l": 22, "r": 26, "t": 112, "b": 115},
                      legend={"orientation": "h", "y": -.14})
    fig.write_html(ROOT / "3D" / "SEPARATION_3D.html", include_plotlyjs=True)


if __name__ == "__main__":
    main()
