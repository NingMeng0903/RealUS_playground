"""Interactive view of the controlled paired knee histories."""
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from summarize import xyz_trace


ROOT = Path(__file__).resolve().parent / "results" / "pose_audit_20261112"
LABELS = {"dynamic_us": "Lag estimate with ultrasound",
          "no_z_us": "Angle and speed model",
          "dynamic_withheld": "Lag estimate without motion ultrasound"}
COLORS = ("#236f99", "#c05c36")


def main():
    raw = json.loads((ROOT / "TRACE.json").read_text())
    histories = raw["histories"]
    fig = make_subplots(
        rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "xy"}]], column_widths=[.58, .42],
        subplot_titles=("Same material sites at +0.5 s", "Transient displacement after the common checkpoint"),
    )
    for i, color in enumerate(COLORS):
        matched = histories["dynamic_us"][i]
        name = matched["history"].capitalize()
        checkpoint = matched["checkpoint"]
        future = matched["future"]
        times = [0., *[f["time_s"] - checkpoint["reading"]["t"] for f in future]]
        truth_z = [checkpoint["true_z_mm"], *[f["z_true_mm"] for f in future]]
        true = 1000. * np.asarray(future[0]["actual_m"])
        fig.add_trace(xyz_trace(true, f"{name} · simulated truth", color, width=5, markers=True,
                                group=f"truth{i}"), row=1, col=1)
        fig.add_trace(go.Scatter(x=times, y=truth_z, mode="lines+markers", name=name,
                                 line={"color": color, "width": 4}, showlegend=False,
                                 legendgroup=f"truth{i}"), row=1, col=2)
        for method, style in (("dynamic_us", "dash"), ("no_z_us", "dot"), ("dynamic_withheld", "dashdot")):
            run = histories[method][i]
            group = f"{method}{i}"
            position = 1000. * np.asarray(run["future"][0]["mean_m"])
            trace = xyz_trace(position, f"{name} · {LABELS[method]}", color, dash=style, width=4,
                              markers=True, group=group)
            if method != "dynamic_us":
                trace.visible = "legendonly"
            fig.add_trace(trace, row=1, col=1)
            z = [run["checkpoint"]["estimated_z_mm"], *[f["z_estimated_mm"] for f in run["future"]]]
            fig.add_trace(go.Scatter(x=times, y=z, mode="lines+markers", name=LABELS[method],
                                     line={"color": color, "dash": style, "width": 3}, showlegend=False,
                                     legendgroup=group, visible=True if method == "dynamic_us" else "legendonly"),
                          row=1, col=2)
    fig.update_scenes(aspectmode="data", xaxis_title="World X (mm)", yaxis_title="World Y (mm)",
                      zaxis_title="World Z (mm)", camera={"eye": {"x": 2.35, "y": 1.14, "z": 1.35}})
    fig.update_xaxes(title_text="Seconds after 15° and zero angular speed", row=1, col=2)
    fig.update_yaxes(title_text="Latent lateral displacement (mm)", row=1, col=2)
    fig.update_layout(template="plotly_white", height=780,
                      title="Knee history changes the internal position at the same angle",
                      legend={"orientation": "h", "y": -.14}, margin={"l": 25, "r": 25, "t": 105, "b": 142})
    fig.write_html(ROOT / "KNEE_HISTORY_3D.html", include_plotlyjs=True)


if __name__ == "__main__":
    main()
