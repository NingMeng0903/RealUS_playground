"""Render the matched-budget Fisher demonstration comparison."""
import argparse
from pathlib import Path
import json

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


HERE = Path(__file__).resolve().parent
METHODS = {
    "fisher": ("Fisher", "#007e9d"),
    "visibility": ("可见性优先", "#c36a25"),
    "symmetric": ("对称倾转", "#943f88"),
    "diverse": ("多样随机", "#46754a"),
    "conditional": ("分漏检估价", "#5164b5"),
}


def make_figure(data, summary, n_basis):
    angles = data["angles"]
    index = int(np.argmin(np.abs(angles)))
    origin = data["contacts"][index]
    anchor = (data["vessel"] - origin) * 1000.
    frame = data["frames"][index]
    depth = max(25., float(anchor @ frame[:, 2]) + 10.)
    time = np.linspace(0., 3., len(data["fisher_learned"]))
    figure = make_subplots(
        rows=1, cols=2, column_widths=[.52, .48],
        specs=[[{"type": "scene"}, {"type": "xy"}]],
        subplot_titles=("同一皮肤接触点的采集方向", "学习轨迹与回放采集角"),
        horizontal_spacing=.08,
    )
    corners = np.array([
        -25. * frame[:, 1], 25. * frame[:, 1],
        25. * frame[:, 1] + depth * frame[:, 2], -25. * frame[:, 1] + depth * frame[:, 2],
    ])
    figure.add_trace(go.Mesh3d(
        x=corners[:, 0], y=corners[:, 1], z=corners[:, 2],
        i=[0, 0], j=[1, 2], k=[2, 3], color="#829aa4",
        opacity=.13, name="中立探头平面", hoverinfo="skip", showlegend=False,
    ), row=1, col=1)
    landmarks = np.stack([np.zeros(3), anchor])
    figure.add_trace(go.Scatter3d(
        x=landmarks[:, 0], y=landmarks[:, 1], z=landmarks[:, 2],
        mode="markers", marker={"size": 5, "color": ["#25344b", "#ad3651"]},
        name="接触点与名义血管点", text=["探头接触点", "图册名义血管点"],
        hovertemplate="%{text}<extra></extra>",
    ), row=1, col=1)
    for key, (name, color) in METHODS.items():
        pair = data[f"{key}_contact_pair"]
        points = angles[pair]
        for slot, beam in enumerate(pair):
            ray = np.stack([np.zeros(3), depth * data["frames"][beam, :, 2]])
            figure.add_trace(go.Scatter3d(
                x=ray[:, 0], y=ray[:, 1], z=ray[:, 2], mode="lines",
                name=name, legendgroup=key, showlegend=False,
                line={"width": 5, "color": color, "dash": "solid" if slot == 0 else "dot"},
                hovertemplate=f"{name} · 观测 {slot + 1} · {points[slot]:+g}°<extra></extra>",
            ), row=1, col=1)
        figure.add_trace(go.Scatter(
            x=time, y=data[f"{key}_learned"], mode="lines", name=name,
            legendgroup=key, line={"width": 2, "color": color},
            hovertemplate=f"{name} · 学习轨迹<br>%{{x:.2f}} s · %{{y:.2f}}°<extra></extra>",
        ), row=1, col=2)
        figure.add_trace(go.Scatter(
            x=time, y=data[f"{key}_taught"], mode="lines", name=name,
            legendgroup=key, showlegend=False, line={"width": 1, "dash": "dot", "color": color},
            opacity=.55, hovertemplate=f"{name} · 示范<br>%{{x:.2f}} s · %{{y:.2f}}°<extra></extra>",
        ), row=1, col=2)
        figure.add_trace(go.Scatter(
            x=[1., 2.], y=points, mode="markers", name=name,
            legendgroup=key, showlegend=False, marker={"size": 8, "color": color, "symbol": "circle-open"},
            hovertemplate=f"{name} · 检查后观测<br>%{{x:.0f}} s · %{{y:.1f}}°<extra></extra>",
        ), row=1, col=2)
    figure.update_xaxes(title_text="时间 (s)", row=1, col=2)
    figure.update_yaxes(title_text="横向倾角 (°)", row=1, col=2)
    figure.update_layout(
        title={
            "text": (
                f"Fisher 示范与 ProDMP · {n_basis} 个积分基 · 等预算 36° 倾转"
                f"<br><sup>留出站点 {int(data['site'])} · 屈膝 {float(data['pose_deg']):g}°"
                f" · 合成粗回波角 {float(data['preferred']):+g}° · 3 秒 / 2 次采集</sup>"
            ),
            "x": .04,
        },
        template="plotly_white", height=710,
        font={"family": "system-ui, sans-serif", "size": 13, "color": "#233046"},
        scene={
            "aspectmode": "data", "xaxis_title": "图册 X (mm)", "yaxis_title": "图册 Y (mm)",
            "zaxis_title": "图册 Z (mm)",
        },
        legend={"orientation": "h", "y": -.14, "groupclick": "togglegroup"},
        margin={"l": 40, "r": 30, "t": 125, "b": 142},
    )
    s = summary["statistics"]["shifted"]
    scores = "　".join(f"{name} {s[f'prodmp_{key}']['anatomy_rmse_mm']:.3f} mm"
                       for key, (name, _) in METHODS.items())
    figure.add_annotation(
        x=.01, y=-.29, xref="paper", yref="paper", showarrow=False, align="left",
        text="实线为学习轨迹　点线为示范　空心圆为回放所用的候选采集角<br>"
             f"留出总体定位 RMSE（回波偏差与噪声变化）　{scores}",
        font={"size": 12},
    )
    return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-basis", type=int, default=12)
    args = parser.parse_args()
    suffix = f"_basis{args.n_basis}" if args.n_basis != 4 else ""
    result = json.loads((HERE / f"DEMO_STUDY{suffix}.json").read_text())["equal_36deg"]
    with np.load(HERE / f"PLOT_equal_36deg{suffix}.npz", allow_pickle=False) as data:
        fig = make_figure(data, result, args.n_basis)
    fig.write_html(HERE / "FISHER_PRODMP_3D.html", include_plotlyjs=True, auto_open=False)


if __name__ == "__main__":
    main()
