"""Show the original vein and the two inner shells in identical coordinates."""
from __future__ import annotations

import json

import numpy as np
from plotly.subplots import make_subplots

from knee_gap_diagnostic import _boundary_faces
from plot_long_scan import line, marker, mesh
from rebind_knee import HERE, OUT, SEALED


def main():
    report = json.loads((OUT/"SUMMARY.json").read_text())
    with np.load(HERE/"LONG_LEG_ASSETS.npz") as asset:
        points = asset["vessel_m"][100:108]
    fig = make_subplots(
        rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("原体网格 · 内孔半径 4%", "实验副本 · 内孔半径 1%"), horizontal_spacing=.055,
    )
    h_min, h_max = .42, .48
    bounds = None
    for col, path in enumerate((SEALED/"atlas_right.npz", OUT/"atlas_right.npz"), 1):
        with np.load(path) as atlas:
            verts = atlas["harmonic_vertices"]
            face, _ = _boundary_faces(atlas["harmonic_tets"])
            h, d = atlas["harmonic_h"], atlas["harmonic_d"]
            inner = (d == 1.) & (h >= h_min) & (h <= h_max)
            face = face[inner[face].all(axis=1)]
            used = np.unique(face)
            core = atlas["core_points"]
            core_h = atlas["core_h"]
            core = core[(core_h >= h_min) & (core_h <= h_max)]
        color = "#4e83a5" if col == 1 else "#328b7e"
        fig.add_trace(mesh(verts[used], np.searchsorted(used, face), "内壳 d=1", color, .26), 1, col)
        fig.add_trace(line(core, "烘焙中心线", "#59616c", dash="dot", width=3, legend=col==1), 1, col)
        fig.add_trace(line(points, "原始静脉连线", "#51316e", width=7, legend=col==1), 1, col)
        fig.add_trace(marker(points[4:6], "旧版未绑定" if col==1 else "新版原位绑定",
                             "#b83531" if col==1 else "#007d61", symbol="x" if col==1 else "diamond",
                             text=["原始点 104", "原始点 105"]), 1, col)
        if col == 1:
            p = np.vstack([verts[used], points, core])*1000.
            bounds = np.column_stack([p.min(axis=0)-.6, p.max(axis=0)+.6])
    fig.update_scenes(
        aspectmode="data", camera=dict(eye=dict(x=1.3,y=.8,z=1.2),up=dict(x=0,y=1,z=0)),
        xaxis=dict(title="X (mm)", range=bounds[0]), yaxis=dict(title="Y (mm)", range=bounds[1]),
        zaxis=dict(title="Z (mm)", range=bounds[2]),
    )
    after = max(row[2] for row in report["new"]["continuous_runs"])
    fig.update_layout(
        title=f"膝部内孔与静脉材料绑定<br><sup>原始 XYZ 一致 · 两段接成 {after:.1f} mm 连续内部坐标</sup>",
        template="plotly_white", height=680, margin=dict(l=8,r=8,t=105,b=72),
        font=dict(family="Noto Sans CJK SC, Arial, sans-serif",color="#233349"),
        legend=dict(orientation="h",y=-.04),
    )
    path = OUT/"KNEE_REBAKE_3D.html"
    fig.write_html(path, include_plotlyjs=True, full_html=True)
    print(path)


if __name__ == "__main__":
    main()
