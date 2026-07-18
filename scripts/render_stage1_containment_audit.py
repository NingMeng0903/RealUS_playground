#!/usr/bin/env python3
"""Render an offline Stage-1 SMPL-X containment audit PNG and NPZ."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--canonical-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "src"))
    from projects.genesis_ue_sync.anatomy_retarget.containment import (
        load_body_surface,
        signed_distance,
    )

    raw = np.load(args.asset, allow_pickle=True)
    vertices = np.asarray(raw["vertices_rest"], dtype=np.float64)
    ranges = np.asarray(raw["source_vertex_ranges"], dtype=np.int64)
    tissues = [str(value).lower() for value in raw["source_tissues"].tolist()]
    surface_vertices, surface_faces = load_body_surface(
        args.canonical_dir / "smpl_canonical_tpose.obj"
    )
    sdf, _closest, _normals = signed_distance(vertices, surface_vertices, surface_faces)
    colors = {"vessel": "#d62728", "nerve": "#ffbf00", "organ": "#9467bd"}
    groups: list[tuple[str, np.ndarray]] = []
    for tissue in ("vessel", "nerve", "organ"):
        mask = np.zeros(len(vertices), dtype=bool)
        for (start, stop), label in zip(ranges, tissues):
            if label == tissue:
                mask[int(start) : int(stop)] = True
        groups.append((tissue, np.flatnonzero(mask & (sdf > 0.0))))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "stage1_outside_points.npz",
        vertices=vertices,
        sdf=sdf.astype(np.float32),
        vessel_indices=groups[0][1], nerve_indices=groups[1][1], organ_indices=groups[2][1],
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(14, 10))
    views = ((10, -90, "front"), (8, 0, "side"), (0, -90, "leg front"), (0, 0, "leg side"))
    for index, (elevation, azimuth, title) in enumerate(views, start=1):
        axis = figure.add_subplot(2, 2, index, projection="3d")
        axis.plot_trisurf(
            surface_vertices[:, 0], surface_vertices[:, 1], surface_vertices[:, 2],
            triangles=surface_faces, color="#9aa7b2", alpha=0.15, linewidth=0.0,
        )
        for tissue, indices in groups:
            if len(indices):
                points = vertices[indices]
                axis.scatter(points[:, 0], points[:, 1], points[:, 2], s=1.5, c=colors[tissue], label=tissue)
        if title.startswith("leg"):
            axis.set_xlim(-0.28, 0.28)
            axis.set_ylim(-1.38, -0.38)
            axis.set_zlim(-0.22, 0.22)
        else:
            axis.set_xlim(-0.95, 0.95)
            axis.set_ylim(-1.4, 0.45)
            axis.set_zlim(-0.28, 0.28)
        axis.set_box_aspect((1, 1, 1))
        axis.view_init(elev=elevation, azim=azimuth)
        axis.set_title(title)
        axis.set_axis_off()
    handles, labels = figure.axes[0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels, loc="lower center", ncol=3)
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    figure.savefig(args.output_dir / "stage1_containment_audit.png", dpi=220)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
