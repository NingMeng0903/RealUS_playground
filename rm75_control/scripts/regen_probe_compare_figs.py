#!/usr/bin/env python3
"""Regenerate capability + Global IRD figures for vertical vs horizontal probe.

Shared colour scale 0–18 (D fraction clim 0–0.18) for cross-probe compare.

Layout::

    ird_playground/data/reports/
      capability/
        vertical_probe.png
        horizontal_probe.png
      global_ird/
        vertical_probe.png
        horizontal_probe.png
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path

os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
os.environ.setdefault("OMP_NUM_THREADS", "1")

REPO = Path(__file__).resolve().parents[1]  # rm75_control/
PKG = REPO / "rm75_control"
IRD = REPO.parent / "ird_playground"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(IRD))
pkg = types.ModuleType("rm75_control")
pkg.__path__ = [str(PKG)]
sys.modules["rm75_control"] = pkg

# Shared display scale (matches horizontal-probe Dmax≈0.179 → bar 0–18).
CLIM = (0.0, 0.18)
BAR_MAX = 18.0
N_LEVELS = 8


def _render_capability(map_dir: Path, out: Path, *, robot_urdf: Path | None, title: str) -> Path:
    from rm75_control.tools.reachability.data_model.capability_map import CapabilityMap
    from rm75_control.tools.reachability.viz.sphere_glyphs import render_reachability_index
    import matplotlib.pyplot as plt

    cm = CapabilityMap.load(map_dir, mmap=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    path = render_reachability_index(
        cm,
        out,
        robot_urdf=robot_urdf,
        d_min=0.02,
        view="cross",
        clim=CLIM,
        clim_auto=False,
        size=(3200, 1100),
        n_color_levels=N_LEVELS,
        bar_max=BAR_MAX,
        fixed_camera=True,
        sphere_radius_m=float(cm.grid.step_m) * 0.48,
    )
    img = plt.imread(path)
    fig, ax = plt.subplots(figsize=(32, 11), dpi=110)
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(title, fontsize=13, pad=6, color="#222222")
    fig.savefig(path, bbox_inches="tight", facecolor="white", dpi=110)
    plt.close(fig)
    print(f"wrote {path}  Dmax={float(cm.d_value.max()):.4f}  clim={CLIM}  bar=0–{BAR_MAX:g}")
    return Path(path)


def _render_global_ird(map_dir: Path, out: Path, *, robot_urdf: Path | None, title: str) -> Path:
    import numpy as np
    from ird_playground.ird.capability_io import load_capability_map_dir
    from ird_playground.viz.global_ird import (
        build_ird_points_from_capability,
        render_global_ird,
        voxelize_max,
    )

    cm = load_capability_map_dir(map_dir)
    order = np.argsort(-cm.d_value)[:12_000]
    class _Sub:
        pass
    sub = _Sub()
    sub.orientations = cm.orientations
    sub.roll = cm.roll
    sub.bitmask = cm.bitmask[order]
    sub.d_value = cm.d_value[order]
    sub.voxel_ids = cm.voxel_ids[order]
    sub.grid = cm.grid
    xyz, q = build_ird_points_from_capability(sub, max_orients_per_voxel=6)
    xyz, q = voxelize_max(xyz, q, step_m=0.05)
    out.parent.mkdir(parents=True, exist_ok=True)
    path = render_global_ird(
        xyz,
        q,
        out,
        d_min=0.02,
        clim=CLIM,
        clim_auto=False,
        title=title,
        sphere_radius_m=0.05 * 0.55,
        robot_urdf=robot_urdf,
    )
    print(f"wrote {path}  n_cells={xyz.shape[0]}  mean={float(q.mean()):.4f}  clim={CLIM}")
    return Path(path)


def main() -> int:
    reports = IRD / "data/reports"
    vert_map = REPO / "data/reachability/rm75_6f_3cm_15deg_coll"
    horiz_map = REPO / "data/reachability/rm75_6f_3cm_15deg_coll_probe"
    if not vert_map.is_dir():
        raise SystemExit(f"missing vertical map: {vert_map}")
    if not horiz_map.is_dir():
        raise SystemExit(f"missing horizontal probe map: {horiz_map}  (build first)")

    vert_urdf = REPO / "rm75_control/assets/robots/rm75_6f_8dof/RM75-6F-8dof.genesis.urdf"
    from ird_playground.probe.transform import ensure_probe_visual_urdf

    horiz_urdf = ensure_probe_visual_urdf(playground_root=IRD)

    _render_capability(
        vert_map,
        reports / "capability" / "vertical_probe.png",
        robot_urdf=vert_urdf,
        title="Capability · vertical probe",
    )
    _render_capability(
        horiz_map,
        reports / "capability" / "horizontal_probe.png",
        robot_urdf=horiz_urdf,
        title="Capability · horizontal probe",
    )
    _render_global_ird(
        vert_map,
        reports / "global_ird" / "vertical_probe.png",
        robot_urdf=vert_urdf,
        title="Global IRD · vertical probe",
    )
    _render_global_ird(
        horiz_map,
        reports / "global_ird" / "horizontal_probe.png",
        robot_urdf=horiz_urdf,
        title="Global IRD · horizontal probe",
    )
    # drop old filenames
    for sub in ("capability", "global_ird"):
        old = reports / sub / "vertical_z220.png"
        if old.is_file():
            old.unlink()
            print(f"removed {old}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
