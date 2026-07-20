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

from ird_playground.viz.viz_style import (  # noqa: E402
    PROBE_COMPARE_BAR_MAX,
    PROBE_COMPARE_CLIM,
    PROBE_COMPARE_D_MIN,
    PROBE_COMPARE_N_LEVELS,
    SPHERE_RADIUS_FACTOR,
)


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
        d_min=PROBE_COMPARE_D_MIN,
        view="cross",
        clim=PROBE_COMPARE_CLIM,
        clim_auto=False,
        size=(3200, 1100),
        n_color_levels=PROBE_COMPARE_N_LEVELS,
        bar_max=PROBE_COMPARE_BAR_MAX,
        fixed_camera=True,
        sphere_radius_m=float(cm.grid.step_m) * SPHERE_RADIUS_FACTOR,
    )
    img = plt.imread(path)
    fig, ax = plt.subplots(figsize=(32, 11), dpi=110)
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(title, fontsize=13, pad=6, color="#222222")
    fig.savefig(path, bbox_inches="tight", facecolor="white", dpi=110)
    plt.close(fig)
    print(
        f"wrote {path}  Dmax={float(cm.d_value.max()):.4f}  "
        f"clim={PROBE_COMPARE_CLIM}  bar=0–{PROBE_COMPARE_BAR_MAX:g}"
    )
    return Path(path)


def _render_global_ird(map_dir: Path, out: Path, *, robot_urdf: Path | None, title: str) -> Path:
    from ird_playground.ird.capability_io import load_capability_map_dir
    from ird_playground.viz.global_ird import render_global_ird_from_capability

    cm = load_capability_map_dir(map_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    path = render_global_ird_from_capability(
        cm,
        out,
        robot_urdf=robot_urdf,
        title=title,
        clim=PROBE_COMPARE_CLIM,
        clim_auto=False,
        d_min=PROBE_COMPARE_D_MIN,
        n_color_levels=PROBE_COMPARE_N_LEVELS,
        bar_max=PROBE_COMPARE_BAR_MAX,
    )
    print(
        f"wrote {path}  n_voxels={int(cm.d_value.shape[0])}  "
        f"clim={PROBE_COMPARE_CLIM}  bar=0–{PROBE_COMPARE_BAR_MAX:g}"
    )
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
    for sub in ("capability", "global_ird"):
        old = reports / sub / "vertical_z220.png"
        if old.is_file():
            old.unlink()
            print(f"removed {old}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
