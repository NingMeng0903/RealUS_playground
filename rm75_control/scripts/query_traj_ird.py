#!/usr/bin/env python3
"""Query a scan trajectory against a capability map + write IRM / placement figures."""
from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
os.environ.setdefault("OMP_NUM_THREADS", "1")

REPO = Path(__file__).resolve().parents[1]
PKG = REPO / "rm75_control"
IRD = Path(__file__).resolve().parents[2] / "ird_playground"
sys.path.insert(0, str(REPO))
pkg = types.ModuleType("rm75_control")
pkg.__path__ = [str(PKG)]
sys.modules["rm75_control"] = pkg

from rm75_control.tools.reachability.data_model.capability_map import CapabilityMap
from rm75_control.tools.reachability.inversion.base_optimizer import full_scan_best_yb
from rm75_control.tools.reachability.inversion.prefix_solver import longest_prefix
from rm75_control.tools.reachability.inversion.trajectory import load_trajectory_json
from rm75_control.tools.reachability.viz.inversion_scene import (
    render_base_candidates,
    render_irm_ground,
    render_scan_line_base_placement,
)


def main() -> int:
    map_dir = Path(
        sys.argv[1]
        if len(sys.argv) > 1
        else REPO / "data/reachability/rm75_6f_3cm_15deg_coll"
    )
    traj_path = Path(
        sys.argv[2]
        if len(sys.argv) > 2
        else IRD / "data/trajectories/front_rail_parallel_z_up.json"
    )
    out_dir = Path(sys.argv[3] if len(sys.argv) > 3 else IRD / "data/reports/traj_query")
    out_dir.mkdir(parents=True, exist_ok=True)

    cm = CapabilityMap.load(map_dir, mmap=True)
    traj = load_trajectory_json(traj_path)
    xz = (0.0, 0.0)
    yb_range = (-0.35, 0.35)
    yb_step = 0.01

    full = full_scan_best_yb(
        cm, traj, xz_base_world=xz, yb_range=yb_range, yb_step=yb_step, rail_travel_half=0.25
    )
    pref = longest_prefix(
        cm, traj, xz_base_world=xz, yb_range=yb_range, yb_step=yb_step, rail_travel_half=0.25
    )

    report = {
        "map": str(map_dir),
        "trajectory": str(traj_path),
        "n_waypoints": traj.n,
        "self_collision": bool((cm.meta.extra or {}).get("self_collision")),
        "mc_samples": int(cm.meta.mc_samples),
        "n_reachable_voxels": int(cm.n_reachable_voxels),
        "full_scan": {
            "feasible": full.feasible,
            "y_b_best": full.y_b_best,
            "score": full.score,
            "feasible_interval_count": len(full.feasible_y_b_intervals),
        },
        "prefix": {
            "feasible": pref.feasible,
            "y_b_best": pref.y_b_best,
            "last_wp_index": pref.last_wp_index,
            "arc_len_m": pref.arc_len_m,
            "rail_y": pref.rail_y,
            "score": pref.score,
            "relaxed": pref.relaxed,
        },
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    render_irm_ground(
        cm, traj, out_dir / "irm_yb.png",
        xz_base_world=xz, yb_range=yb_range, yb_step=yb_step, size=(1600, 600),
    )
    print(f"wrote {out_dir / 'irm_yb.png'}")
    render_base_candidates(
        cm, traj, out_dir / "placement.png",
        result=full, xz_base_world=xz, yb_range=yb_range, yb_step=yb_step, size=(1400, 900),
    )
    print(f"wrote {out_dir / 'placement.png'}")
    # Main figure: TCP scan line in world + scored base candidates + gold y_b*
    render_scan_line_base_placement(
        cm, traj, out_dir / "scan_base.png",
        result=full, xz_base_world=xz, yb_range=yb_range, yb_step=yb_step,
        mode="full", size=(1600, 1000),
    )
    print(f"wrote {out_dir / 'scan_base.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
