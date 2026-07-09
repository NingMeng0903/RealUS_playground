"""``python -m rm75_control.tools.reachability.viz.cli`` — paper-style figures."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from rm75_control.tools.reachability.data_model.capability_map import CapabilityMap
from rm75_control.tools.reachability.inversion.base_optimizer import full_scan_best_yb
from rm75_control.tools.reachability.inversion.prefix_solver import longest_prefix
from rm75_control.tools.reachability.inversion.trajectory import load_trajectory_json
from rm75_control.tools.reachability.viz.inversion_scene import (
    render_base_candidates,
    render_best_placement_pose,
    render_feasible_yb_region,
    render_irm_ground,
)
from rm75_control.tools.reachability.viz.orientation_glyph import render_direction_spheres
from rm75_control.tools.reachability.viz.sphere_glyphs import render_reachability_index, render_slice
from rm75_control.tools.reachability.viz.trajectory_overlay import render_trajectory_from_prefix


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--map", type=Path, required=True, help="capability map dir")
    p.add_argument("--robot-urdf", type=Path, default=None)
    p.add_argument("--d-min", type=float, default=0.02)
    p.add_argument("--sphere-radius", type=float, default=0.0, help="glyph radius (m); 0=auto step*0.52")
    p.add_argument("--figsize", type=int, nargs=2, default=[3200, 1100], metavar=("W", "H"))
    p.add_argument("--out", type=Path, required=True)


def _sphere_radius_arg(v: float) -> float | None:
    return None if v <= 0 else float(v)


def _cmd_capability(args: argparse.Namespace) -> int:
    cm = CapabilityMap.load(args.map, mmap=True)
    clim = tuple(args.clim) if args.clim is not None else None
    out = render_reachability_index(
        cm, args.out, robot_urdf=args.robot_urdf, d_min=args.d_min,
        sphere_radius_m=_sphere_radius_arg(args.sphere_radius), size=tuple(args.figsize),
        clim=clim, clim_auto=bool(args.clim_auto), opacity=float(args.opacity),
        view=str(args.view), n_color_levels=int(args.color_levels),
        fixed_camera=bool(args.fixed_camera),
    )
    print(f"wrote {out}")
    return 0


def _cmd_slice(args: argparse.Namespace) -> int:
    cm = CapabilityMap.load(args.map, mmap=True)
    out = render_slice(
        cm, args.out, plane=args.plane, slab_m=args.slab, robot_urdf=args.robot_urdf,
        d_min=args.d_min, sphere_radius_m=_sphere_radius_arg(args.sphere_radius), size=tuple(args.figsize),
    )
    print(f"wrote {out}")
    return 0


def _cmd_directions(args: argparse.Namespace) -> int:
    cm = CapabilityMap.load(args.map, mmap=True)
    out = render_direction_spheres(
        cm, args.out, stride=args.stride, d_min=args.d_min,
        sphere_radius_m=_sphere_radius_arg(args.sphere_radius) or 0.012, robot_urdf=args.robot_urdf,
        size=tuple(args.figsize), max_voxels=int(args.max_voxels),
    )
    print(f"wrote {out}")
    return 0


def _cmd_irm(args: argparse.Namespace) -> int:
    cm = CapabilityMap.load(args.map, mmap=True)
    traj = load_trajectory_json(args.trajectory)
    out = render_irm_ground(
        cm, traj, args.out, xz_base_world=(args.xb, args.zb),
        yb_range=tuple(args.yb_range), yb_step=args.yb_step, size=tuple(args.figsize),
    )
    print(f"wrote {out}")
    return 0


def _cmd_placement(args: argparse.Namespace) -> int:
    cm = CapabilityMap.load(args.map, mmap=True)
    traj = load_trajectory_json(args.trajectory)
    xz = (float(args.xb), float(args.zb))
    yb_range = (float(args.yb_range[0]), float(args.yb_range[1]))
    if args.mode == "prefix":
        result = longest_prefix(cm, traj, xz_base_world=xz, yb_range=yb_range, yb_step=args.yb_step)
    else:
        result = full_scan_best_yb(cm, traj, xz_base_world=xz, yb_range=yb_range, yb_step=args.yb_step)
    out = render_base_candidates(
        cm, traj, args.out, result=result, xz_base_world=xz,
        yb_range=yb_range, yb_step=args.yb_step, size=tuple(args.figsize),
    )
    print(f"wrote {out}")
    if args.best_pose:
        if result.y_b_best is None:
            print("no feasible placement for best-pose render", file=sys.stderr)
            return 1
        rail_y = getattr(result, "rail_y", 0.0)
        if hasattr(result, "rail_y_series") and result.rail_y_series:
            rail_y = float(result.rail_y_series[0])
        pose_out = args.out.with_name(args.out.stem + "_pose" + args.out.suffix)
        render_best_placement_pose(pose_out, y_b=float(result.y_b_best), rail_y=float(rail_y), robot_urdf=args.robot_urdf)
        print(f"wrote {pose_out}")
    return 0


def _cmd_trajectory(args: argparse.Namespace) -> int:
    traj = load_trajectory_json(args.trajectory)
    cm = CapabilityMap.load(args.map, mmap=True)
    pref = longest_prefix(
        cm, traj, xz_base_world=(args.xb, args.zb),
        yb_range=tuple(args.yb_range), yb_step=args.yb_step,
    )
    out = render_trajectory_from_prefix(traj, pref, args.out, size=tuple(args.figsize))
    print(f"wrote {out}")
    return 0


def _cmd_task(args: argparse.Namespace) -> int:
    """Trajectory-specific feasible y_b set U (Vahrenkamp inversion, not D(x) map)."""
    cm = CapabilityMap.load(args.map, mmap=True)
    traj = load_trajectory_json(args.trajectory)
    out = render_feasible_yb_region(
        cm, traj, args.out, xz_base_world=(args.xb, args.zb),
        yb_range=tuple(args.yb_range), yb_step=args.yb_step, mode=args.mode,
        size=tuple(args.figsize),
    )
    print(f"wrote {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_cap = sub.add_parser("capability", help="Zacharias Fig 3")
    _add_common(p_cap)
    p_cap.add_argument("--clim", type=float, nargs=2, default=None,
                       help="fixed D(x) fraction range (default 0 1 → paper 0–100%% scale)")
    p_cap.add_argument("--clim-auto", action="store_true", default=False,
                       help="auto-scale colours to each map's D max (not comparable across maps)")
    p_cap.add_argument("--no-fixed-camera", action="store_false", dest="fixed_camera", default=True,
                       help="zoom camera to reachable voxels instead of full grid")
    p_cap.add_argument("--opacity", type=float, default=1.0)
    p_cap.add_argument("--view", choices=["cross", "iso"], default="cross", help="cross=Fig3 dual slices (default), iso=3D")
    p_cap.add_argument("--color-levels", type=int, default=21)
    p_cap.set_defaults(func=_cmd_capability)

    p_sl = sub.add_parser("slice", help="Zacharias Fig 8/9")
    _add_common(p_sl)
    p_sl.add_argument("--plane", type=str, required=True)
    p_sl.add_argument("--slab", type=float, default=None)
    p_sl.set_defaults(func=_cmd_slice)

    p_dir = sub.add_parser("directions", help="Zacharias Fig 4/5")
    _add_common(p_dir)
    p_dir.add_argument("--stride", type=int, default=6)
    p_dir.add_argument("--max-voxels", type=int, default=400, help="cap rendered voxels (0 = no cap)")
    p_dir.set_defaults(func=_cmd_directions)

    p_irm = sub.add_parser("irm", help="Vahrenkamp Fig 3")
    p_irm.add_argument("--map", type=Path, required=True)
    p_irm.add_argument("--trajectory", type=Path, required=True)
    p_irm.add_argument("--xb", type=float, default=0.0)
    p_irm.add_argument("--zb", type=float, default=0.0)
    p_irm.add_argument("--yb-range", type=float, nargs=2, default=[-0.5, 0.5])
    p_irm.add_argument("--yb-step", type=float, default=0.02)
    p_irm.add_argument("--figsize", type=int, nargs=2, default=[1400, 500])
    p_irm.add_argument("--out", type=Path, required=True)
    p_irm.set_defaults(func=_cmd_irm)

    p_pl = sub.add_parser("placement", help="Vahrenkamp Fig 4/6")
    p_pl.add_argument("--map", type=Path, required=True)
    p_pl.add_argument("--trajectory", type=Path, required=True)
    p_pl.add_argument("--xb", type=float, default=0.0)
    p_pl.add_argument("--zb", type=float, default=0.0)
    p_pl.add_argument("--yb-range", type=float, nargs=2, default=[-0.5, 0.5])
    p_pl.add_argument("--yb-step", type=float, default=0.02)
    p_pl.add_argument("--mode", choices=["full", "prefix"], default="prefix")
    p_pl.add_argument("--best-pose", action="store_true", help="also render Fig 6 robot pose")
    p_pl.add_argument("--robot-urdf", type=Path, default=None)
    p_pl.add_argument("--figsize", type=int, nargs=2, default=[1200, 900])
    p_pl.add_argument("--out", type=Path, required=True)
    p_pl.set_defaults(func=_cmd_placement)

    p_tr = sub.add_parser("trajectory", help="scan path reachable/unreachable overlay")
    p_tr.add_argument("--map", type=Path, required=True)
    p_tr.add_argument("--trajectory", type=Path, required=True)
    p_tr.add_argument("--xb", type=float, default=0.0)
    p_tr.add_argument("--zb", type=float, default=0.0)
    p_tr.add_argument("--yb-range", type=float, nargs=2, default=[-0.5, 0.5])
    p_tr.add_argument("--yb-step", type=float, default=0.02)
    p_tr.add_argument("--figsize", type=int, nargs=2, default=[1400, 1000])
    p_tr.add_argument("--out", type=Path, required=True)
    p_tr.set_defaults(func=_cmd_trajectory)

    p_task = sub.add_parser("task", help="trajectory feasible y_b region U (Vahrenkamp)")
    p_task.add_argument("--map", type=Path, required=True)
    p_task.add_argument("--trajectory", type=Path, required=True)
    p_task.add_argument("--xb", type=float, default=0.0)
    p_task.add_argument("--zb", type=float, default=0.0)
    p_task.add_argument("--yb-range", type=float, nargs=2, default=[-0.5, 0.5])
    p_task.add_argument("--yb-step", type=float, default=0.01)
    p_task.add_argument("--mode", choices=["full", "prefix"], default="prefix")
    p_task.add_argument("--figsize", type=int, nargs=2, default=[1200, 400])
    p_task.add_argument("--out", type=Path, required=True)
    p_task.set_defaults(func=_cmd_task)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
    sys.exit(main())
