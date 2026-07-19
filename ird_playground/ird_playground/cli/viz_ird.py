"""Visualize **global IRD** (Vahrenkamp): base poses in TCP frame.

Default is Inverse Reachability Distribution — NOT Zacharias forward capability.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def _resolve(root: Path, p: Path) -> Path:
    return p if p.is_absolute() else root / p


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--mode",
        choices=["ird", "ird_gt", "capability", "scatter"],
        default="ird",
        help="ird=invert capability→global IRD (default); ird_gt=from GT npz; "
        "capability=Zacharias forward (legacy); scatter=debug cloud",
    )
    ap.add_argument("--map-dir", type=Path, default=None)
    ap.add_argument("--gt-npz", type=Path, default=Path("data/ird/gt_samples.npz"))
    ap.add_argument("--checkpoint", type=Path, default=None, help="Optional neural IRD overlay grid")
    ap.add_argument("--out", type=Path, default=Path("data/reports/global_ird.png"))
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--d-min", type=float, default=0.02)
    ap.add_argument(
        "--clim",
        type=float,
        nargs=2,
        default=[0.0, 0.30],
        metavar=("LO", "HI"),
        help="Colour limits in fraction units (default 0 0.30 ≈ map D max)",
    )
    ap.add_argument(
        "--clim-auto",
        action="store_true",
        help="Stretch colour bar to [d_min, max(values)] for this figure",
    )
    ap.add_argument(
        "--clim-abs",
        action="store_true",
        help="Absolute 0..1 scale (0–100%% bar, cross-map compare)",
    )
    ap.add_argument("--step-m", type=float, default=0.05, help="IRD voxelize step (max over orients)")
    ap.add_argument("--max-orients", type=int, default=6)
    ap.add_argument("--max-voxels", type=int, default=12_000, help="Cap map voxels when inverting")
    args = ap.parse_args(argv)

    root = Path(__file__).resolve().parents[2]
    out = _resolve(root, args.out)
    if args.clim_abs:
        clim = (0.0, 1.0)
    else:
        clim = (float(args.clim[0]), float(args.clim[1]))
    clim_auto = bool(args.clim_auto)
    if args.mode == "scatter":
        from ird_playground.ird.export_gt import load_ird_gt
        from ird_playground.viz.ird_compare import features_to_xyz, render_ird_comparison

        arrays = load_ird_gt(_resolve(root, args.gt_npz))
        path = render_ird_comparison(
            xyz=features_to_xyz(arrays["features"]),
            gt=arrays["d"],
            pred=None,
            out_path=out if out.suffix else out.with_suffix(".png"),
            value_name="IRD d (base in TCP)",
        )
        print(f"wrote {path}")
        return 0

    if args.mode == "capability":
        from ird_playground.viz.sphere_ird import render_ird_spheres

        map_dir = _map_dir(root, args)
        path = render_ird_spheres(
            map_dir, out if out.suffix else out.with_suffix(".png"),
            channel="reach_D", d_min=args.d_min, clim=clim,
        )
        print(f"wrote [capability-forward] {path}")
        return 0

    from ird_playground.viz.global_ird import (
        build_ird_points_from_capability,
        build_ird_points_from_gt_npz,
        render_global_ird,
        voxelize_max,
    )

    if args.mode == "ird_gt":
        xyz, q = build_ird_points_from_gt_npz(_resolve(root, args.gt_npz))
        title = "Global IRD"
    else:
        from ird_playground.ird.capability_io import load_capability_map_dir

        map_dir = _map_dir(root, args)
        cm = load_capability_map_dir(map_dir)
        # subsample high-d voxels for speed
        import numpy as np

        order = np.argsort(-cm.d_value)[: int(args.max_voxels)]
        # thin wrapper with subset — rebuild light view
        class _Sub:
            pass

        sub = _Sub()
        sub.orientations = cm.orientations
        sub.roll = cm.roll
        sub.bitmask = cm.bitmask[order]
        sub.d_value = cm.d_value[order]
        sub.voxel_ids = cm.voxel_ids[order]
        sub.grid = cm.grid
        xyz, q = build_ird_points_from_capability(sub, max_orients_per_voxel=args.max_orients)
        title = "Global IRD"

    xyz, q = voxelize_max(xyz, q, step_m=args.step_m)
    path = render_global_ird(
        xyz, q, out if out.suffix else out.with_suffix(".png"),
        d_min=args.d_min, clim=clim, clim_auto=clim_auto, title=title,
        sphere_radius_m=float(args.step_m) * 0.55,
    )
    print(
        f"wrote {path}  n_cells={xyz.shape[0]}  mean={float(q.mean()):.4f}  "
        f"max={float(q.max()):.4f}  clim={clim if not clim_auto else 'auto'}"
    )

    if args.checkpoint is not None:
        from ird_playground.neural.model import NeuralIRD
        from ird_playground.viz.global_ird import predict_ird_grid, render_global_ird as _r

        net = NeuralIRD.load(_resolve(root, args.checkpoint), device=args.device)
        lo = xyz.min(axis=0) - 0.05
        hi = xyz.max(axis=0) + 0.05
        gxyz, gd = predict_ird_grid(
            net,
            bbox=((float(lo[0]), float(hi[0])), (float(lo[1]), float(hi[1])), (float(lo[2]), float(hi[2]))),
            step_m=max(args.step_m, 0.05),
            n_orients=4,
        )
        pred_out = out.with_name(out.stem + "_pred" + out.suffix) if out.suffix else Path(str(out) + "_pred.png")
        gxyz, gd = voxelize_max(gxyz, gd, step_m=args.step_m)
        _r(gxyz, gd, pred_out, d_min=args.d_min, clim=clim, title="Neural IRD")
        print(f"wrote {pred_out}")
    return 0


def _map_dir(root: Path, args) -> Path:
    if args.map_dir is not None:
        p = Path(args.map_dir)
        return p if p.is_absolute() else root / p
    meta = _resolve(root, args.gt_npz).with_suffix(".yaml")
    if meta.is_file():
        return Path(yaml.safe_load(meta.read_text(encoding="utf-8"))["map_dir"])
    raise SystemExit("Provide --map-dir or gt npz sibling .yaml with map_dir")


if __name__ == "__main__":
    raise SystemExit(main())
