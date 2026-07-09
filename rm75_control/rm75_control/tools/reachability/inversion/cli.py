"""``python -m rm75_control.tools.reachability.inversion.cli``"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from rm75_control.tools.reachability.inversion.base_optimizer import full_scan_best_yb
from rm75_control.tools.reachability.inversion.loader import load_map
from rm75_control.tools.reachability.inversion.prefix_solver import longest_prefix
from rm75_control.tools.reachability.inversion.quality import QualityWeights
from rm75_control.tools.reachability.inversion.trajectory import load_trajectory_json


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--map", type=Path, required=True)
    ap.add_argument("--trajectory", type=Path, required=True)
    ap.add_argument("--xb", type=float, default=0.0)
    ap.add_argument("--zb", type=float, default=0.0)
    ap.add_argument("--yb-range", type=float, nargs=2, default=[-0.5, 0.5])
    ap.add_argument("--yb-step", type=float, default=0.01)
    ap.add_argument("--rail-half", type=float, default=0.18, help="rail travel half-range (m)")
    ap.add_argument("--mode", choices=["full", "prefix", "both"], default="both")
    ap.add_argument("--report", type=Path, default=None)
    ap.add_argument("--no-relaxed", action="store_true", help="disable relaxed prefix tolerance")
    return ap.parse_args(argv)


def _hash_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.map.exists():
        print(f"map not found: {args.map}", file=sys.stderr)
        return 2
    if not args.trajectory.exists():
        print(f"trajectory not found: {args.trajectory}", file=sys.stderr)
        return 2

    t0 = time.perf_counter()
    cm = load_map(args.map, mmap=True)
    traj = load_trajectory_json(args.trajectory)
    xz = (float(args.xb), float(args.zb))
    yb_range = (float(args.yb_range[0]), float(args.yb_range[1]))
    qw = QualityWeights()

    report: dict = {
        "meta": {
            "map_dir": str(args.map),
            "trajectory": str(args.trajectory),
            "traj_hash": _hash_file(args.trajectory),
            "n_waypoints": traj.n,
            "elapsed_s": 0.0,
        }
    }

    if args.mode in ("full", "both"):
        full = full_scan_best_yb(
            cm, traj, xz_base_world=xz, yb_range=yb_range, yb_step=args.yb_step,
            rail_travel_half=args.rail_half, quality=qw,
        )
        report["full_scan"] = {
            "feasible": full.feasible,
            "y_b_best": full.y_b_best,
            "score": full.score,
            "rail_y_series": full.rail_y_series,
            "feasible_y_b_count": len(full.feasible_y_b_intervals),
        }
        if full.feasible:
            print(f"[full] y_b*={full.y_b_best:.4f} m  score={full.score:.3f}  n_wp={traj.n}")
        else:
            print("[full] no feasible y_b in range")

    if args.mode in ("prefix", "both"):
        pref = longest_prefix(
            cm, traj, xz_base_world=xz, yb_range=yb_range, yb_step=args.yb_step,
            rail_travel_half=args.rail_half, quality=qw, try_relaxed=not args.no_relaxed,
        )
        report["prefix"] = {
            "feasible": pref.feasible,
            "y_b_best": pref.y_b_best,
            "last_wp_index": pref.last_wp_index,
            "arc_len_m": pref.arc_len_m,
            "rail_y": pref.rail_y,
            "rail_y_series": pref.rail_y_series,
            "score": pref.score,
            "relaxed": pref.relaxed,
        }
        if pref.feasible:
            print(
                f"[prefix] y_b*={pref.y_b_best:.4f} m  last_wp={pref.last_wp_index}  "
                f"arc={pref.arc_len_m:.3f} m  rail_y={pref.rail_y:+.4f} m  "
                f"relaxed={pref.relaxed}"
            )
        else:
            print("[prefix] no feasible prefix")

    report["meta"]["elapsed_s"] = round(time.perf_counter() - t0, 3)
    print(f"[inversion] elapsed {report['meta']['elapsed_s']:.3f} s")

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2))
        print(f"wrote {args.report}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
