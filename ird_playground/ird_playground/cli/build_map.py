"""Patch URDF with probe TCP and optionally invoke rm75 reachability build (subprocess)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from ird_playground.probe.transform import load_probe_yaml, patch_urdf_tcp


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe", type=Path, default=Path("configs/probe_default.yaml"))
    ap.add_argument(
        "--src-urdf",
        type=Path,
        default=None,
        help="Base 8-DOF URDF (default: rm75_control asset)",
    )
    ap.add_argument("--out-urdf", type=Path, default=Path("data/maps/RM75-probe.urdf"))
    ap.add_argument(
        "--reachability-config",
        type=Path,
        default=None,
        help="If set, run rm75 reachability build with patched URDF",
    )
    ap.add_argument("--output-map", type=Path, default=Path("data/maps/probe_capability"))
    ap.add_argument("--mc-samples", type=int, default=None, help="Override MC samples for quick builds")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    root = Path(__file__).resolve().parents[2]  # ird_playground/
    rm75 = Path(__file__).resolve().parents[3] / "rm75_control"
    src = args.src_urdf or (
        rm75 / "rm75_control/assets/robots/rm75_6f_8dof/RM75-6F-8dof.urdf"
    )
    probe = load_probe_yaml(args.probe if args.probe.is_absolute() else root / args.probe)
    out_urdf = args.out_urdf if args.out_urdf.is_absolute() else root / args.out_urdf
    patch_urdf_tcp(src, out_urdf, probe)
    print(f"patched URDF → {out_urdf}  probe={probe.name}")

    if args.reachability_config is None:
        return 0

    cfg_path = args.reachability_config
    if not cfg_path.is_absolute():
        # allow configs under rm75_control
        cand = rm75 / cfg_path
        cfg_path = cand if cand.exists() else root / args.reachability_config

    cmd = [
        sys.executable,
        "-m",
        "rm75_control.tools.reachability.build.cli",
        "--config",
        str(cfg_path),
        "--urdf",
        str(out_urdf),
        "--output",
        str(args.output_map if args.output_map.is_absolute() else root / args.output_map),
    ]
    if args.mc_samples is not None:
        cmd += ["--mc-samples", str(args.mc_samples)]
    if args.dry_run:
        cmd.append("--dry-run")
    print(" ".join(cmd))
    return subprocess.call(cmd, cwd=str(rm75))


if __name__ == "__main__":
    raise SystemExit(main())
