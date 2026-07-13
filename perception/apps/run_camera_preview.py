#!/usr/bin/env python3
"""Single-strip OpenCV preview from RealSense ZMQ ingress (N cameras from bundle)."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _load_camera_names(bundle_path: Path, only: list[str] | None) -> list[str]:
    import yaml

    payload = yaml.safe_load(bundle_path.read_text(encoding="utf-8"))
    cams = payload.get("cameras") or {}
    if not isinstance(cams, dict) or not cams:
        raise ValueError(f"No cameras in bundle: {bundle_path}")
    names = [str(k) for k in cams.keys()]
    if only:
        missing = [c for c in only if c not in cams]
        if missing:
            raise ValueError(f"Cameras not in bundle: {missing}")
        return list(only)
    return names


def main() -> int:
    repo = Path(os.environ.get("REALUS_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--bundle",
        type=Path,
        default=Path(os.environ.get("CAMERA_CALIB_BUNDLE", repo / "camera_calibration/calibration_results/genesis_bundle.yaml")),
    )
    ap.add_argument("--connect", type=str, default="tcp://127.0.0.1:17356")
    ap.add_argument(
        "--topic",
        type=str,
        default="amongus_camera_preview_v1",
        help="Preview topic (default: low-res amongus_camera_preview_v1)",
    )
    ap.add_argument("--cameras", type=str, nargs="*", default=None, help="Subset of bundle aliases (default: all)")
    ap.add_argument(
        "--separate-windows",
        action="store_true",
        help="One OS window per camera (default: single horizontal strip).",
    )
    ap.add_argument("--tile-width", type=int, default=480, help="Tile width in strip (default 480 → 4 cams ≈ 1920×270 strip).")
    ap.add_argument("--tile-height", type=int, default=270)
    ap.add_argument("--window-name", type=str, default="RealUS Cam Strip")
    ap.add_argument("--draw-fps", type=float, default=30.0)
    args = ap.parse_args()

    names = _load_camera_names(args.bundle.resolve(), args.cameras)
    py = os.environ.get("PY", sys.executable)
    cmd = [
        py,
        "-m",
        "projects.genesis_ue_sync.cli.render.unreal.watch_ue_camera_frames",
        "--connect",
        str(args.connect),
        "--topic",
        str(args.topic),
        "--camera-names",
        *names,
        "--tile-width",
        str(args.tile_width),
        "--tile-height",
        str(args.tile_height),
        "--window-name",
        str(args.window_name),
        "--draw-fps",
        str(args.draw_fps),
    ]
    if args.separate_windows:
        cmd.append("--separate-windows")
    os.chdir(repo)
    env = dict(os.environ)
    src = str((repo / "src").resolve())
    env["PYTHONPATH"] = src if not env.get("PYTHONPATH") else f"{src}:{env['PYTHONPATH']}"
    mode = "separate_windows" if args.separate_windows else "strip"
    print(f"Preview cameras ({mode}): {' '.join(names)}", flush=True)
    return int(subprocess.call(cmd, cwd=str(repo), env=env))


if __name__ == "__main__":
    raise SystemExit(main())
