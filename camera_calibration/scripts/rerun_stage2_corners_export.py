#!/usr/bin/env python3
"""Re-export Stage 2 corners → extrinsics_world + genesis_bundle from saved captures.

Uses existing ``calibration_results/intrinsics.yaml`` and ``extrinsics_rel.yaml``;
re-runs only the corners phase (floor/bed aligned state must be in the session).

Example::

    # Copy or symlink session first, then:
    python scripts/rerun_stage2_corners_export.py \\
        --session-root data/stage2_world/last

    # Or point at Among_US capture tree:
    python scripts/rerun_stage2_corners_export.py \\
        --session-root /media/camp/EXT_DRIVE/Among_US/camera_calibration/data/stage2_world/last
"""
from __future__ import annotations

import argparse
import os
import site
import sys
from pathlib import Path

if os.environ.get("PYTHONNOUSERSITE") != "1":
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])

sys.path = [p for p in sys.path if not p.startswith(site.getusersitepackages())]
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from multicam_calib.board.apriltag_board import build_board_geometry  # noqa: E402
from multicam_calib.calib.world_align import run_stage2_phase  # noqa: E402
from multicam_calib.io.config import load_app, load_board, load_world  # noqa: E402
from multicam_calib.io.results import (  # noqa: E402
    extrinsics_rel_path,
    intrinsics_path,
    load_extrinsics,
    load_intrinsics_map,
)
from multicam_calib.recording.stage2_session import Stage2SessionBundle  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--session-root",
        type=Path,
        default=REPO / "data" / "stage2_world" / "last",
        help="Stage 2 session with floor/bed/corners captures + aligned_state.json",
    )
    args = ap.parse_args()

    session_root = args.session_root.resolve()
    if not (session_root / "aligned_state.json").is_file():
        print(f"Missing aligned_state.json under {session_root}", file=sys.stderr)
        return 1

    app_cfg = load_app()
    world_cfg = load_world()
    stage1 = load_extrinsics(extrinsics_rel_path())
    if stage1 is None:
        print("Missing extrinsics_rel.yaml — run Stage 1 first.", file=sys.stderr)
        return 1

    bundle = Stage2SessionBundle.open_existing(
        session_root,
        aliases=list(stage1.poses.keys()),
        detector=None,  # type: ignore[arg-type]
        recording_cfg=app_cfg.recording,
    )
    board_geom = build_board_geometry(load_board())
    intrinsics = load_intrinsics_map(intrinsics_path())

    print(f"Session: {session_root}")
    print(f"align_xy_to_bed: {world_cfg.align_xy_to_bed}")
    report = run_stage2_phase(
        bundle=bundle,
        phase="corners",
        board_geom=board_geom,
        intrinsics=intrinsics,
        stage1=stage1,
        app_cfg=app_cfg,
        world_cfg=world_cfg,
    )
    m = report.world_meta
    if m is None:
        print("Corners export produced no world_meta.", file=sys.stderr)
        return 1

    print(
        f"bed: {m.bed_size_m[0]:.3f} x {m.bed_size_m[1]:.3f} m, "
        f"z={m.bed_height_m:.3f} m, rot={m.bed_rotation_deg:.2f} deg"
    )
    if m.xy_aligned_to_bed:
        print(f"  xy aligned to bed (pre-align skew was {m.bed_xy_skew_deg_pre_align:.2f} deg)")
    print(f"  origin floor: {m.bed_center_on_floor}")
    print("Wrote calibration_results/extrinsics_world.yaml, world_meta.yaml, genesis_bundle.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
