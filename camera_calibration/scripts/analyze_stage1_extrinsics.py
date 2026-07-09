#!/usr/bin/env python3
"""Diagnose whether a camera's Stage 1 error comes from intrinsics or extrinsics.

For every camera this prints:

1. Single-view PnP RMSE — each camera fits its own K/dist to its own detected
   corners, independent of every other camera. This isolates INTRINSICS
   quality: it cannot be dragged down by a bad relative pose because no other
   camera's data is involved.
2. Pairwise relative-pose consistency — for every camera pair, the per-frame
   `T_a_b = T_a_board · T_b_board^-1` computed independently frame-by-frame
   (again from single-view PnP only, no bundle adjustment). If a pair's
   estimates disagree a lot from frame to frame even though the physical
   cameras never move, that scatter is caused by whichever camera's lens
   model doesn't fully explain image formation as the board moves around the
   frame — i.e. still an INTRINSICS symptom, not a "wrong extrinsic" one
   (a genuinely wrong but *fixed* extrinsic would not vary frame-to-frame).

Compares against the current `calibration_results/extrinsics_rel.yaml`
per-camera bundle-adjustment RMSE for context.

Usage:
    python scripts/analyze_stage1_extrinsics.py [--last]

By default analyzes `data/stage1_extrinsics/working/`; pass --last to analyze
`data/stage1_extrinsics/last/` instead.
"""
from __future__ import annotations

import argparse
import itertools
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

import numpy as np  # noqa: E402

from multicam_calib.board.apriltag_board import build_board_geometry  # noqa: E402
from multicam_calib.calib.pnp import ViewPose, solve_view_pose  # noqa: E402
from multicam_calib.io.config import DATA_DIR, load_app, load_board  # noqa: E402
from multicam_calib.io.results import extrinsics_rel_path, load_extrinsics, load_intrinsics_map  # noqa: E402
from multicam_calib.recording.session import RecordingSession  # noqa: E402


def _rot_angle_deg(R: np.ndarray) -> float:
    tr = np.clip((np.trace(R) - 1) / 2, -1, 1)
    return float(np.degrees(np.arccos(tr)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--last", action="store_true", help="Analyze data/stage1_extrinsics/last/ instead of working/")
    args = ap.parse_args()

    app = load_app()
    board_geom = build_board_geometry(load_board())
    intrinsics = load_intrinsics_map()
    min_tags = int(app.calibration.min_tags_per_view)
    aliases = sorted(intrinsics.keys())

    sub = "last" if args.last else "working"
    sess_dir = DATA_DIR / "stage1_extrinsics" / sub
    session = RecordingSession(
        session_dir=sess_dir, stage="stage1_extrinsics", aliases=aliases, detector=None, recording_cfg=app.recording
    )
    session.load_existing()
    print(f"Analyzing {sess_dir} — {len(session.samples)} samples\n")
    if not session.samples:
        print("No samples found.")
        return

    def get_pose(sample, alias: str) -> ViewPose | None:
        det = sample.views.get(alias)
        if det is None or det.num_tags() < min_tags:
            return None
        intr = intrinsics.get(alias)
        if intr is None:
            return None
        return solve_view_pose(board_geom, det.tags, intr, min_tags=min_tags)

    per_cam_rmse: dict[str, list[float]] = {a: [] for a in aliases}
    for sample in session.samples:
        for alias in aliases:
            pose = get_pose(sample, alias)
            if pose is not None:
                per_cam_rmse[alias].append(pose.reprojection_rmse_px)

    ba_rmse: dict[str, float] = {}
    ext = load_extrinsics(extrinsics_rel_path())
    if ext is not None:
        ba_rmse = dict(ext.metadata.get("per_camera_rmse_px") or {})

    print("=== 1. Single-view PnP RMSE (intrinsics-only signal) ===")
    print(f"{'cam':>6} {'n_views':>8} {'mean_px':>9} {'median_px':>10} {'p90_px':>8} {'BA_rmse_px':>11}  ratio(BA/PnP)")
    for a in aliases:
        arr = np.array(per_cam_rmse[a])
        if len(arr) == 0:
            print(f"{a:>6}   no qualifying views")
            continue
        ba = ba_rmse.get(a)
        ba_txt = f"{ba:>11.3f}" if ba is not None else f"{'n/a':>11}"
        ratio_txt = f"  {ba / arr.mean():.2f}x" if ba is not None else ""
        print(
            f"{a:>6} {len(arr):>8} {arr.mean():>9.3f} {np.median(arr):>10.3f} "
            f"{np.percentile(arr, 90):>8.3f} {ba_txt}{ratio_txt}"
        )
    print(
        "\nInterpretation: mean_px already ~2x another camera's even in this single-camera\n"
        "test (no cross-camera info involved) => that camera's OWN intrinsics/distortion\n"
        "model is the weaker link. A large BA/PnP ratio on top of that means bundle\n"
        "adjustment is additionally compromising that camera's pose to reconcile it with\n"
        "the rest of the rig — i.e. its noisy intrinsics are also destabilizing the\n"
        "estimated extrinsic.\n"
    )

    print("=== 2. Pairwise co-visibility (frames where both cameras qualify) ===")
    qual_per_frame = [sorted(a for a in aliases if get_pose(s, a) is not None) for s in session.samples]
    pair_counts = {p: 0 for p in itertools.combinations(aliases, 2)}
    for qual in qual_per_frame:
        for p in itertools.combinations(qual, 2):
            pair_counts[p] += 1
    for p, n in sorted(pair_counts.items()):
        print(f"  {p[0]}-{p[1]}: {n}")

    print("\n=== 3. Pairwise relative-pose consistency (single-view PnP only, no BA) ===")
    print("(large spread here that is NOT shared by pairs excluding a given camera implicates that camera)\n")
    for a, b in itertools.combinations(aliases, 2):
        Ts = []
        for s in session.samples:
            pa, pb = get_pose(s, a), get_pose(s, b)
            if pa is None or pb is None:
                continue
            Ts.append(pa.T_cam_board @ np.linalg.inv(pb.T_cam_board))
        if len(Ts) < 2:
            print(f"  {a}-{b}: not enough co-observed frames ({len(Ts)})")
            continue
        Ts_arr = np.array(Ts)
        mean_t = Ts_arr[:, :3, 3].mean(axis=0)
        t_dev_mm = np.linalg.norm(Ts_arr[:, :3, 3] - mean_t, axis=1) * 1000
        R0 = Ts_arr[0, :3, :3]
        ang_devs = np.array([_rot_angle_deg(R0.T @ T[:3, :3]) for T in Ts_arr])
        print(
            f"  {a}-{b}: n={len(Ts)}  translation spread mean={t_dev_mm.mean():.2f}mm "
            f"max={t_dev_mm.max():.2f}mm | rotation spread mean={ang_devs.mean():.3f}deg "
            f"max={ang_devs.max():.3f}deg"
        )


if __name__ == "__main__":
    main()
