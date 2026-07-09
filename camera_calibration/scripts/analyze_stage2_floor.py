#!/usr/bin/env python3
"""Diagnose Stage 2 floor plane RMSE from saved captures."""
from __future__ import annotations

import json
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
from multicam_calib.calib.plane_fit import fit_plane_svd  # noqa: E402
from multicam_calib.calib.world_align import (  # noqa: E402
    _collect_ref_points,
    _estimate_T_ref_board_per_sample,
)
from multicam_calib.io.config import load_app, load_board  # noqa: E402
from multicam_calib.io.results import extrinsics_rel_path, load_extrinsics, load_intrinsics_map  # noqa: E402
from multicam_calib.recording.stage2_session import (  # noqa: E402
    WORKING_DIR_NAME,
    LAST_DIR_NAME,
    Stage2SessionBundle,
)


def _per_sample_plane_rmse_mm(pts: np.ndarray, sample_ids: np.ndarray, plane) -> list[tuple[int, float, int]]:
    resid_m = pts @ plane.normal - plane.d
    out: list[tuple[int, float, int]] = []
    for sid in sorted(set(sample_ids.tolist())):
        mask = sample_ids == sid
        r = resid_m[mask]
        rmse = float(np.sqrt(np.mean(r * r)) * 1000.0)
        out.append((int(sid), rmse, int(mask.sum())))
    return out


def _collect_with_sample_ids(session, board_geom, intrinsics, stage1, min_tags: int):
    pts: list[np.ndarray] = []
    ids: list[int] = []
    min_tags = int(min_tags)
    for sample in session.samples:
        T_ref_board, n_cams = _estimate_T_ref_board_per_sample(
            sample, board_geom, intrinsics, stage1, min_tags=min_tags
        )
        if T_ref_board is None:
            continue
        tag_ids = set()
        for det in sample.views.values():
            tag_ids.update(int(t) for t in det.tags.keys())
        for tag_id in sorted(tag_ids):
            model = board_geom.corners_by_tag.get(tag_id)
            if model is None:
                continue
            for k in range(4):
                ph = np.ones(4, dtype=np.float64)
                ph[:3] = model[k]
                pts.append((T_ref_board @ ph)[:3])
                ids.append(sample.index)
    if not pts:
        return np.empty((0, 3)), np.empty((0,), dtype=int)
    return np.stack(pts, axis=0), np.asarray(ids, dtype=int)


def main() -> int:
    app = load_app()
    board_geom = build_board_geometry(load_board())
    stage1 = load_extrinsics(extrinsics_rel_path())
    if stage1 is None:
        print("No extrinsics_rel.yaml")
        return 1

    intrinsics = load_intrinsics_map()
    aliases = list(stage1.poses.keys())

    for label, sub in [("working", WORKING_DIR_NAME), ("last", LAST_DIR_NAME)]:
        root = REPO / "data" / "stage2_world" / sub
        if not root.is_dir():
            continue
        bundle = Stage2SessionBundle.open_existing(
            root,
            aliases=aliases,
            detector=None,  # type: ignore[arg-type]
            recording_cfg=app.recording,
        )
        floor = bundle.as_legacy_session("floor")
        if not floor.samples:
            continue

        print(f"\n=== {label} / floor ({len(floor.samples)} samples) ===")
        print(f"Stage1 total_rmse_px: {stage1.metadata.get('total_rmse_px', '?'):.3f}")
        for alias, rmse in (stage1.metadata.get("per_camera_rmse_px") or {}).items():
            print(f"  {alias}: {rmse:.3f} px")

        min_tags = int(app.calibration.min_tags_per_view)
        pts, x_axes = _collect_ref_points(
            floor, board_geom, intrinsics, stage1, min_tags=min_tags
        )
        pts2, sids = _collect_with_sample_ids(
            floor, board_geom, intrinsics, stage1, min_tags=min_tags
        )
        fit = fit_plane_svd(pts)

        print(f"Points for plane fit: {fit.n_points}")
        print(f"Global plane RMSE: {fit.residual_mm:.2f} mm")
        print(f"Plane normal (ref): {fit.normal}")
        resid_mm = (pts @ fit.normal - fit.d) * 1000.0
        print(f"Residual range: [{resid_mm.min():.2f}, {resid_mm.max():.2f}] mm")
        print(f"Residual median |.|: {np.median(np.abs(resid_mm)):.2f} mm")

        print("\nPer-sample (fused board pose per capture):")
        for sid, rmse, n in _per_sample_plane_rmse_mm(pts2, sids, fit):
            sample = floor.samples[sid]
            cam_counts = ", ".join(f"{a}:{v.num_tags()}" for a, v in sample.views.items())
            spread = sample.metadata.get("host_ts_spread_ms", "?")
            phase = sample.metadata.get("phase", "?")
            T_rb, n_cams = _estimate_T_ref_board_per_sample(
                sample, board_geom, intrinsics, stage1, min_tags=min_tags
            )
            z_board = float(T_rb[:3, 3][2]) if T_rb is not None else float("nan")
            print(
                f"  #{sid:03d}  {n:4d} pts  plane RMSE {rmse:5.2f} mm  "
                f"fused_cams={n_cams}  board_z_ref={z_board:.3f}m  phase={phase}  spread={spread}ms"
            )
            print(f"         tags: {cam_counts}")

        # Height spread of fused board origins across samples
        origins = []
        for sample in floor.samples:
            T_rb, _ = _estimate_T_ref_board_per_sample(
                sample, board_geom, intrinsics, stage1, min_tags=min_tags
            )
            if T_rb is not None:
                origins.append(T_rb[:3, 3])
        if origins:
            oz = np.array([o[2] for o in origins])
            print(f"\nFused board origin Z in ref frame across samples:")
            print(f"  min={oz.min():.4f}  max={oz.max():.4f}  span={(oz.max()-oz.min())*1000:.1f} mm  std={oz.std()*1000:.2f} mm")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
