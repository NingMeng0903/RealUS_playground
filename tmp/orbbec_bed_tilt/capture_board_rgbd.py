#!/usr/bin/env python3
"""Same-frame Orbbec RGB + D2C depth + AprilTag vs depth plane (camera frame).

Board may be tilted; it must not move. Does not write calibration yaml.

  source camera_calibration/env.sh
  # camera USB exclusive — stop the cloud publisher first
  python tmp/orbbec_bed_tilt/capture_board_rgbd.py --group 01
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_REPO = Path(os.environ.get("REALUS_PROJECT_ROOT") or Path(__file__).resolve().parents[2]).resolve()
for _p in (_REPO / "rm75_control", _REPO / "camera_calibration" / "src"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from multicam_calib.board.apriltag_board import build_board_geometry  # noqa: E402
from multicam_calib.board.detector import AprilTagDetector  # noqa: E402
from multicam_calib.calib.orbbec_handeye import load_orbbec_color_intrinsics  # noqa: E402
from multicam_calib.calib.orbbec_rgbd import unproject_aligned_depth  # noqa: E402
from multicam_calib.calib.pnp import solve_view_pose  # noqa: E402
from multicam_calib.devices.orbbec import OrbbecRGBDSession  # noqa: E402
from multicam_calib.ingress.robot_state import RobotStateReader  # noqa: E402
from multicam_calib.io.config import load_app, load_board, load_orbbec  # noqa: E402

OUT_ROOT = Path(__file__).resolve().parent / "board_rgbd"


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    return v / (float(np.linalg.norm(v)) + 1e-12)


def _ang_deg(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.rad2deg(np.arccos(np.clip(_unit(a) @ _unit(b), -1.0, 1.0))))


def _fit_plane_cam(xyz: np.ndarray) -> tuple[np.ndarray, float, int]:
    """RANSAC the dominant plane (board), then SVD on inliers. Ignores floor/arm."""
    pts = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0] < 20:
        raise RuntimeError(f"too few points: {pts.shape[0]}")
    zmed = float(np.median(pts[:, 2]))
    band = np.abs(pts[:, 2] - zmed) < 0.12
    if int(np.count_nonzero(band)) >= 20:
        pts = pts[band]
    rng = np.random.default_rng(0)
    n = int(pts.shape[0])
    thresh = 0.008
    best_inl = None
    best_c = 0
    for _ in range(120):
        idx = rng.choice(n, 3, replace=False)
        p0, p1, p2 = pts[idx]
        nrm = np.cross(p1 - p0, p2 - p0)
        ln = float(np.linalg.norm(nrm))
        if ln < 1e-9:
            continue
        nrm = nrm / ln
        resid = np.abs(pts @ nrm - float(nrm @ p0))
        inl = resid < thresh
        c = int(np.count_nonzero(inl))
        if c > best_c:
            best_c = c
            best_inl = inl
    if best_inl is None or best_c < 20:
        used = pts
    else:
        used = pts[best_inl]
    cxyz = used.mean(axis=0)
    _, _, vh = np.linalg.svd(used - cxyz, full_matrices=False)
    normal = _unit(vh[-1])
    if float(normal @ cxyz) > 0.0:
        normal = -normal
    resid = used @ normal - float(cxyz @ normal)
    rms = float(np.sqrt(np.mean(resid * resid)))
    return normal, rms, int(used.shape[0])


def _board_mask(shape_hw: tuple[int, int], corners: np.ndarray) -> np.ndarray:
    hull = cv2.convexHull(corners.astype(np.float32).reshape(-1, 1, 2))
    mask = np.zeros(shape_hw, dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull.astype(np.int32), 1)
    mask = cv2.erode(mask, np.ones((21, 21), np.uint8), iterations=1)
    return mask.astype(bool)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True)
    ap.add_argument("--note", default="")
    args = ap.parse_args()
    gid = str(args.group).strip().zfill(2)
    dest = OUT_ROOT / f"g{gid}"
    dest.mkdir(parents=True, exist_ok=True)

    cfg = load_orbbec()
    cfg.fps = 30
    cfg.depth_fps = 30
    board = build_board_geometry(load_board())
    app = load_app()
    detector = AprilTagDetector(board.config, app.detector)
    reader = RobotStateReader()
    snap = reader.read()

    session = OrbbecRGBDSession(cfg)
    try:
        params = session.open()
        frame = None
        for _ in range(8):
            frame = session.read(timeout_ms=2000)
            if frame is not None and frame.depth_m is not None:
                break
        if frame is None or frame.depth_m is None:
            print("no RGB-D frame", file=sys.stderr)
            return 2
        snap2 = reader.read() or snap
        color = frame.color_bgr
        depth = np.asarray(frame.depth_m, dtype=np.float32)
        h, w = color.shape[:2]
        K_intr = load_orbbec_color_intrinsics(image_size=(w, h))
        K = np.asarray(K_intr.K, dtype=np.float64).reshape(3, 3)
        dets = detector.detect(color)
        det_map = AprilTagDetector.detections_to_dict(dets)
        pose = solve_view_pose(board, det_map, K_intr, min_tags=4)
        xyz_all, rgb_all = unproject_aligned_depth(
            depth, K, color_bgr=color, stride=2, min_m=cfg.min_depth_m, max_m=cfg.max_depth_m
        )
        n_all, rms_all, npts_all = _fit_plane_cam(xyz_all)

        n_board = None
        rms_board = None
        npts_board = 0
        xyz_board = np.zeros((0, 3), dtype=np.float32)
        if det_map:
            corners = np.concatenate([np.asarray(c).reshape(-1, 2) for c in det_map.values()], axis=0)
            mask = _board_mask((h, w), corners)
            depth_b = np.where(mask, depth, 0.0).astype(np.float32)
            xyz_board, _ = unproject_aligned_depth(
                depth_b, K, color_bgr=color, stride=2, min_m=cfg.min_depth_m, max_m=cfg.max_depth_m
            )
            if xyz_board.shape[0] >= 20:
                n_board, rms_board, npts_board = _fit_plane_cam(xyz_board)

        n_tag = None
        pnp_rmse = None
        T_cb = None
        if pose is not None:
            T_cb = np.asarray(pose.T_cam_board, dtype=np.float64)
            n_tag = _unit(T_cb[:3, 2])
            # board +Z is out the front; flip toward camera if needed
            if float(n_tag @ T_cb[:3, 3]) > 0.0:
                n_tag = -n_tag
            pnp_rmse = float(pose.reprojection_rmse_px)

        ang_tag_board = _ang_deg(n_tag, n_board) if n_tag is not None and n_board is not None else None
        ang_tag_all = _ang_deg(n_tag, n_all) if n_tag is not None else None

        vis = color.copy()
        for d in dets:
            pts = d.corners.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(vis, [pts], True, (0, 255, 0), 1)
        cv2.imwrite(str(dest / "color.png"), color)
        cv2.imwrite(str(dest / "color_tags.png"), vis)
        np.save(dest / "depth_m.npy", depth)
        np.savez_compressed(
            dest / "cloud.npz",
            xyz_all=xyz_all.astype(np.float32),
            rgb_all=(rgb_all if rgb_all is not None else np.zeros((0, 3), np.uint8)),
            xyz_board=xyz_board.astype(np.float32),
            K=K,
            T_cam_board=T_cb if T_cb is not None else np.eye(4),
            n_tag=n_tag if n_tag is not None else np.zeros(3),
            n_depth_board=n_board if n_board is not None else np.zeros(3),
            n_depth_all=n_all,
        )
        summary = {
            "group": gid,
            "note": str(args.note),
            "wall_time_ns": int(time.time_ns()),
            "serial": getattr(params, "serial", ""),
            "backend": session.backend,
            "color_size": [w, h],
            "n_tags": int(len(det_map)),
            "tag_ids": sorted(int(i) for i in det_map),
            "pnp_rmse_px": pnp_rmse,
            "n_depth_all": npts_all,
            "n_depth_board": npts_board,
            "depth_all_rms_mm": rms_all * 1000.0,
            "depth_board_rms_mm": None if rms_board is None else rms_board * 1000.0,
            "n_tag_cam": None if n_tag is None else n_tag.tolist(),
            "n_depth_board_cam": None if n_board is None else n_board.tolist(),
            "n_depth_all_cam": n_all.tolist(),
            "angle_tag_vs_depth_board_deg": ang_tag_board,
            "angle_tag_vs_depth_all_deg": ang_tag_all,
            "K": K.tolist(),
            "K_source": str(K_intr.source),
            "T_cam_board": None if T_cb is None else T_cb.tolist(),
            "robot": None
            if snap2 is None
            else {
                "seq": int(snap2.seq),
                "rail_m": float(snap2.rail_m),
                "q_deg": snap2.q_deg.tolist(),
                "pose6": snap2.pose.tolist(),
            },
        }
        (dest / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(
            f"saved {dest} tags={len(det_map)} pnp={pnp_rmse} "
            f"ang_tag_vs_depth_board={ang_tag_board} "
            f"ang_tag_vs_depth_all={ang_tag_all} "
            f"board_n={npts_board} all_n={npts_all}",
            flush=True,
        )
    finally:
        session.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
