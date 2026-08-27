#!/usr/bin/env python3
"""Append one Stage-5 RGB-D view into a *new* yaml (does not touch Stage 5 yaml).

USB exclusive — stop ``run_orbbec_cloud_publisher.py`` first.

  source camera_calibration/env.sh
  python camera_calibration/scripts/capture_stage5_rgbd.py
  python camera_calibration/scripts/capture_stage5_rgbd.py --note "j6 pitch near"
  python camera_calibration/scripts/capture_stage5_rgbd.py --status
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import yaml

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
from multicam_calib.io.config import RESULTS_DIR, load_app, load_board, load_orbbec, load_robot  # noqa: E402

SESSION_YAML = RESULTS_DIR / "orbbec_handeye_captures_rgbd.yaml"
FRAMES_DIR = Path(__file__).resolve().parents[1] / "data" / "stage5_rgbd"
SCHEMA = "orbbec_handeye_captures_rgbd_v1"
MIN_TAGS = 8


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else np.array([0.0, 0.0, 1.0])


def _ang_deg(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.rad2deg(np.arccos(np.clip(_unit(a) @ _unit(b), -1.0, 1.0))))


def _load_session(path: Path) -> dict:
    if not path.is_file():
        return {
            "schema": SCHEMA,
            "session": "stage5_rgbd",
            "created_utc": _utc(),
            "n_samples": 0,
            "target_min": 25,
            "target_max": 40,
            "frames_dir": "../data/stage5_rgbd",
            "does_not_overwrite": ["orbbec_handeye.yaml", "orbbec_handeye_captures.yaml"],
            "captures": [],
        }
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    data.setdefault("captures", [])
    return data


def _save_session(path: Path, data: dict) -> None:
    data["updated_utc"] = _utc()
    data["n_samples"] = len(data.get("captures") or [])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    tmp.replace(path)


def _next_id(captures: list[dict]) -> str:
    used = set()
    for c in captures:
        name = str(c.get("id") or "")
        if name.startswith("g") and name[1:].isdigit():
            used.add(int(name[1:]))
    n = 1
    while n in used:
        n += 1
    return f"g{n:02d}"


def _status(data: dict) -> str:
    caps = data.get("captures") or []
    lo = int(data.get("target_min") or 25)
    hi = int(data.get("target_max") or 40)
    lines = [f"session {data.get('session')}  n={len(caps)}  target={lo}-{hi}  yaml={SESSION_YAML}"]
    for c in caps:
        ang = c.get("angle_tag_vs_depth_board_deg")
        ang_s = "   n/a" if ang is None else f"{float(ang):6.2f}°"
        lines.append(
            f"  {c.get('id')}: tags={c.get('n_tags')} pnp={float(c.get('pnp_rmse_px') or 0):.2f}px "
            f"tag-depth={ang_s} rail={float(c.get('rail_m') or 0):.3f}  {c.get('note') or ''}"
        )
    return "\n".join(lines)


def _fit_plane(xyz: np.ndarray) -> tuple[np.ndarray | None, int]:
    """Dominant board plane. Depth can be holed / mixed with arm; RANSAC first."""
    pts = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0] < 30:
        return None, 0
    zmed = float(np.median(pts[:, 2]))
    band = np.abs(pts[:, 2] - zmed) < 0.12
    if int(np.count_nonzero(band)) >= 30:
        pts = pts[band]
    rng = np.random.default_rng(0)
    n = int(pts.shape[0])
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
        inl = np.abs(pts @ nrm - float(nrm @ p0)) < 0.008
        c = int(np.count_nonzero(inl))
        if c > best_c:
            best_c = c
            best_inl = inl
    used = pts if best_inl is None or best_c < 30 else pts[best_inl]
    cxyz = used.mean(axis=0)
    _, _, vh = np.linalg.svd(used - cxyz, full_matrices=False)
    normal = _unit(vh[-1])
    if float(normal @ cxyz) > 0.0:
        normal = -normal
    return normal, int(used.shape[0])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yaml", type=Path, default=SESSION_YAML)
    ap.add_argument("--frames", type=Path, default=FRAMES_DIR)
    ap.add_argument("--note", type=str, default="")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--force", action="store_true", help="Skip stillness gate")
    args = ap.parse_args()

    data = _load_session(args.yaml)
    if args.status:
        print(_status(data), flush=True)
        return 0

    board = build_board_geometry(load_board())
    detector = AprilTagDetector(board.config, load_app().detector)
    robot_cfg = load_robot()
    reader = RobotStateReader()
    if args.force:
        snap = reader.read()
        if snap is None:
            print(f"rm75_state missing: {reader.last_error}", file=sys.stderr)
            return 2
        still_msg = "force (stillness skipped)"
    else:
        snap, still = reader.wait_still(
            window_s=robot_cfg.stillness.window_s,
            trans_m=robot_cfg.stillness.trans_m,
            rot_deg=robot_cfg.stillness.rot_deg,
            rail_m=robot_cfg.stillness.rail_m,
        )
        if snap is None or not still.ok:
            print(f"rejected: {still.message if snap is not None else reader.last_error}", file=sys.stderr)
            return 3
        still_msg = still.message

    cfg = load_orbbec()
    cfg.fps = 30
    cfg.depth_fps = 30
    session = OrbbecRGBDSession(cfg)
    try:
        params = session.open()
        if not session.has_depth:
            print("Orbbec opened without depth. Stop the cloud publisher and retry.", file=sys.stderr)
            return 4
        frame = None
        for _ in range(8):
            frame = session.read(timeout_ms=2000)
            if frame is not None and frame.depth_m is not None:
                break
        if frame is None or frame.depth_m is None:
            print("no RGB-D frame", file=sys.stderr)
            return 5
        snap2 = reader.read() or snap
        color = frame.color_bgr
        depth = np.asarray(frame.depth_m, dtype=np.float32)
        h, w = color.shape[:2]
        K_intr = load_orbbec_color_intrinsics(image_size=(w, h))
        K = np.asarray(K_intr.K, dtype=np.float64).reshape(3, 3)
        dets = detector.detect(color)
        det_map = AprilTagDetector.detections_to_dict(dets)
        if len(det_map) < MIN_TAGS:
            print(f"rejected: only {len(det_map)} tags (need ≥{MIN_TAGS})", file=sys.stderr)
            return 6
        pose = solve_view_pose(board, det_map, K_intr, min_tags=MIN_TAGS)
        if pose is None:
            print("rejected: PnP failed", file=sys.stderr)
            return 7

        corners = np.concatenate([np.asarray(c).reshape(-1, 2) for c in det_map.values()], axis=0)
        hull = cv2.convexHull(corners.astype(np.float32).reshape(-1, 1, 2))
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull.astype(np.int32), 1)
        mask = cv2.erode(mask, np.ones((15, 15), np.uint8), iterations=1)
        depth_b = np.where(mask.astype(bool), depth, 0.0).astype(np.float32)
        xyz_board, _ = unproject_aligned_depth(
            depth_b, K, color_bgr=color, stride=2, min_m=cfg.min_depth_m, max_m=cfg.max_depth_m
        )
        n_tag = _unit(pose.T_cam_board[:3, 2])
        if float(n_tag @ pose.T_cam_board[:3, 3]) > 0.0:
            n_tag = -n_tag
        n_dep, n_plane = _fit_plane(xyz_board)
        ang = None if n_dep is None else _ang_deg(n_tag, n_dep)

        gid = _next_id(data["captures"])
        dest = args.frames / gid
        dest.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dest / "color.png"), color)
        vis = color.copy()
        for d in dets:
            cv2.polylines(vis, [d.corners.astype(np.int32).reshape(-1, 1, 2)], True, (0, 255, 0), 1)
        cv2.imwrite(str(dest / "color_tags.png"), vis)
        np.save(dest / "depth_m.npy", depth)
        t_cd = getattr(params, "T_color_depth", None)
        np.savez_compressed(
            dest / "cloud.npz",
            xyz_board=xyz_board.astype(np.float32),
            K=K,
            T_cam_board=np.asarray(pose.T_cam_board, dtype=np.float64),
            n_tag=n_tag,
            n_depth_board=np.zeros(3) if n_dep is None else n_dep,
        )
        (dest / "summary.json").write_text(
            json.dumps(
                {
                    "id": gid,
                    "note": args.note,
                    "n_tags": len(det_map),
                    "pnp_rmse_px": float(pose.reprojection_rmse_px),
                    "angle_tag_vs_depth_board_deg": ang,
                    "rail_m": float(snap2.rail_m),
                    "q_deg": snap2.q_deg.tolist(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        cap = {
            "id": gid,
            "note": str(args.note),
            "n_tags": int(len(det_map)),
            "rail_m": float(snap2.rail_m),
            "q_deg": [float(v) for v in snap2.q_deg.tolist()],
            "image_size": [int(w), int(h)],
            "T_railbase_tcp": snap2.T_railbase_tcp().tolist(),
            "detections": {int(k): np.asarray(v, dtype=float).reshape(-1, 2).tolist() for k, v in det_map.items()},
            "pnp_rmse_px": float(pose.reprojection_rmse_px),
            "angle_tag_vs_depth_board_deg": None if ang is None else float(ang),
            "n_depth_board": int(xyz_board.shape[0]),
            "n_depth_plane_inliers": int(n_plane),
            "color_png": f"{gid}/color.png",
            "depth_npy": f"{gid}/depth_m.npy",
            "serial": str(getattr(params, "serial", "") or ""),
            "backend": str(session.backend),
            "K_source": str(K_intr.source),
            "still": still_msg,
            "wall_time_ns": int(time.time_ns()),
        }
        if t_cd is not None:
            data.setdefault("T_color_depth", np.asarray(t_cd, dtype=np.float64).reshape(4, 4).tolist())
        data.setdefault("serial", cap["serial"])
        data["captures"].append(cap)
        _save_session(args.yaml, data)
        n = len(data["captures"])
        lo = int(data.get("target_min") or 25)
        hi = int(data.get("target_max") or 40)
        print(
            f"saved {gid}  tags={cap['n_tags']} pnp={cap['pnp_rmse_px']:.2f}px "
            f"tag-depth={ang if ang is None else f'{ang:.2f}°'}  "
            f"rail={cap['rail_m']:.3f}  q7={cap['q_deg'][6]:.1f}  "
            f"n={n}/{lo}-{hi}\n{_status(data)}",
            flush=True,
        )
    finally:
        session.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
