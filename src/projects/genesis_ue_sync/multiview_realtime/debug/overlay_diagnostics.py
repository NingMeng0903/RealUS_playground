"""Diagnostics for multiview track debug overlay (third-party red/green review)."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

BODY25_JOINT_NAMES: tuple[str, ...] = (
    "Nose",
    "Neck",
    "RShoulder",
    "RElbow",
    "RWrist",
    "LShoulder",
    "LElbow",
    "LWrist",
    "MidHip",
    "RHip",
    "RKnee",
    "RAnkle",
    "LHip",
    "LKnee",
    "LAnkle",
    "REye",
    "LEye",
    "REar",
    "LEar",
    "LBigToe",
    "LSmallToe",
    "LHeel",
    "RBigToe",
    "RSmallToe",
    "RHeel",
)

JOINT_GROUPS: dict[str, tuple[int, ...]] = {
    "torso": (1, 2, 5, 8, 9, 12),
    "head": (0, 15, 16, 17, 18),
    "limbs": (3, 4, 6, 7, 10, 11, 13, 14),
    "feet": (19, 20, 21, 22, 23, 24),
}


def _stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "median": None, "p95": None, "max": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": round(float(np.mean(arr)), 3),
        "median": round(float(np.median(arr)), 3),
        "p95": round(float(np.percentile(arr, 95)), 3),
        "max": round(float(np.max(arr)), 3),
    }


def _project(P: np.ndarray, points_world: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    homo = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float64)], axis=1)
    proj = homo @ np.asarray(P, dtype=np.float64).T
    z = proj[:, 2:3]
    z[np.abs(z) < 1e-9] = 1e-9
    return (proj[:, :2] / z).astype(np.float32)


def _ingress_camera_meta(meta: dict[str, Any] | None) -> dict[str, Any]:
    meta = dict(meta or {})
    intr = dict(meta.get("intrinsics") or {})
    ext = dict(meta.get("extrinsics") or {})
    return {
        "width": meta.get("width"),
        "height": meta.get("height"),
        "scene_capture_flip_u": meta.get("scene_capture_flip_u"),
        "scene_capture_flip_v": meta.get("scene_capture_flip_v"),
        "ue_fov_deg": intr.get("fov_degrees"),
        "ue_rotation_deg": ext.get("ue_rotation_deg"),
        "ue_location_cm": ext.get("ue_location_cm"),
    }


def build_config_snapshot(cfg: Any, *, mode: str, overrides: dict[str, dict[str, Any]]) -> dict[str, Any]:
    pb = dict(cfg.pose_backend or {})
    tri = dict(pb.get("triangulation") or {})
    return {
        "calibration_path": str(cfg.calibration_path),
        "scene_spec_path": str(cfg.scene_spec_path),
        "camera_ids": list(cfg.camera_ids),
        "image_correction_mode": mode,
        "image_correction_overrides": overrides,
        "scale_intrinsics_to_ingress": bool(pb.get("scale_intrinsics_to_ingress", True)),
        "triangulation": tri,
    }


def compute_frame_overlay_diagnostics(
    *,
    frame_index: int,
    camera_ids: list[str],
    kp3d: np.ndarray,
    kp2d_by_cam: dict[str, np.ndarray],
    cam_P: np.ndarray,
    tri_diag: dict[str, Any],
    image_corrections: dict[str, Any],
    metadata_by_camera: dict[str, dict[str, Any]] | None,
    intrinsics_scale: dict[str, Any] | None,
    confidence_threshold: float = 0.28,
) -> dict[str, Any]:
    kp3d = np.asarray(kp3d, dtype=np.float32)
    tri_valid = kp3d[:, 3] > 0.0
    reproj_err = np.asarray(tri_diag.get("reproj_err") or [], dtype=np.float32)
    used_views: list[list[int]] = list(tri_diag.get("used_views") or [])
    view_idx = {cid: i for i, cid in enumerate(camera_ids)}

    per_camera: dict[str, Any] = {}
    worst_cases: list[dict[str, Any]] = []

    for ci, cid in enumerate(camera_ids):
        det = np.asarray(kp2d_by_cam.get(cid), dtype=np.float32)
        det_valid = np.isfinite(det[:, 0]) & (det[:, 2] > 0.0)
        uv_tri = _project(cam_P[ci], kp3d[:, :3])

        red_green_filt: list[float] = []
        red_green_by_joint: dict[str, float | None] = {}
        tri_reproj_by_joint: dict[str, float | None] = {}
        det_ok_tri_excluded: list[str] = []
        group_deltas: dict[str, list[float]] = {g: [] for g in JOINT_GROUPS}
        group_deltas_in_subset: dict[str, list[float]] = {g: [] for g in JOINT_GROUPS}
        group_deltas_out_subset: dict[str, list[float]] = {g: [] for g in JOINT_GROUPS}
        red_green_in_subset: list[float] = []
        red_green_out_subset: list[float] = []

        for j in range(min(25, det.shape[0])):
            jname = BODY25_JOINT_NAMES[j]
            tri_reproj_px = None
            if reproj_err.size and ci < reproj_err.shape[0] and j < reproj_err.shape[1]:
                v = float(reproj_err[ci, j])
                if np.isfinite(v):
                    tri_reproj_px = v
            tri_reproj_by_joint[jname] = None if tri_reproj_px is None else round(tri_reproj_px, 3)

            used = used_views[j] if j < len(used_views) else []
            in_subset = ci in used

            if not bool(det_valid[j]):
                red_green_by_joint[jname] = None
                continue

            if bool(tri_valid[j]):
                du = float(uv_tri[j, 0] - det[j, 0])
                dv = float(uv_tri[j, 1] - det[j, 1])
                d = float((du * du + dv * dv) ** 0.5)
                red_green_filt.append(d)
                red_green_by_joint[jname] = round(d, 3)
                if in_subset:
                    red_green_in_subset.append(d)
                else:
                    red_green_out_subset.append(d)
                for gname, idxs in JOINT_GROUPS.items():
                    if j in idxs:
                        group_deltas[gname].append(d)
                        if in_subset:
                            group_deltas_in_subset[gname].append(d)
                        else:
                            group_deltas_out_subset[gname].append(d)
            else:
                red_green_by_joint[jname] = None

            det_conf_ok = float(det[j, 2]) >= float(confidence_threshold)
            if det_conf_ok and not in_subset and tri_reproj_px is not None and tri_reproj_px > 15.0:
                det_ok_tri_excluded.append(jname)

        per_camera[cid] = {
            "image_correction": image_corrections.get(cid),
            "ingress_meta": _ingress_camera_meta((metadata_by_camera or {}).get(cid)),
            "intrinsics_scale": dict((intrinsics_scale or {}).get(cid) or {}),
            "det_valid": int(np.sum(det_valid)),
            "det_mean_conf": round(float(np.nanmean(det[det_valid, 2])), 4) if np.any(det_valid) else None,
            "red_green_filtered_px": _stats(red_green_filt),
            "red_green_in_tri_subset_px": _stats(red_green_in_subset),
            "red_green_out_tri_subset_px": _stats(red_green_out_subset),
            "red_green_filtered_by_group_px": {g: _stats(v) for g, v in group_deltas.items()},
            "red_green_in_subset_by_group_px": {g: _stats(v) for g, v in group_deltas_in_subset.items()},
            "red_green_out_subset_by_group_px": {g: _stats(v) for g, v in group_deltas_out_subset.items()},
            "tri_reproj_raw_px": _stats(
                [float(reproj_err[ci, j]) for j in range(reproj_err.shape[1]) if np.isfinite(reproj_err[ci, j])]
                if reproj_err.size and ci < reproj_err.shape[0]
                else []
            ),
            "det_good_but_view_excluded_joints": det_ok_tri_excluded,
            "red_green_filtered_by_joint_px": red_green_by_joint,
            "tri_reproj_raw_by_joint_px": tri_reproj_by_joint,
        }

        for jname, d in red_green_by_joint.items():
            if d is not None and d >= 25.0:
                worst_cases.append(
                    {
                        "camera_id": cid,
                        "joint": jname,
                        "red_green_filtered_px": d,
                        "tri_reproj_raw_px": tri_reproj_by_joint.get(jname),
                    }
                )

    view_usage = {cid: 0 for cid in camera_ids}
    for j, used in enumerate(used_views):
        if j >= len(BODY25_JOINT_NAMES) or not bool(tri_valid[j]):
            continue
        for vi in used:
            if 0 <= int(vi) < len(camera_ids):
                view_usage[camera_ids[int(vi)]] += 1

    worst_cases.sort(key=lambda r: float(r["red_green_filtered_px"]), reverse=True)

    return {
        "frame_index": int(frame_index),
        "tri_valid_joints": int(np.sum(tri_valid)),
        "tri_dropped_view_indices": list(tri_diag.get("dropped_view_indices") or []),
        "tri_view_usage_count_among_valid_joints": view_usage,
        "per_camera": per_camera,
        "worst_red_green_cases_px": worst_cases[:12],
    }


def aggregate_run_diagnostics(frame_rows: list[dict[str, Any]], camera_ids: list[str]) -> dict[str, Any]:
    measured = [r for r in frame_rows if r.get("overlay_diagnostics")]
    saved = [r for r in frame_rows if r.get("saved")]
    per_cam_filt: dict[str, list[float]] = {cid: [] for cid in camera_ids}
    per_cam_raw: dict[str, list[float]] = {cid: [] for cid in camera_ids}
    per_cam_in: dict[str, list[float]] = {cid: [] for cid in camera_ids}
    per_cam_out: dict[str, list[float]] = {cid: [] for cid in camera_ids}
    per_cam_excluded: dict[str, int] = defaultdict(int)
    tri_valids: list[int] = []

    for row in measured:
        tri_valids.append(int(row.get("tri_valid_joints") or row.get("tri_valid") or 0))
        diag = dict(row.get("overlay_diagnostics") or {})
        for cid in camera_ids:
            pc = dict((diag.get("per_camera") or {}).get(cid) or {})
            rg = dict(pc.get("red_green_filtered_px") or {})
            rin = dict(pc.get("red_green_in_tri_subset_px") or {})
            rout = dict(pc.get("red_green_out_tri_subset_px") or {})
            tr = dict(pc.get("tri_reproj_raw_px") or {})
            if rg.get("mean") is not None:
                per_cam_filt[cid].append(float(rg["mean"]))
            if rin.get("mean") is not None:
                per_cam_in[cid].append(float(rin["mean"]))
            if rout.get("mean") is not None:
                per_cam_out[cid].append(float(rout["mean"]))
            if tr.get("mean") is not None:
                per_cam_raw[cid].append(float(tr["mean"]))
            per_cam_excluded[cid] += len(pc.get("det_good_but_view_excluded_joints") or [])

    return {
        "saved_frames": len(saved),
        "measured_frames": len(measured),
        "tri_valid_mean": round(float(np.mean(tri_valids)), 2) if tri_valids else None,
        "tri_valid_min": int(min(tri_valids)) if tri_valids else None,
        "tri_valid_max": int(max(tri_valids)) if tri_valids else None,
        "per_camera_red_green_filtered_mean_px": {cid: _stats(per_cam_filt[cid]) for cid in camera_ids},
        "per_camera_red_green_in_tri_subset_mean_px": {cid: _stats(per_cam_in[cid]) for cid in camera_ids},
        "per_camera_red_green_out_tri_subset_mean_px": {cid: _stats(per_cam_out[cid]) for cid in camera_ids},
        "per_camera_tri_reproj_raw_mean_px": {cid: _stats(per_cam_raw[cid]) for cid in camera_ids},
        "det_good_view_excluded_joint_hits": dict(per_cam_excluded),
    }


def write_third_party_review_logs(
    *,
    out_root: Path,
    config_snapshot: dict[str, Any],
    frame_rows: list[dict[str, Any]],
    camera_ids: list[str],
    elapsed_s: float,
    saved_frames: int,
    processed_triplets: int,
) -> tuple[Path, Path]:
    aggregate = aggregate_run_diagnostics(frame_rows, camera_ids)
    review = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Third-party review: red=DWPose 2D; green=robust triangulation 3D reprojected "
            "(in-subset); gray ring=same 3D when this view was excluded from that joint's subset."
        ),
        "legend": {
            "red_green_filtered_px": "Red vs green (all joints with valid 3D).",
            "tri_reproj_raw_px": "Internal robust tri reprojection error for in-subset views.",
            "red_green_in_tri_subset_px": "Red-green when this camera participated in that joint's tri subset.",
            "red_green_out_tri_subset_px": (
                "Red-green when this camera did NOT participate (shown as gray ring on PNG)."
            ),
            "det_good_but_view_excluded_joints": "2D confident but camera dropped from that joint's tri subset.",
        },
        "run": {
            "elapsed_s": round(float(elapsed_s), 3),
            "saved_frames": int(saved_frames),
            "processed_triplets": int(processed_triplets),
            "output_root": str(out_root),
        },
        "config_snapshot": config_snapshot,
        "aggregate": aggregate,
        "saved_frame_diagnostics": [
            {
                "frame_index": r.get("frame_index"),
                "tri_valid_joints": r.get("tri_valid_joints", r.get("tri_valid")),
                "overlay_diagnostics": r.get("overlay_diagnostics"),
            }
            for r in frame_rows
            if r.get("saved")
        ],
    }

    json_path = out_root / "review_log.json"
    json_path.write_text(json.dumps(review, ensure_ascii=True, indent=2), encoding="utf-8")

    lines = [
        "Multiview track overlay — third-party review log",
        f"Generated UTC: {review['generated_at_utc']}",
        "",
        "HOW TO READ PNGs",
        "  Red  = DWPose Body25 2D on ingress-corrected RGB",
        "  Green = robust triangulation 3D reprojected (in-subset for that joint)",
        "  Gray ring = same 3D when this camera was OUT of subset",
        "  red-green ~3-8px on green => geometry healthy; gray far off => out-subset, not bad P",
        "",
        "CONFIG (triangulation)",
    ]
    tri = dict(config_snapshot.get("triangulation") or {})
    for k in (
        "confidence_threshold",
        "min_conf",
        "min_view",
        "min_joints",
        "dist_max_px",
        "thres_outlier_view",
        "thres_outlier_joint",
    ):
        if k in tri:
            lines.append(f"  {k}: {tri[k]}")
    lines.extend(
        [
            "",
            f"RUN: saved_frames={saved_frames} measured_frames={aggregate.get('measured_frames', saved_frames)} processed={processed_triplets} elapsed_s={elapsed_s:.1f}",
            f"Aggregate tri_valid: mean={aggregate.get('tri_valid_mean')} min={aggregate.get('tri_valid_min')} max={aggregate.get('tri_valid_max')}",
            "",
            "PER-CAMERA mean px over saved frames:",
            "  in_subset = camera used in robust tri for that joint; out_subset = consensus from other views",
        ]
    )
    for cid in camera_ids:
        st = dict((aggregate.get("per_camera_red_green_filtered_mean_px") or {}).get(cid) or {})
        st_in = dict((aggregate.get("per_camera_red_green_in_tri_subset_mean_px") or {}).get(cid) or {})
        st_out = dict((aggregate.get("per_camera_red_green_out_tri_subset_mean_px") or {}).get(cid) or {})
        raw = dict((aggregate.get("per_camera_tri_reproj_raw_mean_px") or {}).get(cid) or {})
        lines.append(
            f"  {cid}: all={st.get('mean')} in_subset={st_in.get('mean')} out_subset={st_out.get('mean')} "
            f"raw_tri={raw.get('mean')}"
        )
    lines.extend(["", "WORST FRAMES (top red-green per saved frame):"])
    for row in frame_rows:
        if not row.get("saved"):
            continue
        od = dict(row.get("overlay_diagnostics") or {})
        worst = od.get("worst_red_green_cases_px") or []
        if not worst:
            continue
        w0 = worst[0]
        lines.append(
            f"  frame={row.get('frame_index')} tri_valid={od.get('tri_valid_joints')} "
            f"worst {w0.get('camera_id')}/{w0.get('joint')} {w0.get('red_green_filtered_px')}px "
            f"(raw_tri={w0.get('tri_reproj_raw_px')}px)"
        )

    txt_path = out_root / "review_log.txt"
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, txt_path
