#!/usr/bin/env python3
"""Sanity-check multiview cameras.yaml vs optional scene spec and run_meta.

Checks:
  - world_from_camera @ camera_from_world ~ I, det(R)~1, orthogonality
  - fx, fy, cx, cy vs image size
  - Pairwise camera-center distances (baseline sanity)
  - Optional: scene_spec camera pos/lookat/FOV vs bundle (after sync they should match)
  - Optional: run_meta PNG dirs exist and first PNG size matches calibration resolution
  - Round-trip: project a 3D world point to all views, triangulate, mean reprojection error

Usage:
  PYTHONPATH=src python scripts/calibration/check_multiview_calibration.py \\
    --calibration configs/calibration/ue_exec2_bedroom/cameras.yaml \\
    --scene-spec configs/scenes/amass_lie_sync_scene.yaml \\
    --run-meta dataset/demo_video/ue_render_exec2/ue_render/run_meta.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from common.project import project_paths
from projects.genesis_ue_sync.tracking.calibration import load_calibration_bundle
from projects.genesis_ue_sync.tracking.multiview_io import camera_sequences_from_run_meta
from projects.genesis_ue_sync.tracking.triangulation import fundamental_from_calibrations, triangulate_linear


def _horizontal_fov_deg(*, fx: float, width: int) -> float:
    if fx <= 0.0 or width <= 0:
        return float("nan")
    return float(math.degrees(2.0 * math.atan(0.5 * float(width) / float(fx))))


def _project_world(cam, xyz: np.ndarray) -> tuple[float, float, bool]:
    Pw = np.asarray(xyz, dtype=np.float64).reshape(3)
    hom = np.concatenate([Pw, [1.0]], axis=0)
    proj = np.asarray(cam.intrinsics, dtype=np.float64) @ np.asarray(cam.camera_from_world, dtype=np.float64)[:3, :] @ hom
    z = float(proj[2])
    if z <= 1e-9:
        return float("nan"), float("nan"), False
    return float(proj[0] / z), float(proj[1] / z), True


def _scene_lookat_from_wfc(cam) -> np.ndarray:
    center = cam.camera_center_world
    forward = cam.world_from_camera[:3, 2].astype(np.float64)
    return center + forward


def _scene_up_from_wfc(cam) -> np.ndarray:
    return (-cam.world_from_camera[:3, 1]).astype(np.float64)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--calibration", type=Path, required=True, help="cameras.yaml path (repo-relative or absolute).")
    p.add_argument("--scene-spec", type=Path, default=None, help="Sync scene spec for cross-check (optional).")
    p.add_argument("--run-meta", type=Path, default=None, help="run_meta.json to verify PNG resolution vs calibration (optional).")
    p.add_argument("--json", action="store_true", help="Print one JSON object only (for tooling).")
    args = p.parse_args()

    root = project_paths(__file__).root
    cal_path = Path(args.calibration)
    if not cal_path.is_absolute():
        cal_path = (root / cal_path).resolve()
    scene_path = None
    if args.scene_spec is not None:
        scene_path = Path(args.scene_spec)
        if not scene_path.is_absolute():
            scene_path = (root / scene_path).resolve()

    bundle = load_calibration_bundle(cal_path, scene_spec_path=scene_path)
    issues: list[str] = []
    warnings: list[str] = []
    details: dict[str, Any] = {"calibration_path": str(cal_path), "cameras": {}}

    ids = bundle.ordered_camera_ids()
    if len(ids) < 2:
        issues.append(f"Need at least two cameras for multiview checks, got {len(ids)}.")

    # Per-camera matrix / intrinsics
    for cid in ids:
        cam = bundle.camera(cid)
        R = cam.camera_from_world[:3, :3].astype(np.float64)
        t = cam.camera_from_world[:3, 3].astype(np.float64)
        wfc = cam.world_from_camera.astype(np.float64)
        cfw = cam.camera_from_world.astype(np.float64)
        prod = wfc @ cfw
        frob = float(np.linalg.norm(prod - np.eye(4)))
        det = float(np.linalg.det(R))
        ortho = float(np.linalg.norm(R.T @ R - np.eye(3)))
        K = cam.intrinsics
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        w, h = cam.width, cam.height
        fov_k = _horizontal_fov_deg(fx=fx, width=w)
        entry: dict[str, Any] = {
            "image_size": [w, h],
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "horizontal_fov_deg_from_fx": fov_k,
            "camera_center_world": cam.camera_center_world.tolist(),
            "world_from_camera_times_camera_from_world_frob": frob,
            "det_camera_rotation": det,
            "rotation_orthogonality_frob": ortho,
        }
        if frob > 1e-4:
            issues.append(f"{cid}: world_from_camera @ camera_from_world not identity (frob={frob:.2e}).")
        if abs(det - 1.0) > 1e-3:
            issues.append(f"{cid}: det(camera R)={det:.6f} (expected ~1).")
        if ortho > 1e-4:
            issues.append(f"{cid}: R not orthogonal (frob={ortho:.2e}).")
        if fx <= 0 or fy <= 0:
            issues.append(f"{cid}: non-positive focal length.")
        if not (-0.5 <= cx <= w + 0.5) or not (-0.5 <= cy <= h + 0.5):
            warnings.append(f"{cid}: principal point ({cx:.1f},{cy:.1f}) outside image [0,{w})x[0,{h}).")
        details["cameras"][cid] = entry

    # Pairwise baselines + fundamental norm
    baselines: dict[str, float] = {}
    for i, a in enumerate(ids):
        ca = bundle.camera(a)
        for b in ids[i + 1 :]:
            cb = bundle.camera(b)
            d = float(np.linalg.norm(ca.camera_center_world - cb.camera_center_world))
            baselines[f"{a}<->{b}"] = d
            F = fundamental_from_calibrations(ca, cb)
            baselines[f"{a}<->{b}_F_norm"] = float(np.linalg.norm(F))
    details["pairwise_camera_center_distance_m"] = {k: v for k, v in baselines.items() if not k.endswith("_F_norm")}
    details["pairwise_F_frobenius"] = {k: v for k, v in baselines.items() if k.endswith("_F_norm")}

    # Scene spec cross-check
    if scene_path is not None and bundle.scene_spec is not None:
        scene = bundle.scene_spec
        by_name = {c.name: c for c in scene.cameras}
        scene_cmp: dict[str, Any] = {}
        for cid in ids:
            sc = by_name.get(cid)
            if sc is None:
                issues.append(f"Scene spec has no camera named '{cid}' (have {list(by_name.keys())}).")
                continue
            cam = bundle.camera(cid)
            pos_e = np.asarray(sc.pos, dtype=np.float64).reshape(3)
            pos_c = cam.camera_center_world.astype(np.float64).reshape(3)
            pos_err = float(np.linalg.norm(pos_e - pos_c))
            look_e = np.asarray(sc.lookat, dtype=np.float64).reshape(3)
            look_c = _scene_lookat_from_wfc(cam).reshape(3)
            look_err = float(np.linalg.norm(look_e - look_c))
            fov_scene = float(sc.fov)
            fov_k = _horizontal_fov_deg(fx=float(cam.intrinsics[0, 0]), width=int(cam.width))
            fov_diff = abs(fov_scene - fov_k)
            scene_cmp[cid] = {
                "position_l2_m": pos_err,
                "lookat_l2_m": look_err,
                "fov_scene_deg": fov_scene,
                "fov_from_intrinsics_deg": fov_k,
                "abs_fov_diff_deg": fov_diff,
            }
            if pos_err > 0.02:
                issues.append(f"{cid}: scene pos vs calibration center mismatch {pos_err:.3f} m (>2cm).")
            if look_err > 0.05:
                issues.append(f"{cid}: scene lookat vs calibration lookat mismatch {look_err:.3f} m (>5cm).")
            if fov_diff > 1.0:
                warnings.append(f"{cid}: FOV scene={fov_scene:.2f} vs from fx={fov_k:.2f} (diff {fov_diff:.2f} deg).")
        details["scene_spec_cross_check"] = scene_cmp

    # run_meta PNG resolution
    if args.run_meta is not None:
        rm = Path(args.run_meta)
        if not rm.is_absolute():
            rm = (root / rm).resolve()
        seqs = camera_sequences_from_run_meta(rm)
        rm_details: dict[str, Any] = {}
        for seq in seqs:
            cid = seq.camera_id
            if cid not in ids:
                warnings.append(f"run_meta camera_id '{cid}' not in calibration keys {ids}.")
                continue
            cam = bundle.camera(cid)
            if not seq.frames:
                issues.append(f"{cid}: no positive PNG frames under {seq.png_dir}")
                continue
            _, path0 = seq.frames[0]
            im = Image.open(path0)
            rw, rh = im.size
            ew, eh = cam.width, cam.height
            rm_details[cid] = {"first_png": str(path0), "png_size": [rw, rh], "calibration_size": [ew, eh]}
            if (rw, rh) != (ew, eh):
                issues.append(f"{cid}: PNG size {rw}x{rh} != calibration {ew}x{eh}.")
        details["run_meta"] = {"path": str(rm), "per_camera": rm_details}

    # Round-trip triangulation using a world point in front of all cameras
    if len(ids) >= 2:
        centers = np.stack([bundle.camera(c).camera_center_world for c in ids], axis=0)
        test_pt = np.mean(centers, axis=0) + np.array([0.0, 0.0, 0.5], dtype=np.float64)
        obs = []
        valid_all = True
        for cid in ids:
            cam = bundle.camera(cid)
            u, v, ok = _project_world(cam, test_pt)
            if not ok:
                valid_all = False
                issues.append(f"{cid}: test world point behind camera or invalid projection.")
                break
            obs.append((cam, (u, v)))
        if valid_all and len(obs) >= 2:
            xyz_hat, reproj = triangulate_linear(obs)
            err = float(np.linalg.norm(xyz_hat - test_pt.astype(np.float32)))
            details["triangulation_roundtrip"] = {
                "test_point_world": test_pt.tolist(),
                "triangulated_world": xyz_hat.tolist(),
                "l2_world_error_m": err,
                "mean_reprojection_px": reproj,
            }
            if err > 0.05:
                issues.append(f"Triangulation round-trip world error {err:.3f} m (>5cm) — extrinsics/intrinsics inconsistent.")
            if reproj > 2.0:
                warnings.append(f"Triangulation mean reprojection {reproj:.2f} px (>2).")

    ok = len(issues) == 0
    details["ok"] = ok
    details["issues"] = issues
    details["warnings"] = warnings
    details["convention"] = bundle.convention.as_dict()

    if args.json:
        print(json.dumps(details, indent=2))
        raise SystemExit(0 if ok else 1)

    print(json.dumps(details, indent=2))
    if issues:
        print("\nFAIL:", len(issues), "issue(s)")
        for line in issues:
            print(" -", line)
    else:
        print("\nPASS: no hard issues.")
    if warnings:
        print("\nWarnings:", len(warnings))
        for line in warnings:
            print(" -", line)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
