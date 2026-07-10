#!/usr/bin/env python3
"""Compare LevelSequence camera dump (ue_dump_sequencer_camera_poses.py) to cameras.yaml.

The dump stores the CineCamera actor world rotation as a quaternion. That 3x3 is **not**
identical to ``world_from_camera[:3,:3]`` in ``cameras.yaml`` (which follows OpenCV /
``build_camera_from_scene_camera``): the usual relation is a **fixed column permutation**
(UE body axes vs OpenCV camera axes) plus **per-column sign flips** (left/right vs top).

This script aligns ``R_ue`` to ``R_yaml`` by searching sign patterns on columns
``[s1*R_ue[:,1], s2*R_ue[:,2], s3*R_ue[:,0]]`` and reports the residual Frobenius norm.

Exit code 1 if aligned rotation error or camera-center error exceeds thresholds.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import itertools

import numpy as np

_THIS_FILE = Path(__file__).resolve()
_SRC = next(parent for parent in (_THIS_FILE.parent, *_THIS_FILE.parents) if parent.name == "src")
_REPO = _SRC.parent
for p in (_SRC, _THIS_FILE.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from projects.genesis_ue_sync.tracking.calibration import load_calibration_bundle
from ue_camera_calibration_io import _load_world_from_ue, extrinsics_world_camera_from_ue_pose


def _frob(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64), ord="fro"))


def _align_ue_actor_rotation_to_yaml_columns(r_ue: np.ndarray, r_yaml: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Return R_try closest to r_yaml using columns [s1*r_ue[:,1], s2*r_ue[:,2], s3*r_ue[:,0]]."""
    r_ue = np.asarray(r_ue, dtype=np.float64).reshape(3, 3)
    r_yaml = np.asarray(r_yaml, dtype=np.float64).reshape(3, 3)
    best_r = r_ue
    best_err = float("inf")
    best_signs = (1, 1, 1)
    for s1, s2, s3 in itertools.product((-1, 1), repeat=3):
        r_try = np.column_stack([s1 * r_ue[:, 1], s2 * r_ue[:, 2], s3 * r_ue[:, 0]])
        err = _frob(r_try, r_yaml)
        if err < best_err:
            best_err = err
            best_r = r_try
            best_signs = (s1, s2, s3)
    return best_r, best_signs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dump-json", type=Path, required=True)
    p.add_argument("--calibration", type=Path, required=True)
    p.add_argument("--world-from-ue", type=Path, default=None, help="genesis_alignment.yaml or JSON with world_from_ue")
    p.add_argument("--camera-basis-json", type=Path, default=None)
    p.add_argument(
        "--rot-threshold",
        type=float,
        default=1e-4,
        help="Frobenius norm on 3x3 R after UE-vs-OpenCV column alignment (see script docstring).",
    )
    p.add_argument(
        "--raw-rot-threshold",
        type=float,
        default=None,
        help="If set, also require naive quaternion->R Frobenius error below this (usually fails).",
    )
    p.add_argument("--t-threshold", type=float, default=5e-3, help="L2 meters on camera center in world")
    args = p.parse_args()

    basis = None
    if args.camera_basis_json is not None:
        basis = np.asarray(json.loads(args.camera_basis_json.read_text(encoding="utf-8")), dtype=np.float64).reshape(3, 3)
    world_from_ue = _load_world_from_ue(args.world_from_ue) if args.world_from_ue else None

    dump = json.loads(args.dump_json.read_text(encoding="utf-8"))
    bundle = load_calibration_bundle(args.calibration)

    issues: list[str] = []
    report: dict[str, dict[str, float | tuple[int, int, int]]] = {}

    for cam in dump.get("cameras", []):
        cid = str(cam["id"])
        if cid not in bundle.cameras:
            issues.append(f"dump camera {cid!r} not in calibration bundle")
            continue
        q = cam.get("quaternion")
        loc = cam.get("location_m")
        if not isinstance(q, list) or len(q) != 4:
            issues.append(f"{cid}: bad quaternion")
            continue
        if not isinstance(loc, list) or len(loc) != 3:
            issues.append(f"{cid}: bad location_m")
            continue
        order = str(cam.get("quat_order", "xyzw"))
        wfc_ue, _ = extrinsics_world_camera_from_ue_pose(
            location_m=loc,
            quaternion=q,
            quat_order=order,
            world_from_ue=world_from_ue,
            camera_basis=basis,
        )
        wfc_yaml = bundle.camera(cid).world_from_camera
        r_ue = wfc_ue[:3, :3]
        r_yaml = wfc_yaml[:3, :3]
        r_aligned, signs = _align_ue_actor_rotation_to_yaml_columns(r_ue, r_yaml)
        r_err_raw = _frob(r_ue, r_yaml)
        r_err = _frob(r_aligned, r_yaml)
        c_ue = wfc_ue[:3, 3]
        c_yaml = wfc_yaml[:3, 3]
        t_err = float(np.linalg.norm(c_ue - c_yaml))
        report[cid] = {
            "rotation_frobenius_raw": r_err_raw,
            "rotation_frobenius_aligned": r_err,
            "ue_column_signs_on_perm_1_2_0": signs,
            "camera_center_l2_m": t_err,
        }
        if r_err > float(args.rot_threshold):
            issues.append(f"{cid}: aligned R frobenius {r_err:.6g} > {args.rot_threshold}")
        if args.raw_rot_threshold is not None and r_err_raw > float(args.raw_rot_threshold):
            issues.append(f"{cid}: raw R frobenius {r_err_raw:.6g} > {args.raw_rot_threshold}")
        if t_err > float(args.t_threshold):
            issues.append(f"{cid}: camera center L2 {t_err:.6g} m > {args.t_threshold}")

    def _jsonify(obj: object) -> object:
        if isinstance(obj, dict):
            return {k: _jsonify(v) for k, v in obj.items()}
        if isinstance(obj, tuple):
            return list(obj)
        return obj

    out = {"report": _jsonify(report), "issues": issues, "ok": len(issues) == 0}
    print(json.dumps(out, indent=2))
    if issues:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
