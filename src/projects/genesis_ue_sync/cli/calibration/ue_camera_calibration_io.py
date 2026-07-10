#!/usr/bin/env python3
"""Bidirectional conversion between UE-style camera JSON and cameras.yaml fragments.

Export from UE (recommended): use the same CineCamera / camera actor pose you render with,
after placing it in the level, and dump **world-space** ``location_m`` (meters) plus
**absolute rotation** quaternion in a JSON array (see ``--quat-order``). MRQ / Sequencer
keyframes should be baked to one pose per camera id before export so this matches a single frame rig.

UE JSON (per camera): location_m [x,y,z], quaternion (order via --quat-order), width, height, fov_deg, id.

Pipeline cameras.yaml uses OpenCV-style intrinsics plus world_from_camera / camera_from_world (4x4, meters).

Optional: multiply UE poses by world_from_ue from a genesis_alignment-style YAML before writing cameras.yaml.
Optional: apply a fixed 3x3 basis R_fix post-multiplied on the camera rotation (UE camera axes vs OpenCV).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from bridge.adapters.ue import quaternion_xyzw_from_order, ue_camera_world_pose_from_location_quaternion_m
from bridge.core.camera import build_intrinsics_from_fov
from projects.genesis_ue_sync.tracking.calibration import load_calibration_bundle

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


def _mat4_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (np.asarray(a, dtype=np.float64).reshape(4, 4) @ np.asarray(b, dtype=np.float64).reshape(4, 4)).astype(
        np.float64
    )


def _mat4_inv(t: np.ndarray) -> np.ndarray:
    return np.linalg.inv(np.asarray(t, dtype=np.float64).reshape(4, 4))


def _quat_to_xyzw(q: list[float], order: str) -> np.ndarray:
    return quaternion_xyzw_from_order(q, order)


def _world_from_camera_from_ue_json(
    *,
    location_m: list[float],
    quaternion: list[float],
    quat_order: str,
    world_from_ue: np.ndarray | None,
    camera_basis: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    return ue_camera_world_pose_from_location_quaternion_m(
        location_m=location_m,
        quaternion=quaternion,
        quat_order=quat_order,
        world_from_ue=world_from_ue,
        camera_basis=camera_basis,
    )


def extrinsics_world_camera_from_ue_pose(
    *,
    location_m: Sequence[float],
    quaternion: Sequence[float],
    quat_order: str = "xyzw",
    world_from_ue: np.ndarray | None = None,
    camera_basis: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """World-space camera extrinsics from UE-style location (m) and quaternion."""
    return _world_from_camera_from_ue_json(
        location_m=[float(v) for v in location_m],
        quaternion=[float(v) for v in quaternion],
        quat_order=str(quat_order),
        world_from_ue=world_from_ue,
        camera_basis=camera_basis,
    )


def _load_world_from_ue(path: Path | None) -> np.ndarray | None:
    if path is None:
        return None
    raw = path.read_text(encoding="utf-8")
    if yaml is not None and path.suffix.lower() in {".yaml", ".yml"}:
        payload = yaml.safe_load(raw)
    else:
        payload = json.loads(raw)
    rows = payload.get("world_from_ue")
    if rows is None:
        return None
    return np.asarray(rows, dtype=np.float64).reshape(4, 4)


def cmd_to_yaml(args: argparse.Namespace) -> None:
    if yaml is None:
        raise RuntimeError("PyYAML is required for YAML output. pip install pyyaml")
    payload = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    world_from_ue = _load_world_from_ue(Path(args.world_from_ue)) if args.world_from_ue else None
    basis = None
    if args.camera_basis_json:
        basis = np.asarray(json.loads(Path(args.camera_basis_json).read_text(encoding="utf-8")), dtype=np.float64)
    cameras_out: dict[str, Any] = {}
    for cam in payload.get("cameras", []):
        cid = str(cam["id"])
        w, h = int(cam["width"]), int(cam["height"])
        fov = float(cam["fov_deg"])
        k = build_intrinsics_from_fov(width=w, height=h, fov_deg=fov)
        wfc, cfw = _world_from_camera_from_ue_json(
            location_m=list(cam["location_m"]),
            quaternion=list(cam["quaternion"]),
            quat_order=str(args.quat_order),
            world_from_ue=world_from_ue,
            camera_basis=basis,
        )
        cameras_out[cid] = {
            "image_size": [w, h],
            "intrinsics": k.tolist(),
            "camera_from_world": cfw.tolist(),
            "world_from_camera": wfc.tolist(),
            "distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
            "source": "ue_camera_calibration_io",
            "metadata": {"fov_deg": fov, "quat_order": str(args.quat_order)},
        }
    doc: dict[str, Any] = {
        "metadata": dict(payload.get("metadata", {})),
        "cameras": cameras_out,
    }
    if args.scene_spec:
        doc["scene_spec"] = str(args.scene_spec)
    if args.alignment_path:
        doc["alignment_path"] = str(args.alignment_path)
    out = Path(args.output_yaml)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"Wrote {out}")


def cmd_to_ue_json(args: argparse.Namespace) -> None:
    bundle = load_calibration_bundle(Path(args.input_yaml))
    out_cams = []
    for cid in bundle.ordered_camera_ids():
        cam = bundle.camera(cid)
        wfc = cam.world_from_camera
        r = wfc[:3, :3]
        t = wfc[:3, 3]
        q = Rotation.from_matrix(r).as_quat()
        if str(args.quat_order) == "wxyz":
            q_out = [float(q[3]), float(q[0]), float(q[1]), float(q[2])]
        else:
            q_out = [float(q[0]), float(q[1]), float(q[2]), float(q[3])]
        out_cams.append(
            {
                "id": cid,
                "location_m": [float(v) for v in t.tolist()],
                "quaternion": q_out,
                "quat_order": str(args.quat_order),
                "width": int(cam.width),
                "height": int(cam.height),
                "fov_deg": float(cam.metadata.get("fov_deg", 0.0) or 0.0),
            }
        )
    doc = {"metadata": {"source": "from_cameras_yaml", "calibration_path": str(args.input_yaml)}, "cameras": out_cams}
    outp = Path(args.output_json)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"Wrote {outp}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    p_y = sub.add_parser("to-yaml", help="UE JSON -> cameras.yaml fragment")
    p_y.add_argument("--input-json", type=Path, required=True)
    p_y.add_argument("--output-yaml", type=Path, required=True)
    p_y.add_argument("--world-from-ue", type=Path, default=None, help="YAML/JSON with world_from_ue 4x4.")
    p_y.add_argument("--quat-order", choices=("xyzw", "wxyz"), default="xyzw")
    p_y.add_argument("--camera-basis-json", type=Path, default=None, help="3x3 JSON array: R_cam_cv = R_ue @ basis.")
    p_y.add_argument("--scene-spec", type=str, default=None)
    p_y.add_argument("--alignment-path", type=str, default=None)
    p_y.set_defaults(func=cmd_to_yaml)

    p_j = sub.add_parser("to-ue-json", help="cameras.yaml -> UE JSON")
    p_j.add_argument("--input-yaml", type=Path, required=True)
    p_j.add_argument("--output-json", type=Path, required=True)
    p_j.add_argument("--quat-order", choices=("xyzw", "wxyz"), default="xyzw")
    p_j.set_defaults(func=cmd_to_ue_json)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
