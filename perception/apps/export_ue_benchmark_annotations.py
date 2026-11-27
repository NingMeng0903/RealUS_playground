#!/usr/bin/env python3
"""Export portable calibration and native-joint GT beside four-camera MP4s."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation


def export_annotations(root: Path) -> dict:
    root = root.resolve()
    records = [json.loads(line) for line in (root / "manifest.jsonl").read_text().splitlines() if line.strip()]
    if not records or any(row.get("status") != "ok" for row in records):
        raise ValueError("Only complete successful capture groups can be exported")
    meta = json.loads((root / records[0]["metadata_path"]).read_text())["by_camera"]
    mirror = np.diag([1., -1., 1.])
    cv_to_ue_local = np.array([[0., 1., 0.], [0., 0., -1.], [1., 0., 0.]])
    cameras = {}
    for name, camera in meta.items():
        pitch, yaw, roll = camera["extrinsics"]["ue_rotation_deg"]
        rotation = (Rotation.from_euler("z", yaw, degrees=True).as_matrix()
                    @ Rotation.from_euler("y", -pitch, degrees=True).as_matrix()
                    @ Rotation.from_euler("x", -roll, degrees=True).as_matrix())
        world_from_camera = np.eye(4)
        world_from_camera[:3, :3] = mirror @ rotation @ cv_to_ue_local.T
        world_from_camera[:3, 3] = mirror @ np.array(camera["extrinsics"]["ue_location_cm"]) / 100
        camera_from_world = np.linalg.inv(world_from_camera)
        cameras[name] = {"K": camera["intrinsics"]["K"], "distortion": [0.] * 5,
                         "width": int(camera["width"]), "height": int(camera["height"]),
                         "world_from_camera": world_from_camera.tolist(),
                         "R_world_to_camera": camera_from_world[:3, :3].tolist(),
                         "t_world_to_camera_m": camera_from_world[:3, 3].tolist()}
    calibration = {"world_convention": "Genesis RH, Z up, metres",
                   "camera_convention": "OpenCV X right, Y down, Z forward",
                   "source": "actual UE capture component metadata", "cameras": cameras}
    (root / "calibration.json").write_text(json.dumps(calibration, indent=2) + "\n")
    with (root / "joints_gt.jsonl").open("w") as output:
        for index, record in enumerate(records):
            canonical = json.loads((root / record["canonical_path"]).read_text())
            gt = json.loads((root / record["gt_path"]).read_text())
            if gt["missing_joints"]:
                raise ValueError("Recording contains incomplete actual UE skeleton GT")
            bones = {bone["joint_name"]: bone for bone in gt["bones"]}
            joints = np.array([bones[name]["world_location_cm"] for name in gt["joint_names"]], dtype=np.float64)
            joints *= np.array([.01, -.01, .01])
            projected = {}
            for name, camera in cameras.items():
                points = joints @ np.array(camera["R_world_to_camera"]).T + camera["t_world_to_camera_m"]
                pixels = points @ np.array(camera["K"]).T
                pixels = pixels[:, :2] / pixels[:, 2:]
                mask = ((points[:, 2] > 0) & (pixels[:, 0] >= 0) & (pixels[:, 0] < camera["width"])
                        & (pixels[:, 1] >= 0) & (pixels[:, 1] < camera["height"]))
                projected[name] = {"pixels": pixels.tolist(), "in_frame": mask.tolist()}
            row = {"video_frame_index": index, "frame_index": record["frame_index"],
                   "capture_group_id": record["capture_group_id"], "joint_names": gt["joint_names"],
                   "joints_world_m": joints.tolist(), "projected_by_camera": projected,
                   "benchmark_time_ns": canonical.get("benchmark_time_ns"),
                   "benchmark_phase": canonical.get("benchmark_phase"),
                   "source_motion_frame_index": (canonical.get("human") or {}).get("motion_frame_index"),
                   "body_shape": gt["body_shape"],
                   "in_frame_definition": "field-of-view bounds only; scene/robot occlusion is not labelled"}
            output.write(json.dumps(row, separators=(",", ":")) + "\n")
    return {"frames": len(records), "cameras": list(cameras),
            "calibration": str(root / "calibration.json"), "joint_gt": str(root / "joints_gt.jsonl")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_annotations(args.recording_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
