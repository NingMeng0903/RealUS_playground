"""Break down anatomy-vs-SMPL-X-body containment per mesh.

``vessel.containment`` reports the whole skeleton exiting the body surface, which
invalidates the vessel reference.  This locates the protrusion: a handful of
local regions means a fit/authoring problem, while a broad spread means a global
scale mismatch between the anatomy and the SMPL-X body.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from projects.genesis_ue_sync.anatomy_retarget.containment import signed_distance
from projects.genesis_ue_sync.anatomy_retarget.pose_adapter import (
    _default_smplx_model_path,
)
from projects.genesis_ue_sync.anatomy_retarget.smplx_body_surface_v7 import (
    load_smplx_model_v7,
    smplx_body_surface_v7,
)
from projects.genesis_ue_sync.anatomy_retarget.v7_artifacts import (
    apply_subject_pose,
    load_subject_asset,
)


def _load_pose(spec: str) -> np.ndarray:
    if spec == "zero":
        return np.zeros((55, 3), dtype=np.float64)
    return np.asarray(
        np.load(spec)["pose_axis_angle"], dtype=np.float64
    ).reshape(55, 3)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True)
    parser.add_argument("--pose", default="zero")
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()

    subject = load_subject_asset(args.subject)
    asset = subject.rigged_asset
    pose = _load_pose(args.pose)
    vertices = apply_subject_pose(
        subject, pose_axis_angle=pose, transl=None, validate=False
    ).astype(np.float64)

    model = load_smplx_model_v7(_default_smplx_model_path(str(subject.gender)))
    body_vertices, body_faces = smplx_body_surface_v7(
        model,
        betas=np.asarray(subject.betas, dtype=np.float64).reshape(-1),
        pose_axis_angle=pose,
        transl=None,
    )
    distances, _, _ = signed_distance(vertices, body_vertices, body_faces)

    names = list(asset.source_mesh_names or [])
    ranges = np.asarray(asset.source_vertex_ranges, dtype=np.int64)
    rows = []
    for index, name in enumerate(names):
        start, end = int(ranges[index, 0]), int(ranges[index, 1])
        if end <= start:
            continue
        chunk = distances[start:end]
        rows.append(
            {
                "mesh": name,
                "vertices": int(end - start),
                "inside_ratio": float(np.mean(chunk <= 0.0)),
                "max_outside_mm": float(np.max(chunk)) * 1000.0,
                "p99_outside_mm": float(np.percentile(chunk, 99.0)) * 1000.0,
            }
        )
    rows.sort(key=lambda row: -row["max_outside_mm"])

    summary = {
        "pose": args.pose,
        "overall_inside_ratio": float(np.mean(distances <= 0.0)),
        "overall_max_outside_mm": float(np.max(distances)) * 1000.0,
        "meshes_outside_5mm": int(
            sum(1 for row in rows if row["max_outside_mm"] > 5.0)
        ),
        "mesh_count": len(rows),
        "worst": rows[: args.top],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
