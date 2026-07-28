"""Does binding costal cartilage with LBS instead of station-translation fix the ribs?

The rib sternal-end gate fails by 4-8 mm on the lower attached ribs.  Costal
cartilage is currently baked as ``station_translation``, which keeps translation
and throws away the rib's rotation, so the cartilage cannot follow the rib as it
swings.  This flips the mode in memory only, so the answer is known before paying
for a full template rebake.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace

import numpy as np
from scipy.spatial import cKDTree

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


def _mesh_slice(names, ranges, name):
    index = names.index(name)
    return int(ranges[index, 0]), int(ranges[index, 1])


def _sternal_gap(vertices, names, ranges, rib, targets):
    lo, hi = _mesh_slice(names, ranges, rib)
    rib_points = vertices[lo:hi]
    target_points = np.concatenate(
        [
            vertices[slice(*_mesh_slice(names, ranges, target))]
            for target in targets
            if target in names
        ]
    )
    distances, _ = cKDTree(target_points).query(rib_points, k=1)
    # The gate scores the 27 vertices nearest the target at rest.
    order = np.argsort(distances)[:27]
    return distances, order


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True)
    parser.add_argument("--pose", required=True)
    parser.add_argument(
        "--ribs", default="Rib_8L,Rib_8R,Rib_9L,Rib_9R,Rib_7L,Rib_7R"
    )
    args = parser.parse_args()

    subject = load_subject_asset(args.subject)
    asset = subject.rigged_asset
    names = list(asset.source_mesh_names)
    ranges = np.asarray(asset.source_vertex_ranges, dtype=np.int64)
    tissues = [str(t).lower() for t in asset.source_tissues]
    modes = list(asset.source_mesh_follow_modes)

    flipped = list(modes)
    changed = []
    for index, (tissue, mode) in enumerate(zip(tissues, modes)):
        if tissue == "connective_tissue" and mode == "station_translation":
            flipped[index] = "final_bind_lbs"
            changed.append(names[index])

    lbs_subject = replace(
        subject,
        rigged_asset=replace(asset, source_mesh_follow_modes=flipped),
    )

    pose = _load_pose(args.pose)
    zero = np.zeros((55, 3), dtype=np.float64)
    runs = {}
    for tag, subj in (("station", subject), ("lbs", lbs_subject)):
        runs[tag] = {
            "rest": apply_subject_pose(
                subj, pose_axis_angle=zero, transl=None, validate=False
            ).astype(np.float64),
            "posed": apply_subject_pose(
                subj, pose_axis_angle=pose, transl=None, validate=False
            ).astype(np.float64),
        }

    report: dict[str, object] = {"flipped_meshes": changed}
    rows = {}
    for rib in args.ribs.split(","):
        rib = rib.strip()
        if rib not in names:
            continue
        side = "L" if rib.endswith("L") else "R"
        targets = [f"Costal_Cartilage_{side}", "Sternum"]
        entry = {}
        for tag in ("station", "lbs"):
            rest_d, order = _sternal_gap(
                runs[tag]["rest"], names, ranges, rib, targets
            )
            posed_d, _ = _sternal_gap(
                runs[tag]["posed"], names, ranges, rib, targets
            )
            entry[tag] = {
                "rest_min_mm": float(np.min(rest_d[order])) * 1000.0,
                "posed_min_mm": float(np.min(posed_d[order])) * 1000.0,
                "increase_mm": float(
                    np.max(posed_d[order]) - np.max(rest_d[order])
                )
                * 1000.0,
            }
        rows[rib] = entry
    report["ribs"] = rows
    report["gate_limit_mm"] = 2.0
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
