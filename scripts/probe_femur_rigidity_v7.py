"""Locate why the femur mesh stops being rigid during the synthetic knee sweep.

The acceptance sweep reports femur edge-length ratios up to ~2.9x at 10 deg of
knee flexion while the tibia and patella stay rigid, which points at the femur
being blended between two disagreeing drivers rather than carried by one.
"""

from __future__ import annotations

import argparse

import numpy as np

from projects.genesis_ue_sync.anatomy_retarget.acceptance_matrix_v7 import (
    synthetic_knee_sweep_poses_v7,
)
from projects.genesis_ue_sync.anatomy_retarget.v7_artifacts import (
    apply_subject_pose,
    load_subject_asset,
)


def _mesh_edges(faces: np.ndarray, lo: int, hi: int) -> np.ndarray:
    tris = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    keep = np.all((tris >= lo) & (tris < hi), axis=1)
    tris = tris[keep]
    edges = np.concatenate(
        (tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]), axis=0
    )
    return np.unique(np.sort(edges, axis=1), axis=0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True)
    parser.add_argument("--mesh", default="Femur_L")
    parser.add_argument("--count", type=int, default=13)
    args = parser.parse_args()

    subject = load_subject_asset(args.subject)
    asset = subject.rigged_asset
    names = [str(name) for name in asset.source_mesh_names]
    ranges = np.asarray(asset.source_vertex_ranges, dtype=np.int64)
    index = names.index(args.mesh)
    lo, hi = int(ranges[index][0]), int(ranges[index][1])
    edges = _mesh_edges(asset.faces, lo, hi)
    rest = np.asarray(asset.vertices_rest, dtype=np.float64)
    rest_lengths = np.linalg.norm(rest[edges[:, 0]] - rest[edges[:, 1]], axis=1)
    usable = rest_lengths > 1.0e-9
    edges = edges[usable]
    rest_lengths = rest_lengths[usable]

    print(f"{args.mesh}: vertices [{lo},{hi}) edges={len(edges)}")
    offsets = np.asarray(asset.source_influence_offsets, dtype=np.int64)
    group_ids = np.asarray(asset.source_influence_group_indices, dtype=np.int64)
    values = np.asarray(asset.source_influence_values, dtype=np.float64)
    group_bone = np.asarray(asset.source_group_bone_indices, dtype=np.int64)
    bone_names = [str(name) for name in asset.source_bone_names]
    span = slice(int(offsets[lo]), int(offsets[hi]))
    bones = group_bone[group_ids[span]]
    totals: dict[int, float] = {}
    for bone, value in zip(bones, values[span]):
        totals[int(bone)] = totals.get(int(bone), 0.0) + float(value)
    print("  drivers:")
    for bone, total in sorted(totals.items(), key=lambda kv: -kv[1])[:8]:
        print(f"    {bone_names[bone]}: total_weight={total:.3f}")
    counts = np.diff(offsets[lo : hi + 1])
    print(
        f"  influences per vertex: min={counts.min()} max={counts.max()}"
        f" mean={counts.mean():.2f}"
    )
    print(f"  follow_mode={asset.source_mesh_follow_modes[index]}")
    print(f"  controller_bone={bone_names[int(asset.source_mesh_controller_bones[index])]}")

    for spec in synthetic_knee_sweep_poses_v7(count=int(args.count)):
        posed = np.asarray(
            apply_subject_pose(
                subject,
                pose_axis_angle=spec.pose_axis_angle,
                transl=spec.transl,
                validate=False,
            ),
            dtype=np.float64,
        )
        lengths = np.linalg.norm(
            posed[edges[:, 0]] - posed[edges[:, 1]], axis=1
        )
        ratio = lengths / rest_lengths
        worst = int(np.argmax(np.abs(np.log(np.maximum(ratio, 1.0e-12)))))
        print(
            f"  {spec.label}: ratio_min={ratio.min():.4f}"
            f" ratio_max={ratio.max():.4f}"
            f" worst_edge={edges[worst].tolist()}"
            f" rest_len={rest_lengths[worst] * 1000.0:.2f}mm"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
